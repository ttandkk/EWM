"""Eight-task runner wiring on CPU; GPU physics is deliberately not mocked as evidence."""
import json
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
from collections import deque

import numpy as np
import pytest
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'Event-DataCollection/scripts'))
from piper_tasks import load_task
from piper_tasks.dynamic_collection import TASKS, VERSION, make_plan, physics_hz
from piper_tasks.collection_protocol import base_config
from piper_tasks.dynamic_evaluation import supported_conditions, scene_manifest
from piper_tasks.evaluation_imaging import TrialImaging
from scripts.eval_ewm import eval_ewm_piper_tasks_sim as evaluator
from scripts.eval_ewm.ewm_eval_wire import encode_message, decode_message

CELLS=[(t,c) for t in TASKS for c in supported_conditions(t)]


@pytest.mark.parametrize('task,condition,latency',[(t,c,False) for t,c in CELLS]+[('tennis_container','id',True)])
def test_dynamic_runner_inputs_hooks_and_failed_trial_denominator(tmp_path,monkeypatch,task,condition,latency):
    import websockets.sync.client
    import piper_tasks.collection_runtime as capture
    import piper_tasks.tennis.collection as tennis
    import piper_tasks.event_preview as sensors
    import piper_tasks.lighting as lighting
    import piper_tasks.evaluation_imaging as imaging
    import piper_tasks.ood_protocol as backgrounds
    module=load_task(task);plan=make_plan(task,50,20261021,'train')
    collection=dict(version=plan['version'],task=task,base_config=base_config(task,plan).metadata(),plan=plan,
                    camera=module.camera_metadata(),
                    pathtracing_spp=64,physics_hz=physics_hz(task),success_version='test')
    if task.startswith('tennis_'):
        # An old training export must not mislabel the current render camera.
        collection['camera'] = dict(collection['camera'], calibration_id='historical_camera')
    scenes=scene_manifest(collection,trials=2,condition=condition)
    checkpoint=tmp_path/'checkpoint';checkpoint.mkdir()
    (checkpoint/'config.yaml').write_text('model: {}')
    (checkpoint/'checkpoint_step_000010.pt').write_bytes(b'mock')
    ready=tmp_path/'ready.json';ready.write_text('{"url":"ws://test"}')
    out=tmp_path/'eval'
    args=evaluator.build_parser().parse_args(['--task',task,'--condition',condition,'--checkpoint-dir',str(checkpoint),
        '--ready-file',str(ready),'--output-dir',str(out),'--trials','2','--max-steps','4','--interface-only',
        '--latency-aware' if latency else '--no-latency-aware',
        '--latency-controller', 'continuous' if latency else 'serial'])
    evaluator.shared.validate_args(args)
    calls=[];applied=[];background_calls=[]
    scene={name:SimpleNamespace(data=SimpleNamespace(output={'rgb':torch.zeros((1,256,256,4),dtype=torch.uint8)})) for name in ('cam0','cam_wrist')}
    scene['robot']=SimpleNamespace(data=SimpleNamespace(joint_pos=torch.zeros((1,8))))
    env=SimpleNamespace(device='cpu',scene=scene,closed=False,sim=SimpleNamespace())
    env.reset=lambda **kw:None;env.step=lambda action:calls.append(('physics',float(action[0,-1])))
    env.close=lambda:setattr(env,'closed',True)
    runtime=SimpleNamespace(env=env,cfg=module.Config(**collection['base_config']),time=0.,stable_steps=0,
        tracker=SimpleNamespace(window_samples=16,steps=0),initial_gripper_action=-1. if task=='tennis_container' else 1.)
    runtime.before_step=lambda *a:calls.append(('before',len(a)))
    monkeypatch.setitem(sys.modules,'gymnasium',SimpleNamespace(make=lambda *a,**k:SimpleNamespace(unwrapped=env)))
    monkeypatch.setitem(sys.modules,'omni',SimpleNamespace(usd=SimpleNamespace(get_context=lambda:SimpleNamespace(get_stage=lambda:None))))
    monkeypatch.setitem(sys.modules,'omni.usd',sys.modules['omni'].usd)
    monkeypatch.setattr(evaluator,'register_tasks',lambda:None)
    monkeypatch.setattr(evaluator,'load_task',lambda name:SimpleNamespace(TASK=module.TASK,PROMPT=module.PROMPT,
        Runtime=lambda *a:runtime,make_env_cfg=lambda *a,**k:SimpleNamespace(scene=scene),
        camera_metadata=module.camera_metadata,set_views=lambda env:None))
    monkeypatch.setattr(backgrounds,'configure_background',lambda scene, **kw:background_calls.append(scene) or {'changed':True})
    class Lights:
        def __init__(self,stage):pass
        def apply(self,factor):applied.append(factor);return {'multiplier':factor}
        def verify(self,factor):assert factor==applied[-1]
    monkeypatch.setattr(lighting,'FixedSceneLighting',Lights)
    monkeypatch.setattr(evaluator.shared,'configure_renderer',lambda *a:None)
    dvs=SimpleNamespace(recorder=SimpleNamespace(_events={}),reset=lambda x:None,warp_and_process=lambda *a:None)
    monkeypatch.setattr(sensors,'create_event_camera',lambda *a:dvs)
    monkeypatch.setattr(evaluator.shared,'snapshot',lambda *a:{'snapshot':True})
    original_tensor=torch.tensor
    monkeypatch.setattr(torch,'tensor',lambda *a,**kw:original_tensor(*a,**(dict(kw,device='cpu') if kw.get('device')=='cuda:0' else kw)))
    initialized=[]
    def initialize(rt,spec):
        initialized.append(spec);rt.time=0.;rt.trial=spec['slot'];return {'object_poses':spec['object_poses']}
    def step_result(rt,name,action):
        rt.time+=1/30;done=rt.time>=3/30-1e-9
        return done and rt.trial==1,None,False
    monkeypatch.setattr(capture,'initialize',initialize)
    monkeypatch.setattr(capture,'step_result',step_result)
    monkeypatch.setattr(capture,'terminal_state',lambda *a:{'stable_steps':0})
    monkeypatch.setattr(capture,'terminal_valid',lambda *a:True)
    monkeypatch.setattr(tennis,'success_sample',lambda rt:{'time':rt.time})
    class Blur:
        def __init__(self,dvs):self.tick=0
        def __call__(self,frame,snapshot):
            assert snapshot=={'snapshot':True}
            assert abs(frame['observation.sim_time'][0]-self.tick/30)<1e-8
            frame['_blurred_rgb']={n:{c:frame['observation.images.'+c]+n for c in ('cam0','cam_wrist')} for n in range(4,9)}
            self.tick+=1
    monkeypatch.setattr(imaging,'RGBExposureObserver',Blur)
    class Video:
        def __init__(self,path,*a,**kw):self.path=path;self.ffmpeg='mock'
        def write(self,*a,**kw):pass
        def close(self):self.path.write_bytes(b'mock')
    monkeypatch.setattr(evaluator.shared,'TrialVideo',Video)
    monkeypatch.setattr(evaluator.subprocess,'run',lambda *a,**kw:None)
    # Keep the real post-trial protocol active; only the fake video's encoder
    # is replaced. The server fixture must advertise the current default API.
    from scripts.eval_ewm import ewm_prediction_recording as recording
    def append_predictions(path, rows, *, decode, **kwargs):
        anchors = {target[0] for row in rows if (target := recording.prediction_target(row)) is not None}
        for anchor in anchors:
            value, metadata = decode(anchor)
            assert value.shape == (24, 2, 2, 256, 256)
            assert metadata['source_observation_step'] == anchor
        return dict(decoded_predictions=len(anchors))
    monkeypatch.setattr(recording, 'append_prediction_video', append_predictions)
    inputs=[]
    class Connection:
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def send(self,message):
            self.header,arrays=decode_message(message)
            if self.header['op']=='predict':
                inputs.append(arrays)
                assert not any('object' in k or 'launch' in k for k in arrays)
                expected=scenes['episodes'][runtime.trial]['imaging'].get('intervals',0) if condition=='blur_ood' else 0
                assert np.all(arrays['obs_video']==expected)
                assert not np.any(arrays['obs_voxel'])
        def recv(self,**kw):
            if self.header['op']=='hello':
                return encode_message(dict(ok=True,checkpoints=[10],horizon=24,
                    config_sha256=hashlib.sha256((checkpoint/'config.yaml').read_bytes()).hexdigest(),normalization_sha256='test',
                    checkpoint_sha256={'10': hashlib.sha256(b'mock').hexdigest()},
                    event_time_contract=evaluator.shared.EVENT_TIME_CONTRACT,inference_observation_row_offset=0,
                    training_observation_row_offset=0,event_camera_mode='dual',event_cameras=2,voxel_bins=1,count_clip=15,
                    deferred_visualization_supported=True,post_trial_visualization_supported=True))
            if self.header['op']=='visualization_begin':
                self.session=self.header['session'];self.recorded=0
                return encode_message(dict(ok=True))
            if self.header['op']=='visualization_end':
                assert self.header['session']==self.session
                return encode_message(dict(ok=True,recorded_predictions=self.recorded,peak_archive_bytes=0))
            if self.header['op']=='visualization_decode':
                assert self.header['session']==self.session
                return encode_message(dict(ok=True,source_observation_step=self.header['observation_step'],prediction_kind='events'),
                                      {'future_voxel':np.zeros((24,2,2,256,256),np.float32)})
            self.recorded+=1
            return encode_message(dict(ok=True,inference_seconds=.1 if latency else .001,prediction_recorded=True),{'executed_actions':np.zeros((self.header['action_chunk_size'],7),np.float32)})
    monkeypatch.setattr(websockets.sync.client,'connect',lambda *a,**kw:Connection())
    dataset = tmp_path / 'dataset'
    (dataset / 'meta').mkdir(parents=True)
    (dataset / 'meta/piper_collection_config.json').write_text(json.dumps(collection))
    evaluator.run(args,dataset,collection,runtime.cfg,scenes)
    result=json.loads((out/'complete.json').read_text())
    assert result['trials']==2 and result['successes']==1 and result['success_rate'] is None
    assert initialized==scenes['episodes'] and len(inputs)>=2 and env.closed
    assert result['results'][0]['steps']==4 and result['results'][0]['reason']=='step_limit'
    assert result['results'][1]['steps']==3
    if condition == 'speed_ood':
        assert result['breakdown'] == {
            'slow': dict(trials=1, successes=0, success_rate=None),
            'fast': dict(trials=1, successes=1, success_rate=None),
        }
        summary = (out/'summary.txt').read_text()
        assert 'speed_ood_slow: successes=0, completed=1, requested=1, success_rate=None' in summary
        assert 'speed_ood_fast: successes=1, completed=1, requested=1, success_rate=None' in summary
    for r in result['results']:
        assert r['event_archive']['frames']==r['steps']+1
        assert Path(r['event_archive']['directory'],'cam0_raw.npz').is_file()
        assert r['prediction_recording']['prediction_decode_phase']=='post_trial'
        assert r['prediction_recording']['prediction_decode_in_equivalent_latency'] is False
    assert bool(background_calls)==(condition=='background_ood')
    assert applied==[r['imaging'].get('intensity_multiplier',1.) for r in scenes['episodes']]
    if task.startswith('tennis_'):
        assert sum(c==('before',1) for c in calls)==7
        assert len(runtime.collection_catch_window)==3
    if task=='tennis_container':assert calls[0]==('physics',-1.)
    if latency:
        rows=json.loads((out/'trial_000_steps.json').read_text())
        assert rows[0]['action']==[0.,0.,0.,0.,0.,0.,-1.]
        assert result['results'][0]['latency_timing']['bootstrap_wait_steps'] >= 1
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['physics_hz']==physics_hz(task)
    if task.startswith('tennis_'):
        from piper_tasks.tennis.camera import camera_metadata_for_config
        assert manifest['camera_calibration'] == camera_metadata_for_config(runtime.cfg)
        assert manifest['collection_camera_calibration'] == collection['camera']
    else:
        assert manifest['camera_calibration'] == collection['camera']


def test_suite_plan_resume_guards_and_no_gpu(tmp_path, monkeypatch):
    from piper_bench.evaluation import suite
    def settings(args):
        args.max_steps=150
        return None,None,None,dict(task=args.task,condition=args.condition,episodes=[])
    monkeypatch.setattr(suite.client,'load_settings',settings)
    monkeypatch.setattr(suite,'source_hashes',lambda *a:{'source':'unchanged'})
    monkeypatch.setattr(suite.subprocess,'run',lambda *a,**kw:pytest.fail('plan-only must not execute a process'))
    args=suite.parser().parse_args(['--plan-only','--output-dir',str(tmp_path/'plan')])
    suite.run(args)
    manifest=json.loads((tmp_path/'plan/suite_manifest.json').read_text())
    assert len(manifest['cells'])==40 and manifest['requested_trials']==2000
    assert set(c['condition'] for c in manifest['cells']) == {'id', 'lighting_ood', 'blur_ood', 'background_ood', 'speed_ood'}
    assert manifest['skipped']==[]
    preflight=suite.parser().parse_args(['--preflight','--output-dir',str(tmp_path/'no-policy')])
    with pytest.raises(ValueError,match='requires policy-url'):suite.run(preflight)
    assert not (tmp_path/'no-policy').exists()
    suite.run(args)
    args.scene_seed+=1
    with pytest.raises(ValueError,match='changed'):suite.run(args)
    args.scene_seed-=1
    monkeypatch.setattr(suite,'source_hashes',lambda *a:{'source':'modified'})
    with pytest.raises(ValueError,match='changed'):suite.run(args)
    monkeypatch.setattr(suite,'source_hashes',lambda *a:{'source':'unchanged'})
    path=tmp_path/'plan/conveyor/id/planned_scenes.json'
    path.write_text('{}')
    with pytest.raises(ValueError,match='manifest differs'):suite.run(args)


def test_imaging_breakdown_keeps_failures():
    from piper_tasks.dynamic_evaluation import condition_breakdown
    rows=[dict(success=ok,spec=dict(imaging=dict(condition='lighting_ood',group=group)))
          for group,ok in [('dark',False),('dark',True),('bright',True)]]
    assert condition_breakdown(rows)['dark']==dict(trials=2,successes=1,success_rate=.5)
    assert condition_breakdown(rows,interface_only=True)['bright']['success_rate'] is None


@pytest.mark.parametrize('filename', ['checkpoint_step_40000.pt', 'checkpoint_step_040000.pt'])
def test_suite_preflight_resolves_training_and_legacy_checkpoint_names(tmp_path, monkeypatch, filename):
    from piper_bench.evaluation import suite
    from piper_policies.metadata import checkpoint_file, load_metadata
    from piper_bench.protocols.suite import load_suite
    checkpoint = tmp_path/'checkpoint'
    checkpoint.mkdir()
    (checkpoint/filename).write_bytes(b'policy')
    (checkpoint/'config.yaml').write_text('model: {training_mode: joint}\ndataloader: {chunk_size: 24}\n')
    collection_hash = load_suite()['tasks']['conveyor']['collection_file_sha256']
    (checkpoint/'normalization.json').write_text(json.dumps(dict(voxel=dict(num_bins=1, count_clip=15),
        sources=dict(conveyor=dict(collection_sha256=collection_hash)))))
    descriptor = load_metadata(checkpoint)[3]
    def settings(args):
        args.max_steps = 350
        args.latency_controller = args.latency_controller or 'continuous'
        args.min_execution_steps = args.min_execution_steps or 8
        return None, None, None, dict(task=args.task, condition=args.condition, episodes=[])
    monkeypatch.setattr(suite.client, 'load_settings', settings)
    monkeypatch.setattr(suite, 'source_hashes', lambda *args: {'source': 'test'})
    monkeypatch.setattr(suite.subprocess, 'run', lambda *a, **kw: pytest.fail('preflight must not execute simulation'))
    args = suite.parser().parse_args(['--preflight', '--policy-url', 'ws://test-policy.invalid',
        '--tasks', 'conveyor', '--conditions', 'id',
        '--output-dir', str(tmp_path/'plan')])
    suite.run(args, descriptor_provider=lambda task: descriptor)
    manifest = json.loads((tmp_path/'plan/suite_manifest.json').read_text())
    assert manifest['policy_descriptors']['conveyor']['checkpoint_step'] == 40000
    assert manifest['policy_descriptors']['conveyor']['checkpoint_sha256'] == hashlib.sha256(b'policy').hexdigest()
    assert manifest['policy_descriptors']['conveyor']['training_data']['conveyor']['collection_sha256'] == collection_hash
    assert checkpoint_file(checkpoint, 40000) == checkpoint/filename
    with pytest.raises(ValueError, match='checkpoint-step'):
        checkpoint_file(checkpoint, 60000)
    other = 'checkpoint_step_040000.pt' if filename == 'checkpoint_step_40000.pt' else 'checkpoint_step_40000.pt'
    (checkpoint/other).write_bytes(b'different policy')
    with pytest.raises(ValueError, match='ambiguous'):
        checkpoint_file(checkpoint, 40000)
