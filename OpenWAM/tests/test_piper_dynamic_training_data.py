"""Current task binding, train-only pooled stats, and stale input rejection."""
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from hydra import compose,initialize_config_dir
from omegaconf import OmegaConf

from openwam.dataloader.piper_training_data import (
    DYNAMIC_TASKS,PROTOCOL,prepare_normalization,source_signatures,validate_exports)


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def exports(root):
    for task_id,task in enumerate(DYNAMIC_TASKS):
        d=root/'datasets'/f'piper_{task}'
        m=d/'meta'
        write(m/'info.json',dict(fps=30,total_episodes=2,total_frames=52,splits={'train':'0:2'}))
        write(m/'ewm_normalization_stats.json',dict(rgb={'type':'backbone'},voxel={'count_clip':15,'num_bins':1}))
        write(m/'piper_collection_config.json',dict(version=PROTOCOL,task=task,plan={'split':'train'},
            standard_imaging=True,control_hz=30,event_hz=240,physics_hz=240 if task=='tennis_container' else 120,camera={'calibration':'current'}))
        write(m/'piper_camera_calibration.json',{'calibration':'current'})
        write(m/'piper_temporal_contract.json',dict(temporal_contract='aligned_event_frame_pre_action_v1',
            task_selector=task,collection_protocol=PROTOCOL,control_hz=30,event_hz=240))
        write(m/'piper_validation.json',dict(passed=True,task=task,episodes=2,frames=52))
        for name in ('.collection_complete','.export_complete'):(d/name).touch()
        for cam in ('cam0','cam_wrist'):(d/'events'/cam).mkdir(parents=True)
        (m/'episodes').mkdir()
        pq.write_table(pa.Table.from_pylist([{'episode_index':i,'length':26} for i in range(2)]),m/'episodes/file.parquet')
        (d/'data').mkdir()
        rows=np.arange(52,dtype=float)+task_id*100
        pq.write_table(pa.table({'observation.state':np.tile(rows[:,None],(1,8)).tolist(),
                                 'action':np.tile(rows[:,None],(1,7)).tolist()}),d/'data/file.parquet')


def test_pools_actual_training_rows_and_reuses_only_identical_sources(tmp_path):
    exports(tmp_path)
    report=validate_exports(tmp_path,expected_episodes=2)
    assert len(report)==8 and sum(r['windows'] for r in report.values())==32
    out=tmp_path/'cache/stats.json'
    p=prepare_normalization(tmp_path,out)
    actual=np.concatenate([np.arange(52)+i*100 for i in range(8)])
    assert p['frames']==416 and p['tasks']==list(DYNAMIC_TASKS)
    np.testing.assert_allclose(p['action']['q99'],np.quantile(actual,.99))
    assert p['action']['type']=='minmax'
    np.testing.assert_allclose(p['action']['min'],actual.min())
    np.testing.assert_allclose(p['action']['max'],actual.max())
    np.testing.assert_allclose(p['proprio']['mean'],actual.mean())
    assert prepare_normalization(tmp_path,out)==p
    stats=tmp_path/'datasets/piper_conveyor/meta/ewm_normalization_stats.json'
    stats.write_text(stats.read_text()+'\n')
    with pytest.raises(ValueError,match='stale'):prepare_normalization(tmp_path,out)


def test_legacy_quantile_cache_cannot_silently_start_new_training(tmp_path):
    exports(tmp_path)
    out=tmp_path/'cache/stats.json'
    payload=prepare_normalization(tmp_path,out)
    payload['action']['type']='quantile'
    write(out,payload)
    with pytest.raises(ValueError,match='legacy quantile'):
        prepare_normalization(tmp_path,out)


@pytest.mark.parametrize('task', ['moving_carton', 'conveyor'])
def test_lowwall_protocol_is_carton_only(tmp_path, task):
    exports(tmp_path)
    meta=tmp_path/'datasets'/f'piper_{task}'/'meta'
    for name,key in [('piper_collection_config.json','version'),
                     ('piper_temporal_contract.json','collection_protocol')]:
        value=json.loads((meta/name).read_text())
        value[key]='piper_carton_lowwall_v4'
        write(meta/name,value)
    if task=='moving_carton':
        assert validate_exports(tmp_path,expected_episodes=2)[task]['episodes']==2
    else:
        with pytest.raises(ValueError,match='standard-ID'):
            validate_exports(tmp_path,expected_episodes=2)


@pytest.mark.parametrize('change',['incomplete','test_split','camera','count','clock'])
def test_invalid_export_stops_before_cache(tmp_path,change):
    exports(tmp_path);d=tmp_path/'datasets/piper_conveyor'
    if change=='incomplete':(d/'.export_complete').unlink()
    else:
        file={'test_split':'piper_collection_config.json','camera':'piper_camera_calibration.json',
              'count':'info.json','clock':'piper_temporal_contract.json'}[change]
        p=d/'meta'/file;value=json.loads(p.read_text())
        if change=='test_split':value['plan']['split']='evaluation'
        if change=='camera':value['calibration']='old'
        if change=='count':value['total_frames']-=1
        if change=='clock':value['control_hz']=25
        write(p,value)
    with pytest.raises(ValueError):validate_exports(tmp_path,expected_episodes=2)


def test_preset_uses_current_eight_and_separate_cache(monkeypatch,tmp_path):
    root=Path(__file__).resolve().parents[2]
    for key in ('EWM_EIGHT_CACHE','EWM_EIGHT_STATS'):
        monkeypatch.delenv(key,raising=False)
    for key,value in dict(NEW_PROJECT_ROOT=str(tmp_path),EWM_EIGHT_OUTPUT=str(tmp_path/'output'),
                          EWM_EIGHT_RUN_NAME='test',EWM_DATASET_DIR=str(tmp_path/'old')).items():
        monkeypatch.setenv(key,value)
    with initialize_config_dir(config_dir=str(root/'OpenWAM/configs'),version_base=None):
        cfg=compose(config_name='train_ewm_eight')
    OmegaConf.resolve(cfg)
    assert tuple(cfg.dataloader.tasks)==DYNAMIC_TASKS
    assert cfg.dataloader.cache_root==str(tmp_path/'tmp/training_data_cache/piper_dynamic_v1')
    assert cfg.dataloader.normalization_stats_path==cfg.dataloader.cache_root+'/shared_normalization.json'
    assert cfg.dataloader.event_camera_mode==cfg.model.event_camera_mode=='dual'
    assert cfg.dataloader.observation_row_offset==0 and cfg.dataloader.chunk_size==24
    assert cfg.training.init_policy_checkpoint is None and cfg.training.resume_ckpt_path is None


@pytest.mark.parametrize('fail_preflight',[False,True])
def test_preparation_reuses_cpu_checks_and_never_launches_training(tmp_path,monkeypatch,fail_preflight):
    from types import SimpleNamespace
    from scripts.train_ewm import prepare_piper_dynamic_data as pipeline
    from scripts.train_ewm import build_ewm_event_frame_cache as events
    cache=tmp_path/'cache';cache.mkdir()
    write(cache/'cpu_validation.json',{'passed':True,'format':'ewm_current_inputs_v1'})
    for task in DYNAMIC_TASKS:
        write(cache/f'piper_{task}/event_frames/manifest.json',{})
    monkeypatch.setattr(pipeline,'validate_exports',lambda *a:{t:dict(episodes=50,frames=1000) for t in DYNAMIC_TASKS})
    monkeypatch.setattr(pipeline,'source_signatures',lambda *a:{'source':'same'})
    monkeypatch.setattr(pipeline,'prepare_normalization',lambda *a:None)
    monkeypatch.setattr(pipeline,'verify_cpu',lambda *a:pytest.fail('completed CPU check must not repeat'))
    monkeypatch.setattr(events,'build',lambda *a:pytest.fail('complete event cache must not rebuild'))
    calls=[]
    def launch(command,**kwargs):
        calls.append(command)
        assert Path(command[1]).name in ('prepare_piper_current_dino.py', 'prepare_piper_eight.py')
        assert 'train.py' not in command
        if fail_preflight and Path(command[1]).name == 'prepare_piper_eight.py':raise RuntimeError('loader failed')
    monkeypatch.setattr(pipeline.subprocess,'run',launch)
    args=SimpleNamespace(root=tmp_path,cache_root=cache,stage='all',batch_size=32)
    if fail_preflight:
        with pytest.raises(RuntimeError,match='loader failed'):pipeline.run(args)
        assert not (cache/'complete.json').exists()
    else:
        pipeline.run(args)
        assert json.loads((cache/'complete.json').read_text())['training_started'] is False
    assert len(calls)==2


@pytest.mark.parametrize('task', ['moving_carton', 'turntable_cups', 'conveyor'])
def test_low_geometry_protocol_is_scoped_to_changed_tasks(tmp_path, task):
    exports(tmp_path)
    meta=tmp_path/'datasets'/f'piper_{task}'/'meta'
    for name,key in [('piper_collection_config.json','version'),
                     ('piper_temporal_contract.json','collection_protocol')]:
        value=json.loads((meta/name).read_text())
        value[key]='piper_low_geometry_v5'
        write(meta/name,value)
    if task in ('moving_carton', 'turntable_cups'):
        assert validate_exports(tmp_path,expected_episodes=2)[task]['episodes']==2
    else:
        with pytest.raises(ValueError,match='standard-ID'):
            validate_exports(tmp_path,expected_episodes=2)


@pytest.mark.parametrize('task', ['tennis_container', 'conveyor'])
def test_wide_container_protocol_is_container_only(tmp_path, task):
    exports(tmp_path)
    meta = tmp_path/'datasets'/f'piper_{task}'/'meta'
    for name, key in [('piper_collection_config.json', 'version'),
                      ('piper_temporal_contract.json', 'collection_protocol')]:
        value = json.loads((meta/name).read_text())
        value[key] = 'piper_container_spatial_v4'
        write(meta/name, value)
    if task == 'tennis_container':
        assert validate_exports(tmp_path, expected_episodes=2)[task]['episodes'] == 2
    else:
        with pytest.raises(ValueError, match='standard-ID'):
            validate_exports(tmp_path, expected_episodes=2)
