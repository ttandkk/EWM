"""Preview a real rotating rack; contact-only validation is separate from robot success."""
import argparse
import json
import shutil
import subprocess
import traceback
from pathlib import Path
from piper_bench.backend.isaac import launcher_class, launch
AppLauncher = launcher_class()
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',type=Path,required=True)
parser.add_argument('--mode',choices=['preview','contact','oracle','both'],default='both')
parser.add_argument('--trials',type=int,default=1)
parser.add_argument('--max-steps',type=int,default=900)
parser.add_argument('--seed',type=int,default=20261016)
parser.add_argument('--events',action='store_true')
parser.add_argument('--pathtracing-spp',type=int,default=32)
AppLauncher.add_app_launcher_args(parser);parser.set_defaults(device='cpu')
args=parser.parse_args()
if args.trials<1 or args.max_steps<1:parser.error('trials and max-steps must be positive')

if args.device!='cpu':parser.error('use --device cpu; RTX and EVIS still run on GPU')
app=launch(args)
import gymnasium as gym
import numpy as np
import torch
from PIL import Image
from piper_bench.tasks.conveyor.core import CAMERAS
from piper_bench.sensors.capture import configure_renderer
from piper_bench.tasks.rotating_tube.scene import TASK
from piper_bench.tasks.rotating_tube.scene import TubeConfig
from piper_bench.tasks.rotating_tube.scene import TubeRuntime
from piper_bench.tasks.rotating_tube.scene import make_env_cfg
from piper_bench.tasks.rotating_tube.scene import set_views
from piper_bench.tasks.rotating_tube.scene import camera_metadata
from piper_bench.tasks.rotating_tube.scene import insertion_metrics
from piper_bench.tasks.rotating_tube.oracle import TubeOracle


def panel(env):
    result=Image.new('RGB',(512,256))
    for i,name in enumerate(CAMERAS):
        rgb=env.scene[name].data.output['rgb'][0,...,:3].detach().cpu().numpy().astype(np.uint8)
        result.paste(Image.fromarray(rgb),(i*256,0))
    return result


def episode(env,runtime,out,mode,limit,ffmpeg,dvs,*,spec):
    out.mkdir();runtime.motion_enabled=False
    zero=torch.tensor([[0.,0.,0.,0.,0.,0.,1.]],device=env.device)
    env.reset()
    for _ in range(60):set_views(env);env.step(zero)
    from piper_bench.collection.runtime import initialize
    initial=initialize(runtime,spec)
    (out/'initial_spec.json').write_text(json.dumps(spec,indent=2)+'\n')
    (out/'initial_verified.json').write_text(json.dumps(initial,indent=2)+'\n')
    if mode=='contact':runtime.preload_contact_probe()
    set_views(env);env.sim.forward();env.sim.render()
    for name in CAMERAS:env.scene[name].update(0.,force_recompute=True)
    (out/'initial_state.json').write_text(json.dumps(runtime.metadata(),indent=2)+'\n')
    first=panel(env);first.save(out/'initial.png')
    events=None
    if dvs is not None:
        from piper_bench.sensors.preview import EventPreview
        events=EventPreview(dvs,runtime,out,ffmpeg,labels=False)
    path=out/'preview.mp4'
    video=subprocess.Popen([ffmpeg,'-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','512x256',
        '-r','30','-i','pipe:0','-an','-c:v','libx264','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    oracle=TubeOracle(runtime);rows=[];stages=set();success=False;reason=None
    try:
        for step in range(limit):
            action=oracle.action() if mode=='oracle' and step else zero
            set_views(env);env.step(action)
            joints=env.scene['robot'].data.joint_pos[0].detach().cpu().numpy()
            gripper_open=abs(joints[-2]-joints[-1])>.045
            success,reason=runtime.after_step(gripper_open=gripper_open)
            poses,vel,tcp=runtime.state()
            row=dict(step=step+1,time=runtime.time,stage=oracle.stage,poses=poses.tolist(),
                velocities=vel.tolist(),tcp=tcp.tolist(),joints=joints.tolist(),
                tcp_quaternion=env.scene['ee_frame'].data.target_quat_w[0,0].detach().cpu().tolist(),
                target=None if oracle.target is None else oracle.target.tolist(),
                action=action[0].detach().cpu().tolist(),hole=runtime.hole_target()[0].tolist(),
                gripper_open=bool(gripper_open),success=bool(success),reason=reason,
                insertion=insertion_metrics(poses[0],vel[0],tcp,runtime.cfg,runtime.hole_target(),gripper_open))
            rows.append(row);frame=panel(env);video.stdin.write(frame.tobytes())
            if events is not None:row['events']=events.step(frame)
            if oracle.stage not in stages or step in (29,89,149):
                stages.add(oracle.stage);frame.save(out/f'step_{step+1:04d}_{oracle.stage}.png')
            if step%30==0 or (mode=='oracle' and (success or reason)):
                (out/'progress.json').write_text(json.dumps(row,indent=2)+'\n')
                print(f'[tube] {out.name} step={step+1} stage={oracle.stage} xyz={poses[0,:3].round(4).tolist()} '
                      f'tilt={row["insertion"]["tilt_deg"]:.1f} xy={row["insertion"]["axis_xy_error_m"]} success={success}',flush=True)
            if mode=='oracle' and (success or reason):break
    finally:
        try:
            video.stdin.close()
            if video.wait(timeout=60):raise RuntimeError('ffmpeg encode failed')
        finally:
            (out/'steps.json').write_text(json.dumps(rows,indent=2)+'\n')
            if events is not None:events.close()
    subprocess.run([ffmpeg,'-v','error','-i',str(path),'-f','null','-'],check=True)
    panel(env).save(out/'terminal.png')
    passed=bool(success and (mode=='contact' or oracle.grasp_lifted))
    if mode=='scene':
        passed=bool(-.005<poses[0,2]<.02 and 70<rows[-1]['insertion']['tilt_deg']<110)
    result=dict(mode=mode,seed=spec['seed'],passed=passed,task_success=bool(success) if mode=='oracle' else None,
                contact_success=bool(success) if mode=='contact' else None,
                grasp_lifted=oracle.grasp_lifted,retries=oracle.retries,frames=len(rows),
                stable_steps=runtime.tracker.steps,reason=reason,video_decoded=True,
                metrics=rows[-1]['insertion'])
    if events is not None:result['events']=events.report;result['passed']&=events.report['passed']
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    from piper_bench.collection.protocol import make_plan
    from piper_bench.collection.protocol import base_config
    from piper_bench.collection.protocol import episode_spec
    plan=make_plan('rotating_tube',args.trials,args.seed,'validation')
    specs=[episode_spec(plan,i) for i in range(args.trials)]
    cfg=base_config('rotating_tube',plan);ffmpeg=shutil.which('ffmpeg');assert ffmpeg
    (out/'sampling_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    configure_renderer(args.pathtracing_spp)
    env=gym.make(TASK,cfg=make_env_cfg(cfg,device=args.device,seed=args.seed)).unwrapped
    results=[]
    try:
        runtime=TubeRuntime(env,cfg);dvs=None;event_protocol=None
        if args.events:
            from piper_bench.sensors.preview import create_event_camera
            from piper_bench.sensors.preview import event_metadata
            dvs=create_event_camera(env,out/'dvs_tmp');event_protocol=event_metadata(dvs)
        from isaaclab.sim.utils.stage import get_current_stage
        from pxr import UsdPhysics
        from pxr import UsdGeom
        stage=get_current_stage();base=str(env.scene.env_prim_paths[0])
        rack=stage.GetPrimAtPath(base+'/RotatingRack')
        assert UsdPhysics.RigidBodyAPI(rack).GetKinematicEnabledAttr().Get()
        for i in range(cfg.holes):
            prim=stage.GetPrimAtPath(base+f'/RotatingRack/geometry/perforated_sector_{i}')
            assert UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()=='none'
        assert not stage.GetPrimAtPath(base+'/RotatingRack/geometry/mesh')
        (out/'protocol.json').write_text(json.dumps(dict(config=cfg.metadata(),camera=camera_metadata(),
            control_hz=30,physics_hz=120,oracle=TubeOracle.PROFILE,events=event_protocol,
            text_overlays=False,training_data=False,geometry_checked=True),indent=2)+'\n')
        if args.mode in ('preview','both'):
            results.append(episode(env,runtime,out/'scene','scene',120,ffmpeg,dvs,spec=specs[0]))
        if args.mode in ('preview','contact','both'):
            results.append(episode(env,runtime,out/'contact','contact',180,ffmpeg,dvs,spec=specs[0]))
        if args.mode in ('oracle','both'):
            for i in range(args.trials):
                results.append(episode(env,runtime,out/f'oracle_{i:03d}','oracle',args.max_steps,ffmpeg,dvs,spec=specs[i]))
        complete=dict(complete=True,passed=all(r['passed'] for r in results),results=results)
        (out/'complete.json').write_text(json.dumps(complete,indent=2)+'\n')
        print(json.dumps(complete),flush=True)
        if not complete['passed']:raise RuntimeError('some physical oracle checks failed')
    finally:env.close()

try:main()
except Exception:
    traceback.print_exc();raise
finally:app.close()
