"""Collect successful physical Piper conveyor episodes using the stack EVIS/LeRobot contract."""
from __future__ import annotations
import argparse,json,subprocess,tempfile,os,sys,traceback
from pathlib import Path
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--episodes',type=int,default=50)
parser.add_argument('--max-attempts',type=int,default=200)
parser.add_argument('--seed',type=int,default=20263002)
parser.add_argument('--conveyor-speed-range',type=float,nargs=2,metavar=('MIN','MAX'),help='Uniform m/s per saved episode; retries keep its speed.')
parser.add_argument('--yaw-jitter-deg',type=float,default=10.)
parser.add_argument('--spacing-jitter',type=float,default=0.)
parser.add_argument('--repo-id',default='local/piper_conveyor_evis_event_frames')
parser.add_argument('--resume',action='store_true')
parser.add_argument('--initial-hold-steps',type=int,default=1,help='Record initial open-gripper hold before tracking; matches the validated reach rollout.')
parser.add_argument('--probe-steps',type=int,default=0,help='Write a short real RGB/event interface probe, excluded from successful demos.')
parser.add_argument('--pathtracing-spp',type=int,default=64)
parser.add_argument('--writer-python',type=Path,default=Path(__file__).resolve().parents[2]/'envs/lerobot/bin/python')
AppLauncher.add_app_launcher_args(parser);parser.set_defaults(device='cpu')
args=parser.parse_args();app=AppLauncher(args).app
import gymnasium as gym
import numpy as np
import torch
from dvs_gen.sensors import DVSCamera
from piper_conveyor import TASK,CAMERAS,CONTROL_HZ,ConveyorConfig,ConveyorRuntime,camera_metadata,make_env_cfg,set_views
from piper_conveyor_oracle import ConveyorOracle
from piper_conveyor_io import configure_renderer,snapshot,capture_frame,send_writer_message,writer_reply,refresh_spawn_views
from piper_event_frames import events_to_frame
from piper_conveyor_resume import prepare_resume
from piper_conveyor_speed import uniform_speed_spec,sample_episode_speed,set_episode_speed


def main():
    out=args.output.resolve();cfg=ConveyorConfig(yaw_jitter=float(np.deg2rad(args.yaw_jitter_deg)),object_spacing_jitter=args.spacing_jitter)
    speed_spec=uniform_speed_spec(args.conveyor_speed_range)
    if args.device!='cpu': raise ValueError('Conveyor contact validation currently requires --device cpu; RTX/events still run on GPU.')
    if args.episodes<=0 or args.max_attempts<=0 or args.initial_hold_steps<0:raise ValueError('episodes/attempts must be positive and initial hold nonnegative')
    if args.resume:
        saved_cal=json.loads((out/'meta/piper_camera_calibration.json').read_text())
        if saved_cal!=camera_metadata():raise ValueError('cannot resume with changed camera calibration')
        saved_config=json.loads((out/'meta/piper_collection_config.json').read_text())
        if saved_config.get('speed_sampling')!=speed_spec:raise ValueError('cannot resume with changed speed distribution')
        if saved_config['scene_config']!=cfg.metadata() or saved_config['seed']!=args.seed or saved_config['probe_only']!=bool(args.probe_steps) or saved_config.get('initial_hold_steps',0)!=args.initial_hold_steps or saved_config.get('oracle_profile')!=ConveyorOracle.PROFILE:
            raise ValueError('cannot resume with changed scene, seeds, or probe mode')
        if saved_config.get('pathtracing_spp')!=args.pathtracing_spp:
            raise ValueError('cannot resume with changed renderer sampling')
        resume_attempts,resume_saved=prepare_resume(out,args.episodes)
    elif out.exists():raise FileExistsError(f'use a fresh dataset path: {out}')
    configure_renderer(args.pathtracing_spp)
    env=gym.make(TASK,cfg=make_env_cfg(cfg,device=args.device,seed=args.seed)).unwrapped
    writer=None
    attempts=[]
    try:
        writer=subprocess.Popen([str(args.writer_python),'-u',str(Path(__file__).with_name('write_piper_conveyor_lerobot.py')),
            '--output',str(out),'--repo-id',args.repo_id,*(['--resume'] if args.resume else [])],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
        saved=writer_reply(writer,'READY')
        if args.resume:
            if saved!=resume_saved:raise ValueError('writer episode count differs from resume metadata')
            attempts=resume_attempts
        else:
            (out/'meta/piper_camera_calibration.json').write_text(json.dumps(camera_metadata(),indent=2)+'\n')
            (out/'meta/piper_collection_config.json').write_text(json.dumps(dict(scene_config=cfg.metadata(),seed=args.seed,speed_sampling=speed_spec,
                requested_successful_episodes=args.episodes,probe_only=bool(args.probe_steps),initial_hold_steps=args.initial_hold_steps,oracle_profile=ConveyorOracle.PROFILE,renderer_mode='pathtracing',
                pathtracing_spp=args.pathtracing_spp,physics_backend='CPU PhysX, GPU RTX rendering',
                simulation_clock='fixed_simulation_time_inference_paused'),indent=2)+'\n')
        runtime=ConveyorRuntime(env,cfg)
        zero=torch.tensor([[0.,0.,0.,0.,0.,0.,1.]],device=env.device)
        with tempfile.TemporaryDirectory(prefix='piper_conveyor_evis_') as event_dir:
            dvs=DVSCamera.from_scene(env.scene,list(CAMERAS),out_dir=event_dir,compression='lzf',event_source='hdr',antialiasing=None)
            while saved<args.episodes and len(attempts)<args.max_attempts:
                seed=args.seed+len(attempts)
                env.reset()
                for _ in range(60):set_views(env);env.step(zero)
                runtime.reset(seed)
                if speed_spec is not None:
                    speed=sample_episode_speed(speed_spec,args.seed,saved)
                    set_episode_speed(runtime,speed)
                    print(f'[conveyor-speed] episode={saved} attempt={len(attempts)+1} speed={speed:.9f} m/s',flush=True)
                runtime.supply();refresh_spawn_views(env)
                dvs.reset(torch.tensor([0],dtype=torch.long,device='cuda:0'));dvs.recorder._events.pop(0,None)
                previous=snapshot(dvs,env);t_previous=0.
                oracle=ConveyorOracle(runtime);buffer=[];success=False;reason=None
                limit=args.probe_steps or round(cfg.max_seconds*CONTROL_HZ)
                event_counts={name:0 for name in CAMERAS}
                for step in range(limit):
                    action=zero if step<args.initial_hold_steps else oracle.action()
                    dvs.warp_and_process(previous,previous,1,t_previous,1/240)
                    chunks=dvs.recorder._events.pop(0,{})
                    events={name:events_to_frame(chunks.get('DVS/'+name,[]),t_previous-1/CONTROL_HZ,t_previous) for name in CAMERAS}
                    if step==0 and any(np.any(frame) for frame in events.values()):raise RuntimeError('first event frame must be zero after reset')
                    for name,frame in events.items():event_counts[name]+=int(frame[:,:,:2].sum())
                    buffer.append(capture_frame(runtime,action,events))
                    set_views(env);env.step(action)
                    success,reason=runtime.after_step()
                    if runtime.supply():refresh_spawn_views(env)
                    current=snapshot(dvs,env)
                    dvs.warp_and_process(previous,current,8,t_previous,1/240)
                    previous=current;t_previous=runtime.time
                    if step%60==0:print(f'[conveyor-collect] attempt={len(attempts)+1} step={step} stage={oracle.stage}',flush=True)
                    if success or reason:break
                poses,velocities,tcp=runtime.state()
                record=dict(attempt=len(attempts)+1,seed=seed,speed_episode_index=saved,conveyor_speed_m_s=runtime.cfg.speed,success=success,reason=reason or ('success' if success else 'probe_only' if args.probe_steps else 'step_limit'),
                    frames=len(buffer),stability_steps=runtime.tracker.steps,event_counts=event_counts,grasp_lifted=oracle.grasp_lifted,oracle_profile=oracle.PROFILE,retries=oracle.retries,scene=runtime.metadata(),
                    terminal_poses=poses.tolist(),terminal_velocities=velocities.tolist(),terminal_tcp=tcp.tolist())
                if success or args.probe_steps:
                    for frame in buffer:send_writer_message(writer,dict(command='frame',frame=frame))
                    send_writer_message(writer,dict(command='save'));writer.stdin.flush();saved=writer_reply(writer,'SAVED')
                    record['saved_episode']=saved-1
                attempts.append(record)
                (out/'meta/piper_conveyor_attempts.json').write_text(json.dumps(attempts,indent=2)+'\n')
                print('[conveyor-attempt] '+json.dumps(record),flush=True)
                dvs.recorder._events.pop(0,None)
            if saved<args.episodes:raise RuntimeError(f'collected only {saved}/{args.episodes} successful episodes in {len(attempts)} attempts')
    finally:
        env.close()
        if writer is not None:
            if writer.stdin and not writer.stdin.closed:writer.stdin.close()
            if writer.wait(timeout=300):raise RuntimeError('LeRobot writer failed')
    (out/'.collection_complete').write_text(json.dumps(dict(episodes=saved,attempts=len(attempts),probe_only=bool(args.probe_steps)))+'\n')
    print(f'[conveyor-collection] completed {saved} episodes -> {out}',flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
    else:app.close()
