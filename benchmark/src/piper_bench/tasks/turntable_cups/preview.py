"""Render circular transport and physical pick/place without collecting training data."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

from piper_bench.backend.isaac import launcher_class, launch

AppLauncher = launcher_class()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', type=Path, required=True)
parser.add_argument('--mode', choices=['transport', 'nesting', 'oracle', 'both'], default='both')
parser.add_argument('--trials', type=int, default=3)
parser.add_argument('--max-steps', type=int, default=1050)
parser.add_argument('--transport-steps', type=int, default=150)
parser.add_argument('--seed', type=int, default=20261015)
parser.add_argument('--direction', type=int, choices=[-1, 1], default=1)
parser.add_argument('--speed', type=float, help='Tangential m/s; defaults to the task configuration')
parser.add_argument('--disk-radius', type=float, default=.18)
parser.add_argument('--orbit-radius', type=float, default=.095)
parser.add_argument('--center-y', type=float, default=-.05)
parser.add_argument('--object-size', type=float, default=.03)
parser.add_argument('--pathtracing-spp', type=int, default=64)
parser.add_argument('--events', action='store_true', help='Save synchronized HDR EVIS four-panel video and count frames')
parser.add_argument('--no-labels', action='store_true', help='Render RGB/events without any text overlays')
parser.add_argument('--labels', dest='no_labels', action='store_false', help='Opt in to diagnostic text overlays')
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device='cpu', no_labels=True)
args = parser.parse_args()
if args.device != 'cpu':parser.error('use --device cpu; RTX and EVIS still run on GPU')
app = launch(args)

import gymnasium as gym
import numpy as np
import torch
from PIL import Image
from PIL import ImageDraw
from piper_bench.tasks.conveyor.core import CAMERAS
from piper_bench.tasks.conveyor.core import CONTROL_HZ
from piper_bench.sensors.capture import configure_renderer
from piper_bench.tasks.turntable_cups.scene import TASK
from piper_bench.tasks.turntable_cups.scene import DISK_SEGMENTS
from piper_bench.tasks.turntable_cups.scene import BowlConfig as TurntableConfig
from piper_bench.tasks.turntable_cups.scene import CupRuntime as TurntableRuntime
from piper_bench.tasks.turntable_cups.scene import make_env_cfg
from piper_bench.tasks.turntable_cups.scene import camera_metadata
from piper_bench.tasks.turntable_cups.scene import cup_in_basket as object_in_basket
from piper_bench.tasks.turntable_cups.scene import nesting_metrics
from piper_bench.tasks.turntable_cups.scene import set_views
from piper_bench.tasks.turntable_cups.oracle import CupOracle as TurntableOracle


def panel(env, label):
    result = Image.new('RGB', (512, 256))
    for i, name in enumerate(CAMERAS):
        rgb = env.scene[name].data.output['rgb'][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
        image = Image.fromarray(rgb)
        if not args.no_labels:
            ImageDraw.Draw(image).text((5, 5), name+' '+label, fill='white', stroke_width=1, stroke_fill='black')
        result.paste(image, (i*256, 0))
    return result


def transport_metrics(rows, cfg):
    output = []
    for i in range(2):
        selected = [r for r in rows if r['time'] > 1. and
                    abs(r['poses'][i][2]-(cfg.belt_top+cfg.object_size/2)) < .005]
        if len(selected) < 30:
            output.append(dict(object=i+1, passed=False, samples=len(selected)))
            continue
        pos = np.array([r['poses'][i][:2] for r in selected])-cfg.center
        vel = np.array([r['velocities'][i][:2] for r in selected])
        radius = np.linalg.norm(pos, axis=1)
        omega = (pos[:, 0]*vel[:, 1]-pos[:, 1]*vel[:, 0])/(radius**2)
        measured = float(np.median(omega))
        drift = float(np.ptp(radius))
        output.append(dict(object=i+1, samples=len(selected), expected_omega_rad_s=cfg.omega,
            measured_omega_rad_s=measured, median_radius_m=float(np.median(radius)),
            radius_range_m=drift, median_tangential_speed_m_s=float(np.median(omega*radius)),
            passed=bool(abs(measured-cfg.omega) <= .1*abs(cfg.omega) and drift < .015)))
    return dict(passed=all(r['passed'] for r in output), objects=output)


def episode(env, runtime, out, mode, seed, limit, ffmpeg, dvs=None):
    out.mkdir()
    zero = torch.tensor([[0.,0.,0.,0.,0.,0.,1.]], device=env.device)
    env.reset()
    for _ in range(60):
        set_views(env)
        env.step(zero)
    runtime.reset(seed)
    runtime.supply()
    if mode=='nesting':
        # Contact-only validation: reset placement above the lower cavity, then gravity.
        # This is separate from expert trials and never contributes a grasp success.
        c=runtime.cfg
        for i,obj in enumerate(runtime.objects):
            pose=[[c.basket_x,c.basket_y,c.wall_thickness+c.cup_height/2+.001+i*(c.cup_height-.01),0.,0.,0.,1.]]
            obj.write_root_pose_to_sim(torch.tensor(pose,dtype=torch.float32,device=env.device))
            obj.write_root_velocity_to_sim(torch.zeros((1,6),device=env.device))
    runtime.before_step()
    set_views(env)
    env.sim.forward()
    env.sim.render()
    for name in CAMERAS:
        env.scene[name].update(0.0,force_recompute=True)
    (out/'initial_state.json').write_text(json.dumps(runtime.metadata(), indent=2)+'\n')
    initial = panel(env, 'initial')
    initial.save(out/'initial.png')
    event_preview = None
    if dvs is not None:
        from piper_bench.sensors.preview import EventPreview
        event_preview = EventPreview(dvs, runtime, out, ffmpeg, labels=not args.no_labels)
        event_preview.panel(initial).save(out/'initial_events.png')
    video_path = out/'preview.mp4'
    video = subprocess.Popen([ffmpeg, '-loglevel', 'error', '-y', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', '512x256', '-r', '30', '-i', 'pipe:0', '-an',
        '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video_path)],
        stdin=subprocess.PIPE)
    oracle = TurntableOracle(runtime)
    rows, keyframes, stages = [], [initial], set()
    success, reason = False, None
    try:
        for step in range(limit):
            action = oracle.action() if mode == 'oracle' and step else zero
            runtime.before_step()
            set_views(env)
            env.step(action)
            success, reason = runtime.after_step()
            poses, velocities, tcp = runtime.state()
            row = dict(step=step+1, time=runtime.time, stage=oracle.stage, object_index=oracle.index,
                poses=poses.tolist(), velocities=velocities.tolist(), tcp=tcp.tolist(),
                action=action[0].detach().cpu().tolist(),
                tcp_quaternion=env.scene['ee_frame'].data.target_quat_w[0,0].detach().cpu().tolist(),
                joints=env.scene['robot'].data.joint_pos[0].detach().cpu().tolist(),
                target=None if oracle.target is None else oracle.target.tolist(),
                success=success, reason=reason)
            rows.append(row)
            frame = panel(env, f'{runtime.time:.2f}s {oracle.index}:{oracle.stage}')
            video.stdin.write(frame.tobytes())
            if event_preview is not None:
                row['events'] = event_preview.step(frame)
                row['event_window_s'] = [runtime.time-1/CONTROL_HZ, runtime.time]
            stage_key = (oracle.index, oracle.stage)
            if (mode == 'transport' and step in (29, 74, limit-1)) or (mode == 'oracle' and stage_key not in stages):
                stages.add(stage_key)
                frame.save(out/f'step_{step+1:04d}.png')
                if event_preview is not None:
                    event_preview.last_panel.save(out/f'step_{step+1:04d}_events.png')
                if mode == 'transport' or oracle.stage in ('close', 'carry', 'release'):
                    keyframes.append(frame.copy())
            if step % 30 == 0 or success or reason:
                (out/'progress.json').write_text(json.dumps(row, indent=2)+'\n')
                print(f'[turntable] {out.name} step={step+1} stage={oracle.index}:{oracle.stage} '
                      f'objects={poses[:,:3].round(3).tolist()} success={success} reason={reason}', flush=True)
            if mode == 'oracle' and (success or reason):
                break
    finally:
        try:
            video.stdin.close()
            if video.wait(timeout=60):
                raise RuntimeError('ffmpeg encoding failed')
        finally:
            (out/'steps.json').write_text(json.dumps(rows, indent=2)+'\n')
            if event_preview is not None:
                event_preview.close()
    subprocess.run([ffmpeg, '-v', 'error', '-i', str(video_path), '-f', 'null', '-'], check=True)
    final = panel(env, f'final success={success}')
    final.save(out/'terminal.png')
    keyframes.append(final)
    sheet = Image.new('RGB', (512, 256*len(keyframes)))
    for i, frame in enumerate(keyframes):
        sheet.paste(frame, (0, i*256))
    sheet.save(out/'contact_sheet.png')
    result = dict(mode=mode, seed=seed, task_success=success, reason=reason or ('success' if success else 'step_limit'),
        frames=len(rows), video_decoded=True, scene=runtime.metadata(),
        grasp_lifted=oracle.grasp_lifted, retries=oracle.retries,
        objects_in_basket=[object_in_basket(p, runtime.cfg) for p in poses],
        stability_steps=runtime.tracker.steps, nesting=nesting_metrics(poses,runtime.cfg))
    if mode == 'transport':
        result['transport'] = transport_metrics(rows, runtime.cfg)
        result['passed'] = result['transport']['passed']
    elif mode=='nesting':
        result['passed']=bool(success)
    else:
        result['passed'] = bool(success and all(oracle.grasp_lifted))
    if event_preview is not None:
        result['events'] = event_preview.report
        result['passed'] = result['passed'] and result['events']['passed']
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def main():
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    if args.trials < 1 or args.transport_steps < 60 or args.max_steps < 1:
        raise ValueError('invalid trial or step count')
    cfg = TurntableConfig(speed=TurntableConfig().speed if args.speed is None else args.speed, direction=args.direction,
                          disk_radius=args.disk_radius, orbit_radius=args.orbit_radius,
                          center_y=args.center_y, object_size=args.object_size, cup_height=args.object_size)
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('ffmpeg missing from PATH')
    configure_renderer(args.pathtracing_spp)
    env_cfg = make_env_cfg(cfg, device=args.device, seed=args.seed)
    env = gym.make(TASK, cfg=env_cfg).unwrapped
    results = []
    try:
        runtime = TurntableRuntime(env, cfg)
        dvs, event_protocol = None, None
        if args.events:
            from piper_bench.sensors.preview import create_event_camera
            from piper_bench.sensors.preview import event_metadata
            dvs = create_event_camera(env, out/'dvs_tmp')
            event_protocol = event_metadata(dvs)
        from pxr import PhysxSchema
        from pxr import UsdGeom
        from pxr import UsdPhysics
        from isaaclab.sim.utils.stage import get_current_stage
        stage = get_current_stage()
        prim = stage.GetPrimAtPath(str(env.scene.env_prim_paths[0])+'/Turntable')
        surface = PhysxSchema.PhysxSurfaceVelocityAPI(prim)
        observed = float(surface.GetSurfaceAngularVelocityAttr().Get()[2])
        assert math.isclose(observed, math.degrees(cfg.omega), rel_tol=1e-6)
        camera = UsdGeom.Camera(stage.GetPrimAtPath(str(env.scene.env_prim_paths[0])+'/cam0'))
        assert math.isclose(camera.GetFocalLengthAttr().Get(), camera_metadata()['cam0_focal_length_cm'])
        base = str(prim.GetPath())
        collision = stage.GetPrimAtPath(base+'/geometry/mesh')
        visual = stage.GetPrimAtPath(base+'/geometry/smooth_disk')
        assert UsdGeom.Imageable(collision).GetPurposeAttr().Get() == 'guide'
        assert UsdPhysics.CollisionAPI(collision).GetCollisionEnabledAttr().Get()
        assert not visual.HasAPI(UsdPhysics.CollisionAPI)
        assert len(UsdGeom.Mesh(visual).GetPointsAttr().Get()) == 2*DISK_SEGMENTS
        assert math.isclose(UsdGeom.Cylinder(collision).GetRadiusAttr().Get(), cfg.disk_radius)
        for i in (1, 2):
            obj = env.scene[f'cube_{i}'].cfg.spawn
            assert obj.cup_config['cup_height'] == cfg.cup_height
            root=str(env.scene.env_prim_paths[0])+f'/ConveyorObject_{i}/geometry'
            assert not stage.GetPrimAtPath(root+'/mesh')
            assert stage.GetPrimAtPath(root+'/open_cup')
            assert sum(stage.GetPrimAtPath(root+f'/wall_{j:02d}').HasAPI(UsdPhysics.CollisionAPI) for j in range(32))==32
        (out/'protocol.json').write_text(json.dumps(dict(task=TASK, config=cfg.metadata(),
            camera_calibration=camera_metadata(), control_hz=30, physics_hz=120,
            authored_surface_omega_deg_s=observed, oracle=TurntableOracle.PROFILE,
            disk_visual_segments=DISK_SEGMENTS, geometry_checked=True, text_overlays=not args.no_labels,
            events=event_protocol, training_data=False, collection_contract_verified=False), indent=2)+'\n')
        if args.mode in ('transport', 'both'):
            r = episode(env, runtime, out/'transport', 'transport', args.seed, args.transport_steps, ffmpeg, dvs)
            results.append(r)
            if not r['passed']:
                raise RuntimeError('circular contact transport failed; see transport/result.json')
        if args.mode in ('nesting','both'):
            r=episode(env,runtime,out/'nesting_contact','nesting',args.seed,150,ffmpeg,dvs)
            results.append(r)
            if not r['passed']:raise RuntimeError('physical cup nesting contact check failed')
        if args.mode in ('oracle', 'both'):
            for i in range(args.trials):
                results.append(episode(env, runtime, out/f'oracle_{i:03d}', 'oracle',
                                       args.seed+i, args.max_steps, ffmpeg, dvs))
        summary = dict(complete=True, passed=all(r['passed'] for r in results),
                       results=results, successes=sum(r['task_success'] for r in results if r['mode']=='oracle'),
                       oracle_trials=sum(r['mode']=='oracle' for r in results))
        (out/'complete.json').write_text(json.dumps(summary, indent=2)+'\n')
        if not summary['passed']:
            raise RuntimeError('some physical oracle trials failed')
    finally:
        env.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        traceback.print_exc()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir/'failure.json').write_text(json.dumps(dict(error=type(exc).__name__, message=str(exc)))+'\n')
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    else:
        app.close()
