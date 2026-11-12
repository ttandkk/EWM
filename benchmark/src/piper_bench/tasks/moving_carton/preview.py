"""Render moving-basket contact transport and physical pick/place without collecting training data."""
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
parser.add_argument('--mode', choices=['contact', 'oracle', 'both'], default='both')
parser.add_argument('--trials', type=int, default=3)
parser.add_argument('--max-steps', type=int, default=1050)
parser.add_argument('--contact-steps', type=int, default=300)
parser.add_argument('--seed', type=int, default=20261015)
parser.add_argument('--basket-amplitude', type=float, default=.06)
parser.add_argument('--basket-peak-speed', type=float, default=.05)
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
from piper_bench.tasks.moving_carton.scene import TASK
from piper_bench.tasks.moving_carton.scene import HalfWallConfig as MovingBasketConfig
from piper_bench.tasks.moving_carton.scene import MovingBasketRuntime
from piper_bench.tasks.moving_carton.scene import make_env_cfg
from piper_bench.tasks.moving_carton.scene import camera_metadata
from piper_bench.tasks.moving_carton.scene import stack_geometry
from piper_bench.tasks.moving_carton.scene import set_views
from piper_bench.tasks.moving_carton.oracle import MovingBasketOracle


def panel(env, label):
    result = Image.new('RGB', (512, 256))
    for i, name in enumerate(CAMERAS):
        rgb = env.scene[name].data.output['rgb'][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
        image = Image.fromarray(rgb)
        if not args.no_labels:
            ImageDraw.Draw(image).text((5, 5), name+' '+label, fill='white', stroke_width=1, stroke_fill='black')
        result.paste(image, (i*256, 0))
    return result


def contact_metrics(rows, cfg):
    selected = [r for r in rows if r['time'] > 1.]
    basket = np.array([r['basket_pose'][:3] for r in selected])
    reference = np.array([r['basket_target'] for r in selected])
    velocities = np.array([r['basket_velocity'][:3] for r in selected])
    target_error = float(np.linalg.norm(basket-reference,axis=1).max())
    span = float(np.ptp(basket[:,1]))
    median_speed = float(np.median(np.linalg.norm(velocities,axis=1)))
    objects = []
    for i in range(2):
        positions = np.array([r['poses'][i][:3] for r in selected])
        relative = positions-basket
        object_velocity = np.array([r['velocities'][i][:3] for r in selected])
        slip = float(np.median(np.linalg.norm(object_velocity-velocities,axis=1)))
        drift = float(np.linalg.norm(np.ptp(relative[:,:2],axis=0)))
        contained = float(np.mean([r['objects_in_basket'][i] for r in selected]))
        objects.append(dict(object=i+1,relative_drift_m=drift,median_relative_speed_m_s=slip,
                            contained_fraction=contained,passed=bool(drift<.012 and slip<.015 and contained>.98)))
    stack_fraction=float(np.mean([r['stack']['valid'] for r in selected]))
    if cfg.stack_success_mode == 'lowwall_v4':
        from piper_bench.tasks.moving_carton.lowwall import support_metrics
        # A correctly supported upper cube is above the rim, so the legacy
        # below-rim containment diagnostic is not applicable to it.
        support_fraction=float(np.mean([support_metrics(np.array(r['poses']),
            np.array(r['basket_pose']), np.array(r['tcp']), [True,True], cfg)['valid'] for r in selected]))
        stack_fraction=support_fraction
        for obj in objects:
            obj['supported_fraction']=support_fraction
            obj['passed']=bool(obj['relative_drift_m']<.012 and obj['median_relative_speed_m_s']<.015 and support_fraction>.98)
    return dict(stack_fraction=stack_fraction,passed=bool(stack_fraction>.98 and target_error<.002 and span>1.8*cfg.amplitude and
                           median_speed>.35*cfg.peak_speed and all(r['passed'] for r in objects)),
                basket_max_target_error_m=target_error,basket_travel_span_m=span,
                basket_median_speed_m_s=median_speed,objects=objects)


def episode(env, runtime, out, mode, seed, limit, ffmpeg, dvs=None):
    out.mkdir()
    zero = torch.tensor([[0.,0.,0.,0.,0.,0.,1.]], device=env.device)
    runtime.motion_enabled = False
    env.reset()
    for _ in range(60):
        set_views(env)
        env.step(zero)
    runtime.reset(seed)
    if mode == 'contact':
        runtime.preload_contact_probe()
    else:
        runtime.supply()
    set_views(env)
    env.sim.forward()
    env.sim.render()
    for name in CAMERAS:
        env.scene[name].update(0.0,force_recompute=True)
    initial_state = dict(scene=runtime.metadata(),object_pose=runtime.state()[0].tolist(),
                         basket_pose=runtime.basket_state()[0].tolist(),initialization=mode)
    (out/'initial_state.json').write_text(json.dumps(initial_state, indent=2)+'\n')
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
    oracle = MovingBasketOracle(runtime)
    rows, keyframes, stages = [], [initial], set()
    success, reason = False, None
    try:
        for step in range(limit):
            action = oracle.action() if mode == 'oracle' and step else zero
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
            basket_pose, basket_velocity = runtime.basket_state()
            row.update(basket_pose=basket_pose.tolist(),basket_velocity=basket_velocity.tolist(),
                       basket_target=runtime.basket_target(runtime.time)[0].tolist(),
                       objects_in_basket=runtime.objects_in_basket(poses),
                       stack=stack_geometry(poses,basket_pose,runtime.cfg),
                       relative_object_velocity=(velocities[:,:3]-basket_velocity[:3]).tolist())
            rows.append(row)
            frame = panel(env, f'{runtime.time:.2f}s {oracle.index}:{oracle.stage}')
            video.stdin.write(frame.tobytes())
            if event_preview is not None:
                row['events'] = event_preview.step(frame)
                row['event_window_s'] = [runtime.time-1/CONTROL_HZ, runtime.time]
            stage_key = (oracle.index, oracle.stage)
            if (mode == 'contact' and step in (29, 74, limit-1)) or (mode == 'oracle' and stage_key not in stages):
                stages.add(stage_key)
                frame.save(out/f'step_{step+1:04d}.png')
                if event_preview is not None:
                    event_preview.last_panel.save(out/f'step_{step+1:04d}_events.png')
                if mode == 'contact' or oracle.stage in ('close', 'carry', 'release'):
                    keyframes.append(frame.copy())
            if step % 30 == 0 or (mode == 'oracle' and (success or reason)):
                (out/'progress.json').write_text(json.dumps(row, indent=2)+'\n')
                print(f'[moving-basket] {out.name} step={step+1} stage={oracle.index}:{oracle.stage} '
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
        objects_in_basket=runtime.objects_in_basket(poses),
        stability_steps=runtime.tracker.steps,
        stack=stack_geometry(poses,runtime.basket_state()[0],runtime.cfg))
    if mode == 'contact':
        result['contact'] = contact_metrics(rows, runtime.cfg)
        result['passed'] = result['contact']['passed']
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
    if args.trials < 1 or args.contact_steps < 240 or args.max_steps < 1:
        raise ValueError('invalid trial or step count')
    cfg = MovingBasketConfig(amplitude=args.basket_amplitude,peak_speed=args.basket_peak_speed,
                             object_size=args.object_size)
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        raise RuntimeError('ffmpeg missing from PATH')
    configure_renderer(args.pathtracing_spp)
    env_cfg = make_env_cfg(cfg, device=args.device, seed=args.seed)
    env = gym.make(TASK, cfg=env_cfg).unwrapped
    results = []
    try:
        runtime = MovingBasketRuntime(env, cfg)
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
        prim = stage.GetPrimAtPath(str(env.scene.env_prim_paths[0])+'/Basket')
        assert UsdPhysics.RigidBodyAPI(prim).GetKinematicEnabledAttr().Get()
        colliders = [p for p in prim.GetChildren() if p.HasAPI(UsdPhysics.CollisionAPI)]
        assert len(colliders) == 5
        platform = stage.GetPrimAtPath(str(env.scene.env_prim_paths[0])+'/SourcePlatform')
        assert not platform.IsValid()
        belt = stage.GetPrimAtPath(str(env.scene.env_prim_paths[0])+'/Belt')
        assert belt.HasAPI(PhysxSchema.PhysxSurfaceVelocityAPI)
        assert math.isclose(cfg.basket_z,cfg.belt_top)
        for i in range(2):
            assert stage.GetPrimAtPath(str(env.scene.env_prim_paths[0])+f'/BeltRoller_{i}').IsValid()
            rail=stage.GetPrimAtPath(str(env.scene.env_prim_paths[0])+f'/ConveyorSide_{i}')
            assert rail.IsValid()
        assert not stage.GetPrimAtPath(str(prim.GetPath())+'/red_lattice').IsValid()
        (out/'protocol.json').write_text(json.dumps(dict(task=TASK, config=cfg.metadata(),
            camera_calibration=camera_metadata(), control_hz=30, physics_hz=120,
            oracle=MovingBasketOracle.PROFILE, basket_colliders=len(colliders), geometry_checked=True,
            conveyor_length_m=cfg.belt_length, conveyor_width_m=cfg.belt_width,
            basket_on_conveyor=True, basket_travel_axis='Y',container_style='open_cardboard_box',
            raised_side_frames_m=.008,stack_order=['gift_box','cardboard_box'],
            events=event_protocol, text_overlays=not args.no_labels, training_data=False, collection_contract_verified=False), indent=2)+'\n')
        if args.mode in ('contact', 'both'):
            r = episode(env, runtime, out/'contact', 'contact', args.seed, args.contact_steps, ffmpeg, dvs)
            results.append(r)
            if not r['passed']:
                raise RuntimeError('basket contact transport failed; see contact/result.json')
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
