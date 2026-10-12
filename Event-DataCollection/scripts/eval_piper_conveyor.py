"""Render and evaluate the shared Piper conveyor scene on a Slurm GPU node."""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir', type=Path, required=True)
parser.add_argument('--mode', choices=['scene', 'transport', 'reach', 'ik_check', 'jacobian_check', 'oracle_eval'], default='scene')
parser.add_argument('--trials', type=int, default=1)
parser.add_argument('--initial-hold-steps', type=int, default=1)
parser.add_argument('--max-steps', '--max_steps', dest='max_steps', type=int, default=180)
parser.add_argument('--seed', type=int, default=20261002)
parser.add_argument('--yaw-jitter-deg',type=float,default=10.)
parser.add_argument('--spacing-jitter',type=float,default=0.)
parser.add_argument('--pathtracing-spp', type=int, default=64)
parser.add_argument('--preview-resolution', type=int, choices=[256,512,1024], default=256)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(device="cpu")
args = parser.parse_args()
app = AppLauncher(args).app

import carb
import gymnasium as gym
import numpy as np
import torch
from PIL import Image, ImageDraw
from piper_conveyor import CAMERAS, TASK, ConveyorConfig, ConveyorRuntime, camera_metadata, make_env_cfg, set_views


def configure_renderer():
    from piper_conveyor_io import configure_renderer as configure
    configure(args.pathtracing_spp)


def camera_image(env, label):
    panels = []
    for name in CAMERAS:
        value = env.scene[name].data.output['rgb'][0, ..., :3].detach().cpu().numpy().astype(np.uint8)
        im = Image.fromarray(value)
        ImageDraw.Draw(im).text((5, 5), f'{name} {label}', fill='white', stroke_width=1, stroke_fill='black')
        panels.append(im)
    resolution = panels[0].width
    combined = Image.new('RGB', (2*resolution, resolution))
    for i, panel in enumerate(panels):
        combined.paste(panel, (resolution*i, 0))
    return combined


def evaluate_trials(env, runtime, zero, out):
    """Independent seeded episodes; sparse rendering leaves fixed-step physics unchanged."""
    from piper_conveyor_oracle import ConveyorOracle
    from piper_conveyor_io import refresh_spawn_views
    if args.trials < 1 or args.initial_hold_steps < 0:
        raise ValueError('trials must be positive and initial hold must be nonnegative')
    results = []
    original_render_interval = env.cfg.sim.render_interval
    try:
        # Oracle measurements need physics and state, without RGB/event generation.
        env.cfg.sim.render_interval = 1000000
        for trial in range(args.trials):
            seed = args.seed + trial
            env.reset()
            for _ in range(60):
                env.step(zero)
            runtime.reset(seed)
            runtime.supply()
            oracle = ConveyorOracle(runtime)
            if trial == 0:
                set_views(env)
                refresh_spawn_views(env)
                camera_image(env, 'initial').save(out/'scene_dual.png')
            success, reason = False, None
            rows = []
            for step in range(args.max_steps):
                runtime.supply()
                action = zero if step < args.initial_hold_steps else oracle.action()
                env.step(action)
                success, reason = runtime.after_step()
                poses, velocities, tcp = runtime.state()
                if step % 30 == 0 or success or reason:
                    rows.append(dict(step=step,time=runtime.time,stage=oracle.stage,
                                     poses=poses.tolist(),velocities=velocities.tolist(),tcp=tcp.tolist(),
                                     success=success,reason=reason))
                if success or reason:
                    break
            record = dict(trial=trial,seed=seed,success=success,reason=reason or ('success' if success else 'step_limit'),
                          frames=step+1,sim_seconds=runtime.time,stability_steps=runtime.tracker.steps,
                          grasp_lifted=oracle.grasp_lifted,oracle_profile=oracle.PROFILE,retries=oracle.retries,scene=runtime.metadata(),
                          terminal_poses=poses.tolist(),terminal_velocities=velocities.tolist(),terminal_tcp=tcp.tolist())
            results.append(record)
            (out/f'trial_{trial:03d}.json').write_text(json.dumps(dict(**record,trajectory=rows),indent=2)+'\n')
            if trial == 0 or not success:
                set_views(env)
                refresh_spawn_views(env)
                camera_image(env, f'trial {trial} success={success}').save(out/f'trial_{trial:03d}_terminal.png')
            passed = sum(r['success'] for r in results)
            summary = dict(mode='oracle_eval',policy='physical_feedback_oracle',requested_trials=args.trials,
                           completed_trials=len(results),successes=passed,success_rate=passed/len(results),
                           initial_hold_steps=args.initial_hold_steps,seed_base=args.seed,
                           target_success_rate=.95,passed=len(results)==args.trials and passed/len(results)>=.95,
                           simulation_clock='fixed_simulation_time_inference_paused',control_hz=30,physics_hz=120,
                           camera_calibration=camera_metadata(),results=results)
            (out/'result.json').write_text(json.dumps(summary,indent=2)+'\n')
            print('[conveyor-evaluation] '+json.dumps({k:v for k,v in record.items() if k not in ('scene','terminal_poses','terminal_velocities','terminal_tcp')}),flush=True)
        if not summary['passed']:
            raise RuntimeError(f"oracle success rate {summary['success_rate']:.1%} is below 95%")
    finally:
        env.cfg.sim.render_interval = original_render_interval
        env.close()


def main():
    out = args.output_dir.resolve()
    if out.exists():
        raise FileExistsError(f'use a fresh output directory: {out}')
    out.mkdir(parents=True)
    cfg = ConveyorConfig(yaw_jitter=float(np.deg2rad(args.yaw_jitter_deg)),object_spacing_jitter=args.spacing_jitter)
    configure_renderer()
    env_cfg = make_env_cfg(cfg, device=args.device, seed=args.seed)
    for name in CAMERAS:
        getattr(env_cfg.scene, name).height = args.preview_resolution
        getattr(env_cfg.scene, name).width = args.preview_resolution
    env = gym.make(TASK, cfg=env_cfg).unwrapped
    from pxr import Usd, UsdGeom, UsdPhysics
    import isaaclab.sim as sim_utils
    stage = sim_utils.get_current_stage()
    debug = {}
    for name in ('belt', 'cube_1', 'cube_2'):
        asset = env.scene[name]
        path = asset.root_view.prim_paths[0]
        root = stage.GetPrimAtPath(path)
        debug[name] = dict(path=path, max_shapes=asset.root_view.max_shapes,
            disable_simulation=asset.root_view.get_disable_simulations().numpy().tolist(),
            transforms=asset.root_view.get_transforms().numpy().tolist(),
            contact_offsets=asset.root_view.get_contact_offsets().numpy().tolist(),
            pose=asset.data.root_pos_w[0].detach().cpu().tolist(),
            prims=[dict(path=str(prim.GetPath()), schemas=prim.GetAppliedSchemas(),
                        attributes={a.GetName(): str(a.Get()) for a in prim.GetAttributes()
                                    if a.GetName().startswith(('physics:', 'physx', 'xformOp:', 'size'))})
                   for prim in Usd.PrimRange(root)])
    (out/'physics_authoring.json').write_text(json.dumps(debug, indent=2)+'\n')
    print('[conveyor-physics] authoring saved', flush=True)
    runtime = ConveyorRuntime(env, cfg)
    zero = torch.tensor([[0., 0., 0., 0., 0., 0., 1.]], device=env.device)
    from piper_conveyor_oracle import ConveyorOracle
    oracle = ConveyorOracle(runtime)
    if args.mode == 'oracle_eval':
        evaluate_trials(env, runtime, zero, out)
        return
    rows = []
    video = None
    try:
        env.reset()
        for _ in range(60):
            set_views(env)
            env.step(zero)
        if args.mode == 'jacobian_check':
            from isaaclab.utils.math import compute_pose_error
            arm = env.action_manager.get_term('arm_action')
            robot = env.scene['robot']
            analytical = arm._compute_frame_jacobian()[0].detach().cpu().numpy()
            original = robot.data.joint_pos.torch.clone()
            numerical = np.zeros_like(analytical)
            epsilon = .001
            for j in range(6):
                measurements = []
                for sign in (-1,1):
                    positions = original.clone(); positions[0,j] += sign*epsilon
                    robot.write_joint_position_to_sim_index(position=positions)
                    robot.set_joint_position_target_index(target=positions)
                    robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(positions))
                    env.sim.forward()
                    robot.update(.001)
                    measurements.append(tuple(v.clone() for v in arm._compute_frame_pose()))
                dp, dq = compute_pose_error(*measurements[0],*measurements[1],rot_error_type='axis_angle')
                numerical[:,j] = torch.cat((dp[0],dq[0])).detach().cpu().numpy()/(2*epsilon)
            error = float(np.abs(analytical-numerical).max())
            result = dict(mode=args.mode, analytical=analytical.tolist(),numerical=numerical.tolist(),
                          max_error=error, jacobian_passed=error<.01)
            (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
            print('[jacobian-check] '+json.dumps(result),flush=True)
            if not result['jacobian_passed']: raise RuntimeError('TCP Jacobian finite-difference check failed')
            return
        import omni.physx
        queries = omni.physx.get_physx_scene_query_interface()
        probe = dict(raycast=queries.raycast_closest((cfg.belt_x,cfg.spawn_y,.30),(0.,0.,-1.),1.0))
        from pxr import Usd, UsdGeom
        bounds = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_]).ComputeWorldBound(
            stage.GetPrimAtPath('/World/envs/env_0/Belt')).ComputeAlignedRange()
        probe['usd_bounds'] = [list(bounds.GetMin()),list(bounds.GetMax())]
        probe['relationships'] = {str(p.GetPath()):{r.GetName():[str(v) for v in r.GetTargets()] for r in p.GetRelationships()}
                                  for p in Usd.PrimRange(stage.GetPrimAtPath('/World/envs/env_0'))
                                  if p.GetRelationships()}
        (out/'contact_probe.json').write_text(json.dumps(probe, default=str, indent=2)+'\n')
        print('[contact-probe] '+str(probe['raycast']),flush=True)
        runtime.reset(args.seed)
        runtime.supply()
        set_views(env)
        env.step(zero)
        runtime.after_step()
        initial = camera_image(env, 'initial')
        initial.save(out/'scene_dual.png')
        for i, name in enumerate(CAMERAS):
            initial.crop((i*args.preview_resolution, 0, (i+1)*args.preview_resolution, args.preview_resolution)).save(out/f'scene_{name}.png')
        ffmpeg = shutil.which('ffmpeg') or str(Path(__file__).resolve().parents[2]/'envs/torchcodec_ffmpeg7/bin/ffmpeg')
        if not Path(ffmpeg).is_file():
            raise RuntimeError('ffmpeg is required for validated H.264 previews')
        video = subprocess.Popen([ffmpeg, '-loglevel', 'error', '-y', '-f', 'rawvideo',
            '-pix_fmt', 'rgb24', '-s', f'{2*args.preview_resolution}x{args.preview_resolution}', '-r', '30', '-i', 'pipe:0', '-an',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out/'transport.mp4')],
            stdin=subprocess.PIPE)
        keyframes = [initial.copy()]
        for step in range(args.max_steps):
            runtime.supply()
            set_views(env)
            action = oracle.action() if args.mode == "reach" else zero
            if args.mode == 'ik_check':
                action = zero.clone()
                action[0, (step//30)%3] = .002
            env.step(action)
            success, reason = runtime.after_step()
            poses, velocities, tcp = runtime.state()
            rows.append(dict(step=step, time=runtime.time, oracle_stage=oracle.stage,
                             oracle_target=None if oracle.target is None else oracle.target.tolist(),
                             orientation_error=float(oracle.orientation_error),
                             action=action[0].detach().cpu().tolist(), belt_pose=env.scene['belt'].data.root_pos_w[0].detach().cpu().tolist(), poses=poses.tolist(),
                             velocities=velocities.tolist(), tcp=tcp.tolist(), tcp_quat=env.scene["ee_frame"].data.target_quat_w[0,0].detach().cpu().tolist(),
                             active=runtime.active.copy(),
                             joint_pos=env.scene['robot'].data.joint_pos[0].detach().cpu().tolist(),
                             joint_vel=env.scene['robot'].data.joint_vel[0].detach().cpu().tolist(),
                             success=success, reason=reason))
            if step % 60 == 0:
                print(f'[conveyor-step] step={step} time={runtime.time:.2f} stage={oracle.stage} '
                      f'tcp={tcp.round(3).tolist()} objects={poses[:,:3].round(3).tolist()}', flush=True)
                (out/'progress.json').write_text(json.dumps(rows[-1], indent=2)+'\n')
            frame = camera_image(env, f'{runtime.time:.2f}s')
            video.stdin.write(frame.tobytes())
            if step in (29, 89, args.max_steps-1) or (args.mode == "reach" and step % 60 == 0):
                frame.save(out/f'transport_step_{step+1:04d}.png')
                keyframes.append(frame.copy())
            if args.mode == 'reach' and (success or reason):
                frame.save(out/'terminal_dual.png')
                keyframes.append(frame.copy())
                break
        video.stdin.close()
        if video.wait(timeout=60):
            raise RuntimeError('video encoding failed')
        video = None
        subprocess.run([ffmpeg, '-v', 'error', '-i', str(out/'transport.mp4'), '-f', 'null', '-'], check=True)
        # Measure free transport before contact; placed objects have zero velocity.
        speeds = np.array([r['velocities'][0][1] for r in rows
                           if r['time'] > 1.0 and r['active'][0]
                           and abs(r['poses'][0][2]-(cfg.belt_top+cfg.object_size/2)) < .005
                           and np.linalg.norm(np.asarray(r['poses'][0][:3])-r['tcp']) > .07
                           and abs(r['poses'][0][1]) < cfg.belt_length/2])
        measured = float(np.median(speeds)) if len(speeds) else None
        passed = measured is not None and abs(measured-cfg.speed) <= cfg.speed*0.1
        sheet = Image.new('RGB', (2*args.preview_resolution, args.preview_resolution*len(keyframes)))
        for i, im in enumerate(keyframes):
            sheet.paste(im, (0, args.preview_resolution*i))
        sheet.save(out/'transport_contact_sheet.png')
        (out/'steps.json').write_text(json.dumps(rows, indent=2)+'\n')
        calibration = camera_metadata()
        calibration["resolution_hw"] = [args.preview_resolution]*2
        result = dict(mode=args.mode, expected_speed=cfg.speed, measured_speed=measured,
                      transport_passed=passed, camera_calibration=calibration,
                      scene=runtime.metadata(), frame_count=len(rows), video_decoded=True, grasp_lifted=oracle.grasp_lifted,
                      task_success=any(r["success"] for r in rows), stability_steps=runtime.tracker.steps,oracle_profile=oracle.PROFILE,retries=oracle.retries)
        (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        print('[conveyor-probe] '+json.dumps(result), flush=True)
        if args.mode == 'reach' and not result['task_success']:
            raise RuntimeError('physical pick-and-place reach test failed')
        if args.mode == 'transport' and not passed:
            raise RuntimeError(f'conveyor speed failed: {measured} != {cfg.speed}')
    finally:
        if video is not None:
            if video.stdin and not video.stdin.closed:
                video.stdin.close()
            video.wait(timeout=60)
        env.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        import os
        import sys
        import traceback
        traceback.print_exc()
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir/'failure.json').write_text(json.dumps({
            'error': type(exc).__name__, 'message': str(exc)}, indent=2)+'\n')
        sys.stdout.flush()
        sys.stderr.flush()
        # Kit's shutdown handlers can mask unhandled Python exceptions.
        os._exit(1)
    else:
        app.close()
