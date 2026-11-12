"""Reset-only initialization and capture for the six existing physical experts."""
import math
import numpy as np

from piper_bench.tasks import load_task
from piper_bench.tasks.conveyor.core import CAMERAS
from piper_bench.sensors.capture import capture_frame


def initialize(runtime, spec):
    """Apply the precomputed initial condition before the first recorded step.

    No pose or joint writes in this module run during an expert rollout. Dynamic
    supports subsequently use their existing physical-substep kinematic drivers.
    """
    import torch
    name, env = spec["task"], runtime.env
    runtime.cfg = load_task(name).Config(**spec["config"])
    runtime.reset(spec["seed"])
    c = runtime.cfg
    if name.startswith("tennis_"):
        from piper_bench.tasks.tennis.collection import initialize as initialize_tennis
        initialize_tennis(runtime, spec)
        refresh_views(runtime, name)
        actual = runtime.state()[0]
        if not np.allclose(actual, spec["object_poses"], atol=1e-5):
            raise RuntimeError("tennis initial pose differs from manifest")
        value = dict(object_poses=actual.tolist(), object_velocities=runtime.state()[1].tolist(),
                     object_active=list(runtime.active), launch=runtime.metadata(), cabinet_pose=None,
                     appearance=runtime.tabletop_appearance,
                     robot_joints=env.scene["robot"].data.joint_pos[0].detach().cpu().tolist())
        if name == "tennis_container":
            value["container_pose"], value["container_velocity"] = [v.tolist() for v in runtime.receiver_state()]
        return value
    if name == "conveyor":
        from piper_bench.collection.conveyor_speed import set_episode_speed
        set_episode_speed(runtime, c.speed)
    if name in ("moving_carton", "rotating_tube"):
        runtime.phase = spec["phase"]
        if name == "moving_carton":
            from pxr import Gf
            position, velocity = runtime.basket_target(0.)
            runtime.target_pose[0, :3] = torch.as_tensor(position, dtype=torch.float32, device=env.device)
            runtime.basket.write_root_pose_to_sim(runtime.target_pose)
            runtime.basket_view.set_kinematic_targets(runtime.target_pose, runtime.target_ids)
            runtime.surface_velocity.Set(Gf.Vec3f(*velocity))
        else:
            runtime.target_pose[0, 5] = math.sin(runtime.phase/2)
            runtime.target_pose[0, 6] = math.cos(runtime.phase/2)
            runtime.rack.write_root_pose_to_sim(runtime.target_pose)
            runtime.rack_view.set_kinematic_targets(runtime.target_pose, runtime.target_ids)
            runtime.tube_yaw = spec["object_yaws"][0]
    if name == "turntable_cups":
        from isaaclab.sim.utils.stage import get_current_stage
        from pxr import PhysxSchema
        from pxr import Gf
        prim = get_current_stage().GetPrimAtPath(str(env.scene.env_prim_paths[0]) + "/Turntable")
        PhysxSchema.PhysxSurfaceVelocityAPI(prim).GetSurfaceAngularVelocityAttr().Set(
            Gf.Vec3f(0., 0., math.degrees(c.omega)))
        runtime.rotation_op.Set(0.)
    if spec["cabinet_pose"] is not None:
        cabinet = env.scene["cabinet"]
        cabinet.write_root_pose_to_sim(torch.tensor([spec["cabinet_pose"]], dtype=torch.float32, device=env.device))
    poses = np.asarray(spec["object_poses"])
    for obj, pose in zip(runtime.objects, poses):
        obj.write_root_pose_to_sim(torch.tensor(pose[None], dtype=torch.float32, device=env.device))
        obj.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))
    runtime.active = [True] * len(poses)
    if hasattr(runtime, "supplied"):
        runtime.supplied = True
    if hasattr(runtime, "spawn_x"):
        runtime.spawn_x, runtime.spawn_y, runtime.spawn_yaw = poses[:, 0], poses[:, 1], np.array(spec["object_yaws"])
    refresh_views(runtime, name)
    actual = runtime.state()[0]
    if not np.allclose(actual, poses, atol=1e-5):
        raise RuntimeError("initial object poses differ from sampling manifest")
    if name == "drawer_cup":
        actual_cabinet = env.scene["cabinet"].data.root_pose_w[0].detach().cpu().numpy()
        if not np.allclose(actual_cabinet, spec["cabinet_pose"], atol=1e-5):
            raise RuntimeError("cabinet initialization did not reach sampled pose")
    result = dict(object_poses=actual.tolist(), object_velocities=runtime.state()[1].tolist(),
                  cabinet_pose=None if name != "drawer_cup" else actual_cabinet.tolist(),
                  robot_joints=env.scene["robot"].data.joint_pos[0].detach().cpu().tolist())
    if name in ("conveyor", "moving_carton", "turntable_cups", "rotating_tube"):
        result['appearance'] = runtime.tabletop_appearance
    return result


def refresh_views(runtime, name):
    env = runtime.env
    load_task(name).set_views(env)
    env.sim.forward()
    env.sim.render()
    for camera in CAMERAS:
        env.scene[camera].update(0., force_recompute=True)


def step_result(runtime, name, action):
    joints = runtime.env.scene["robot"].data.joint_pos[0].detach().cpu().numpy()
    opened = bool(action[0, -1] > 0 and abs(joints[-2]-joints[-1]) > .045)
    if name in ("moving_carton", "turntable_cups"):
        success, reason = runtime.after_step()
    else:
        success, reason = runtime.after_step(opened)
    return bool(success), reason, opened


def terminal_state(runtime, name, opened):
    poses, velocities, tcp = runtime.state()
    state = dict(poses=poses.tolist(), velocities=velocities.tolist(), tcp=tcp.tolist(),
                 gripper_open=opened, time=runtime.time)
    if name == "moving_carton":
        pose, velocity = runtime.basket_state()
        state.update(basket_pose=pose.tolist(), basket_velocity=velocity.tolist())
    elif name == "rotating_tube":
        state.update(phase=runtime.phase, accepted_index=runtime.accepted_index)
    elif name == "drawer_cup":
        opening, speed = runtime.opening()
        state.update(opening=opening, drawer_speed=speed, opened=runtime.tracker.opened,
                     placed=runtime.tracker.placed)
    tracker = getattr(runtime, "tracker", None)
    state["stable_steps"] = int(getattr(runtime, "stable_steps", getattr(tracker, "steps", 0)))
    if name == 'moving_carton' and runtime.cfg.stack_success_mode == 'lowwall_v4':
        state['stack_pose_window'] = list(tracker.pose_window)
    if name.startswith("tennis_"):
        state["catch_window"] = list(runtime.collection_catch_window)
    return state


def terminal_valid(name, config, state):
    """Recheck saved terminal truth with the existing task predicates on CPU."""
    cfg = load_task(name).Config(**config)
    if name.startswith("tennis_"):
        from piper_bench.tasks.tennis.collection import terminal_valid as tennis_valid
        return tennis_valid(cfg, state)
    p, v, tcp = (np.array(state[key]) for key in ("poses", "velocities", "tcp"))
    if name == "conveyor":
        from piper_bench.tasks.conveyor.core import success_now
        valid = success_now(p, v, tcp, [True, True], cfg)
    elif name == "moving_carton":
        from piper_bench.tasks.moving_carton.scene import moving_success_now
        valid = moving_success_now(p, v, tcp, [True, True], np.array(state["basket_pose"]),
                                   np.array(state["basket_velocity"]), cfg)
        if cfg.stack_success_mode == 'lowwall_v4':
            from piper_bench.tasks.moving_carton.lowwall import stable_window
            from piper_bench.tasks.moving_carton.lowwall import relative_corners
            window = state.get('stack_pose_window', [])
            valid = (valid and stable_window(window, cfg)
                     and np.allclose(window[-1], relative_corners(p, state['basket_pose'], cfg), atol=1e-8))
    elif name == "turntable_cups":
        from piper_bench.tasks.turntable_cups.scene import NestingTracker
        tracker = NestingTracker(cfg)
        tracker.steps = round(cfg.stable_seconds*30)-1
        valid = tracker.update(p, v, tcp, [True, True])
    elif name == "rotating_tube":
        from piper_bench.tasks.rotating_tube.scene import insertion_metrics
        from piper_bench.tasks.rotating_tube.scene import hole_motion
        hole = hole_motion(cfg, state["time"], state["phase"], state["accepted_index"])
        valid = insertion_metrics(p[0], v[0], tcp, cfg, hole, state["gripper_open"])["valid"]
    elif name == "drawer_cup":
        from piper_bench.tasks.drawer_cup.scene import SequenceTracker
        tracker = SequenceTracker()
        tracker.opened, tracker.placed = state["opened"], state["placed"]
        valid = tracker.update(p[0], v[0], tcp, state["opening"], state["drawer_speed"],
                               state["gripper_open"], cfg)["valid"]
    elif name == "push_t":
        from piper_bench.tasks.push_t.scene import pose_metrics
        valid = pose_metrics(p[0], v[0], tcp, cfg)["valid"]
    else:
        from piper_bench.tasks.square_nut.scene import assembly_metrics
        valid = assembly_metrics(p[0], v[0], tcp, state["gripper_open"], cfg)["valid"]
    return bool(valid and state["stable_steps"] >= round(cfg.stable_seconds*30))


def frame(runtime, name, action, events):
    result = capture_frame(runtime, action, events)
    # Drawer position is diagnostic ground truth, never added to the 8-D policy state.
    if name == "drawer_cup":
        result["observation.drawer_state"] = np.asarray(runtime.opening(), dtype=np.float32)
    if name == "tennis_container":
        pose, velocity = runtime.receiver_state()
        result["observation.container_pose"] = pose.astype(np.float32)
        result["observation.container_velocity"] = velocity.astype(np.float32)
    return result
