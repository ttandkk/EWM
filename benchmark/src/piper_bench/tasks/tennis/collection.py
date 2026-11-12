"""Collection-only tennis initialization and independently replayed terminal truth."""
import numpy as np

from piper_bench.tasks.tennis.metrics import CatchTracker
from piper_bench.tasks.tennis.metrics import container_metrics
from piper_bench.tasks.tennis.metrics import gripper_metrics
from piper_bench.tasks.tennis.motion import launch_state
from piper_bench.tasks.tennis.motion import launch_time


def initialize(runtime, spec):
    # reset() already parked the ball. Verify the manifest instead of overriding
    # its sampled trajectory or borrowing any interception target from the expert.
    p, v = launch_state(runtime.cfg, spec["seed"])
    if (not np.allclose(p, spec["launch_position_m"], atol=1e-12)
            or not np.allclose(v, spec["launch_velocity"], atol=1e-12)
            or launch_time(runtime.cfg, spec["seed"]) != spec["launch_time_s"]):
        raise ValueError("tennis launch differs from the sampling manifest")
    if runtime.launch_time_s == 0:
        runtime.supply()


def success_sample(runtime):
    poses, velocities, tcp = runtime.state()
    robot = runtime.env.scene["robot"].data
    joints = robot.joint_pos[0].detach().cpu().numpy()
    q = runtime.env.scene["ee_frame"].data.target_quat_w[0, 0].detach().cpu().numpy()
    sample = dict(time=runtime.time, ball=poses[0].tolist(), velocity=velocities[0].tolist(),
                  tcp=tcp.tolist(), tool_quat=q.tolist(), tcp_velocity=runtime.tcp_velocity.tolist(),
                  aperture=float(abs(joints[-2]-joints[-1])), closed=runtime.gripper_closed,
                  launched=runtime.launched, table_hit=runtime.table_hit)
    if runtime.cfg.mode == "container":
        pose, velocity = runtime.receiver_state()
        sample.update(container=pose.tolist(), container_velocity=velocity.tolist())
    return sample


def terminal_valid(cfg, state):
    samples = state.get("catch_window", [])
    tracker = CatchTracker(cfg)
    if len(samples) != tracker.window_samples:
        return False
    times = np.asarray([s["time"] for s in samples])
    if (not np.allclose(np.diff(times), 1/30, atol=1e-9, rtol=0)
            or abs(times[-1]-state["time"]) > 1e-9
            or not np.allclose(samples[-1]["ball"], state["poses"][0], atol=1e-9)):
        return False
    for sample in samples:
        if cfg.mode == "container":
            metric = container_metrics(np.array(sample["ball"]), np.array(sample["velocity"]),
                np.array(sample["container"]), np.array(sample["container_velocity"]),
                np.array(sample["tcp"]), sample["aperture"], sample["closed"], cfg)
        else:
            metric = gripper_metrics(np.array(sample["ball"]), np.array(sample["velocity"]),
                np.array(sample["tcp"]), np.array(sample["tool_quat"]),
                np.array(sample["tcp_velocity"]), sample["aperture"], sample["closed"], cfg)
        success = tracker.update(metric, sample["launched"], sample["table_hit"])
    return bool(success and state["stable_steps"] >= tracker.window_samples)
