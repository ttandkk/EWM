"""Versioned Push-T pose coverage and shared, immutable evaluation manifests."""
import argparse
import hashlib
import json
import math
from pathlib import Path

VERSION = "piper_push_t_pose_v2"
POSITION_JITTER = .03
YAW_JITTER = math.pi / 4


def configure_plan(plan):
    """Latin-hypercube marginals; 50 slots give ten in each 18-degree band."""
    from piper_bench.collection.protocol import rng
    count = plan["episodes"]
    orders = [rng(plan["seed"], "push_t", plan["split"], stream=stream).permutation(count)
              for stream in (8, 9, 10)]
    plan.update(version=VERSION, position_jitter_m=POSITION_JITTER, yaw_jitter_rad=YAW_JITTER,
                pose_sampling="shuffled_uniform_strata", yaw_groups=5, retry="keep_pose_strata")
    for i, slot in enumerate(plan["slots"]):
        slot["pose_strata"] = [int(order[i]) for order in orders]


def sample_pose(plan, slot, retry=0):
    from piper_bench.collection.protocol import rng
    if plan["version"] != VERSION:
        raise ValueError("unsupported Push-T pose protocol")
    record = plan["slots"][slot]
    if "fixed_pose" in record:
        return record["fixed_pose"]["offset_xy"], record["fixed_pose"]["yaw"]
    count = plan["episodes"]
    strata = record["pose_strata"]
    position = rng(plan["seed"], "push_t", plan["split"], slot, retry, 1).random(2)
    orientation = rng(plan["seed"], "push_t", plan["split"], slot, retry, 2).random()
    xy = [-POSITION_JITTER + 2*POSITION_JITTER*(strata[i]+position[i])/count for i in range(2)]
    yaw = -YAW_JITTER + 2*YAW_JITTER*(strata[2]+orientation)/count
    return xy, float(yaw)


def boundary_plan(seed=20261016):
    from piper_bench.collection.protocol import make_plan
    positions = [(0., 0.), (-.03, -.03), (-.03, .03), (.03, -.03), (.03, .03)]
    angles = (-45, -30, -15, 0, 15, 30, 45)
    plan = make_plan("push_t", len(positions)*len(angles), seed, "acceptance")
    plan.update(pose_sampling="fixed_boundary_cases", retry="same_pose")
    for slot, (xy, angle) in zip(plan["slots"], ((xy, a) for xy in positions for a in angles)):
        slot.pop("pose_strata")
        slot["fixed_pose"] = dict(offset_xy=list(xy), yaw=math.radians(angle))
    return plan


def manifest(plan):
    from piper_bench.collection.protocol import episode_spec
    return dict(plan=plan, episodes=[episode_spec(plan, i) for i in range(plan["episodes"])])


def load_manifest(path):
    """Reject stale/tampered poses rather than silently regenerating an eval set."""
    from piper_bench.collection.protocol import make_plan
    data = json.loads(Path(path).read_text())
    plan = data["plan"]
    if plan["task"] != "push_t" or plan["version"] != VERSION:
        raise ValueError("expected Push-T pose-v2 manifest")
    expected = (boundary_plan(plan['seed']) if plan['pose_sampling'] == 'fixed_boundary_cases'
                else make_plan('push_t', plan['episodes'], plan['seed'], plan['split']))
    if plan != expected:
        raise ValueError("manifest sampling plan differs from the versioned protocol")
    if data != manifest(plan):
        raise ValueError("manifest initial states differ from their sampling plan")
    return data


def write_manifests(directory, seed=20261016):
    from piper_bench.collection.protocol import make_plan
    from piper_bench.collection.protocol import write_json
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    plans = {"train_50": make_plan("push_t", 50, seed, "train"),
             "validation_10": make_plan("push_t", 10, seed, "validation"),
             "evaluation_50": make_plan("push_t", 50, seed, "evaluation"),
             "boundary_35": boundary_plan(seed)}
    for name, plan in plans.items():
        write_json(directory / (name+".json"), manifest(plan))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.glob("*.json"))}
    write_json(directory / "sha256.json", hashes)
    return hashes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261016)
    args = parser.parse_args()
    print(json.dumps(write_manifests(args.output_dir, args.seed), indent=2))
