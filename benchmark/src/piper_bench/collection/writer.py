"""Piper task LeRobot writer; preserves the observation/action contract."""
import argparse
import sys

from piper_bench.tasks import load_task
from piper_bench.collection.protocol import OBJECT_COUNTS
from piper_bench.collection.protocol import COLLECTION_TASKS
from piper_bench.collection.protocol import VERSION
from piper_bench.collection.protocol import CONTAINER_SPATIAL_VERSIONS
from piper_bench.collection.sampling import physics_hz
from piper_bench.collection.sampling import CARTON_LOWWALL_VERSION
from piper_bench.collection.sampling import LOW_GEOMETRY_VERSION
from piper_bench.collection.sampling import VERSIONS as DYNAMIC_VERSIONS


def features(task):
    from piper_bench.collection.lerobot_writer import features as stack_features
    result = stack_features()
    result.pop("observation.cube_pose")
    count = OBJECT_COUNTS[task]
    for name, size in (("object_pose", 7*count), ("object_velocity", 6*count),
                       ("object_active", count), ("sim_time", 1)):
        result["observation."+name] = dict(dtype="float32", shape=(size,), names=None)
    if task == "drawer_cup":
        result["observation.drawer_state"] = dict(dtype="float32", shape=(2,), names=None)
    if task == "tennis_container":
        result["observation.container_pose"] = dict(dtype="float32", shape=(7,), names=None)
        result["observation.container_velocity"] = dict(dtype="float32", shape=(6,), names=None)
    return result


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task", choices=COLLECTION_TASKS, required=True)
    parser.add_argument("--raw-event-archive", action="store_true")
    from piper_bench.tasks.push_t.protocol import VERSION as PUSH_VERSION
    parser.add_argument("--collection-protocol", choices=(VERSION, PUSH_VERSION, *DYNAMIC_VERSIONS), default=VERSION)
    args, rest = parser.parse_known_args()
    if args.collection_protocol == PUSH_VERSION and args.task != "push_t":
        parser.error("Push-T pose-v2 protocol requires --task push_t")
    if args.collection_protocol in CONTAINER_SPATIAL_VERSIONS and args.task != "tennis_container":
        parser.error("Container spatial protocol requires --task tennis_container")
    if args.collection_protocol == CARTON_LOWWALL_VERSION and args.task != 'moving_carton':
        parser.error('Carton low-wall-v4 protocol requires --task moving_carton')
    if args.collection_protocol == LOW_GEOMETRY_VERSION and args.task not in ('moving_carton', 'turntable_cups'):
        parser.error('Low-geometry-v5 protocol requires moving_carton or turntable_cups')
    sys.argv = [sys.argv[0], *rest]
    from piper_bench.collection.lerobot_writer import main as write
    task = load_task(args.task)
    extra = {}
    if args.raw_event_archive:
        from piper_bench.sensors.event_archive import CONTRACT
        extra["event_archive"] = CONTRACT
    write(task_prompt=task.PROMPT, feature_builder=lambda: features(args.task), contract_overrides={
        "task_id": task.TASK, "task_selector": args.task, "collection_protocol": args.collection_protocol,
        "object_count": OBJECT_COUNTS[args.task], "physics_hz": physics_hz(args.task),
        "physics_backend": "CPU PhysX, GPU RTX rendering",
        "simulation_clock": "fixed_simulation_time_inference_paused", **extra})


if __name__ == "__main__":
    main()
