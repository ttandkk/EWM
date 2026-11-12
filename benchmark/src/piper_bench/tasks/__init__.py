"""Piper task registry; importing it does not start Isaac Sim."""
from importlib import import_module


TASK_MODULES = {
    "conveyor": 'piper_bench.tasks.conveyor',
    "moving_carton": 'piper_bench.tasks.moving_carton',
    "turntable_cups": 'piper_bench.tasks.turntable_cups',
    "rotating_tube": 'piper_bench.tasks.rotating_tube',
    "push_t": 'piper_bench.tasks.push_t',
    "drawer_cup": 'piper_bench.tasks.drawer_cup',
    "square_nut": 'piper_bench.tasks.square_nut',
    "tennis_rolling": 'piper_bench.tasks.tennis_rolling',
    "tennis_falling": 'piper_bench.tasks.tennis_falling',
    "tennis_parabolic": 'piper_bench.tasks.tennis_parabolic',
    "tennis_container": 'piper_bench.tasks.tennis_container',
}


def load_task(name):
    """Return a task's Config, Runtime, Oracle, camera and scene factory API."""
    try:
        module = TASK_MODULES[name]
    except KeyError as exc:
        raise ValueError(f"unknown task {name!r}; choose from {', '.join(TASK_MODULES)}") from exc
    return import_module(module)


def register_tasks():
    """Register fully qualified Gym config entry points, without constructing scenes."""
    import gymnasium as gym

    for name in TASK_MODULES:
        task = load_task(name)
        if task.TASK not in gym.registry:
            gym.register(
                task.TASK,
                entry_point="isaaclab.envs:ManagerBasedRLEnv",
                disable_env_checker=True,
                kwargs={"env_cfg_entry_point": f"{task.__name__}.scene:make_env_cfg"},
            )

