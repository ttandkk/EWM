"""Independent tennis_container task entry point."""
from piper_bench.tasks.tennis.config import ContainerConfig as Config
from piper_bench.tasks.tennis.scene import TASK_INFO
from piper_bench.tasks.tennis.scene import TennisRuntime as Runtime
from piper_bench.tasks.tennis.scene import make_tennis_env_cfg

TASK, PROMPT = TASK_INFO['container']


def make_env_cfg(config=Config(), *, device='cpu', seed=42):
    if config.mode != 'container':
        raise ValueError('configuration mode differs from task')
    return make_tennis_env_cfg(config, device=device, seed=seed)
