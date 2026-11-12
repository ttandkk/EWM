"""tennis_container: contact-only dynamic tennis manipulation."""
from piper_bench.tasks.tennis_container.scene import TASK
from piper_bench.tasks.tennis_container.scene import PROMPT
from piper_bench.tasks.tennis_container.scene import Config
from piper_bench.tasks.tennis_container.scene import Runtime
from piper_bench.tasks.tennis_container.scene import make_env_cfg
from piper_bench.tasks.tennis.camera import camera_metadata as _camera_metadata
from piper_bench.tasks.tennis.camera import set_views
from piper_bench.tasks.tennis_container.oracle import Oracle


def camera_metadata():
    return _camera_metadata('container')
