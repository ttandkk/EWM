"""tennis_falling: contact-only dynamic tennis manipulation."""
from piper_bench.tasks.tennis_falling.scene import TASK
from piper_bench.tasks.tennis_falling.scene import PROMPT
from piper_bench.tasks.tennis_falling.scene import Config
from piper_bench.tasks.tennis_falling.scene import Runtime
from piper_bench.tasks.tennis_falling.scene import make_env_cfg
from piper_bench.tasks.tennis.camera import camera_metadata as shared_camera_metadata
from piper_bench.tasks.tennis.camera import set_views
from piper_bench.tasks.tennis.oracle import TennisOracle


def camera_metadata():
    return shared_camera_metadata('fall')


class Oracle(TennisOracle):
    PROFILE = 'tennis_falling_sampled_speed_contact_v2'
