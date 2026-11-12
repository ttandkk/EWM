"""tennis_parabolic: contact-only dynamic tennis manipulation."""
from piper_bench.tasks.tennis_parabolic.scene import TASK
from piper_bench.tasks.tennis_parabolic.scene import PROMPT
from piper_bench.tasks.tennis_parabolic.scene import Config
from piper_bench.tasks.tennis_parabolic.scene import Runtime
from piper_bench.tasks.tennis_parabolic.scene import make_env_cfg
from piper_bench.tasks.tennis.camera import camera_metadata as shared_camera_metadata
from piper_bench.tasks.tennis.camera import set_views


def camera_metadata():
    return shared_camera_metadata('lob')
from piper_bench.tasks.tennis.oracle import TennisOracle


class Oracle(TennisOracle):
    PROFILE = 'tennis_parabolic_frontal_velocity_aligned_v3'


SUCCESS_VERSION = "lob_finger_region_v3"
