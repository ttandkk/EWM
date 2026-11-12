"""tennis_rolling: contact-only dynamic tennis manipulation."""
from piper_bench.tasks.tennis_rolling.scene import TASK
from piper_bench.tasks.tennis_rolling.scene import PROMPT
from piper_bench.tasks.tennis_rolling.scene import Config
from piper_bench.tasks.tennis_rolling.scene import Runtime
from piper_bench.tasks.tennis_rolling.scene import make_env_cfg
from piper_bench.tasks.tennis.camera import camera_metadata as shared_camera_metadata
from piper_bench.tasks.tennis.camera import set_views
from piper_bench.tasks.tennis.oracle import TennisOracle

SUCCESS_VERSION = 'rolling_lifted_hold_v3'


def camera_metadata():
    return shared_camera_metadata('roll')


class Oracle(TennisOracle):
    PROFILE = 'tennis_rolling_predictive_contact_v1'
