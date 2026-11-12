"""Adapter for collecting the existing conveyor with the shared task writer."""
from piper_bench.tasks.conveyor.scene import Config
from piper_bench.tasks.conveyor.scene import Runtime
from piper_bench.tasks.conveyor.scene import TASK
from piper_bench.tasks.conveyor.scene import PROMPT
from piper_bench.tasks.conveyor.scene import make_env_cfg
from piper_bench.tasks.conveyor.core import camera_metadata
from piper_bench.tasks.conveyor.core import set_views
from piper_bench.tasks.conveyor.oracle import ConveyorOracle

SUCCESS_VERSION = "clipped_basket_v2"


class Oracle(ConveyorOracle):
    failure = None
