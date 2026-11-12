"""Pick a horizontal tube and insert it into a continuously rotating rack."""
from piper_bench.tasks.rotating_tube.config import TubeConfig as Config
from piper_bench.tasks.rotating_tube.scene import TASK
from piper_bench.tasks.rotating_tube.scene import PROMPT
from piper_bench.tasks.rotating_tube.scene import SUCCESS_VERSION
from piper_bench.tasks.rotating_tube.scene import TubeRuntime as Runtime
from piper_bench.tasks.rotating_tube.scene import camera_metadata
from piper_bench.tasks.rotating_tube.scene import make_env_cfg
from piper_bench.tasks.rotating_tube.scene import set_views
from piper_bench.tasks.rotating_tube.oracle import TubeOracle as Oracle

