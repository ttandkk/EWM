"""Static square-hole nut assembly on the existing Piper tabletop."""
from piper_bench.tasks.square_nut.config import SquareNutConfig as Config
from piper_bench.tasks.square_nut.scene import TASK
from piper_bench.tasks.square_nut.scene import PROMPT
from piper_bench.tasks.square_nut.scene import SquareNutRuntime as Runtime
from piper_bench.tasks.square_nut.scene import make_env_cfg
from piper_bench.tasks.square_nut.oracle import SquareNutOracle as Oracle
from piper_bench.tasks.common import camera_metadata
from piper_bench.tasks.common import set_views
