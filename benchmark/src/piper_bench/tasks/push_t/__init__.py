"""Contact-only Push-T on the existing Piper tabletop."""
from piper_bench.tasks.push_t.scene import TASK
from piper_bench.tasks.push_t.scene import PROMPT
from piper_bench.tasks.push_t.scene import PushTConfig as Config
from piper_bench.tasks.push_t.scene import PushTRuntime as Runtime
from piper_bench.tasks.push_t.scene import make_env_cfg
from piper_bench.tasks.common import camera_metadata
from piper_bench.tasks.common import set_views
from piper_bench.tasks.push_t.oracle import PushTOracle as Oracle
