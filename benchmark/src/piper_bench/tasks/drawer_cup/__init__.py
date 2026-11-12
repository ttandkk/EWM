"""Open a passive drawer, place a bowl held by one rim inside, then close it.

The package/CLI name is retained for existing launch scripts.
"""
from piper_bench.tasks.drawer_cup.scene import TASK
from piper_bench.tasks.drawer_cup.scene import PROMPT
from piper_bench.tasks.drawer_cup.scene import DrawerCupConfig as Config
from piper_bench.tasks.drawer_cup.scene import DrawerCupRuntime as Runtime
from piper_bench.tasks.drawer_cup.scene import make_env_cfg
from piper_bench.tasks.drawer_cup.scene import camera_metadata
from piper_bench.tasks.drawer_cup.scene import set_views
from piper_bench.tasks.drawer_cup.oracle import DrawerCupOracle as Oracle
