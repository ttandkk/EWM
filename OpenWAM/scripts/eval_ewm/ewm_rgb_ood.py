"""Compatibility import; implementation lives in piper_bench.evaluation.rgb_ood."""
import sys
from pathlib import Path
try:
    import piper_bench
except ModuleNotFoundError as exc:
    if exc.name != "piper_bench":
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmark/src"))
from piper_bench.evaluation import rgb_ood as _implementation
sys.modules[__name__] = _implementation
