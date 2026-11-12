"""Compatibility entry point for the package-owned Piper evaluator."""
import sys
from pathlib import Path
try:
    import piper_bench
except ModuleNotFoundError as exc:
    if exc.name != "piper_bench":
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmark/src"))
from piper_bench.evaluation.legacy import conveyor_module
_impl = conveyor_module()
if __name__ == "__main__":
    _impl.main()
else:
    sys.modules[__name__] = _impl
