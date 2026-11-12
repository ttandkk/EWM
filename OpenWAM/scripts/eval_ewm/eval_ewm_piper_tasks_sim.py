"""Compatibility entry point; one implementation lives in piper_bench.evaluation.runner."""
import sys
from pathlib import Path
try:
    import piper_bench
except ModuleNotFoundError as exc:
    if exc.name != "piper_bench":
        raise
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmark/src"))
from piper_bench.evaluation import runner as _impl

# Old suite callers may still ask for model fingerprints. Resolve them on the
# model side without adding a model package dependency to the benchmark.
def checkpoint_file(directory, step):
    from piper_policies.metadata import checkpoint_file as resolve
    return resolve(directory, step)


def file_digest(path):
    from piper_policies.metadata import file_sha256
    return file_sha256(path)


_impl.checkpoint_file = checkpoint_file
_impl.file_digest = file_digest
if __name__ == "__main__":
    _impl.main()
else:
    sys.modules[__name__] = _impl
