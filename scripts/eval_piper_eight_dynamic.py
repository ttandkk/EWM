"""Compatibility entry point for the packaged eight-task benchmark."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'benchmark/src'), str(ROOT / 'policies/src')]
from piper_bench.evaluation import suite as _suite

parser = _suite.parser
run = _suite.run
client = _suite.client
trial_counts = _suite.trial_counts
suite_summary = _suite.suite_summary
failed_cell = _suite.failed_cell
source_hashes = _suite.source_hashes
subprocess = _suite.subprocess


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if any(a.split('=', 1)[0] in ('--checkpoint-dir', '--checkpoint-map') for a in argv):
        from piper_policies.legacy_suite import main as legacy
        return legacy(argv)
    return _suite.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
