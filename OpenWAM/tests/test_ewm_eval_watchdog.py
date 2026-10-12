"""End-to-end smoke test for an interrupted evaluation client."""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


WATCHDOG = Path(__file__).resolve().parents[1] / "scripts/eval_ewm/ewm_eval_watchdog.py"


class EvalWatchdogTests(unittest.TestCase):
    def test_stalled_client_restarts_with_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            client = (
                "import pathlib, sys, time; "
                "out = pathlib.Path(sys.argv[1]); "
                "(out / 'progress.heartbeat').touch(); "
                "(out / 'manifest.json').write_text('{}'); "
                "resume = '--resume' in sys.argv; "
                "(out / 'complete.json').write_text('{}') if resume else time.sleep(10)"
            )
            result = subprocess.run(
                [sys.executable, str(WATCHDOG), "--output-dir", str(output),
                 "--stall-timeout", "0.3", "--startup-timeout", "1",
                 "--poll-interval", "0.05", "--max-restarts", "1", "--",
                 sys.executable, "-c", client, str(output)],
                capture_output=True, text=True, timeout=15, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("no client progress", result.stdout)
            self.assertIn("attempt=2 resume=True", result.stdout)
            self.assertIn("all trials completed", result.stdout)


if __name__ == "__main__":
    unittest.main()
