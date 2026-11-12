import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('bounded_probe', Path(__file__).with_name('bounded_probe.py'))
bounded = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bounded)


class BoundedProbeTests(unittest.TestCase):
    def test_caches_stay_under_output_without_changing_home(self):
        with tempfile.TemporaryDirectory() as directory:
            values = bounded.cache_environment(Path(directory))
            self.assertNotIn('HOME', values)
            for key in ('TMPDIR', 'HF_HOME', 'HF_HUB_CACHE', 'TORCH_HOME', 'CUDA_CACHE_PATH',
                        'TORCHINDUCTOR_CACHE_DIR', 'TRITON_CACHE_DIR', 'MODELSCOPE_CACHE'):
                self.assertTrue(Path(values[key]).is_relative_to(directory))
                self.assertTrue(Path(values[key]).is_dir())
            self.assertEqual(values['DIFFSYNTH_SKIP_DOWNLOAD'], 'true')

    def test_storage_does_not_follow_external_symlinks_or_double_count_hardlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            watched = root / 'watched'
            watched.mkdir()
            (root / 'external').write_bytes(b'x' * 100000)
            (watched / 'source').write_bytes(b'x' * 100)
            os.link(watched / 'source', watched / 'hardlink')
            (watched / 'link').symlink_to(root / 'external')
            value = bounded.storage_usage([watched, watched])
            self.assertEqual(value['files'], 2)
            self.assertLess(value['logical_bytes'], 1000)

    def run_probe(self, root, code, stop_bytes=1024**2, extra=()):
        return subprocess.run([sys.executable, '-B', str(Path(bounded.__file__)),
            '--output', str(root/'supervisor'), '--cache-root', str(root/'cache'),
            '--storage-root', str(root), '--stop-bytes', str(stop_bytes),
            '--interval', '.05', '--timeout', '10', *extra, '--', sys.executable, '-B', '-c', code],
            capture_output=True, text=True, timeout=20)

    def test_actual_subprocess_success_and_resource_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = self.run_probe(root, 'print("actual CPU child")')
            self.assertEqual(result.returncode, 0, result.stderr)
            summary = json.loads((root/'supervisor/summary.json').read_text())
            self.assertEqual(summary['exit_code'], 0)
            self.assertIsNone(summary['stop_reason'])
            self.assertTrue((root/'supervisor/resources.jsonl').stat().st_size)

    def test_budget_stops_own_child_and_separate_service_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # The model server starts a separate session just like this child.
            code = ('import pathlib,subprocess,sys,time; '
                    'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(60)"],start_new_session=True); '
                    f'pathlib.Path({str(root/"pid")!r}).write_text(str(p.pid)); '
                    f'pathlib.Path({str(root/"big")!r}).write_bytes(b"x"*300000); '
                    'time.sleep(60)')
            result = self.run_probe(root, code, stop_bytes=200000)
            self.assertEqual(result.returncode, 1, result.stderr)
            summary = json.loads((root/'supervisor/summary.json').read_text())
            self.assertEqual(summary['stop_reason'], 'storage_stop_threshold')
            pid = int((root/'pid').read_text())
            stat = Path(f'/proc/{pid}/stat')
            if stat.exists():
                self.assertEqual(stat.read_text().split(') ')[1][0], 'Z')

    def test_separate_cache_limit_stops_before_large_checkpoint_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code = ('import os,pathlib,time; '
                    'pathlib.Path(os.environ["TMPDIR"],"cache").write_bytes(b"x"*300000); '
                    'time.sleep(60)')
            result = self.run_probe(root, code, extra=['--cache-stop-bytes', '200000'])
            self.assertEqual(result.returncode, 1, result.stderr)
            summary = json.loads((root/'supervisor/summary.json').read_text())
            self.assertEqual(summary['stop_reason'], 'cache_stop_threshold')

    def test_extra_native_compile_cache_is_in_the_same_cache_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            extra = root/'prediction/inductor_cache'
            extra.mkdir(parents=True)
            code = (f'import pathlib,time; pathlib.Path({str(extra/"kernel")!r}).write_bytes(b"x"*300000); '
                    'time.sleep(60)')
            result = self.run_probe(root, code, extra=['--cache-stop-bytes', '200000', '--extra-cache-root', str(extra)])
            self.assertEqual(result.returncode, 1, result.stderr)
            summary = json.loads((root/'supervisor/summary.json').read_text())
            self.assertEqual(summary['stop_reason'], 'cache_stop_threshold')


if __name__ == '__main__':
    unittest.main()
