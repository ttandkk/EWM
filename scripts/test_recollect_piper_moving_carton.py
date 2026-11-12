"""Small filesystem tests for failure-safe data/cache replacement."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('recollect', Path(__file__).with_name('recollect_piper_moving_carton.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReplacementTest(unittest.TestCase):
    def test_rollback_restores_ready_marker_and_original_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ready = root / 'complete.json'
            ready.write_text('old-ready')
            data = root / 'data'
            data.mkdir()
            (data / 'old').write_text('original')
            with self.assertRaises(RuntimeError):
                with module.replacement([('ready', ready), ('data', data)], root/'backup', root/'journal'):
                    self.assertFalse(ready.exists())
                    data.mkdir()
                    (data / 'new').write_text('partial')
                    raise RuntimeError('GPU failure')
            self.assertEqual(ready.read_text(), 'old-ready')
            self.assertEqual((data / 'old').read_text(), 'original')
            self.assertFalse((data / 'new').exists())

    def test_commit_preserves_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / 'data'
            data.write_text('old')
            with module.replacement([('data', data)], root/'backup', root/'journal'):
                data.write_text('new')
            self.assertEqual(data.read_text(), 'new')
            self.assertEqual((root / 'backup/data').read_text(), 'old')

    def test_symlink_rejection_restores_already_moved_ready_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ready = root / 'ready'
            ready.write_text('ready')
            link = root / 'link'
            link.symlink_to(ready)
            with self.assertRaises(ValueError):
                with module.replacement([('ready', ready), ('link', link)], root/'backup', root/'journal'):
                    self.fail('must reject symlink')
            self.assertEqual(ready.read_text(), 'ready')
            self.assertTrue(link.is_symlink())


if __name__ == '__main__':
    unittest.main()
