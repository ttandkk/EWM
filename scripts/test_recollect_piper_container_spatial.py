import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

path = Path(__file__).with_name('recollect_piper_container_spatial.py')
spec = importlib.util.spec_from_file_location('container_replace', path)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class CleanupTest(unittest.TestCase):
    def test_recollection_uses_wide_protocol(self):
        self.assertEqual(m.PROTOCOL, "piper_container_spatial_v4")

    def test_allowlist_keeps_other_tasks_and_removes_before_rebuild(self):
        original = m.ROOT
        with tempfile.TemporaryDirectory() as tmp:
            m.ROOT = Path(tmp)
            try:
                run = m.ROOT/'run'
                run.mkdir()
                (run/'protected_fingerprints.json').write_text('{}')
                (m.ROOT/'datasets').mkdir()
                for p in m.deletion_targets():
                    if p.suffix=='.json':
                        p.parent.mkdir(parents=True,exist_ok=True);p.write_text('{}')
                    else:
                        p.mkdir(parents=True,exist_ok=True);(p/'payload').write_text('old')
                keep=m.ROOT/'datasets/piper_conveyor/payload'
                keep.parent.mkdir();keep.write_text('preserve')
                class Helper:
                    @staticmethod
                    def write(p,v):
                        p.write_text(json.dumps(v))
                    @staticmethod
                    def fingerprint(p):
                        return {}
                old_helper=m.helper;m.helper=lambda run:Helper
                try:
                    m.clean(run)
                    self.assertTrue(all(not p.exists() for p in m.deletion_targets()))
                    self.assertEqual(keep.read_text(),'preserve')
                    self.assertTrue((run/'cache_cleanup_complete.json').exists())
                    m.clean(run)
                finally:
                    m.helper=old_helper
            finally:
                m.ROOT=original


if __name__=='__main__':
    unittest.main()
