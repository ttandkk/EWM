import contextlib
import importlib.util
import io
import json
import os
import signal
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe, gpu = load('ephemeral_openwam_probe'), load('monitor_gpu')


class EphemeralProbeTests(unittest.TestCase):
    def make_output(self, directory, partial=False):
        output = Path(directory)/'output'
        output.mkdir()
        probe.write_json(output/'probe_owner.json', dict(kind='untrained_openwam_ephemeral_probe', output=str(output)))
        checkpoint = output/probe.CHECKPOINT
        checkpoint.parent.mkdir()
        header = json.dumps({'weight': {'dtype': 'BF16', 'shape': [1], 'data_offsets': [0, 2]}}).encode()
        checkpoint.write_bytes(len(header).to_bytes(8, 'little')+header+(b'x' if partial else b'xx'))
        (output/'native.log').write_text('CPU fixture only\n')
        (output/'cache').mkdir()
        (output/'cache/scratch').write_text('disposable')
        (output/'prediction/inductor_cache').mkdir(parents=True)
        (output/'prediction/inductor_cache/kernel').write_text('disposable compiled artifact')
        return output, checkpoint

    def test_success_archives_header_hash_and_retires_only_generated_weight(self):
        with tempfile.TemporaryDirectory() as directory:
            original = Path(directory)/'trained.pt'
            original.write_bytes(b'protected original')
            output, checkpoint = self.make_output(directory)
            identity = probe.sha256(checkpoint)
            probe.archive_and_retire(output, [dict(stage=name, exit_code=0) for name in ['export', 'prepare', 'predict']])
            record = json.loads((output/'checkpoint_lifecycle.json').read_text())
            self.assertFalse(checkpoint.exists())
            self.assertFalse((output/'cache').exists())
            self.assertFalse((output/'prediction/inductor_cache').exists())
            self.assertEqual(original.read_bytes(), b'protected original')
            self.assertEqual(record['identity']['sha256'], identity)
            self.assertTrue(record['strict_reload_and_predict_passed'])
            self.assertFalse(record['plan_is_executable'])
            self.assertFalse(record['partial'])

    def test_partial_failed_export_retired_but_never_claimed_as_strict_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            output, checkpoint = self.make_output(directory, partial=True)
            probe.archive_and_retire(output, [dict(stage='export', exit_code=1)])
            record = json.loads((output/'checkpoint_lifecycle.json').read_text())
            self.assertTrue(record['partial'])
            self.assertFalse(record['strict_reload_and_predict_passed'])
            self.assertFalse(checkpoint.exists())

    def test_stops_retirement_if_a_recorded_process_is_alive(self):
        with tempfile.TemporaryDirectory() as directory:
            output, checkpoint = self.make_output(directory)
            (output/'supervisor_export').mkdir()
            (output/'supervisor_export/resources.jsonl').write_text(json.dumps({'processes':[{'pid':os.getpid()}]})+'\n')
            with self.assertRaisesRegex(RuntimeError, 'still alive'):
                probe.archive_and_retire(output, [])
            self.assertTrue(checkpoint.exists())

    def test_refuses_checkpoint_symlink_to_original(self):
        with tempfile.TemporaryDirectory() as directory:
            output, checkpoint = self.make_output(directory)
            original = Path(directory)/'original.safetensors'
            checkpoint.rename(original)
            checkpoint.symlink_to(original)
            with self.assertRaisesRegex(ValueError, 'symlink'):
                probe.archive_and_retire(output, [])
            self.assertTrue(original.exists())

    def test_refuses_external_parent_symlinks_before_any_retirement(self):
        for parent_name in ('initialization', 'plan', 'prediction'):
            with self.subTest(parent=parent_name), tempfile.TemporaryDirectory() as directory:
                output, checkpoint = self.make_output(directory)
                parent = output/parent_name
                parent.mkdir(exist_ok=True)
                cache = parent/'triton_cache'
                cache.mkdir(exist_ok=True)
                (cache/'protected').write_bytes(b'external data')
                external = Path(directory)/'external'
                parent.rename(external)
                parent.symlink_to(external, target_is_directory=True)
                self.assertFalse(cache.is_symlink())
                with self.assertRaisesRegex(ValueError, 'symlink|outside output'):
                    probe.archive_and_retire(output, [])
                self.assertEqual((external/'triton_cache/protected').read_bytes(), b'external data')
                self.assertTrue(checkpoint.exists())
                self.assertFalse((output/'checkpoint_audit.json').exists())

    def test_cleanup_ignores_repeated_term_int_and_restores_handlers(self):
        received = []
        def handler(signum, frame):
            received.append(signum)
        original = {sig: signal.signal(sig, handler) for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            with tempfile.TemporaryDirectory() as directory:
                output, checkpoint = self.make_output(directory)
                with probe.cleanup_signals():
                    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGTERM, signal.SIGINT):
                        os.kill(os.getpid(), sig)
                    probe.archive_and_retire(output, [dict(stage='export', exit_code=1)])
                    os.kill(os.getpid(), signal.SIGTERM)
                self.assertEqual(received, [])
                self.assertFalse(checkpoint.exists())
                for sig in original:
                    self.assertIs(signal.getsignal(sig), handler)
        finally:
            for sig, old in original.items():
                signal.signal(sig, old)

    def test_plan_only_creates_no_output_or_model(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            output = Path(directory)/'never-created'
            code = probe.main(['--source-root', '/nonexistent', '--source-manifest-sha256', 'unused',
                               '--model-python', '/nonexistent/python', '--pretrained-root', '/nonexistent',
                               '--ffmpeg-lib', '/nonexistent',
                               '--output', str(output)])
            self.assertEqual(code, 0)
            self.assertFalse(output.exists())

    def test_reclamation_requires_root_receipt_and_idle_installation(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory)/'receipt.json'
            record=dict(version='piper_probe_storage_release_v1',
                rootfs_path='/projects/_hdd/isaachdd/piper-container-build-20261111-v2/data/piper-simulation-v2',
                archived_before_removal=True, rootfs_removed=True, policy_installation_idle=False,
                approved_temporary_budget_bytes=26*1024**3)
            p.write_text(json.dumps(record))
            with patch.object(Path, 'exists', return_value=False), patch.object(Path, 'is_symlink', return_value=False):
                with self.assertRaisesRegex(ValueError, 'not confirmed'):
                    probe.verify_reclamation(p)
                record['policy_installation_idle']=True
                p.write_text(json.dumps(record))
                self.assertEqual(probe.verify_reclamation(p), record)


class GPUDeviceMappingTests(unittest.TestCase):
    def test_remapped_namespace_uses_visible_uuids(self):
        rows=[dict(index='0',uuid='GPU-a'),dict(index='1',uuid='GPU-b')]
        self.assertEqual(gpu.select_device_uuids(rows,dict(SLURM_JOB_GPUS='5,6',CUDA_VISIBLE_DEVICES='0,1')),['GPU-a','GPU-b'])

    def test_global_namespace_respects_slurm_allocation(self):
        rows=[dict(index=str(i),uuid=f'GPU-{i}') for i in range(8)]
        self.assertEqual(gpu.select_device_uuids(rows,dict(SLURM_JOB_GPUS='5,6',CUDA_VISIBLE_DEVICES='0,1')),['GPU-5','GPU-6'])

    def test_explicit_uuids_and_invalid_mapping(self):
        rows=[dict(index='0',uuid='GPU-abcd')]
        self.assertEqual(gpu.select_device_uuids(rows,dict(CUDA_VISIBLE_DEVICES='GPU-abc')),['GPU-abcd'])
        with self.assertRaises(ValueError):
            gpu.select_device_uuids(rows,dict(SLURM_JOB_GPUS='5'))


if __name__ == '__main__':
    unittest.main()
