import importlib.util
from pathlib import Path
import json

import pytest

spec = importlib.util.spec_from_file_location('gpu_probe', Path(__file__).with_name('service_gpu_probe.py'))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


@pytest.mark.parametrize('mode', ['joint', 'action_only', 'rgb_action'])
def test_probe_observation_respects_real_contract(mode):
    from piper_policies.contract import validate_request, EVENT_CONTRACT, RGB_CONTRACT, RGB_LATENT_CONTRACT
    descriptor = dict(mode=mode, checkpoint_step=80, control_hz=30, action_horizon=24,
                      event_cameras=2, voxel_bins=1, event_time_contract=EVENT_CONTRACT,
                      observation_contract=RGB_LATENT_CONTRACT if mode == 'rgb_action' else RGB_CONTRACT)
    request, arrays = probe.observation(descriptor, [0.] * 8)
    value = validate_request(request, arrays, descriptor)
    assert value.chunk_size == 1 and value.sampling_steps == 4 and value.step == 1
    assert arrays['obs_video'][0, 0, 0, -1] == 255
    assert arrays['obs_video'][1, 2, -1, 0] == 255


def test_snapshot_rejects_changed_or_injected_source(tmp_path):
    code = tmp_path / 'code.py'
    code.write_text('# frozen\n')
    (tmp_path / 'probe_source_manifest.json').write_text(json.dumps({'sha256': {'code.py': probe.sha256(code)}}))
    probe.verify_source(tmp_path)
    code.write_text('# changed\n')
    with pytest.raises(ValueError, match='changed'):
        probe.verify_source(tmp_path)
    code.write_text('# frozen\n')
    (tmp_path / 'injected.py').write_text('# extra\n')
    with pytest.raises(ValueError, match='inventory differs'):
        probe.verify_source(tmp_path)


def test_prepare_preserves_venv_python_symlink(tmp_path, monkeypatch):
    from types import SimpleNamespace
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'probe_source_manifest.json').write_text(json.dumps({'sha256': {}}))
    interpreter = tmp_path / 'venv/bin/python'
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to('/usr/bin/python3')
    stats = tmp_path / 'stats.json'
    stats.write_text(json.dumps({'observation.state': {'mean': [0.] * 8}}))
    calls = []
    def describe(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout=json.dumps({'checkpoint_sha256': 'test'}))
    monkeypatch.setattr(probe.subprocess, 'run', describe)
    probe.prepare(SimpleNamespace(source_root=source, model_python=interpreter,
        output=tmp_path/'plan', family='fastwam', checkpoint_dir=tmp_path,
        checkpoint='step_000002.pt', checkpoint_step=2, pretrained_root=tmp_path,
        project_root=tmp_path, pooled_stats=stats, evidence_label='test',
        formal_optimization=False, service_timeout=1200))
    assert calls[0][0] == str(interpreter)
    assert calls[0][0] != str(interpreter.resolve())


@pytest.mark.parametrize('changed', [None, 'compile_core_active', 'compile_mode', 'cuda_graphs_enabled',
                                     'cache_context_kv', 'warmup_iters', 'iterations', 'action_max_abs_error'])
def test_formal_probe_checks_actual_startup_settings(changed):
    settings = dict(compile_core_active=True, compile_mode='default', cuda_graphs_enabled=False,
                    cache_context_kv=True, warmup_iters=3, cache_bf16_weights=True, warmup_sampling_steps=4,
                    startup_benchmark=dict(iterations=3, action_max_abs_error=.03,
                        baseline_eager_no_kv_cache_mean_seconds=.2, eager_kv_cache_mean_seconds=.15,
                        optimized_mean_seconds=.1))
    hello = dict(policy_descriptor=dict(family='ewm'), optimization=settings)
    if changed is None:
        # Formal protocol records action error above .02; it does not reject it.
        assert probe.validate_optimization(hello, True)
    else:
        if changed in settings['startup_benchmark']:
            settings['startup_benchmark'][changed] = None
        else:
            settings[changed] = None
        with pytest.raises(ValueError):
            probe.validate_optimization(hello, True)


@pytest.mark.parametrize('cuda_initialized', [False, True])
def test_instrumented_native_failure_still_records_resources(tmp_path, monkeypatch, cuda_initialized):
    from types import SimpleNamespace
    from piper_policies import cli
    import sys
    location = Path(__file__).with_name('instrumented_policy.py')
    spec = importlib.util.spec_from_file_location('instrumented_probe', location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def fail():
        raise ValueError('native failure')
    monkeypatch.setattr(cli, 'main', fail)
    monkeypatch.setattr(sys, 'argv', ['instrumented_policy', 'serve'])
    monkeypatch.setenv('PROBE_RESOURCE_METRICS', str(tmp_path/'resources.json'))
    fake_cuda = SimpleNamespace(is_initialized=lambda: cuda_initialized,
        max_memory_allocated=lambda device: 123 if device == 0 else None,
        max_memory_reserved=lambda device: 456 if device == 0 else None,
        get_device_name=lambda device: 'test GPU 0')
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=fake_cuda))
    with pytest.raises(ValueError, match='native failure'):
        module.main()
    record = json.loads((tmp_path/'resources.json').read_text())
    assert record['process_max_rss_kib'] > 0 and record['requested_device'] == 'cuda:0'
    assert record['cuda_initialized'] == cuda_initialized
    if cuda_initialized:
        assert record['cuda0_peak_allocated_bytes'] == 123
        assert record['cuda0_peak_reserved_bytes'] == 456


def test_describe_instrumentation_does_not_touch_cuda(tmp_path, monkeypatch):
    from piper_policies import cli
    import sys
    location = Path(__file__).with_name('instrumented_policy.py')
    spec = importlib.util.spec_from_file_location('instrumented_probe_describe', location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(cli, 'main', lambda: 0)
    monkeypatch.setattr(sys, 'argv', ['instrumented_policy', 'describe'])
    monkeypatch.setenv('PROBE_RESOURCE_METRICS', str(tmp_path/'should_not_exist.json'))
    assert module.main() == 0
    assert not (tmp_path/'should_not_exist.json').exists()


def test_exported_legacy_piper_stats_match_native_state_and_action_coordinates(tmp_path):
    import numpy as np
    from openwam.dataloader.transforms.normalize import Normalizer, load_mode_stats
    location = Path(__file__).with_name('export_openwam_initialization.py')
    spec = importlib.util.spec_from_file_location('openwam_export_probe', location)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    stats = {'eef': {'min': np.full(7, -2., np.float32), 'max': np.full(7, 4., np.float32)},
             'state': {'min': np.zeros(8, np.float32), 'max': np.full(8, 8., np.float32)}}
    source, target = tmp_path/'original.npy', tmp_path/'canonical.npy'
    np.save(source, stats)
    original_bytes = source.read_bytes()
    # This is the actual existing export mismatch, not a fake model boundary.
    with pytest.raises(ValueError):
        Normalizer(mode='min_max', stats=load_mode_stats(source, 'eef')).normalize(np.full(8, 4., np.float32))
    record = module.export_normalization(source, target)
    assert source.read_bytes() == original_bytes
    assert record['mapping'] == {'state': 'eef_state'} and record['exact_shapes_dtypes_and_values_verified']
    normalizer = Normalizer(mode='min_max', stats=load_mode_stats(target, 'eef'))
    np.testing.assert_array_equal(normalizer.normalize(np.full(8, 4., np.float32)), np.zeros(8, np.float32))
    np.testing.assert_array_equal(normalizer.unnormalize(np.zeros(7, np.float32)), np.ones(7, np.float32))
