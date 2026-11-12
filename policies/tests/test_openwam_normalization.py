"""Real Piper statistics through the public adapter and native normalizer."""
import io
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import pytest

from piper_policies.metadata import load_metadata
from piper_policies.openwam.normalization import normalization_payload, native_statistics_directory

FIXTURE = Path(__file__).parent/'fixtures/openwam_piper_legacy_stats.npy'


def payload():
    return np.load(FIXTURE, allow_pickle=True).item()


@pytest.mark.parametrize('canonical', [False, True])
def test_public_adapter_keeps_physical_state_and_action_scales(tmp_path, monkeypatch, canonical):
    from openwam.deploy import model_loader
    from piper_policies.openwam.adapter import OpenWAMPolicy
    from openwam.model.action_backbone.scheduler import ActionScheduler
    from openwam.model.video_backbone.wan.shared.diffusion.flow_match import FlowMatchScheduler
    stats_path = tmp_path/'normalization_stats.npy'
    summary, converted = normalization_payload(FIXTURE)
    stats_path.write_bytes(converted if canonical else FIXTURE.read_bytes())
    original = stats_path.read_bytes()
    (tmp_path/'checkpoint_step_0.safetensors').write_bytes(b'model weights replaced only for the CPU boundary test')
    (tmp_path/'config.yaml').write_text('''model:
  architecture:
    action_dim: 7
    state_dim: 8
dataloader:
  num_frames: 25
  video_stride: 2
  height: 256
  width: 512
  action_mode: eef
  normalize_mode: min-max
  unify_action: false
''')
    class Architecture:
        video_scheduler = FlowMatchScheduler('Wan')
        action_scheduler = ActionScheduler()
        attached = 0
        def attach_normalizer(self, value):
            self.normalizer = value
            self.attached += 1
    model = Architecture()
    def load(directory, *, device, ckpt_name, config_override):
        # Exercise the exact native initial attachment, including its legacy gap.
        model.attach_normalizer(model_loader._build_normalizer(config_override, directory))
        return config_override, model
    monkeypatch.setattr(model_loader, 'load_from_checkpoint_dir', load)
    policy = OpenWAMPolicy(tmp_path, device='cpu')
    stats = payload()
    state = stats['state']['mean']
    expected = 2*(state-stats['state']['min'])/(stats['state']['max']-stats['state']['min'])-1
    np.testing.assert_allclose(model.normalizer.normalize(state), expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(model.normalizer.unnormalize(np.zeros(7, np.float32)),
                               (stats['eef']['min']+stats['eef']['max'])/2, atol=1e-7)
    assert model.attached == (1 if canonical else 2)
    assert stats_path.read_bytes() == original
    assert policy.descriptor == load_metadata(tmp_path, 'openwam')[3]
    description = policy.descriptor['normalization_adapter']
    assert description['mapping'] == ({} if canonical else {'state': 'eef_state'})
    assert description['source_sha256'] == policy.descriptor['normalization_sha256']
    if canonical:
        assert description['effective_sha256'] == description['source_sha256']
    else:
        assert description['effective_sha256'] == summary['effective_sha256']


def test_translated_directory_lifetime_and_failure_cleanup():
    summary, _ = normalization_payload(FIXTURE)
    with pytest.raises(RuntimeError, match='native factory failed'):
        with native_statistics_directory(FIXTURE, summary) as directory:
            assert (directory/'normalization_stats.npy').exists()
            assert normalization_payload(directory/'normalization_stats.npy')[0]['mapping'] == {}
            raise RuntimeError('native factory failed')
    assert not directory.exists()
    with native_statistics_directory(FIXTURE, summary) as directory:
        restored = np.load(directory/'normalization_stats.npy', allow_pickle=True).item()
    assert not directory.exists() and restored['eef_state']['mean'].shape == (8,)


def test_canonical_path_is_never_rewritten(tmp_path):
    _, encoded = normalization_payload(FIXTURE)
    path = tmp_path/'normalization_stats.npy'
    path.write_bytes(encoded)
    summary, same = normalization_payload(path)
    assert same == encoded
    with native_statistics_directory(path, summary) as directory:
        assert directory == tmp_path
    assert path.read_bytes() == encoded


@pytest.mark.parametrize('corruption', ['unknown_schema', 'ambiguous_state', 'bad_state_shape', 'unknown_statistic'])
def test_unknown_or_ambiguous_statistics_fail_explicitly(tmp_path, corruption):
    stats = payload()
    if corruption == 'unknown_schema':
        stats['other'] = {}
    elif corruption == 'ambiguous_state':
        stats['eef_state'] = stats['state']
    elif corruption == 'bad_state_shape':
        stats['state']['min'] = np.zeros(7, np.float32)
    else:
        stats['state']['count'] = np.array([123])
    path = tmp_path/'stats.npy'
    np.save(path, stats)
    with pytest.raises(ValueError):
        normalization_payload(path)


def test_statistics_changed_after_describe_fail_before_native_factory(tmp_path):
    path = tmp_path/'normalization_stats.npy'
    path.write_bytes(FIXTURE.read_bytes())
    summary, _ = normalization_payload(path)
    stats = payload(); stats['state']['mean'][0] += .01
    np.save(path, stats)
    with pytest.raises(ValueError, match='differ from the described'):
        with native_statistics_directory(path, summary):
            raise AssertionError('native factory must not run')
