"""Temporal input stride preserves action timing and never pads a codec group."""
from types import SimpleNamespace

import pytest
import torch

from openwam.model.architectures.ewm import EWMModel
from openwam.model.architectures.lean_rgb_codec import LeanRGBCodec
from tests.ewm_helpers import options
from tests.test_ewm_model import FakeLeanVAE, small_batch


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(2)


def model(stride=1, mode='joint', **kwargs):
    opts = options(cameras=2, mode=mode)
    if mode == 'rgb_action':
        opts.update(event_codec_impl='leanvae16_rgb_joint', event_codec=LeanRGBCodec(vae=FakeLeanVAE()))
    return EWMModel(**opts, event_temporal_downsample=stride, **kwargs)


@pytest.mark.parametrize('stride,slices,times', [
    (1, 6, [0, 2.5, 6.5, 10.5, 14.5, 18.5, 22.5]),
    (2, 3, [0, 5, 13, 21]),
    (3, 2, [0, 7.5, 19.5]),
])
@pytest.mark.parametrize('mode', ['joint', 'event_only', 'rgb_action'])
def test_train_sample_frame_selection_action_timing_and_decode(monkeypatch, stride, slices, times, mode):
    m = model(stride, mode)
    x = small_batch(24, cameras=2)
    key = 'rgb' if mode == 'rgb_action' else 'voxel'
    if mode == 'rgb_action':
        x['obs_rgb'] = torch.randint(256, (1, 2, 3, 32, 32), dtype=torch.uint8)
        x['future_rgb'] = torch.randint(256, (1, 24, 2, 3, 32, 32), dtype=torch.uint8)
    captured = []
    encode = m.event_codec.encode
    def record(frames):
        captured.append(frames.clone())
        return encode(frames)
    monkeypatch.setattr(m.event_codec, 'encode', record)
    token_counts, event_times = [], []
    def inspect(module, args):
        token_counts.append((args[0].shape[1], None if args[1] is None else args[1].shape[1]))
        event_times.append(args[5][:, 0].reshape(slices + 1, 2)[:, 0].tolist())
    m.blocks[0].register_forward_pre_hook(inspect)
    out = m(x, diagnostics=True)
    assert all(torch.isfinite(v) for v in out.values())
    out['loss'].backward()
    assert m.event_head[-1].weight.grad.abs().sum() > 0
    if mode != 'event_only':
        assert m.action_head[-1].weight.grad.abs().sum() > 0
    assert token_counts[0] == (2 * (slices + 1), None if mode == 'event_only' else 25)
    assert event_times[0] == times
    expected = torch.cat((x[f'obs_{key}'][:, None], x[f'future_{key}'][:, stride-1::stride]), 1)
    torch.testing.assert_close(captured[0], expected, rtol=0, atol=0)
    assert captured[0].shape[1] == 1 + 24 // stride  # no artificial frames
    result = m.sample_policy(x, horizon=24, num_steps=2,
                             normalizer=SimpleNamespace(denormalize_action=lambda a: a))
    assert result['future_latent'].shape == (1, slices, 2, 16, 4, 4)
    assert result[f'future_{key}'].shape == x[f'future_{key}'].shape
    if mode != 'event_only':
        assert result['action'].shape == (1, 24, 7)
    else:
        assert 'action' not in result
    assert all(t == times for t in event_times)


@pytest.mark.parametrize('stride', [0, 4, 5, -1, True, 1.0, '2'])
def test_invalid_strides_rejected(stride):
    with pytest.raises(ValueError, match='event_temporal_downsample'):
        model(stride)


def test_wan_rejects_extra_stride():
    from tests.test_ewm_model import small_model
    with pytest.raises(ValueError, match='LeanVAE'):
        small_model(event_temporal_downsample=2)


@pytest.mark.parametrize('stride,horizon', [(2, 4), (3, 8), (3, 16)])
def test_incompatible_horizon_fails_before_codec_without_padding(monkeypatch, stride, horizon):
    m = model(stride)
    def forbidden(*args, **kwargs):
        raise AssertionError('must reject before encoding')
    monkeypatch.setattr(m.event_codec, 'encode', forbidden)
    with pytest.raises(ValueError, match='no frame padding'):
        m(small_batch(horizon, cameras=2))
    with pytest.raises(ValueError, match='no frame padding'):
        m.sample_policy(small_batch(horizon, cameras=2), horizon=horizon, num_steps=1)




@pytest.mark.parametrize('stride', [2, 3])
@pytest.mark.parametrize('count', [0, 5, 11, 12, 23, 24])
def test_tail_padding_and_gradients_ignore_invalid_labels(stride, count):
    torch.manual_seed(13)
    m = model(stride)
    x = small_batch(24, cameras=2)
    x['future_valid_mask'] = (torch.arange(24) < count)[None]
    x['action_valid_mask'] = (torch.arange(24) < min(count+1, 24))[None]
    fixed = dict(t=torch.tensor([.4]), event_noise=torch.randn(1, 6//stride, 2, 16, 4, 4),
                 action_noise=torch.randn(1, 24, 7), diagnostics=True)
    a = m(x, **fixed)
    a['loss'].backward()
    grads = {name: p.grad.clone() for name, p in m.named_parameters() if p.grad is not None}
    x['future_voxel'][~x['future_valid_mask']] = float('nan')
    x['action'][~x['action_valid_mask']] = float('nan')
    m.zero_grad(set_to_none=True)
    b = m(x, **fixed)
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
        assert torch.isfinite(b[key]), key
    b['loss'].backward()
    for name, p in m.named_parameters():
        if name in grads:
            torch.testing.assert_close(p.grad, grads[name], rtol=0, atol=0)
    if count < 4 * stride:
        assert b['loss_event'] == 0
    assert m.action_head[-1].weight.grad.abs().sum() > 0




@pytest.mark.parametrize('stride', [2, 3])
def test_unsampled_labels_do_not_affect_training_or_decoded_diagnostics(stride):
    m = model(stride)
    x = small_batch(24, cameras=2)
    fixed = dict(t=torch.tensor([.5]), event_noise=torch.randn(1, 6//stride, 2, 16, 4, 4),
                 action_noise=torch.randn(1, 24, 7), diagnostics=True)
    before = m(x, **fixed)
    skipped = (torch.arange(24) + 1) % stride != 0
    x['future_voxel'][:, skipped] = 1
    after = m(x, **fixed)
    for key in before:
        torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    frames = torch.arange(1, 1+24//stride).reshape(1, -1, 1, 1, 1, 1)
    restored = m.temporal_sampling.restore(frames)
    assert restored.shape[1] == 24
    for offset in range(stride):
        torch.testing.assert_close(restored[:, offset::stride], frames)




def test_config_override_and_legacy_resume_fingerprint(tmp_path):
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    from openwam.train.utils.ewm_resume import resume_contract
    from tests.test_ewm_resume import config, TinyDataset
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1]/'configs'), version_base=None):
        assert compose(config_name='train_ewm').model.event_temporal_downsample == 1
        for stride in (2, 3):
            cfg = compose(config_name='train_ewm_eight',
                          overrides=[f'model.event_temporal_downsample={stride}'])
            assert cfg.model.event_temporal_downsample == stride
    cfg = config(tmp_path, 6)
    def contract():
        return resume_contract(cfg, TinyDataset(), [0], SimpleNamespace(num_processes=1), 6)
    before = contract()
    cfg.model.event_temporal_downsample = 1
    assert contract() == before
    cfg.model.event_temporal_downsample = 2
    assert contract()['model'] != before['model']
