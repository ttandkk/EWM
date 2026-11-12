"""Short event forecasts and longer actions share time, but not label lengths."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir

from openwam.dataloader.ewm_dataset import EWMDataset
from openwam.dataloader.ewm_gpu_voxel import materialize_sparse_event_voxels
from openwam.model.architectures.ewm import EWMModel
from scripts.eval_ewm import eval_ewm_piper_conveyor_sim as client
from tests.ewm_helpers import options
from tests.test_ewm_model import small_batch
from tests.test_ewm_tail_mask import toy_dataset


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(2)


def policy():
    return EWMModel(**options(cameras=2), event_horizon=16, event_temporal_downsample=2)


def inputs():
    x = small_batch(16, cameras=2)
    x['action'] = torch.rand(1, 40, 7)
    return x


def test_train_sample_and_physical_time_coordinates():
    m, x = policy(), inputs()
    captured = []
    m.blocks[0].register_forward_pre_hook(lambda _, args: captured.append(
        (args[0].shape[1], args[1].shape[1], args[5][:, 0].tolist(), args[6][:, 0].tolist())))
    out = m(x)
    out['loss'].backward()
    assert torch.isfinite(out['loss'])
    assert m.action_head[-1].weight.grad.abs().sum() > 0
    assert m.event_head[-1].weight.grad.abs().sum() > 0
    result = m.sample_policy(x, horizon=40, num_steps=2,
                             normalizer=SimpleNamespace(denormalize_action=lambda a: a))
    assert result['action'].shape == (1, 40, 7)
    assert result['future_voxel'].shape == (1, 16, 2, 2, 32, 32)
    assert result['future_latent'].shape[1] == 2
    assert all(row[:2] == (6, 41) for row in captured)  # Small test spatial grid.
    assert all(row[2] == [0, 0, 5, 5, 13, 13] for row in captured)
    assert all(row[3] == list(range(41)) for row in captured)



@pytest.mark.parametrize('remaining', [1, 7, 9, 10, 17, 39, 40])
def test_padding_cannot_change_loss_or_gradients(remaining):
    torch.manual_seed(7)
    m, x = policy(), inputs()
    x['action_valid_mask'] = (torch.arange(40) < remaining)[None]
    x['future_valid_mask'] = (torch.arange(16) < remaining - 1)[None]
    fixed = dict(t=torch.tensor([.4]), action_noise=torch.randn(1, 40, 7),
                 event_noise=torch.randn(1, 2, 2, 16, 4, 4))
    clean = m(x, diagnostics=True, **fixed)
    clean['loss'].backward()
    gradients = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    x['action'][~x['action_valid_mask']] = float('nan')
    x['future_voxel'][~x['future_valid_mask']] = float('nan')
    m.zero_grad(set_to_none=True)
    changed = m(x, diagnostics=True, **fixed)
    for key in clean:
        assert torch.isfinite(changed[key]), key
        torch.testing.assert_close(clean[key], changed[key], rtol=0, atol=0)
    changed['loss'].backward()
    for name, p in m.named_parameters():
        if name in gradients:
            torch.testing.assert_close(p.grad, gradients[name], rtol=0, atol=0)
    assert m.action_head[-1].weight.grad.abs().sum() > 0
    if remaining <= 8:
        assert changed['loss_event'] == 0


def test_tail_windows_keep_real_actions_without_crossing_episode(monkeypatch, tmp_path):
    d = toy_dataset(monkeypatch, tmp_path, length=50)
    d.chunk_size, d.event_horizon = 40, 16
    x = d[40]  # Ten action records remain, nine future observations remain.
    assert x['action'].shape == (40, 7) and x['future_voxel'].shape[0] == 16
    assert x['action_valid_mask'].tolist() == [True] * 10 + [False] * 30
    assert x['future_valid_mask'].tolist() == [True] * 9 + [False] * 7
    assert x['action'][:10, 0].tolist() == list(range(41, 51))
    assert not x['action'][10:].any()
    assert x['future_voxel'].max() == 49
    terminal = d[49]
    assert terminal['action_valid_mask'].sum() == 1
    assert not terminal['future_valid_mask'].any()
    assert d[50]['action'][0, 0] == 51
    unpadded = EWMDataset(tmp_path, chunk_size=40, event_horizon=16, voxel_bins=1)
    assert len(unpadded) == 20


def test_sparse_voxelization_uses_event_length():
    batch = dict(action=torch.zeros(2, 40, 7), event_sparse={
        key: torch.zeros(0, dtype=torch.long) for key in ('frame_id', 'x', 'y', 'polarity')})
    result = materialize_sparse_event_voxels(batch, num_cameras=2, count_clip=15,
                                           height=2, width=2, event_horizon=16)
    assert result['future_voxel'].shape == (2, 16, 2, 2, 2, 2)
    assert result['action'].shape == (2, 40, 7)


def test_short_event_reply_is_strict_and_buffer_tail_has_no_fake_prediction():
    args = SimpleNamespace(decode_predicted_events=True, predicted_event_threshold=1/30,
                           predicted_event_min_neighbors=1, event_prediction_horizon=16)
    reply = dict(inference_seconds=.01, future_decoded=True, event_horizon=16)
    values = dict(executed_actions=np.zeros((40, 7), np.float32),
                  future_voxel=np.zeros((16, 2, 2, 256, 256), np.float32))
    actions, predicted, _ = client.validate_reply(reply, values, 40, ('cam0', 'cam_wrist'), args, 15)
    assert len(actions) == 40 and len(predicted) == 16
    with pytest.raises(ValueError, match='event horizon'):
        client.validate_reply(dict(reply, event_horizon=40), values, 40, ('cam0', 'cam_wrist'), args, 15)
    with pytest.raises(ValueError, match='decoded future'):
        client.validate_reply(reply, dict(values, future_voxel=values['future_voxel'][:15]),
                              40, ('cam0', 'cam_wrist'), args, 15)
    calls = []
    def predict(obs, step, horizon):
        calls.append((step, horizon))
        return dict(actions=actions, predicted=predicted, round_trip_seconds=0., inference_seconds=0.)
    stream = client.BufferedLatencyActionStream(predict, horizon=10, plan_horizon=40, max_steps=100)
    obs = dict(obs_video=np.zeros(1), obs_voxel=np.zeros(1), proprio=np.zeros(1), timing={})
    stream.ensure_request(obs, 0)
    stream.decision(0)  # Receive the bootstrap plan at its actual delivery step.
    assert stream.decision(15)['predicted_counts'] is not None
    assert stream.decision(16)['predicted_counts'] is None
    assert stream.decision(39)['action_source'] == 'policy'
    assert stream.decision(40)['action_source'] == 'buffer_underflow_hold'
    stream = client.BufferedLatencyActionStream(predict, horizon=10, plan_horizon=40, max_steps=100)
    calls.clear()
    for step in range(31):
        stream.ensure_request(obs, step)
        stream.decision(step)
    assert calls == [(0, 40), (10, 40), (20, 40), (30, 40)]


def test_checkpoint_restores_both_horizons(monkeypatch, tmp_path):
    from scripts.eval_ewm import eval_ewm_policy as inference
    original = policy()
    checkpoint = tmp_path/'checkpoint.pt'
    torch.save(dict(format='ewm_wan_latent_v2', model=original.policy_state_dict(), horizon=40,
                    model_config=dict(type='ewm', event_horizon=16, event_temporal_downsample=2)), checkpoint)
    monkeypatch.setattr(EWMModel, 'from_config', classmethod(lambda cls, cfg: cls(
        **options(cameras=2), event_horizon=cfg.get('event_horizon'),
        event_temporal_downsample=cfg.get('event_temporal_downsample', 1))))
    monkeypatch.setattr(inference, 'EWMNormalizer', lambda _: SimpleNamespace())
    restored, _, horizon = inference.load_policy(checkpoint)
    assert horizon == 40 and restored.event_prediction_horizon(horizon) == 16
    for name, value in original.policy_state_dict().items():
        torch.testing.assert_close(restored.policy_state_dict()[name], value)


def test_null_event_horizon_keeps_legacy_resume_contract():
    from openwam.train.utils.ewm_resume import resume_contract
    from tests.test_ewm_resume import config, TinyDataset
    cfg = config('/tmp/test-asymmetric-resume', 6)
    dataset, loader, accelerator = TinyDataset(), [0, 1], SimpleNamespace(num_processes=1)
    before = resume_contract(cfg, dataset, loader, accelerator, 6)
    cfg.model.event_horizon = None
    cfg.dataloader.event_horizon = None
    assert resume_contract(cfg, dataset, loader, accelerator, 6) == before
    cfg.model.event_horizon = cfg.dataloader.event_horizon = 16
    changed = resume_contract(cfg, dataset, loader, accelerator, 6)
    assert changed['model'] != before['model'] and changed['dataloader'] != before['dataloader']


def test_presets_match_model_data_and_execution_defaults(monkeypatch):
    monkeypatch.setenv('NEW_PROJECT_ROOT', '/project')
    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).parents[1]/'configs')):
        for name in ('train_ewm_eight',):
            cfg = compose(config_name=name)
            assert cfg.dataloader.chunk_size == 24
            assert cfg.model.event_horizon == cfg.dataloader.event_horizon == 24
            assert cfg.model.event_temporal_downsample == 2
            assert cfg.dataloader.pad_tail
            assert cfg.model.type == 'ewm'
            assert 'rgb_history_steps' not in cfg.model and 'memory_enabled' not in cfg.model
        cfg = compose(config_name='train_ewm_eight_action_only')
        assert cfg.dataloader.chunk_size == 24 and cfg.dataloader.pad_tail
    from scripts.eval_ewm.eval_ewm_piper_tasks_sim import build_parser
    assert build_parser().parse_args(['--task', 'conveyor', '--plan-only']).execution_horizon == 10


@pytest.mark.parametrize('bad', [0, -1, 10, True, 16.0])
def test_invalid_event_lengths_rejected(bad):
    with pytest.raises(ValueError):
        EWMModel(**options(cameras=2), event_horizon=bad, event_temporal_downsample=2)


def test_mismatched_future_labels_rejected():
    with pytest.raises(ValueError, match='target length'):
        policy()(small_batch(40, cameras=2))
