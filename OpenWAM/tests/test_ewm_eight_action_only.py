"""Current-only cached conditioning and terminal labels for the eight-task ablation."""
from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir

from openwam.dataloader.ewm_action_only import PiperActionOnlyDataset
from tests.test_ewm_action_only import make_model, batch, forbidden
from tests.test_ewm_tail_mask import toy_dataset


@pytest.fixture(autouse=True)
def cpu_threads():
    torch.set_num_threads(2)


def test_terminal_action_does_not_cross_episode(monkeypatch, tmp_path):
    from openwam.dataloader import ewm_action_only, ewm_dataset
    d = toy_dataset(monkeypatch, tmp_path)
    d.__class__ = PiperActionOnlyDataset
    d.action_only_mmap_cache_dir = None
    d.current_dino_cache = None
    d._rgb = lambda row, indices: torch.zeros(1, 2, 3, 2, 2, dtype=torch.uint8)
    monkeypatch.setattr(ewm_action_only, '_read_action_table', ewm_dataset._read_table)
    monkeypatch.setattr(d, '_direct_event_frames', forbidden)
    for index, expected, count in [(0, 1, 4), (4, 5, 2), (5, 6, 1), (6, 7, 4), (11, 12, 1)]:
        item = d[index]
        assert set(item) == {'obs_video', 'proprio', 'action', 'prompt', 'action_valid_mask'}
        assert item['action'][0, 0] == expected
        assert item['action_valid_mask'].sum() == count
        assert not item['action'][count:].any()


def test_cached_rgb_matches_online_and_padding_is_ignored(monkeypatch):
    model = make_model()
    inputs = batch(horizon=4)
    inputs['action_valid_mask'] = torch.tensor([[True, True, False, False], [True, False, False, False]])
    features = model.context_encoder.encode_rgb_features(inputs['obs_video'])
    fixed = dict(t=torch.tensor([.3, .6]), action_noise=torch.randn_like(inputs['action']))
    reference = model(inputs, **fixed)['loss']
    cached = {k: v for k, v in inputs.items() if k != 'obs_video'}
    cached['current_dino_features'] = features
    cached['action'] = inputs['action'].clone()
    cached['action'][~inputs['action_valid_mask']] = float('nan')
    monkeypatch.setattr(model.context_encoder.dino, 'batch_encode', forbidden)
    actual = model(cached, **fixed)
    assert set(actual) == {'loss', 'loss_action'}
    torch.testing.assert_close(actual['loss'], reference)
    actual['loss'].backward()
    assert model.event_codec is None and not hasattr(model, "alignment_weight")
    assert model.action_head[-1].weight.grad.abs().sum() > 0
    assert all(block.event is None for block in model.blocks)
    assert not any('memory' in name for name, _ in model.named_parameters())


def test_eight_action_only_config(monkeypatch):
    monkeypatch.setenv('NEW_PROJECT_ROOT', '/project')
    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).parents[1] / 'configs')):
        cfg = compose(config_name='train_ewm_eight_action_only')
    assert cfg.model.type == 'ewm' and cfg.model.training_mode == 'action_only'
    assert "alignment_weight" not in cfg.model and cfg.model.event_vae_path is None
    assert 'memory_enabled' not in cfg.model
    assert "rgb_history_steps" not in cfg.model and "rgb_history_steps" not in cfg.dataloader
    assert len(cfg.dataloader.tasks) == 8 and cfg.dataloader.pad_tail
    assert not cfg.training.wandb_log_checkpoints
    assert cfg.training.batch_size == 32 and cfg.training.max_steps == 80000
