"""Original Wan + ActionDiT action-only mode (distinct from hierarchical EWM)."""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir

from openwam.model.architectures.dual_system.mot_driver import DualSystemMoTDriver
from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture
from openwam.model.architectures.utils.mask_modes import ACTION_SEES_VIDEO
from tests.test_mot_driver import _make_action_dit, _make_states
from tests.test_openwam_trainer import _MockVideoBackbone


@pytest.mark.parametrize("checkpointing", [False, True])
def test_detached_video_matches_joint_attention_and_action_gradients(checkpointing):
    torch.manual_seed(42)
    torch.set_num_threads(2)
    video = _MockVideoBackbone(dim=32, num_layers=2, num_heads=4).requires_grad_(False)
    action = _make_action_dit()
    video_ref, action_ref = copy.deepcopy(video), copy.deepcopy(action)
    rng = torch.random.get_rng_state()
    state, astate = _make_states(video, action)
    # An upstream trainable state/context must not reintroduce a video graph.
    state.hidden_states.requires_grad_(True)
    torch.random.set_rng_state(rng)
    state_ref, astate_ref = _make_states(video_ref, action_ref)
    state_ref.hidden_states.requires_grad_(True)
    expected_driver = DualSystemMoTDriver(video_ref, action_ref, mot_checkpoint_mixed_attn=False,
                                          attention_mask_mode=ACTION_SEES_VIDEO)
    _, expected = expected_driver.run_joint_loop(state_ref, astate_ref)
    pred_ref = action_ref.extract_prediction(expected)
    pred_ref.square().mean().backward()
    driver = DualSystemMoTDriver(video, action, action_only=True, mot_checkpoint_mixed_attn=True,
                                 attention_mask_mode=ACTION_SEES_VIDEO)
    state, actual = driver.run_joint_loop(state, astate, use_gradient_checkpointing=checkpointing)
    assert not state.hidden_states.requires_grad
    pred = action.extract_prediction(actual)
    torch.testing.assert_close(pred, pred_ref)
    pred.square().mean().backward()
    assert all(p.grad is None for p in video.parameters())
    for (name, p), (_, reference) in zip(action.named_parameters(), action_ref.named_parameters()):
        if reference.grad is not None:
            assert p.grad is not None, name
            torch.testing.assert_close(p.grad, reference.grad, atol=1e-6, rtol=1e-4)


def test_driver_rejects_mutual_or_trainable_video():
    video = _MockVideoBackbone(dim=32, num_layers=2, num_heads=4)
    action = _make_action_dit()
    with pytest.raises(ValueError, match="frozen"):
        DualSystemMoTDriver(video, action, action_only=True)
    video.requires_grad_(False)
    with pytest.raises(ValueError, match="independent"):
        DualSystemMoTDriver(video, action, action_only=True, attention_mask_mode="mutual")


def test_architecture_mode_guards_and_freezing():
    arch = DualSystemSelfAttnArchitecture(dict(
        action_only=True, dim=32, ffn_dim=64, num_heads=4, video_dim=32,
        text_dim=32, action_dim=7, bridge_layers=(0, 1),
        attention_mask_mode=ACTION_SEES_VIDEO,
    ))
    arch.video_backbone = _MockVideoBackbone(dim=32, num_layers=2, num_heads=4)
    arch.build_mot_driver()
    arch.train()
    assert not arch.video_backbone.training
    assert not any(p.requires_grad for p in arch.video_backbone.parameters())
    assert all(p.requires_grad for p in arch.action_backbone.parameters())
    with pytest.raises(ValueError, match="lambda_video=0"):
        arch.compute_loss(lambda_video=1.0)
    with pytest.raises(ValueError, match="ground-truth actions"):
        arch.compute_loss(lambda_video=0.0)


def test_original_openwam_action_only_preset():
    from openwam.model import resolve_architecture_config

    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[1] / "configs")):
        cfg = compose(config_name="train_openwam_action_only")
    assert cfg.model.architecture.action_only
    assert list(cfg.model.freeze) == ["video_backbone"]
    assert cfg.training.lambda_video == 0 and cfg.training.lambda_action == 1
    resolved = resolve_architecture_config(cfg.model)
    assert resolved.params["action_only"] is True


def test_zero_video_weight_skips_video_mse(monkeypatch):
    from tests.test_openwam_trainer import _make_tiny_arch, _make_fake_loss_inputs, _MockScheduler

    arch = _make_tiny_arch()
    arch.action_backbone.scheduler = _MockScheduler()
    monkeypatch.setattr(arch, "_compute_video_loss", lambda *a, **k: pytest.fail("video MSE must be skipped"))
    # Base loss routing is shared across original architectures.
    loss = arch.compute_loss(actions=torch.randn(1, 5, 7), lambda_video=0,
                             **_make_fake_loss_inputs())
    assert torch.isfinite(loss["loss"]) and loss["loss_video"].item() == 0
    loss["loss"].backward()


@pytest.mark.parametrize("checkpointing", [False, True])
def test_full_architecture_action_step_and_checkpoint(checkpointing, tmp_path, monkeypatch):
    from tests.test_openwam_trainer import _MockScheduler, _make_fake_loss_inputs
    from openwam.train.utils.optimizer_groups import build_trainable_parameters

    arch = DualSystemSelfAttnArchitecture(dict(
        action_only=True, dim=32, ffn_dim=64, num_heads=4, video_dim=32,
        text_dim=32, action_dim=7, bridge_layers=(0, 1),
        use_proprioception=True, state_dim=8, attention_mask_mode=ACTION_SEES_VIDEO,
    ))
    arch.video_backbone = _MockVideoBackbone(dim=32, num_layers=2, num_heads=4)
    arch._device, arch._dtype = torch.device('cpu'), torch.float32
    arch.action_backbone.scheduler = _MockScheduler()
    arch.build_mot_driver()
    arch.train()
    frozen = copy.deepcopy(arch.video_backbone.state_dict())
    before = copy.deepcopy(arch.action_backbone.state_dict())
    params = build_trainable_parameters(SimpleNamespace(architecture=arch))
    assert {id(p) for p in params} == {id(p) for p in arch.parameters() if p.requires_grad}
    assert not {id(p) for p in params}.intersection(id(p) for p in arch.video_backbone.parameters())
    optimizer = torch.optim.AdamW(params, lr=0.01)
    inputs = _make_fake_loss_inputs()
    inputs.update(context=torch.randn(1, 4, 32), context_mask=torch.ones(1, 4, dtype=torch.bool),
                  proprio=torch.randn(1, 8), use_gradient_checkpointing=checkpointing)
    original_prepare = arch.video_backbone.prepare
    preparation_modes = []

    def prepare(**kwargs):
        preparation_modes.append(torch.is_grad_enabled())
        return original_prepare(**kwargs)

    monkeypatch.setattr(arch.video_backbone, 'prepare', prepare)
    monkeypatch.setattr(arch, '_compute_video_loss', lambda *a, **k: pytest.fail('video loss must be unused'))
    losses = arch.compute_loss(actions=torch.randn(1, 5, 7), lambda_video=0, **inputs)
    losses['loss'].backward()
    assert preparation_modes == [False]
    assert losses['loss_video'].item() == 0
    assert arch.proprio_encoder.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in arch.video_backbone.parameters())
    optimizer.step()
    assert all(torch.equal(v, frozen[k]) for k, v in arch.video_backbone.state_dict().items())
    assert any(not torch.equal(v, before[k]) for k, v in arch.action_backbone.state_dict().items())
    checkpoint = tmp_path / 'action.safetensors'
    arch.save_checkpoint(str(checkpoint))
    saved = copy.deepcopy(arch.state_dict())
    with torch.no_grad():
        for p in arch.action_backbone.parameters():
            p.zero_()
    arch.load_checkpoint(str(checkpoint))
    assert all(torch.equal(v, saved[k]) for k, v in arch.state_dict().items())
    assert not any(p.requires_grad for p in arch.video_backbone.parameters())


def test_action_only_override_survives_finetune_merge():
    from omegaconf import OmegaConf
    from openwam.model import resolve_architecture_config
    from openwam.train.utils.ckpt_model_loader import merge_ckpt_model_cfg
    from tests.test_finetune_config_override import CKPT_CFG

    with initialize_config_dir(version_base=None, config_dir=str(Path(__file__).resolve().parents[1] / 'configs')):
        cfg = compose(config_name='train_openwam_action_only')
    merge_ckpt_model_cfg(OmegaConf.create(CKPT_CFG), cfg, '/unused/checkpoint')
    assert cfg.model.architecture.action_only
    assert cfg.model.architecture.attention_mask_mode == ACTION_SEES_VIDEO
    assert list(cfg.model.freeze) == ['video_backbone']
    assert cfg.model.video_backbone.components  # retain checkpoint reconstruction specs
    assert resolve_architecture_config(cfg.model).params['action_only'] is True
