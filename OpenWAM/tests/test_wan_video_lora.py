"""CPU coverage of native Wan/ActionDiT adapters and the Piper training contract."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn

from openwam.model.architectures.base import BaseWAMArchitecture
from openwam.model.architectures.dual_system.joint_self_attn import DualSystemSelfAttnArchitecture
from openwam.model.architectures.utils.video_lora import LoRALinear, is_adapter_key
from openwam.model.video_backbone.wan.models.dit import WanModel
from openwam.model.video_backbone.wan_backbone import Wan22Ti2v
from openwam.train.utils.optimizer_groups import build_trainable_parameters


def tiny_backbone():
    dit = WanModel(dim=24, in_dim=4, ffn_dim=48, out_dim=4, text_dim=32,
                   freq_dim=16, eps=1e-6, patch_size=(1, 2, 2), num_heads=2,
                   num_layers=2, has_image_input=False, require_vae_embedding=False,
                   require_clip_embedding=False)
    return Wan22Ti2v(SimpleNamespace(dit=dit, vae=nn.Linear(1, 1), text_encoder=nn.Linear(1, 1)))


def make_arch(monkeypatch, lora=None):
    def init_video(self, cfg):
        self.video_backbone = tiny_backbone()
    monkeypatch.setattr(BaseWAMArchitecture, '_init_video_backbone', init_video)
    cfg = dict(dim=16, ffn_dim=32, text_dim=32, action_dim=7, use_proprioception=True,
               state_dim=8, bridge_interval=1, attention_mask_mode='mutual', mot_checkpoint_mixed_attn=True,
               video_backbone=dict(from_scratch=False))
    if lora is not None:
        cfg['video_lora'] = lora
    return DualSystemSelfAttnArchitecture(cfg)


def predict(arch, checkpointing):
    dtype = next(arch.parameters()).dtype
    torch.manual_seed(13)
    ctx = torch.cat([torch.randn(1, 4, 32, dtype=dtype),
                     arch.proprio_encoder(torch.randn(1, 1, 8, dtype=dtype))], dim=1)
    ts = torch.tensor([0.5], dtype=dtype)
    mask = torch.ones(1, 5, dtype=torch.bool)
    vs = arch.video_backbone.prepare(latents=torch.randn(1, 4, 2, 4, 4, dtype=dtype),
                                    timestep=ts, context=ctx, context_mask=mask)
    ac = arch.action_backbone.prepare_state(torch.randn(1, 3, 7, dtype=dtype), ts,
                                           context=ctx, context_mask=mask)
    vs, ac = arch.mot_driver.run_joint_loop(vs, ac, use_gradient_checkpointing=checkpointing)
    return arch.video_backbone.finalize(vs), arch.action_backbone.extract_prediction(ac)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('checkpointing', [False, True])
def test_native_joint_gradients_freezing_and_checkpoint(monkeypatch, tmp_path, dtype, checkpointing):
    torch.set_num_threads(2)
    arch = make_arch(monkeypatch).to(dtype=dtype)
    with torch.no_grad():
        before = predict(arch, False)
    arch.enable_video_lora({'rank': 2, 'alpha': 2, 'dropout': 0.0})
    with torch.no_grad():
        after = predict(arch, False)
    for a, b in zip(before, after):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert sum(isinstance(m, LoRALinear) for m in arch.modules()) == 20
    for n, p in arch.video_backbone.named_parameters():
        assert p.requires_grad == is_adapter_key(n), n
    assert all(p.requires_grad for p in arch.action_backbone.parameters())
    assert all(p.requires_grad for p in arch.proprio_encoder.parameters())
    frozen = {n: p.detach().clone() for n, p in arch.named_parameters() if not p.requires_grad}
    params = build_trainable_parameters(SimpleNamespace(architecture=arch))
    assert {id(p) for p in params} == {id(p) for p in arch.parameters() if p.requires_grad}
    optimizer = torch.optim.AdamW(params, lr=0.01)
    for _ in range(2):
        arch.eval()
        arch.train()
        optimizer.zero_grad()
        video, action = predict(arch, checkpointing)
        (video.float().square().mean() + action.float().square().mean()).backward()
        for n, p in arch.video_backbone.named_parameters():
            if is_adapter_key(n):
                assert p.grad is not None and torch.isfinite(p.grad).all(), n
            else:
                assert p.grad is None, n
        assert arch.proprio_encoder.weight.grad.abs().sum() > 0
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in arch.action_backbone.parameters())
        optimizer.step()
    for n, p in arch.named_parameters():
        if n in frozen:
            assert torch.equal(p, frozen[n]), n
    ckpt = tmp_path / 'lora.safetensors'
    arch.save_checkpoint(str(ckpt))
    # Reconstruct adapters before strict load, as deploy/finetune does via config.
    restored = make_arch(monkeypatch, OmegaConf.create(arch.video_lora_config)).to(dtype=dtype)
    restored.load_checkpoint(str(ckpt))
    for n, v in arch.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[n], v, rtol=0, atol=0)
    for a, b in zip(predict(arch, False), predict(restored, False)):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_invalid_lora_is_atomic(monkeypatch):
    arch = make_arch(monkeypatch)
    before = {n: p.requires_grad for n, p in arch.named_parameters()}
    with pytest.raises(ValueError, match='not found'):
        arch.enable_video_lora({'target_modules': ['self_attn.q', 'missing']})
    assert before == {n: p.requires_grad for n, p in arch.named_parameters()}
    assert not any(isinstance(m, LoRALinear) for m in arch.modules())
    arch.action_only = True
    with pytest.raises(ValueError, match='mutually exclusive'):
        arch.enable_video_lora({'rank': 2})


def test_piper_presets_share_data_lora_and_update_budget():
    from openwam.model import resolve_architecture_config
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(config_dir=str(root/'OpenWAM/configs'), version_base=None):
        ow = compose(config_name='train_openwam_piper_rgb')
    with initialize_config_dir(config_dir=str(root/'FastWAM/configs'), version_base=None):
        fw = compose(config_name='train', overrides=['task=piper_rgb'])
    assert OmegaConf.to_container(ow.model.architecture.video_lora) == OmegaConf.to_container(fw.model.video_lora)
    assert resolve_architecture_config(ow.model).params['video_lora']['rank'] == 16
    assert ow.training.max_optimizer_steps == fw.max_steps == 30000
    assert ow.training.num_epochs is None and ow.training.max_steps is None
    assert ow.training.weight_decay == fw.weight_decay == 0
    assert ow.training.batch_size == fw.batch_size == 1
    assert ow.training.gradient_accumulation_steps == fw.gradient_accumulation_steps == 1
    tasks = ['conveyor','moving_carton','turntable_cups','rotating_tube',
             'tennis_rolling','tennis_falling','tennis_parabolic','tennis_container']
    assert list(ow.dataloader.tasks) == tasks
    assert [Path(p).name for p in fw.data.train.dataset_dirs] == ['piper_'+t for t in tasks]
    assert ow.dataloader.num_frames == fw.data.train.num_frames == 25
    assert ow.dataloader.video_stride == fw.data.train.action_video_freq_ratio == 2
