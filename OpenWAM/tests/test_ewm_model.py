"""EWM integration gates; small models preserve the exact v1 computation graph."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from openwam.model.architectures.ewm import EWMContextEncoder, EWMJointBlock, event_action_coordinates


class FakeDino(nn.Module):
    def __init__(self, tokens=3):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(6))
        self.properties = SimpleNamespace(z_dim=6)
        self.tokens = tokens
        self.calls = 0

    def preprocess_video(self, frames):
        import numpy as np

        return torch.from_numpy(np.array(frames[0]).copy()).permute(2, 0, 1)[None, :, None].float() / 255

    def batch_encode(self, pixels):
        self.calls += 1
        return (pixels.mean((1, 2, 3, 4))[:, None] * self.weight)[..., None, None, None].expand(
            -1, -1, 1, 1, self.tokens
        )


class FakeText(nn.Module):
    hidden_size = 16

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(16))
        self.calls = 0

    def forward(self, prompts):
        self.calls += 1
        return self.weight.expand(len(prompts), 4, 16), torch.tensor([[True, True, False, False]]).expand(
            len(prompts), -1
        )


def test_joint_block_context_mask_and_bidirectional_gradients():
    torch.manual_seed(5)
    block = EWMJointBlock(16, 2, action_hidden_dim=16)
    ec, ac = event_action_coordinates(1, 1, 1)
    event = torch.randn(2, 4, 16, requires_grad=True)
    action = torch.randn(2, 2, 16, requires_grad=True)
    ctx = torch.randn(2, 5, 16)
    mask = torch.tensor([[True, True, True, False, False]]).expand(2, -1)
    args = (torch.tensor([0.2, 0.8]), ctx, mask, ec, ac)
    e, a = block(event, action, *args)
    assert torch.autograd.grad(e.sum(), action, retain_graph=True)[0].abs().sum() > 0
    assert torch.autograd.grad(a.sum(), event, retain_graph=True)[0].abs().sum() > 0
    (e.square().mean() + a.square().mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())
    changed = ctx.clone()
    changed[:, 3:] += 1000
    e2, a2 = block(event, action, args[0], changed, mask, ec, ac)
    torch.testing.assert_close(e, e2)
    torch.testing.assert_close(a, a2)


def test_joint_block_heterogeneous_expert_widths_and_cached_context():
    torch.manual_seed(6)
    block = EWMJointBlock(dim=32, heads=2, action_hidden_dim=16)
    ec, ac = event_action_coordinates(1, 1, 1, num_cameras=1)
    event = torch.randn(2, 2, 32, requires_grad=True)
    action = torch.randn(2, 2, 16, requires_grad=True)
    context = torch.randn(2, 3, 32)
    mask = torch.ones(2, 3, dtype=torch.bool)
    args = (torch.tensor([.2, .8]), context, mask, ec, ac)
    expected = block(event, action, *args)
    assert expected[0].shape == event.shape and expected[1].shape == action.shape
    assert block.action.qkv.out_features == 3 * 32
    assert block.action.out.out_features == 16
    cached = block(event, action, *args,
                   event_context_kv=block.event.context.project_kv(context),
                   action_context_kv=block.action.context.project_kv(context))
    for actual, reference in zip(cached, expected):
        torch.testing.assert_close(actual, reference)
    assert torch.autograd.grad(expected[0].sum(), action, retain_graph=True)[0].abs().sum() > 0
    assert torch.autograd.grad(expected[1].sum(), event, retain_graph=True)[0].abs().sum() > 0


@pytest.mark.parametrize("tokens", [3, 7])
def test_context_frozen_dynamic_tokens_and_camera_order(tokens):
    ctx = EWMContextEncoder(FakeDino(tokens), FakeText(), 16).train()
    rgb = torch.zeros(2, 2, 3, 16, 16, dtype=torch.uint8)
    rgb[:, 1] = 255
    c, mask = ctx(rgb, ["task", "task"])
    assert c.shape == (2, 2 * tokens + 4, 16)
    assert mask.sum(1).tolist() == [2 * tokens + 2] * 2
    assert not ctx.dino.training and not ctx.text.training
    c.sum().backward()
    assert all(p.grad is None and not p.requires_grad for m in (ctx.dino, ctx.text) for p in m.parameters())
    assert ctx.rgb_projection.weight.grad is not None
    assert ctx.camera.grad is not None and ctx.modality.grad is not None
    assert not torch.allclose(c[:, 0], c[:, tokens])


from openwam.model.architectures.ewm import EWMModel, rectified_flow_inputs
from openwam.model.architectures.wan_event_codec import WanEventCodec


class FakeVAE(nn.Module):
    """Channel-preserving spatial mean codec; asserts single-frame invocation."""
    properties = SimpleNamespace(z_dim=48, spatial_compression=16)

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))
        self.encodes = 0
        self.decodes = 0

    def batch_encode(self, x):
        assert x.shape[2] == 1
        self.encodes += 1
        return torch.nn.functional.avg_pool2d(x[:, :, 0], 16).repeat(1, 16, 1, 1)[:, :, None] * self.anchor

    def decode(self, z, *, tiled=False):
        assert z.shape[2] == 1
        self.decodes += 1
        return torch.nn.functional.interpolate(z[:, :3, 0], scale_factor=16, mode='nearest')[:, :, None]


def small_model(**kw):
    args = dict(event_codec_impl="wan22_frame", event_temporal_compression=1, dim=16, action_hidden_dim=16, heads=2, layers=1, gradient_checkpointing=False,
                event_spatial_compression=32, event_patch_size=2,
                context_encoder=EWMContextEncoder(FakeDino(), FakeText(), 16),
                event_codec=WanEventCodec(mode="frame", vae=FakeVAE(), batch_size=3))
    args.update(kw)
    return EWMModel(**args)


def test_heterogeneous_model_training_and_sampling():
    model = small_model(dim=32, action_hidden_dim=16,
                        context_encoder=EWMContextEncoder(FakeDino(), FakeText(), 32))
    batch = small_batch()
    loss = model(batch)["loss"]
    loss.backward()
    assert model.event_input.out_channels == 32
    assert model.action_mlp[-1].out_features == 16
    assert model.blocks[0].action.qkv.weight.grad is not None
    sampled = model.sample_policy(batch, horizon=2,
                                  normalizer=SimpleNamespace(denormalize_action=lambda x: x),
                                  num_steps=2, decode_future=False)
    assert sampled["action"].shape == (1, 2, 7)


def small_batch(k=2, cameras=1, resolution=32):
    return dict(obs_voxel=torch.rand(1, cameras, 2, resolution, resolution),
                future_voxel=torch.rand(1, k, cameras, 2, resolution, resolution),
                proprio=torch.rand(1, 8), action=torch.rand(1, k, 7),
                obs_video=torch.zeros(1, 2, 3, resolution, resolution, dtype=torch.uint8), prompt=['stack'])


def test_opt_in_forward_segment_profile_preserves_loss():
    model = small_model()
    batch = small_batch()
    flow_time = torch.tensor([.3])
    event_noise = torch.zeros_like(model.event_codec.encode(batch['future_voxel']))
    action_noise = torch.zeros_like(batch['action'])
    kwargs = dict(t=flow_time, event_noise=event_noise, action_noise=action_noise)
    ordinary = model(batch, **kwargs)
    profiled = model(batch, profile_timing=True, **kwargs)
    torch.testing.assert_close(profiled['loss'], ordinary['loss'])
    assert '_profile' not in ordinary
    assert set(profiled['_profile']) == {
        'vae_encode', 'flow_prepare', 'context_encode', 'dit_forward', 'loss_and_diagnostics'
    }
    assert all(value >= 0 for value in profiled['_profile'].values())


@pytest.mark.parametrize('cameras', [1, 2])
def test_framewise_rgb_roundtrip_and_microbatch_order(cameras):
    codec = WanEventCodec(mode="frame", vae=FakeVAE(), batch_size=3)
    # Distinct frame/camera/polarity constants survive the fake spatial codec exactly.
    x = torch.linspace(0, 1, 2*5*cameras*2).reshape(2,5,cameras,2,1,1).expand(-1,-1,-1,-1,32,32)
    z = codec.encode(x)
    assert z.shape == (2,5,cameras,48,2,2)
    torch.testing.assert_close(codec.decode(z), x)
    other = WanEventCodec(mode="frame", vae=FakeVAE(), batch_size=100)
    torch.testing.assert_close(other.encode(x), z)
    codec.train()
    assert not codec.vae.training
    assert all(not p.requires_grad for p in codec.parameters())


@pytest.mark.parametrize('cameras', [1, 2])
@pytest.mark.parametrize('horizon', [1, 3, 24])
def test_latent_forward_backward_sampling(cameras, horizon):
    m = small_model(event_camera_mode='dual' if cameras==2 else 'wrist_only')
    b = small_batch(horizon, cameras)
    losses = m(b)
    assert m.event_codec.vae.decodes == 0
    losses['loss'].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)
    assert all(p.grad is None for p in m.event_codec.parameters())
    assert not m.event_codec.training
    diag = m(b, diagnostics=True)
    assert 'event_f1' in diag and 'voxel_x0_mse' in diag
    normalizer = SimpleNamespace(denormalize_action=lambda x:x*2)
    result = m.sample_policy(b, horizon=horizon, normalizer=normalizer, num_steps=2)
    assert result['future_voxel'].shape == b['future_voxel'].shape
    assert result['future_latent'].shape == (1,horizon,cameras,48,2,2)
    torch.testing.assert_close(result['action'],result['action_normalized']*2)
    assert result['future_voxel'].min()>=0 and result['future_voxel'].max()<=1
    assert m.training and not m.event_codec.training


def test_noise_added_after_frozen_encoding_and_shared_time(monkeypatch):
    m=small_model();b=small_batch();z=m.event_codec.encode(b['future_voxel'])
    en=torch.randn_like(z);an=torch.randn_like(b['action']);t=torch.tensor([.37])
    def velocity(obs, state, e, a, actual, ctx):
        torch.testing.assert_close(e,.63*z+.37*en)
        torch.testing.assert_close(a,.63*b['action']+.37*an)
        torch.testing.assert_close(actual,t)
        return en-z,an-b['action']
    monkeypatch.setattr(m,'predict_velocity',velocity)
    result=m(b,t=t,event_noise=en,action_noise=an)
    assert result['loss']==0
    assert result['latent_x0_mse']<1e-12
    with pytest.raises(ValueError,match='noise shapes'):
        m(b,event_noise=torch.randn_like(b['future_voxel']))


def test_sampling_sign_and_observation_encoded_once(monkeypatch):
    m=small_model();b=small_batch();times=[]
    def velocity(obs,state,e,a,t,ctx,context_kv=None):
        assert context_kv is not None
        times.append(t.item());return torch.ones_like(e)*.1,torch.ones_like(a)*.2
    monkeypatch.setattr(m,'predict_velocity',velocity)
    rng=torch.Generator().manual_seed(73)
    result=m.sample_policy(b,horizon=2,normalizer=SimpleNamespace(denormalize_action=lambda x:x),num_steps=4,generator=rng)
    rng.manual_seed(73);en=torch.randn(1,2,1,48,2,2,generator=rng);an=torch.randn(1,2,7,generator=rng)
    torch.testing.assert_close(result['future_latent'],en-.1)
    torch.testing.assert_close(result['action'],(an-.2).clamp(-1,1))
    assert times==[1,.75,.5,.25] and m.event_codec.vae.encodes==1
    assert m.context_encoder.dino.calls == 1


def test_action_only_sampling_caches_text_and_skips_future_decode():
    m = small_model().eval()
    batch = small_batch()
    normalizer = SimpleNamespace(denormalize_action=lambda x: x)
    result = m.sample_policy(
        batch, horizon=2, normalizer=normalizer, num_steps=4,
        generator=torch.Generator().manual_seed(3), decode_future=False,
    )
    assert "future_voxel" not in result
    assert m.event_codec.vae.decodes == 0
    # Context is built once outside all four denoising iterations.
    assert m.context_encoder.dino.calls == 1
    assert m.context_encoder.text.calls == 1
    m.sample_policy(
        batch, horizon=2, normalizer=normalizer, num_steps=1,
        generator=torch.Generator().manual_seed(4), decode_future=False,
    )
    # RGB changes every replan, while the identical frozen prompt stays cached.
    assert m.context_encoder.dino.calls == 2
    assert m.context_encoder.text.calls == 1


def test_context_kv_cache_matches_eager_and_projects_once_per_layer():
    m = small_model().eval()
    batch = small_batch()
    normalizer = SimpleNamespace(denormalize_action=lambda x: x)
    calls = []
    handles = [
        branch.context.k.register_forward_hook(lambda *_: calls.append(1))
        for block in m.blocks for branch in (block.event, block.action)
    ]
    uncached = m.sample_policy(
        batch, horizon=2, normalizer=normalizer, num_steps=4,
        generator=torch.Generator().manual_seed(11), decode_future=False,
        cache_context_kv=False,
    )
    assert len(calls) == 8
    calls.clear()
    cached = m.sample_policy(
        batch, horizon=2, normalizer=normalizer, num_steps=4,
        generator=torch.Generator().manual_seed(11), decode_future=False,
        cache_context_kv=True,
    )
    assert len(calls) == 2
    for handle in handles:
        handle.remove()
    torch.testing.assert_close(cached["action_normalized"], uncached["action_normalized"])
    torch.testing.assert_close(cached["future_latent"], uncached["future_latent"])


def test_checkpoint_excludes_vae_and_strict_policy_loading():
    m=small_model();state=m.policy_state_dict()
    assert not any(k.startswith(('event_codec.','context_encoder.dino.','context_encoder.text.')) for k in state)
    other=small_model();other.load_policy_state_dict(state)
    damaged=dict(state);damaged.pop('camera')
    with pytest.raises(RuntimeError,match='Invalid EWM'):other.load_policy_state_dict(damaged)


def test_checkpointed_gradients_equal():
    torch.manual_seed(2);a=small_model();b=small_model(gradient_checkpointing=True);b.load_state_dict(a.state_dict());batch=small_batch()
    torch.manual_seed(3);la=a(batch)['loss'];la.backward()
    torch.manual_seed(3);lb=b(batch)['loss'];lb.backward()
    torch.testing.assert_close(la,lb)
    for pa,pb in zip(a.parameters(),b.parameters()):
        if pa.requires_grad:torch.testing.assert_close(pa.grad,pb.grad)


def test_dino_schedule():
    m=small_model(layers=3,dino_injection_interval=2);lengths=[]
    for block in m.blocks:
        block.register_forward_pre_hook(lambda module,args:lengths.append(args[3].shape[1]))
    m(small_batch())
    assert lengths==[10,4,10]


@pytest.mark.parametrize('options',[dict(voxel_bins=4),dict(event_temporal_compression=2),dict(event_codec_impl='patch_only_2d')])
def test_dense_configuration_rejected(options):
    with pytest.raises(ValueError,match='dense codecs were removed'):small_model(**options)


def test_token_count_and_time_coordinates():
    m=small_model();b=small_batch(24);seen=[]
    m.blocks[0].register_forward_pre_hook(lambda module,args:seen.append((args[0].shape,args[5].clone())))
    z=torch.randn(1,24,1,48,16,16)
    m.predict_velocity(z[:,0],b['proprio'],z,b['action'],torch.ones(1),(torch.randn(1,3,16),torch.ones(1,3,dtype=torch.bool)))
    shape,coords=seen[0]
    assert shape==(1,1600,16)
    assert coords[:,0].reshape(25,64)[:,0].tolist()==list(range(25))


def test_event_head_unpatches_channels_and_spatial_offsets():
    m=small_model(event_camera_mode='dual')
    class PatternHead(nn.Module):
        def forward(self,x):
            # Unique value for each frame/camera/grid/channel/patch offset.
            return torch.arange(x.numel()//x.shape[-1]*192,dtype=x.dtype).reshape(*x.shape[:-1],192)
    m.event_head=PatternHead()
    z=torch.zeros(1,2,2,48,4,6)
    v,_=m.predict_velocity(z[:,0],torch.zeros(1,8),z,torch.zeros(1,2,7),torch.ones(1),(torch.zeros(1,1,16),torch.ones(1,1,dtype=torch.bool)))
    expected=torch.arange(2*2*2*3*192).reshape(1,2,2,2,3,48,2,2)
    for k,c,gy,gx,ch,dy,dx in [(0,0,0,0,0,0,0),(1,1,1,2,47,1,1),(0,1,0,2,17,1,0)]:
        assert v[0,k,c,ch,gy*2+dy,gx*2+dx]==expected[0,k,c,gy,gx,ch,dy,dx]


def test_new_policy_loader_and_old_format_rejection(monkeypatch,tmp_path):
    from scripts.eval_ewm import eval_ewm_policy as module
    m=small_model()
    payload=dict(format='ewm_wan_latent_v2',model=m.policy_state_dict(),model_config={},horizon=24)
    monkeypatch.setattr(module.torch,'load',lambda *a,**kw:payload)
    monkeypatch.setattr(module.EWMModel,'from_config',lambda cfg:small_model())
    monkeypatch.setattr(module,'EWMNormalizer',lambda path:SimpleNamespace())
    restored,_,horizon=module.load_policy(tmp_path/'checkpoint.pt')
    assert not restored.training and horizon==24
    payload['format']='ewm_trainable_v1'
    with pytest.raises(ValueError,match='incompatible'):module.load_policy(tmp_path/'checkpoint.pt')


class FakeJointVAE(FakeVAE):
    def batch_encode(self,x):
        self.encodes+=1
        n,c,t,h,w=x.shape
        spatial=torch.nn.functional.avg_pool2d(x.permute(0,2,1,3,4).reshape(n*t,c,h,w),16).reshape(n,t,c,h//16,w//16)
        assert (t-1)%4==0
        z=torch.cat((spatial[:,:1],spatial[:,1:].reshape(n,(t-1)//4,4,c,h//16,w//16).mean(2)),1) if t>1 else spatial
        return z.permute(0,2,1,3,4).repeat(1,16,1,1,1)

    def decode(self,z,*,tiled=False):
        self.decodes+=1
        frames=torch.cat((z[:,:3,:1],z[:,:3,1:].repeat_interleave(4,2)),2)
        n,c,t,h,w=frames.shape
        return torch.nn.functional.interpolate(frames.permute(0,2,1,3,4).reshape(n*t,c,h,w),scale_factor=16).reshape(n,t,c,h*16,w*16).permute(0,2,1,3,4)


def test_joint_model_shape_time_and_causal_condition():
    codec=WanEventCodec(vae=FakeJointVAE(),mode='joint',batch_size=1)
    m=small_model(event_codec_impl='wan22_joint',event_temporal_compression=4,event_codec=codec)
    batch=small_batch(24)
    video=torch.cat((batch['obs_voxel'][:,None],batch['future_voxel']),1)
    z=codec.encode(video)
    assert z.shape==(1,7,1,48,2,2)
    torch.testing.assert_close(z[:,:1],codec.encode(video[:,:1]))
    seen=[]
    m.blocks[0].register_forward_pre_hook(lambda module,args:seen.append((args[0].shape,args[1].shape,args[5].clone())))
    losses=m(batch,diagnostics=True);losses['loss'].backward()
    assert seen[0][0][1]==7 and seen[0][1][1]==25
    assert seen[0][2][:,0].tolist()==[0,2.5,6.5,10.5,14.5,18.5,22.5]
    sampled=m.sample_policy(batch,horizon=24,normalizer=SimpleNamespace(denormalize_action=lambda x:x),num_steps=2)
    assert sampled['future_voxel'].shape==batch['future_voxel'].shape
    assert sampled['future_latent'].shape==(1,6,1,48,2,2)
    with pytest.raises(ValueError):m.sample_policy(batch,horizon=3,normalizer=None)


class FakeLeanVAE(nn.Module):
    """Native Lean layout/range with a causal first slice and four-frame pooling."""
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))

    def set_tile_inference(self, enabled):
        assert not enabled

    def encode(self, x):
        assert x.dtype == torch.bfloat16
        assert not torch.is_autocast_enabled('cpu')
        b,c,t,h,w = x.shape
        frames = torch.nn.functional.avg_pool2d(x.transpose(1,2).reshape(b*t,c,h,w),8).reshape(b,t,c,h//8,w//8)
        future = frames[:,1:].reshape(b,(t-1)//4,4,c,h//8,w//8).mean(2)
        z = torch.cat((frames[:,:1],future),1).transpose(1,2)
        return z.repeat(1,6,1,1,1)[:,:16] * self.anchor

    def decode(self, z, is_image=False):
        assert z.dtype == torch.bfloat16
        frames = torch.cat((z[:,:3,:1],z[:,:3,1:].repeat_interleave(4,2)),2)
        b,c,t,h,w = frames.shape
        x = torch.nn.functional.interpolate(frames.transpose(1,2).reshape(b*t,c,h,w),scale_factor=8).reshape(b,t,c,h*8,w*8).transpose(1,2)
        return x[:,:,0] if is_image else x


@pytest.mark.parametrize('patch',[2,4,8])
def test_lean_codec_training_sampling_and_policy(patch):
    from openwam.model.architectures.lean_event_codec import LeanEventCodec
    codec = LeanEventCodec(vae=FakeLeanVAE(), batch_size=1)
    m = small_model(event_codec_impl='leanvae16_joint',event_temporal_compression=4,
                    event_spatial_compression=8*patch,event_patch_size=patch,event_codec=codec,event_camera_mode='dual')
    resolution = 64 if patch == 8 else 32
    batch = small_batch(4,cameras=2,resolution=resolution)
    # Channel/camera/batch order, range mapping and frozen BF16 under outer autocast.
    video = torch.cat((batch['obs_voxel'][:,None],batch['future_voxel']),1)
    constant = torch.arange(20).reshape(2,5,2,1,1,1).float().expand(2,5,2,2,32,32)/20
    with torch.autocast('cpu',dtype=torch.bfloat16):
        z = codec.encode(constant)
        restored = codec.decode(z)
    assert z.shape == (2,2,2,16,4,4)
    torch.testing.assert_close(restored[:,:1],constant[:,:1],atol=.004,rtol=0)
    expected = constant[:,1:].mean(1,keepdim=True).expand(-1,4,-1,-1,-1,-1)
    torch.testing.assert_close(restored[:,1:],expected,atol=.004,rtol=0)
    codec.to(dtype=torch.bfloat16)
    assert next(codec.parameters()).dtype == torch.bfloat16
    seen=[]
    m.blocks[0].register_forward_pre_hook(lambda module,args:seen.append(args[0].shape[1]))
    loss=m(batch,diagnostics=True);loss['loss'].backward()
    assert seen[0] == 2*2*(resolution//8//patch)**2
    assert m.event_input.weight.grad is not None and m.event_head[1].weight.grad is not None
    assert all(p.grad is None and not p.requires_grad for p in codec.parameters())
    sampled=m.sample_policy(batch,horizon=4,normalizer=SimpleNamespace(denormalize_action=lambda x:x),num_steps=2)
    assert sampled['future_voxel'].shape == batch['future_voxel'].shape
    assert sampled['future_latent'].shape == (1,1,2,16,resolution//8,resolution//8)
    state=m.policy_state_dict()
    assert not any(k.startswith('event_codec.') for k in state)
    m.load_policy_state_dict(state)
    with pytest.raises(RuntimeError):small_model().load_policy_state_dict(state)


def test_lean_model_configuration_composes_without_wan_override():
    from hydra import compose, initialize_config_dir
    from pathlib import Path
    with initialize_config_dir(config_dir=str(Path(__file__).resolve().parents[1]/'configs'),version_base=None):
        cfg=compose(config_name='train_ewm',overrides=['model=ewm_leanvae16'])
        default=compose(config_name='train_ewm')
        wan=compose(config_name='train_ewm',overrides=['model=ewm_wan22'])
    assert wan.model.event_codec_impl=='wan22_joint'
    assert wan.model.event_spatial_compression==32
    assert wan.model.event_patch_size==2
    assert not default.model.gradient_checkpointing and not default.training.compile_core
    assert cfg.model.event_codec_impl=='leanvae16_joint'
    assert cfg.model.event_spatial_compression==32
    assert cfg.model.event_patch_size==4
    assert cfg.model.event_temporal_compression==4
    assert default.model.event_codec_impl=='leanvae16_joint'
    assert default.model.event_spatial_compression==32
    assert default.model.event_patch_size==4
    assert default.model.dim==768 and default.model.action_hidden_dim==512


def test_reject_mismatched_injected_codec():
    with pytest.raises(ValueError,match='does not match'):
        small_model(event_codec_impl='leanvae16_joint',event_temporal_compression=4,event_spatial_compression=16)


def test_explicit_patch4_codec_equal_token_geometry():
    from openwam.model.architectures.lean_event_codec import LeanEventCodec
    lean=EWMModel(dim=16,heads=2,layers=1,event_spatial_compression=32,event_codec=LeanEventCodec(vae=FakeLeanVAE()))
    wan=EWMModel(dim=16,heads=2,layers=1,event_spatial_compression=32,event_codec_impl='wan22_joint',event_codec=WanEventCodec(vae=FakeJointVAE()))
    assert lean.latent_patch_size==4 and wan.latent_patch_size==2
    assert 7*(256//lean.event_codec.spatial_compression//lean.latent_patch_size)**2 == 448
    assert 7*(256//wan.event_codec.spatial_compression//wan.latent_patch_size)**2 == 448
    assert not lean.gradient_checkpointing


def test_patch_size_must_match_codec_spatial_stride():
    with pytest.raises(ValueError, match='matching total spatial stride'):
        small_model(event_patch_size=8, event_spatial_compression=32)


def test_event_only_ignores_actions_and_proprio_and_has_no_action_gradients():
    model = small_model(training_mode='event_only', gradient_checkpointing=True)
    batch = small_batch()
    event_noise = torch.zeros_like(model.event_codec.encode(batch['future_voxel']))
    fixed = dict(t=torch.tensor([.4]), event_noise=event_noise)
    first = model(batch, **fixed)
    changed = dict(batch, action=torch.randn_like(batch['action']) * 100,
                   proprio=torch.randn_like(batch['proprio']) * 100)
    second = model(changed, **fixed)
    no_action = {key: value for key, value in batch.items() if key not in ('action', 'proprio')}
    third = model(no_action, **fixed)
    torch.testing.assert_close(first['loss'], first['loss_event'])
    torch.testing.assert_close(first['loss_action'], torch.zeros_like(first['loss_action']))
    torch.testing.assert_close(first['loss_event'], second['loss_event'])
    torch.testing.assert_close(first['loss_event'], third['loss_event'])
    first['loss'].backward()
    assert model.event_input.weight.grad is not None
    assert model.blocks[0].event.qkv.weight.grad is not None
    assert model.blocks[0].action.qkv.weight.grad is None
    assert model.action_mlp[0].weight.grad is None
    assert model.state_mlp[0].weight.grad is None
    assert model.action_head[-1].weight.grad is None
    assert not model.action_mlp[0].weight.requires_grad
    sample = model.sample_policy(no_action, horizon=2, num_steps=2, decode_future=False)
    assert "action" not in sample and "action_normalized" not in sample
    assert sample["future_latent"].shape[1] == 2
    # Retain the exact policy state-dict layout for later joint fine-tuning.
    joint = small_model()
    joint.load_policy_state_dict(model.policy_state_dict())


def test_event_only_valid_region_excludes_letterbox_from_loss(monkeypatch):
    model = small_model(training_mode='event_only')
    batch = small_batch()
    clean = model.event_codec.encode(batch['future_voxel'])
    noise = torch.zeros_like(clean)
    target_velocity = noise - clean

    def velocity(_obs, _state, _event, _action, _time, _context):
        predicted = target_velocity.clone()
        predicted[..., 0, :] += 1  # Error only in the top latent row.
        return predicted, None

    monkeypatch.setattr(model, 'predict_velocity', velocity)
    args = dict(t=torch.tensor([.4]), event_noise=noise)
    unmasked = model(batch, **args)
    batch['valid_region_mask'] = torch.zeros(1, 1, 32, 32, dtype=torch.bool)
    batch['valid_region_mask'][..., 16:, :] = True
    masked = model(batch, **args)
    assert unmasked['loss_event'] > 0
    torch.testing.assert_close(masked['loss_event'], torch.zeros_like(masked['loss_event']))


@pytest.mark.parametrize("camera_mode", ["wrist_only", "dual"])
def test_piper_rgb_action_trains_both_heads_and_samples_uint8_rgb(camera_mode):
    from openwam.model.architectures.lean_rgb_codec import LeanRGBCodec

    codec = LeanRGBCodec(vae=FakeLeanVAE(), batch_size=2)
    model = small_model(
        training_mode="rgb_action", event_codec_impl="leanvae16_rgb_joint",
        event_temporal_compression=4, event_spatial_compression=16,
        event_patch_size=2, event_camera_mode=camera_mode, event_codec=codec,
    )
    cameras = 2 if camera_mode == "dual" else 1
    rgb = torch.randint(0, 256, (1, 5, cameras, 3, 32, 32), dtype=torch.uint8)
    batch = dict(
        obs_video=torch.randint(0, 256, (1, 2, 3, 32, 32), dtype=torch.uint8),
        obs_rgb=rgb[:, 0], future_rgb=rgb[:, 1:],
        action=torch.randn(1, 4, 7), proprio=torch.randn(1, 8), prompt=["stack"],
    )
    z = codec.encode(rgb)
    assert z.shape == (1, 2, cameras, 16, 4, 4)
    reconstructed = codec.decode(z)
    assert reconstructed.shape == (1, 5, cameras, 3, 32, 32)
    assert reconstructed.min() >= 0 and reconstructed.max() <= 1

    losses = model(batch, diagnostics=True)
    assert "loss_rgb" in losses and "loss_event" not in losses
    assert losses["rgb_x0_psnr"].isfinite()
    torch.testing.assert_close(losses["loss"], losses["loss_rgb"] + losses["loss_action"])
    losses["loss"].backward()
    assert model.event_head[-1].weight.grad is not None
    assert model.action_head[-1].weight.grad is not None
    assert all(not p.requires_grad and p.grad is None for p in codec.parameters())

    sampled = model.sample_policy(
        batch, horizon=4, normalizer=SimpleNamespace(denormalize_action=lambda x: x),
        num_steps=2,
    )
    assert sampled["future_rgb"].shape == batch["future_rgb"].shape
    assert sampled["action"].shape == batch["action"].shape
