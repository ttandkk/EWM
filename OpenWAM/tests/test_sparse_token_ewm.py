"""CPU contracts for fixed-budget event-driven spatial token compression."""

from math import prod
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from openwam.model.architectures.ewm import EWMContextEncoder
from openwam.model.architectures.ewm import EWMModel as DenseEWMModel
from openwam.model.architectures.lean_event_codec import LeanEventCodec
from openwam.model.architectures.sparse_token_ewm import (
    SparseTokenContext,
    SparseTokenEWMModel,
    merge_spatial_cells,
    split_spatial_cells,
)
from tests.test_ewm_model import FakeDino, FakeLeanVAE, FakeText


def model_options(cameras=1, training_mode="joint"):
    return dict(
        dim=16, action_hidden_dim=16, heads=2, layers=1,
        event_codec_impl="leanvae16_joint", event_temporal_compression=4,
        event_spatial_compression=32, event_patch_size=4,
        event_camera_mode="dual" if cameras == 2 else "wrist_only",
        event_codec=LeanEventCodec(vae=FakeLeanVAE(), batch_size=2),
        context_encoder=EWMContextEncoder(FakeDino(), FakeText(), 16),
        training_mode=training_mode,
    )


def sparse_model(refined_cells=8, cameras=1, training_mode="joint"):
    return SparseTokenEWMModel(
        refined_cells=refined_cells, **model_options(cameras, training_mode))


def explicit_context(batch_size=1):
    return torch.randn(batch_size, 3, 16), torch.ones(batch_size, 3, dtype=torch.bool)


def small_batch(cameras=1, horizon=4):
    return dict(
        obs_voxel=torch.rand(1, cameras, 2, 256, 256),
        future_voxel=torch.rand(1, horizon, cameras, 2, 256, 256),
        proprio=torch.rand(1, 8), action=torch.rand(1, horizon, 7),
        obs_video=torch.zeros(1, 2, 3, 256, 256, dtype=torch.uint8),
        prompt=["stack"],
    )


def active_cell(obs, batch, camera, cell, *, polarity=0, rows=32):
    """Distinct occupancy, without relying on event-count magnitudes."""
    y, x = divmod(cell, 4)
    obs[batch, camera, polarity, y * 64:y * 64 + rows, x * 64:(x + 1) * 64] = 1


@pytest.mark.parametrize("shape,cell_size", [((2, 3, 2, 3, 16, 24), 8), ((1, 2, 1, 2, 8, 12), 4)])
def test_spatial_cell_roundtrip_keeps_batch_time_camera_channel_and_pixel_order(shape, cell_size):
    x = torch.arange(prod(shape), dtype=torch.float32).reshape(shape)
    x.requires_grad_()
    cells = split_spatial_cells(x, cell_size)
    gh, gw = shape[-2] // cell_size, shape[-1] // cell_size
    assert cells.shape == (*shape[:3], gh * gw, shape[3], cell_size, cell_size)
    for cell in range(gh * gw):
        y, z = divmod(cell, gw)
        torch.testing.assert_close(
            cells[:, :, :, cell],
            x[..., y * cell_size:(y + 1) * cell_size, z * cell_size:(z + 1) * cell_size],
        )
    restored = merge_spatial_cells(cells, gh, gw)
    torch.testing.assert_close(restored, x, rtol=0, atol=0)
    restored.square().sum().backward()
    torch.testing.assert_close(x.grad, 2 * x, rtol=0, atol=0)


def test_routing_selects_both_polarities_independently_per_sample_and_camera():
    model = sparse_model(refined_cells=2, cameras=2)
    obs = torch.zeros(2, 2, 2, 256, 256)
    selected = (((3, 11), (14, 6)), ((0, 10), (8, 1)))
    for b in range(2):
        for camera in range(2):
            active_cell(obs, b, camera, selected[b][camera][0], polarity=0, rows=48)
            active_cell(obs, b, camera, selected[b][camera][1], polarity=1, rows=32)
    layout = model.build_token_layout(obs)
    assert (layout.grid_height, layout.grid_width) == (4, 4)
    assert layout.fine_indices.shape == (2, 2, 2)
    assert layout.coarse_indices.shape == (2, 2, 14)
    for b in range(2):
        for camera in range(2):
            assert set(layout.fine_indices[b, camera].tolist()) == set(selected[b][camera])
            all_cells = torch.cat((layout.fine_indices[b, camera], layout.coarse_indices[b, camera]))
            torch.testing.assert_close(all_cells.sort().values, torch.arange(16))
    repeat = model.build_token_layout(obs)
    torch.testing.assert_close(repeat.fine_indices, layout.fine_indices)
    torch.testing.assert_close(repeat.coarse_indices, layout.coarse_indices)


def test_router_excludes_padding_and_preserves_budget_when_valid_cells_are_few():
    model = sparse_model(refined_cells=1)
    obs = torch.zeros(1, 1, 2, 256, 256)
    active_cell(obs, 0, 0, 0, rows=64)
    active_cell(obs, 0, 0, 15, polarity=1, rows=8)
    valid = torch.ones(1, 1, 256, 256)
    valid[:, :, :64, :64] = 0
    assert model.build_token_layout(obs, valid).fine_indices.item() == 15
    model = sparse_model(refined_cells=8)
    valid.zero_()
    valid[:, :, 192:, 192:] = 1
    layout = model.build_token_layout(obs, valid)
    assert layout.fine_indices.shape == (1, 1, 8)
    assert 15 in layout.fine_indices[0, 0].tolist()
    assert torch.cat((layout.fine_indices, layout.coarse_indices), -1).unique().numel() == 16


@pytest.mark.parametrize("cameras", [1, 2])
@pytest.mark.parametrize("refined_cells,future_tokens", [(0, 16), (8, 40), (16, 64)])
def test_token_budget_matches_dense_observation_and_future_quota(cameras, refined_cells, future_tokens):
    model = sparse_model(refined_cells, cameras)
    counts = model.token_counts(horizon=24, latent_height=32, latent_width=32)
    assert counts["observed_per_camera"] == 64
    assert counts["future_per_camera_per_slice"] == future_tokens
    assert counts["event"] == cameras * (64 + 6 * future_tokens)
    assert counts["action"] == 25
    assert counts["self_attention"] == counts["event"] + 25


class MarkerVelocityHead(nn.Module):
    """Token markers generate identifiable per-channel, per-pixel velocities."""

    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def forward(self, tokens):
        size = self.patch_size
        y = torch.arange(size, device=tokens.device).to(tokens)
        x = torch.arange(size, device=tokens.device).to(tokens)
        if size == 4:
            child = tokens[..., 1]
            y = y + (child // 2)[..., None] * 4
            x = x + (child % 2)[..., None] * 4
        channels = torch.arange(16, device=tokens.device).to(tokens) * 64
        spatial = y[..., :, None] * 8 + x[..., None, :]
        values = (tokens[..., 0, None, None, None] * 10000
                  + tokens[..., 2, None, None, None] * 100000
                  + channels[:, None, None] + spatial[..., None, :, :])
        return values.flatten(-3)


def test_decode_scatter_restores_original_cell_and_fine_subpatch_locations():
    model = sparse_model(cameras=2)
    obs = torch.zeros(2, 2, 2, 256, 256)
    for b in range(2):
        for camera in range(2):
            for cell in range(16):
                active_cell(obs, b, camera, cell, rows=1 + (cell + 3 * b + 5 * camera) % 16)
    layout = model.build_token_layout(obs)
    model.event_head = MarkerVelocityHead(4)
    model.coarse_event_head = MarkerVelocityHead(8)
    tokens = torch.zeros(2, 2, 2, 40, 16)
    tokens[..., :8, 0] = layout.coarse_indices[:, None]
    tokens[..., 8:, 0] = layout.fine_indices[:, None].repeat_interleave(4, -1)
    tokens[..., 8:, 1] = torch.arange(4).repeat(8)
    tags = torch.arange(8).reshape(2, 2, 2)
    tokens[..., 2] = tags[..., None]
    actual = model.decode_future_tokens(tokens, layout)
    expected = torch.empty(2, 2, 2, 16, 32, 32)
    local = (torch.arange(16)[:, None, None] * 64
             + torch.arange(8)[None, :, None] * 8 + torch.arange(8)[None, None, :])
    for cell in range(16):
        y, x = divmod(cell, 4)
        expected[..., y * 8:(y + 1) * 8, x * 8:(x + 1) * 8] = (
            tags[..., None, None, None] * 100000 + cell * 10000 + local)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_batch_rope_uses_each_layout_and_retains_temporal_and_subpatch_coordinates():
    model = sparse_model(refined_cells=1)
    obs = torch.zeros(2, 1, 2, 256, 256)
    active_cell(obs, 0, 0, 0)
    active_cell(obs, 1, 0, 15)
    layout = model.build_token_layout(obs)
    coordinates = model.event_token_coordinates(layout, future_slices=2)
    assert coordinates.shape == (2, 64 + 2 * 19, 3)
    torch.testing.assert_close(coordinates[0, :64], coordinates[1, :64])
    future = coordinates[:, 64:].reshape(2, 2, 19, 3)
    torch.testing.assert_close(future[:, 0, :, 0], torch.full((2, 19), 2.5))
    torch.testing.assert_close(future[:, 1, :, 0], torch.full((2, 19), 6.5))
    torch.testing.assert_close(future[0, 0, -4:, 1:], torch.tensor([[0., 0.], [0., 1.], [1., 0.], [1., 1.]]))
    torch.testing.assert_close(future[1, 0, -4:, 1:], torch.tensor([[6., 6.], [6., 7.], [7., 6.], [7., 7.]]))
    for b in range(2):
        ids = layout.coarse_indices[b, 0]
        expected = torch.stack((ids // 4 * 2 + .5, ids % 4 * 2 + .5), -1)
        torch.testing.assert_close(future[b, 0, :15, 1:], expected)


def test_all_fine_matches_dense_model_velocity_and_input_gradients():
    torch.manual_seed(91)
    dense = DenseEWMModel(**model_options()).eval()
    sparse = sparse_model(refined_cells=16).eval()
    sparse.initialize_from_dense_policy(dense.policy_state_dict())
    layout = sparse.build_token_layout(torch.rand(2, 1, 2, 256, 256))
    obs = torch.randn(2, 1, 16, 32, 32, requires_grad=True)
    future = torch.randn(2, 2, 1, 16, 32, 32, requires_grad=True)
    action = torch.randn(2, 8, 7, requires_grad=True)
    common = (obs, torch.randn(2, 8), future, action, torch.tensor([.25, .75]), explicit_context(2))
    expected = dense.predict_velocity(*common)
    actual = sparse.predict_velocity(*common, layout=layout)
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=2e-5, atol=2e-6)
    actual_grad = torch.autograd.grad(sum(x.square().mean() for x in actual), (obs, future, action), retain_graph=True)
    expected_grad = torch.autograd.grad(sum(x.square().mean() for x in expected), (obs, future, action))
    for result, reference in zip(actual_grad, expected_grad):
        torch.testing.assert_close(result, reference, rtol=3e-4, atol=2e-7)


@pytest.mark.parametrize("refined_cells", [0, 8, 16])
def test_zero_velocity_heads_have_no_noisy_latent_skip(refined_cells):
    model = sparse_model(refined_cells).eval()
    for head in (model.event_head, model.coarse_event_head):
        nn.init.zeros_(head[-1].weight)
        nn.init.zeros_(head[-1].bias)
    layout = model.build_token_layout(torch.zeros(1, 1, 2, 256, 256))
    for noise_scale in (1., 7.):
        velocity, _ = model.predict_velocity(
            torch.randn(1, 1, 16, 32, 32), torch.randn(1, 8),
            torch.randn(1, 1, 1, 16, 32, 32) * noise_scale,
            torch.randn(1, 4, 7), torch.ones(1), explicit_context(), layout=layout)
        assert torch.count_nonzero(velocity) == 0


@pytest.mark.parametrize("training_mode", ["joint", "event_only"])
def test_forward_backprop_trains_both_spatial_scales_and_keeps_codec_frozen(training_mode):
    torch.manual_seed(92)
    model = sparse_model(training_mode=training_mode)
    batch = small_batch()
    if training_mode == "event_only":
        batch.pop("proprio")
        batch.pop("action")
    losses = model(batch, t=torch.tensor([.5]))
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    for parameter in (model.event_input.weight, model.coarse_event_input.weight,
                      model.event_head[-1].weight, model.coarse_event_head[-1].weight,
                      model.blocks[0].event.qkv.weight):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    assert all(parameter.grad is None and not parameter.requires_grad for parameter in model.event_codec.parameters())
    assert not model.event_codec.training
    if training_mode == "joint":
        assert model.action_head[-1].weight.grad.abs().sum() > 0
    else:
        assert all(parameter.grad is None for parameter in model.action_head.parameters())


def test_sampling_builds_layout_once_and_reuses_it_through_every_denoise_step(monkeypatch):
    model = sparse_model()
    batch = small_batch()
    layouts, seen = [], []
    build = model.build_token_layout

    def counted(obs_voxel, valid_region_mask=None):
        layout = build(obs_voxel, valid_region_mask)
        layouts.append(layout)
        return layout

    def velocity(obs, state, event, action, t, context, context_kv=None, *, layout=None):
        chosen = context.layout if layout is None else layout
        seen.append(chosen)
        return torch.zeros_like(event), torch.zeros_like(action)

    monkeypatch.setattr(model, "build_token_layout", counted)
    monkeypatch.setattr(model, "predict_velocity", velocity)
    normalizer = SimpleNamespace(denormalize_action=lambda x: x)
    result = model.sample_policy(batch, horizon=4, normalizer=normalizer, num_steps=3, decode_future=False)
    assert result["future_latent"].shape == (1, 1, 1, 16, 32, 32)
    assert len(layouts) == 1 and len(seen) == 3
    assert all(layout is layouts[0] for layout in seen)
    model.sample_policy(batch, horizon=4, normalizer=normalizer, num_steps=2, decode_future=False)
    assert len(layouts) == 2 and len(seen) == 5
    assert all(layout is layouts[1] for layout in seen[3:])


def test_compiled_core_accepts_changed_layout_values_with_fixed_shapes():
    torch.manual_seed(93)
    model = sparse_model(cameras=2).eval()
    obs_voxel = torch.zeros(2, 2, 2, 256, 256)
    for b in range(2):
        for camera in range(2):
            for cell in range(8):
                active_cell(obs_voxel, b, camera, (cell + 5 * b + 3 * camera) % 16)
    layouts = [model.build_token_layout(obs_voxel), model.build_token_layout(obs_voxel.flip(-1))]
    args = (torch.randn(2, 2, 16, 32, 32), torch.randn(2, 8),
            torch.randn(2, 1, 2, 16, 32, 32), torch.randn(2, 4, 7),
            torch.tensor([.25, .75]), explicit_context(2))
    compiled = torch.compile(model.predict_velocity, backend="eager", fullgraph=True)
    for layout in layouts:
        reference = model.predict_velocity(*args, layout=layout)
        actual = compiled(*args, layout=layout)
        for result, expected in zip(actual, reference):
            torch.testing.assert_close(result, expected)
        # This is the production forward/sample path: the layout is carried
        # inside the context and per-layer K/V were projected outside the core.
        wrapped = SparseTokenContext(args[-1], layout)
        context_kv = model.prepare_context_kv(wrapped)
        reference = model.predict_velocity(*args[:-1], wrapped, context_kv=context_kv)
        actual = compiled(*args[:-1], wrapped, context_kv=context_kv)
        for result, expected in zip(actual, reference):
            torch.testing.assert_close(result, expected)
    sum(value.square().mean() for value in actual).backward()
    assert model.coarse_event_input.weight.grad is not None
    assert model.event_head[-1].weight.grad is not None



def test_policy_loading_rejects_layout_changes_and_incomplete_dense_initialization():
    model = sparse_model(refined_cells=8)
    state = model.policy_state_dict()
    assert "token_layout_spec" in state
    assert not any(key.startswith("event_codec.") for key in state)
    model.load_policy_state_dict(state)
    with pytest.raises(RuntimeError, match="incompatible token layout"):
        sparse_model(refined_cells=4).load_policy_state_dict(state)
    with pytest.raises(RuntimeError, match="token layout"):
        model.load_policy_state_dict(DenseEWMModel(**model_options()).policy_state_dict())
    broken_sparse = dict(state)
    broken_sparse.pop("event_head.1.weight")
    with pytest.raises(RuntimeError, match="missing"):
        model.load_policy_state_dict(broken_sparse)

    dense = DenseEWMModel(**model_options()).policy_state_dict()
    coarse_before = {key: value.clone() for key, value in state.items()
                     if key.startswith(("coarse_event_input.", "coarse_event_head."))}
    model.initialize_from_dense_policy(dense)
    for key, value in coarse_before.items():
        torch.testing.assert_close(model.policy_state_dict()[key], value)
    broken_dense = dict(dense)
    broken_dense.pop("event_input.weight")
    with pytest.raises(RuntimeError, match="Invalid dense EWM initialization"):
        model.initialize_from_dense_policy(broken_dense)
    unexpected_dense = dict(dense, token_layout_spec=model.token_layout_spec)
    with pytest.raises(RuntimeError, match="Invalid dense EWM initialization"):
        model.initialize_from_dense_policy(unexpected_dense)


def test_sparse_policy_eval_loader(monkeypatch, tmp_path):
    from scripts.eval_ewm import eval_ewm_policy as module
    original = sparse_model()
    payload = dict(format='ewm_wan_latent_v2', model=original.policy_state_dict(),
                   model_config={'type': 'ewm_sparse', 'refined_cells': 8, 'cell_size': 8}, horizon=24)
    monkeypatch.setattr(module.torch, 'load', lambda *args, **kwargs: payload)
    monkeypatch.setattr(SparseTokenEWMModel, 'from_config', lambda cfg: sparse_model())
    monkeypatch.setattr(module, 'EWMNormalizer', lambda path: SimpleNamespace())
    restored, _, horizon = module.load_policy(tmp_path / 'checkpoint.pt')
    assert isinstance(restored, SparseTokenEWMModel)
    assert not restored.training and horizon == 24
    for key, value in original.policy_state_dict().items():
        torch.testing.assert_close(restored.policy_state_dict()[key], value)
