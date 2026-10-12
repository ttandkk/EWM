"""Fixed-budget, event-adaptive spatial tokenization for EWM.

The frozen codec, dense EWMJointBlock modules, flow loss and sampler are
inherited from ewm.py. Only future-event tokenization and recovery change:
each fixed cell contributes either one coarse token or four patch tokens.
Recovery predicts dense latent velocity directly, without a noisy-latent skip.

Defaults (Lean patch4, 256px, horizon24): 16 cells per camera, 8 refined
cells, 40 tokens per future latent slice, and 64 observation tokens.
This gives 304/608 event tokens and 329/633 joint tokens for wrist/dual.
RGB/text remain separate cross-attention context. The observation-derived
layout is shared across future times and reused throughout a sampling call.

Use SparseTokenEWMModel.from_config(cfg.model) or import the EWMModel alias
from this module. The existing trainer accepts train(cfg, model=model);
production ewm.py import/dispatch is not replaced. New coarse weights need
training. initialize_from_dense_policy initializes shared weights explicitly;
sparse policies load strictly, including the token-layout configuration.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from openwam.model.action_backbone.components import sinusoidal_embedding_1d
from openwam.model.architectures.ewm import EWMModel as DenseEWMModel
from openwam.model.architectures.ewm import event_action_coordinates, multimodal_rope_frequencies


class SpatialTokenLayout(NamedTuple):
    """Indices are [B,camera,count], shared across all future/flow times."""

    fine_indices: Tensor
    coarse_indices: Tensor
    grid_height: int
    grid_width: int


class SparseTokenContext(NamedTuple):
    """Explicit per-request layout; no mutable model-side routing cache."""

    context: tuple
    layout: SpatialTokenLayout


def split_spatial_cells(latent: Tensor, cell_size: int) -> Tensor:
    """[B,T,V,C,H,W] -> [B,T,V,M,C,S,S], cells in raster order."""
    if latent.ndim != 6 or cell_size <= 0:
        raise ValueError("latent must be [B,T,V,C,H,W] and cell_size positive")
    b, t, v, c, h, w = latent.shape
    if h % cell_size or w % cell_size:
        raise ValueError("latent spatial dimensions must be divisible by cell_size")
    gh, gw = h // cell_size, w // cell_size
    return latent.reshape(b, t, v, c, gh, cell_size, gw, cell_size).permute(
        0, 1, 2, 4, 6, 3, 5, 7
    ).reshape(b, t, v, gh * gw, c, cell_size, cell_size)


def merge_spatial_cells(cells: Tensor, grid_height: int, grid_width: int) -> Tensor:
    """Inverse of split_spatial_cells, preserving all camera/channel/pixel axes."""
    if cells.ndim != 7 or min(grid_height, grid_width) <= 0:
        raise ValueError("cells must be [B,T,V,M,C,S,S] with a positive grid")
    b, t, v, m, c, sh, sw = cells.shape
    if m != grid_height * grid_width or sh != sw:
        raise ValueError("cell count/shape does not match the spatial grid")
    return cells.reshape(b, t, v, grid_height, grid_width, c, sh, sw).permute(
        0, 1, 2, 5, 3, 6, 4, 7
    ).reshape(b, t, v, c, grid_height * sh, grid_width * sw)


def _gather_cells(cells: Tensor, indices: Tensor) -> Tensor:
    b, t, v, _, c, sh, sw = cells.shape
    index = indices[:, None, :, :, None, None, None].expand(b, t, v, indices.shape[-1], c, sh, sw)
    return cells.gather(3, index)


def _batched_rope_frequencies(coordinates: Tensor, head_dim: int, dtype: torch.dtype):
    # EWM already accepts precomputed RoPE tables. Batched tables broadcast
    # over heads, leaving the original RoPE and DiT blocks unchanged.
    b, n, _ = coordinates.shape
    tables = multimodal_rope_frequencies(coordinates.reshape(b * n, 3), head_dim, dtype=dtype)
    return tuple((cos.reshape(b, 1, n, -1), sin.reshape(b, 1, n, -1)) for cos, sin in tables)


class SparseTokenEWMModel(DenseEWMModel):
    """EWM with observation-guided coarse/fine future tokens and dense outputs."""

    _SPARSE_POLICY_PREFIXES = ("coarse_event_input.", "coarse_event_head.")
    _SPARSE_POLICY_KEYS = ("scale_embedding", "token_layout_spec")

    def __init__(self, *, refined_cells=8, cell_size=None, **kwargs):
        if kwargs.get("training_mode", "joint") not in ("joint", "event_only"):
            raise ValueError("SparseTokenEWMModel supports joint and event_only event training")
        if isinstance(refined_cells, bool) or not isinstance(refined_cells, int) or refined_cells < 0:
            raise ValueError("refined_cells must be a nonnegative integer")
        super().__init__(**kwargs)
        size = 2 * self.latent_patch_size if cell_size is None else cell_size
        if isinstance(size, bool) or not isinstance(size, int) or size != 2 * self.latent_patch_size:
            raise ValueError("cell_size must be twice event_patch_size for the 1/4 token quotas")
        self.cell_size = size
        self.refined_cells = refined_cells
        dim = self.event_input.out_channels
        self.coarse_event_input = nn.Conv2d(self.latent_channels, dim, kernel_size=size, stride=size)
        self.coarse_event_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, self.latent_channels * size**2))
        # Zero fine embedding makes all-fine a permutation of the original
        # patch model after loading the shared dense weights.
        self.scale_embedding = nn.Parameter(torch.zeros(2, dim))  # fine, coarse
        self.register_buffer("token_layout_spec", torch.tensor([1, size, refined_cells], dtype=torch.int64))

    def token_counts(self, horizon=24, latent_height=32, latent_width=32):
        if horizon <= 0 or horizon % self.event_temporal_compression:
            raise ValueError("horizon must be positive and divisible by codec temporal compression")
        if min(latent_height, latent_width) <= 0 or latent_height % self.cell_size or latent_width % self.cell_size:
            raise ValueError("latent dimensions must be positive multiples of cell_size")
        cells = (latent_height // self.cell_size) * (latent_width // self.cell_size)
        if self.refined_cells > cells:
            raise ValueError("refined_cells exceeds the number of spatial cells")
        observed = (latent_height // self.latent_patch_size) * (latent_width // self.latent_patch_size)
        future = cells + 3 * self.refined_cells
        event = self.num_event_cameras * (observed + horizon // self.event_temporal_compression * future)
        action = 0 if self.training_mode == "event_only" else horizon + 1
        return dict(event=event, action=action, self_attention=event + action,
                    observed_per_camera=observed, future_per_camera_per_slice=future)

    @torch.no_grad()
    def build_token_layout(self, obs_voxel: Tensor, valid_region_mask: Tensor | None = None) -> SpatialTokenLayout:
        """Rank observed OFF/ON occupancy, never future/noisy latents.

        Padding is excluded from density estimates. Fully invalid cells rank
        last; if fewer than R cells are valid, padding still fills the fixed
        quota. Empty-score ties prefer a deterministic checkerboard.
        """
        if obs_voxel.ndim != 5 or obs_voxel.shape[1:3] != (self.num_event_cameras, 2):
            raise ValueError("obs_voxel must be [B,event_cameras,2,H,W]")
        b, v, _, h, w = obs_voxel.shape
        pixel_cell = self.cell_size * self.event_codec.spatial_compression
        if min(b, h, w) <= 0 or h % pixel_cell or w % pixel_cell:
            raise ValueError("observation dimensions must be divisible by codec stride times cell_size")
        gh, gw = h // pixel_cell, w // pixel_cell
        if self.refined_cells > gh * gw:
            raise ValueError("refined_cells exceeds the number of spatial cells")
        occupancy = (obs_voxel > 0).float().mean(2)
        if valid_region_mask is None:
            scores = F.avg_pool2d(occupancy, pixel_cell, pixel_cell)
        else:
            if valid_region_mask.shape != (b, 1, h, w):
                raise ValueError("valid_region_mask must be [B,1,H,W]")
            valid = valid_region_mask.to(device=obs_voxel.device, dtype=torch.float32).clamp(0, 1)
            numerator = F.avg_pool2d(occupancy * valid, pixel_cell, pixel_cell)
            denominator = F.avg_pool2d(valid, pixel_cell, pixel_cell)
            scores = torch.where(denominator > 0, numerator / denominator.clamp_min(1e-8), -1.0)
        scores = scores.flatten(2)
        cell_ids = torch.arange(gh * gw, device=obs_voxel.device)
        tie_order = torch.argsort((cell_ids // gw + cell_ids % gw) % 2, stable=True)
        ranks = torch.argsort(scores.index_select(-1, tie_order), descending=True, stable=True)
        ordered = tie_order[ranks]
        fine = ordered[..., :self.refined_cells].sort(dim=-1).values
        coarse = ordered[..., self.refined_cells:].sort(dim=-1).values
        return SpatialTokenLayout(fine, coarse, gh, gw)

    def _context(self, batch, context):
        if isinstance(context, SparseTokenContext):
            return context
        raw_context = super()._context(batch, context)
        layout = self.build_token_layout(batch["obs_voxel"], batch.get("valid_region_mask"))
        return SparseTokenContext(raw_context, layout)

    def prepare_context_kv(self, context):
        raw_context = context.context if isinstance(context, SparseTokenContext) else context
        return super().prepare_context_kv(raw_context)

    def _check_layout(self, layout, batch, cameras, height, width):
        gh, gw = layout.grid_height, layout.grid_width
        if (height, width) != (gh * self.cell_size, gw * self.cell_size):
            raise ValueError("layout grid does not match the latent spatial dimensions")
        if cameras != self.num_event_cameras or not 0 <= self.refined_cells <= gh * gw:
            raise ValueError("invalid camera count or refinement budget")
        if (layout.fine_indices.shape != (batch, cameras, self.refined_cells) or
                layout.coarse_indices.shape != (batch, cameras, gh * gw - self.refined_cells)):
            raise ValueError("layout indices do not match batch, cameras or refinement budget")
        if layout.fine_indices.dtype != torch.long or layout.coarse_indices.dtype != torch.long:
            raise ValueError("layout indices must have dtype torch.long")

    def encode_future_tokens(self, event: Tensor, layout: SpatialTokenLayout) -> Tensor:
        """Pack coarse tokens then cell-major 2x2 fine tokens: [B,T,V,S,D]."""
        if event.ndim != 6 or event.shape[3] != self.latent_channels:
            raise ValueError("future latent must be [B,T,V,latent_channels,H,W]")
        b, t, v, c, h, w = event.shape
        self._check_layout(layout, b, v, h, w)
        cells = split_spatial_cells(event, self.cell_size)
        nc, nf = layout.coarse_indices.shape[-1], layout.fine_indices.shape[-1]
        dim = self.event_input.out_channels
        coarse = _gather_cells(cells, layout.coarse_indices).reshape(b * t * v * nc, c, self.cell_size, self.cell_size)
        fine = _gather_cells(cells, layout.fine_indices).reshape(b * t * v * nf, c, self.cell_size, self.cell_size)
        coarse = self.coarse_event_input(coarse).reshape(b, t, v, nc, dim)
        fine = self.event_input(fine).flatten(2).transpose(1, 2).reshape(b, t, v, nf * 4, dim)
        coarse = coarse + self.scale_embedding[1].to(coarse)
        fine = fine + self.scale_embedding[0].to(fine)
        tokens = torch.cat((coarse, fine), dim=3)
        return tokens + self.camera.to(tokens)[None, None, :, None]

    def decode_future_tokens(self, tokens: Tensor, layout: SpatialTokenLayout) -> Tensor:
        """Predict every dense latent velocity value from DiT tokens, no skip."""
        if tokens.ndim != 5 or tokens.shape[-1] != self.event_input.out_channels:
            raise ValueError("future tokens must be [B,T,V,S,D]")
        b, t, v, n, _ = tokens.shape
        gh, gw, s, p = layout.grid_height, layout.grid_width, self.cell_size, self.latent_patch_size
        self._check_layout(layout, b, v, gh * s, gw * s)
        nc, nf = layout.coarse_indices.shape[-1], layout.fine_indices.shape[-1]
        if n != nc + 4 * nf:
            raise ValueError("future token count does not match the fixed layout budget")
        c = self.latent_channels
        coarse = self.coarse_event_head(tokens[..., :nc, :]).reshape(b, t, v, nc, c, s, s)
        fine = self.event_head(tokens[..., nc:, :]).reshape(b, t, v, nf, 2, 2, c, p, p)
        fine = fine.permute(0, 1, 2, 3, 6, 4, 7, 5, 8).reshape(b, t, v, nf, c, s, s)
        values = torch.cat((coarse, fine), dim=3)
        indices = torch.cat((layout.coarse_indices, layout.fine_indices), dim=-1)
        indices = indices[:, None, :, :, None, None, None].expand(b, t, v, gh * gw, c, s, s)
        cells = torch.zeros_like(values).scatter(3, indices, values)
        return merge_spatial_cells(cells, gh, gw)

    def event_token_coordinates(self, layout: SpatialTokenLayout, future_slices: int) -> Tensor:
        """Batched [time,y,x] coordinates in the original patch-grid units."""
        if future_slices <= 0:
            raise ValueError("future_slices must be positive")
        b, v, _ = layout.fine_indices.shape
        gh, gw = layout.grid_height, layout.grid_width
        device = layout.fine_indices.device
        y, x = torch.meshgrid(torch.arange(gh * 2, device=device),
                              torch.arange(gw * 2, device=device), indexing="ij")
        obs = torch.stack((torch.zeros_like(y), y, x), dim=-1).float()
        obs = obs.reshape(1, 1, gh * gw * 4, 3).expand(b, v, gh * gw * 4, 3).flatten(1, 2)
        coarse_ids = layout.coarse_indices
        coarse_yx = torch.stack((2 * (coarse_ids // gw) + .5, 2 * (coarse_ids % gw) + .5), dim=-1)
        fine_ids = layout.fine_indices
        fine_yx = torch.stack((2 * (fine_ids // gw), 2 * (fine_ids % gw)), dim=-1)
        offsets = fine_yx.new_tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
        fine_yx = (fine_yx[..., None, :] + offsets).reshape(b, v, 4 * self.refined_cells, 2).float()
        yx = torch.cat((coarse_yx, fine_yx), dim=2)
        n = yx.shape[2]
        times = torch.arange(1, future_slices + 1, device=device, dtype=torch.float32)
        if self.event_temporal_compression == 4:
            times = times * 4 - 1.5
        times = times[None, :, None, None, None].expand(b, future_slices, v, n, 1)
        spatial = yx[:, None].expand(b, future_slices, v, n, 2)
        future = torch.cat((times, spatial), dim=-1).reshape(b, future_slices * v * n, 3)
        return torch.cat((obs, future), dim=1)

    def predict_velocity(self, obs_latent, proprio, event, action, t, context, context_kv=None, *, layout=None):
        if isinstance(context, SparseTokenContext):
            if layout is not None:
                raise ValueError("provide layout either in SparseTokenContext or as a keyword")
            context, layout = context
        if layout is None:
            raise ValueError("predict_velocity requires an observation-derived layout; use build_token_layout or forward/sample_policy")
        if event.ndim != 6 or event.shape[2:4] != (self.num_event_cameras, self.latent_channels):
            raise ValueError("future latent must be [B,T,event_cameras,latent_channels,H,W]")
        b, future_slices, v, c, h, w = event.shape
        if future_slices <= 0 or obs_latent.shape != (b, v, c, h, w) or t.shape != (b,):
            raise ValueError("invalid observation latent, future horizon or flow time shape")
        event_only = self.training_mode == "event_only"
        if not event_only and (action is None or proprio is None or proprio.shape != (b, 8) or
                               action.shape != (b, future_slices * self.event_temporal_compression, 7)):
            raise ValueError("invalid action or proprio shape")
        future_tokens = self.encode_future_tokens(event, layout)
        dim, per_future = self.event_input.out_channels, future_tokens.shape[3]
        observed = self.event_input(obs_latent.reshape(b * v, c, h, w)).flatten(2).transpose(1, 2)
        per_observed = observed.shape[1]
        observed = observed.reshape(b, v, per_observed, dim)
        observed = observed + self.camera.to(observed)[None, :, None] + self.scale_embedding[0].to(observed)
        e = torch.cat((observed.flatten(1, 2), future_tokens.flatten(1, 3)), dim=1)
        ec = self.event_token_coordinates(layout, future_slices)
        a = None if event_only else torch.cat((self.state_mlp(proprio)[:, None], self.action_mlp(action)), dim=1)
        ac = (None if event_only else event_action_coordinates(action.shape[1], 1, 1, num_cameras=1, device=action.device)[1])
        if len(context) == 3:
            ctx, mask, visual_tokens = context
        elif len(context) == 2:
            ctx, mask = context
            visual_tokens = 0
        else:
            raise ValueError("context must be (tokens, mask) or (tokens, mask, visual_token_count)")
        head_dim = dim // self.blocks[0].heads
        event_rope = _batched_rope_frequencies(ec, head_dim, e.dtype)
        action_rope = None if event_only else multimodal_rope_frequencies(ac, head_dim, dtype=a.dtype)
        event_time = sinusoidal_embedding_1d(self.blocks[0].event.time.freq_dim, t)
        action_time = (event_time if self.blocks[0].action.time.freq_dim == self.blocks[0].event.time.freq_dim
                       else sinusoidal_embedding_1d(self.blocks[0].action.time.freq_dim, t))
        if context_kv is not None and len(context_kv) != len(self.blocks):
            raise ValueError("context K/V cache must contain one entry per EWM block")
        for index, block in enumerate(self.blocks):
            if visual_tokens and index % self.dino_injection_interval:
                block_ctx, block_mask = ctx[:, visual_tokens:], mask[:, visual_tokens:]
            else:
                block_ctx, block_mask = ctx, mask
            args = (e, a, t, block_ctx, block_mask, ec, ac, event_rope, action_rope, (event_time, action_time))
            options = dict(event_only=event_only)
            if context_kv is not None:
                event_kv, action_kv = context_kv[index]
                options.update(event_context_kv=event_kv, action_context_kv=action_kv)
            if context_kv is None and index >= self.gradient_checkpointing_skip_first_blocks:
                e, a = self._checkpoint(block, *args, **options)
            else:
                e, a = block(*args, **options)
        future = e[:, v * per_observed:].reshape(b, future_slices, v, per_future, dim)
        velocity = self.decode_future_tokens(future, layout)
        return velocity, None if event_only else self.action_head(a[:, 1:])

    def load_policy_state_dict(self, state):
        spec = state.get("token_layout_spec")
        if spec is None or not torch.equal(spec.detach().cpu(), self.token_layout_spec.detach().cpu()):
            raise RuntimeError("Sparse EWM policy has a missing or incompatible token layout; use initialize_from_dense_policy for dense weights")
        return super().load_policy_state_dict(state)

    def initialize_from_dense_policy(self, state):
        """Load exactly the shared dense-policy keys; coarse weights stay new."""
        shared = {key for key in self.policy_state_dict()
                  if key not in self._SPARSE_POLICY_KEYS and not key.startswith(self._SPARSE_POLICY_PREFIXES)}
        missing, unexpected = shared - set(state), set(state) - shared
        if missing or unexpected:
            raise RuntimeError(f"Invalid dense EWM initialization: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
        return self.load_state_dict(state, strict=False)


# Explicit drop-in import without altering the existing production module.
EWMModel = SparseTokenEWMModel

__all__ = ["SparseTokenEWMModel", "EWMModel", "SpatialTokenLayout", "SparseTokenContext",
           "split_spatial_cells", "merge_spatial_cells"]
