"""Event/action rectified flow with optional single-expert ablations."""

from __future__ import annotations

from pathlib import Path

import torch
import time
import torch.nn.functional as F
from openwam.model.architectures.wan_event_codec import WanEventCodec
from openwam.model.architectures.lean_event_codec import LeanEventCodec
from openwam.model.architectures.lean_rgb_codec import LeanRGBCodec
from torch import Tensor, nn

from openwam.model.action_backbone.components import RMSNorm, TimestepEmbedding, TimestepModulation, sinusoidal_embedding_1d
from openwam.model.action_backbone.separate_action_dit import BridgeCrossAttention


def canonical_voxel(voxel: Tensor, *, voxel_bins: int = 4, num_cameras: int = 2) -> Tensor:
    """Restore p-major polarity/bin axes without mixing intervals or cameras."""
    if voxel_bins not in (1, 4) or num_cameras not in (1, 2):
        raise ValueError("voxel_bins must be 1/4 and num_cameras must be 1/2")
    if voxel.ndim not in (5, 6) or voxel.shape[-4:-2] != (num_cameras, 2 * voxel_bins):
        raise ValueError(f"voxel must be [B,(K),camera={num_cameras},{2 * voxel_bins},H,W]")
    return voxel.unflatten(-3, (2, voxel_bins))


def event_action_coordinates(horizon: int, height: int, width: int, *, num_cameras: int = 2, device=None):
    if min(horizon, height, width) <= 0 or num_cameras not in (1, 2):
        raise ValueError("horizon/grid dimensions must be positive and num_cameras must be 1 or 2")
    t, cam, y, x = torch.meshgrid(
        torch.arange(horizon + 1, device=device),
        torch.arange(num_cameras, device=device),
        torch.arange(height, device=device),
        torch.arange(width, device=device),
        indexing="ij",
    )
    event = torch.stack((t, y, x), -1).reshape(-1, 3)
    action = torch.zeros(horizon + 1, 3, device=device, dtype=torch.long)
    action[:, 0] = torch.arange(horizon + 1, device=device)
    return event, action


def multimodal_rope_frequencies(coordinates: Tensor, head_dim: int, *, dtype: torch.dtype) -> tuple[tuple[Tensor, Tensor], ...]:
    """Build the three-axis RoPE cos/sin tables once per EWM forward."""
    if head_dim % 8:
        raise ValueError("RoPE head dimension must be divisible by 8")
    time_dim = 2 * (head_dim // 8)
    y_dim = 2 * ((head_dim - time_dim) // 4)
    dims = (time_dim, y_dim, head_dim - time_dim - y_dim)
    tables = []
    for axis, d in enumerate(dims):
        frequency = 10000.0 ** (-torch.arange(0, d, 2, device=coordinates.device).float() / d)
        phase = coordinates[:, axis].float()[:, None] * frequency
        tables.append((phase.cos().to(dtype), phase.sin().to(dtype)))
    return tuple(tables)


def multimodal_rope(x: Tensor, coordinates: Tensor | None = None, *, frequencies=None) -> Tensor:
    """Apply shared 16/24/24 RoPE, optionally using precomputed tables."""
    if frequencies is None and coordinates is None:
        raise ValueError("coordinates or precomputed frequencies are required")
    head_dim = x.shape[-1]
    if head_dim % 8:
        raise ValueError("RoPE head dimension must be divisible by 8")
    time_dim = 2 * (head_dim // 8)
    y_dim = 2 * ((head_dim - time_dim) // 4)
    dims = (time_dim, y_dim, head_dim - time_dim - y_dim)
    frequencies = multimodal_rope_frequencies(coordinates, head_dim, dtype=x.dtype) if frequencies is None else frequencies
    parts = []
    for part, (c, s) in zip(x.split(dims, dim=-1), frequencies):
        even, odd = part[..., 0::2], part[..., 1::2]
        parts.append(torch.stack((even * c - odd * s, even * s + odd * c), -1).flatten(-2))
    return torch.cat(parts, -1)


def full_mixed_self_attention(
    q_event,
    k_event,
    v_event,
    q_action,
    k_action,
    v_action,
    *,
    num_heads,
    event_coordinates=None,
    action_coordinates=None,
    event_rope=None,
    action_rope=None,
    memory_kv=None,
):
    tensors = (q_event, k_event, v_event, q_action, k_action, v_action)
    dim = q_event.shape[-1]
    if num_heads <= 0 or dim % num_heads:
        raise ValueError("hidden dimension must be divisible by num_heads")
    if any(x.ndim != 3 or x.shape[0] != q_event.shape[0] or x.shape[-1] != dim for x in tensors):
        raise ValueError("all streams must share batch and head geometry")
    qe, ke, ve, qa, ka, va = [x.unflatten(-1, (num_heads, dim // num_heads)).transpose(1, 2) for x in tensors]
    if (event_coordinates is None) != (action_coordinates is None):
        raise ValueError("both streams require coordinates")
    if event_coordinates is not None:
        qe, ke = [multimodal_rope(x, event_coordinates, frequencies=event_rope) for x in (qe, ke)]
        qa, ka = [multimodal_rope(x, action_coordinates, frequencies=action_rope) for x in (qa, ka)]
    key, value = torch.cat((ke, ka), 2), torch.cat((ve, va), 2)
    query = torch.cat((qe, qa), 2)
    attention_mask = None
    if memory_kv is not None:
        mk, mv, valid = memory_kv
        if mk.shape != mv.shape or mk.shape[:2] != key.shape[:2] or mk.shape[-1] != key.shape[-1]:
            raise ValueError("memory K/V must share batch and head geometry")
        if valid.shape != (key.shape[0], mk.shape[2]):
            raise ValueError("memory validity mask must match its token axes")
        attention_mask = torch.cat((torch.ones((key.shape[0], key.shape[2]),
                                               device=key.device, dtype=torch.bool), valid), -1)[:, None, None]
        key, value = torch.cat((key, mk.to(key)), 2), torch.cat((value, mv.to(value)), 2)
    output = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask).transpose(1, 2).flatten(2)
    return output.split((q_event.shape[1], q_action.shape[1]), dim=1)


class _EWMBranch(nn.Module):
    def __init__(self, dim, heads, attention_dim, context_dim):
        super().__init__()
        self.time = TimestepEmbedding(dim, dim)
        self.modulation = TimestepModulation(dim, 6)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.qkv = nn.Linear(dim, 3 * attention_dim)
        self.q_norm, self.k_norm = RMSNorm(attention_dim), RMSNorm(attention_dim)
        self.out = nn.Linear(attention_dim, dim)
        self.context_norm = nn.LayerNorm(dim)
        self.context = BridgeCrossAttention(dim, heads, attention_dim // heads, kv_hidden_dim=context_dim)
        self.ffn_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.ffn = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def prepare(self, x, t, time_sinusoid=None):
        # Learned timestep MLPs remain layer-specific; only their fixed input is shared.
        time_embed = self.time.mlp(time_sinusoid) if time_sinusoid is not None else self.time(t)
        mod = self.modulation(time_embed).unbind(1)
        shift, scale = [v[:, None] for v in mod[:2]]
        q, k, v = self.qkv(self.norm(x) * (1 + scale) + shift).chunk(3, -1)
        return (self.q_norm(q), self.k_norm(k), v), mod

    def finish(self, x, mixed, mod, context, mask, context_kv=None):
        _, _, gate, shift, scale, ffn_gate = [v[:, None] for v in mod]
        x = x + gate * self.out(mixed)
        normalized = self.context_norm(x)
        if context_kv is None:
            context_output = self.context(normalized, context, ctx_mask=mask[:, None, None, :])
        else:
            context_output = self.context.forward_with_kv(
                normalized, context_kv, ctx_mask=mask[:, None, None, :])
        x = x + context_output
        return x + ffn_gate * self.ffn(self.ffn_norm(x) * (1 + scale) + shift)


class EWMJointBlock(nn.Module):
    """Independent experts, full mixed self-attention, external context, FFN."""

    def __init__(self, dim=768, heads=8, action_hidden_dim=512):
        super().__init__()
        if heads <= 0 or dim % heads or (dim // heads) % 8:
            raise ValueError("shared head dimension must be divisible by 8")
        if action_hidden_dim <= 0:
            raise ValueError("action hidden dimension must be positive")
        self.heads = heads
        # OpenWAM MoT: residual widths may differ while Q/K/V share geometry.
        self.event = _EWMBranch(dim, heads, dim, dim)
        self.action = _EWMBranch(action_hidden_dim, heads, dim, dim)

    def forward(self, event, action, t, context, mask, event_coords, action_coords,
                event_rope=None, action_rope=None, time_sinusoid=None,
                event_context_kv=None, action_context_kv=None, event_only=False, action_only=False, memory_kv=None):
        if action_only and memory_kv is not None:
            raise ValueError("event memory is unsupported in action_only mode")
        if event_only and action_only:
            raise ValueError("event_only and action_only are mutually exclusive")
        event_time, action_time = (time_sinusoid if isinstance(time_sinusoid, tuple)
                                   else (time_sinusoid, time_sinusoid))
        if action_only:
            # No event queries, keys, values, context attention, or FFN.
            aqkv, amod = self.action.prepare(action, t, action_time)
            qa, ka, va = [x.unflatten(-1, (self.heads, x.shape[-1] // self.heads)).transpose(1, 2)
                          for x in aqkv]
            qa, ka = [multimodal_rope(x, action_coords, frequencies=action_rope) for x in (qa, ka)]
            ao = F.scaled_dot_product_attention(qa, ka, va).transpose(1, 2).flatten(2)
            return None, self.action.finish(action, ao, amod, context, mask, action_context_kv)
        eqkv, emod = self.event.prepare(event, t, event_time)
        if event_only:
            # Equivalent to masking every action/state key for event queries.
            # Skip the unused action queries and action expert as well.
            qe, ke, ve = [x.unflatten(-1, (self.heads, x.shape[-1] // self.heads)).transpose(1, 2)
                          for x in eqkv]
            qe, ke = [multimodal_rope(x, event_coords, frequencies=event_rope) for x in (qe, ke)]
            attention_mask = None
            if memory_kv is not None:
                mk, mv, valid = memory_kv
                attention_mask = torch.cat((torch.ones((ke.shape[0], ke.shape[2]),
                    device=ke.device, dtype=torch.bool), valid), -1)[:, None, None]
                ke, ve = torch.cat((ke, mk.to(ke)), 2), torch.cat((ve, mv.to(ve)), 2)
            eo = F.scaled_dot_product_attention(qe, ke, ve, attn_mask=attention_mask).transpose(1, 2).flatten(2)
            return self.event.finish(event, eo, emod, context, mask, event_context_kv), None
        aqkv, amod = self.action.prepare(action, t, action_time)
        eo, ao = full_mixed_self_attention(
            *eqkv, *aqkv, num_heads=self.heads, event_coordinates=event_coords, action_coordinates=action_coords, event_rope=event_rope, action_rope=action_rope, memory_kv=memory_kv
        )
        return (
            self.event.finish(event, eo, emod, context, mask, event_context_kv),
            self.action.finish(action, ao, amod, context, mask, action_context_kv),
        )


class EWMContextEncoder(nn.Module):
    """Frozen OpenWAM DINO/T5; only projections and modality/camera embeddings train."""

    def __init__(self, dino, text, dim=768):
        super().__init__()
        self.dino, self.text = dino, text
        for encoder in (dino, text):
            encoder.requires_grad_(False).eval()
        self.rgb_projection = nn.Linear(dino.properties.z_dim, dim)
        self.text_projection = nn.Linear(text.hidden_size, dim) if text.hidden_size != dim else nn.Identity()
        self.camera = nn.Parameter(torch.randn(2, dim) * 0.02)
        self.modality = nn.Parameter(torch.randn(2, dim) * 0.02)
        # The simulator uses one fixed instruction for an entire trial.  Keep a
        # single unprojected T5 result so policy replans do not rerun the frozen
        # text tower; learned projections still run on every call.
        self._cached_text_key = None
        self._cached_text = None

    def _clear_text_cache(self):
        self._cached_text_key = None
        self._cached_text = None

    def _apply(self, fn):
        # Cached tensors are intentionally not registered buffers.  Invalidate
        # them whenever the module changes device or dtype.
        self._clear_text_cache()
        return super()._apply(fn)

    def train(self, mode=True):
        super().train(mode)
        self.dino.eval()
        self.text.eval()
        if mode:
            self._clear_text_cache()
        return self

    @staticmethod
    def preprocess_rgb(rgb, *, device, dtype):
        """OpenWAM DINO normalization, batched on the encoder device, without host round trips."""
        from openwam.model.video_backbone.encoder.dinov3 import _IMAGENET_MEAN, _IMAGENET_STD

        pixels = rgb.flatten(0, 1).to(device=device, dtype=dtype, non_blocking=True) / 255.0
        mean = pixels.new_tensor(_IMAGENET_MEAN)[None, :, None, None]
        std = pixels.new_tensor(_IMAGENET_STD)[None, :, None, None]
        return ((pixels - mean) / std).unsqueeze(2)

    @torch.no_grad()
    def encode_rgb_features(self, rgb):
        """Frozen per-frame DINO grids reusable by context and memory branches."""
        if rgb.ndim != 5 or rgb.shape[1:3] != (2, 3) or rgb.dtype != torch.uint8:
            raise ValueError("RGB must be uint8 [B,2,3,H,W]")
        anchor = next(self.dino.parameters())
        pixels = self.preprocess_rgb(rgb, device=anchor.device, dtype=anchor.dtype)
        grid = self.dino.batch_encode(pixels)
        if grid.ndim != 5 or grid.shape[2] != 1:
            raise ValueError("DINO must independently encode one frame per camera")
        return grid[:, :, 0].reshape(rgb.shape[0], 2, grid.shape[1], *grid.shape[-2:])

    def encode(self, rgb, prompts, *, rgb_features=None):
        """Return context plus the visual-token prefix length for EWM scheduling."""
        if rgb.ndim != 5 or rgb.shape[1:3] != (2, 3) or rgb.dtype != torch.uint8:
            raise ValueError("RGB must be uint8 [B,2,3,H,W]")
        b = rgb.shape[0]
        if len(prompts) != b:
            raise ValueError("one prompt per batch sample is required")
        # Match DINO's normalization, keeping one independent frame per camera.
        with torch.no_grad():
            grid = self.encode_rgb_features(rgb) if rgb_features is None else rgb_features
            if grid.ndim != 5 or grid.shape[:3] != (b, 2, self.dino.properties.z_dim):
                raise ValueError("RGB features must be [B,2,DINO_D,Hpatch,Wpatch]")
            features = grid.flatten(0, 1).flatten(2).transpose(1, 2)
            text_key = tuple(prompts)
            if not self.training and self._cached_text_key == text_key:
                text, text_mask = self._cached_text
            else:
                text, text_mask = self.text(prompts)
                if not self.training:
                    self._cached_text_key = text_key
                    self._cached_text = (text.detach(), text_mask.detach())
        dtype = self.rgb_projection.weight.dtype
        device = self.rgb_projection.weight.device
        features = self.rgb_projection(features.to(device=device, dtype=dtype)).reshape(b, 2, -1, self.camera.shape[-1])
        # Preserve autocast's BF16 residual stream while retaining FP32
        # master weights for the learned embeddings.
        camera = self.camera.to(device=features.device, dtype=features.dtype)
        features = features + camera[None, :, None] + self.modality[0].to(features)
        text = self.text_projection(text.to(device=device, dtype=dtype))
        text = text + self.modality[1].to(text)
        features = features.flatten(1, 2)
        mask = torch.cat(
            (
                torch.ones(b, features.shape[1], dtype=torch.bool, device=device),
                text_mask.to(device=device, dtype=torch.bool),
            ),
            1,
        )
        return torch.cat((features, text), 1), mask, features.shape[1]

    def forward(self, rgb, prompts):
        """Backward-compatible context pair for direct encoder callers."""
        tokens, mask, _ = self.encode(rgb, prompts)
        return tokens, mask


def rectified_flow_inputs(event, action, *, t=None, event_noise=None, action_noise=None):
    """One continuous flow time per sample, shared exactly across modalities."""
    if event.shape[0] != action.shape[0] or action.shape[-1] != 7:
        raise ValueError("event/action batch must match; action dimension is 7")
    b = event.shape[0]
    t = torch.rand(b, device=event.device, dtype=event.dtype) if t is None else t.to(event)
    if t.shape != (b,) or not torch.isfinite(t).all() or ((t < 0) | (t > 1)).any():
        raise ValueError("flow time must be finite [B] in [0,1]")
    en = torch.randn_like(event) if event_noise is None else event_noise
    an = torch.randn_like(action) if action_noise is None else action_noise
    if en.shape != event.shape or an.shape != action.shape:
        raise ValueError("noise shapes must match targets")
    te, ta = t.reshape(b, *([1] * (event.ndim - 1))), t[:, None, None]
    return (1 - te) * event + te * en, (1 - ta) * action + ta * an, en - event, an - action, t


def rectified_event_flow_inputs(event, *, t=None, event_noise=None):
    """Event flow without an action target, noise, or state dependency."""
    b = event.shape[0]
    t = torch.rand(b, device=event.device, dtype=event.dtype) if t is None else t.to(event)
    if t.shape != (b,) or not torch.isfinite(t).all() or ((t < 0) | (t > 1)).any():
        raise ValueError("flow time must be finite [B] in [0,1]")
    noise = torch.randn_like(event) if event_noise is None else event_noise
    if noise.shape != event.shape:
        raise ValueError("event noise shape must match target")
    weight = t.reshape(b, *([1] * (event.ndim - 1)))
    return (1 - weight) * event + weight * noise, noise - event, t


def rectified_action_flow_inputs(action, *, t=None, action_noise=None):
    """Action flow without any event target, latent encoding, or event noise."""
    if action.ndim != 3 or action.shape[1] <= 0 or action.shape[-1] != 7:
        raise ValueError("action must be [B,K,7] with a positive horizon")
    if action_noise is not None and action_noise.shape != action.shape:
        raise ValueError("action noise shape must match target")
    return rectified_event_flow_inputs(action, t=t, event_noise=action_noise)


class EWMModel(nn.Module):
    """Joint event/action flow or a single active expert; no dense voxel denoiser."""

    def __init__(self, *, dim=768, action_hidden_dim=512, heads=8, layers=12, context_encoder=None,
                 gradient_checkpointing=False, gradient_checkpointing_skip_first_blocks=0,
                 codec_batch_size=16, dino_injection_interval=1,
                 event_codec_impl="leanvae16_joint", event_vae_path=None, event_codec=None,
                 voxel_bins=1, event_spatial_compression=32, event_patch_size=None, event_temporal_compression=4,
                 event_camera_mode="wrist_only", diagnostic_occupancy_threshold=1/30,
                 diagnostic_samples=1, training_mode="joint"):
        super().__init__()
        if layers <= 0 or not 0 <= gradient_checkpointing_skip_first_blocks <= layers:
            raise ValueError("invalid layers or gradient checkpointing skip count")
        codecs = {"wan22_frame": (WanEventCodec, 48, 16, 1),
                  "wan22_joint": (WanEventCodec, 48, 16, 4),
                  "leanvae16_joint": (LeanEventCodec, 16, 8, 4),
                  "leanvae16_rgb_joint": (LeanRGBCodec, 16, 8, 4)}
        if event_codec_impl not in codecs:
            raise ValueError("unsupported event codec; dense codecs were removed")
        codec_class, channels, spatial_stride, temporal_stride = codecs[event_codec_impl]
        if event_patch_size is None:
            # Older checkpoint configs recorded only the total spatial stride.
            event_patch_size = event_spatial_compression // spatial_stride
        if (voxel_bins != 1 or event_temporal_compression != temporal_stride or
                event_patch_size not in (2, 4, 8) or
                event_spatial_compression != spatial_stride * event_patch_size):
            raise ValueError("EWM requires bins1, matching codec temporal stride and latent patch2/4/8 with matching total spatial stride; dense codecs were removed")
        self.event_codec_impl = event_codec_impl
        if training_mode not in ("joint", "event_only", "rgb_action", "action_only"):
            raise ValueError("training_mode must be 'joint', 'event_only', 'rgb_action', or 'action_only'")
        if (training_mode == "rgb_action") != (event_codec_impl == "leanvae16_rgb_joint"):
            raise ValueError("rgb_action mode requires the leanvae16_rgb_joint codec")
        self.training_mode = training_mode
        self.latent_channels = channels
        self.latent_patch_size = int(event_patch_size)
        if event_camera_mode not in ("dual", "wrist_only", "global_only") or dino_injection_interval <= 0:
            raise ValueError("invalid event camera mode or DINO injection interval")
        if not 0 < diagnostic_occupancy_threshold <= 1 or diagnostic_samples <= 0:
            raise ValueError("invalid diagnostic threshold or sample count")
        self.voxel_bins = 1
        self.event_temporal_compression = int(event_temporal_compression)
        self.event_camera_mode = event_camera_mode
        self.num_event_cameras = 2 if event_camera_mode == "dual" else 1
        self.diagnostic_occupancy_threshold = diagnostic_occupancy_threshold
        self.diagnostic_samples = int(diagnostic_samples)
        if training_mode == "action_only":
            # The action ablation has no codec dependency, including at startup.
            event_codec = None
        elif event_codec is None:
            options = dict(batch_size=codec_batch_size)
            if codec_class is WanEventCodec:
                options["mode"] = "joint" if temporal_stride == 4 else "frame"
            event_codec = codec_class(event_vae_path, **options)
        if event_codec is not None and (event_codec.latent_channels != channels or event_codec.spatial_compression != spatial_stride or
                event_codec.mode != ("joint" if temporal_stride == 4 else "frame") or
                (training_mode == "rgb_action" and not isinstance(event_codec, LeanRGBCodec))):
            raise ValueError("event codec does not match configured representation")
        self.event_codec = event_codec.requires_grad_(False).eval() if event_codec is not None else None
        self.event_input = nn.Conv2d(channels, dim, kernel_size=self.latent_patch_size, stride=self.latent_patch_size)
        self.event_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, channels * self.latent_patch_size ** 2))
        self.state_mlp = nn.Sequential(nn.Linear(8, action_hidden_dim), nn.SiLU(), nn.Linear(action_hidden_dim, action_hidden_dim))
        self.action_mlp = nn.Sequential(nn.Linear(7, action_hidden_dim), nn.SiLU(), nn.Linear(action_hidden_dim, action_hidden_dim))
        self.action_head = nn.Sequential(nn.LayerNorm(action_hidden_dim), nn.Linear(action_hidden_dim, 7))
        self.camera = nn.Parameter(torch.randn(self.num_event_cameras, dim) * .02)
        self.blocks = nn.ModuleList([EWMJointBlock(dim, heads, action_hidden_dim) for _ in range(layers)])
        if self.training_mode == "event_only":
            # DDP must not track permanently unused action parameters.
            for module in (self.state_mlp, self.action_mlp, self.action_head,
                           *(block.action for block in self.blocks)):
                module.requires_grad_(False)
        elif self.training_mode == "action_only":
            # Mirror event_only: retain the expert layout but exclude unused
            # event weights from gradients/optimizer/DDP tracking.
            for module in (self.event_input, self.event_head,
                           *(block.event for block in self.blocks)):
                module.requires_grad_(False)
            self.camera.requires_grad_(False)
        self.context_encoder = context_encoder
        self.gradient_checkpointing = gradient_checkpointing
        self.gradient_checkpointing_skip_first_blocks = int(gradient_checkpointing_skip_first_blocks)
        self.dino_injection_interval = int(dino_injection_interval)

    def train(self, mode=True):
        if mode and getattr(self, "_inference_bf16_weights", False):
            self.disable_inference_compile()
        super().train(mode)
        if self.event_codec is not None:
            self.event_codec.eval()
        return self

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if getattr(self, "_inference_bf16_weights", False):
            self.disable_inference_compile()
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def _apply(self, fn, recurse=True):
        if getattr(self, "_inference_bf16_weights", False):
            self.disable_inference_compile()
        return super()._apply(fn, recurse=recurse)

    @classmethod
    def from_config(cls, config):
        from openwam.model.text_encoder.t5_small import T5SmallTextEncoder
        from openwam.model.video_backbone.encoder.dinov3 import DinoV3VideoEncoder

        options = dict(config)
        options.pop("type", None)
        dino_path, text_path = options.pop("dino_path"), options.pop("text_path")
        for name, path in (("dino_path", dino_path), ("text_path", text_path)):
            if path is None or not Path(str(path)).is_dir():
                raise ValueError(f"model.{name} must point to a local pretrained checkpoint directory")
        max_length = options.pop("text_max_length", 64)
        dino = DinoV3VideoEncoder.from_pretrained(str(dino_path))
        text = T5SmallTextEncoder(str(text_path), max_length=max_length, local_files_only=True)
        context = EWMContextEncoder(dino, text, options.get("dim", 768))
        return cls(context_encoder=context, **options)

    _FROZEN_CONTEXT_PREFIXES = ("context_encoder.dino.", "context_encoder.text.", "event_codec.")

    def policy_state_dict(self, state=None):
        """Save EWM weights/buffers; frozen DINO/T5/VAE are external checkpoint dependencies."""
        state = self.state_dict() if state is None else state
        return {key: value for key, value in state.items() if not key.startswith(self._FROZEN_CONTEXT_PREFIXES)}

    def load_policy_state_dict(self, state):
        # Do not silently accept a damaged EWM checkpoint via generic strict=False.
        expected = set(self.policy_state_dict())
        missing, unexpected = expected - set(state), set(state) - expected
        if missing or unexpected:
            raise RuntimeError(
                f"Invalid EWM policy weights: missing={sorted(missing)}, unexpected={sorted(unexpected)}"
            )
        return self.load_state_dict(state, strict=False)

    def _checkpoint(self, function, *args, **kwargs):
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint

            return checkpoint(function, *args, use_reentrant=False, **kwargs)
        return function(*args, **kwargs)

    def _context(self, batch, context):
        if context is not None:
            return context
        if self.context_encoder is None:
            raise ValueError("a frozen context_encoder or explicit (tokens, mask) is required")
        # Keep the public encoder forward return compatible; EWM additionally
        # consumes its visual-prefix length to schedule DINO injection.
        return self.context_encoder.encode(batch["obs_video"], batch["prompt"])

    def prepare_context_kv(self, context):
        """Pre-project per-layer context K/V once for all denoising steps."""
        if len(context) == 3:
            ctx, _, visual_tokens = context
        elif len(context) == 2:
            ctx, _ = context
            visual_tokens = 0
        else:
            raise ValueError("context must be (tokens, mask) or (tokens, mask, visual_token_count)")
        projected = []
        for index, block in enumerate(self.blocks):
            block_ctx = ctx[:, visual_tokens:] if visual_tokens and index % self.dino_injection_interval else ctx
            projected.append((
                block.event.context.project_kv(block_ctx) if self.training_mode != "action_only" else None,
                block.action.context.project_kv(block_ctx) if self.training_mode != "event_only" else None,
            ))
        return tuple(projected)

    def compile_inference(self, *, mode="reduce-overhead", cache_bf16_weights=False, **compile_options):
        """Compile sampling only, after moving the model to its inference device.

        cache_bf16_weights keeps an inference-only BF16 copy of core Linear
        parameters, avoiding repeated FP32->BF16 casts. It requires eval mode
        and sampling under BF16 autocast. Original training parameters remain
        unchanged. Training, state-dict loading and device/dtype moves discard
        the cache; call again afterwards. Direct in-place weight edits require
        explicitly disabling/recompiling inference. Compilation is lazy.
        """
        if cache_bf16_weights and self.training:
            raise ValueError("cache_bf16_weights requires model.eval() before compile_inference")
        self.disable_inference_compile()
        function = self.predict_velocity
        if cache_bf16_weights:
            from openwam.model.architectures.ewm_inference import bf16_velocity_snapshot

            function = bf16_velocity_snapshot(self)
        self._inference_velocity = torch.compile(
            function, mode=mode, dynamic=False, fullgraph=True, **compile_options)
        self._inference_bf16_weights = bool(cache_bf16_weights)
        self._inference_cache_code = function.__code__ if cache_bf16_weights else None

    def disable_inference_compile(self):
        code = getattr(self, "_inference_cache_code", None)
        if code is not None:
            # Release compiled references to the fixed-address weight snapshot.
            torch._dynamo.reset_code(code)
        self._inference_cache_code = None
        self._inference_velocity = None
        self._inference_bf16_weights = False

    def prepare_memory_kv(self, batch):
        """Optional read-only physical-event memory, prepared once per request."""
        return None

    def predict_velocity(self, obs_latent, proprio, event, action, t, context, context_kv=None, memory_kv=None):
        # Gaussian noise and predicted velocity live entirely in the selected codec latent space.
        action_only = self.training_mode == "action_only"
        if action_only:
            if action is None or action.ndim != 3 or action.shape[1] <= 0 or action.shape[-1] != 7:
                raise ValueError("action must be [B,K,7] with a positive horizon")
            b = action.shape[0]
            if proprio is None or proprio.shape != (b, 8) or t.shape != (b,):
                raise ValueError("invalid proprio or time shape")
            e = ec = None
        elif event.ndim != 6 or event.shape[2:4] != (self.num_event_cameras, self.latent_channels):
            raise ValueError("future latent must be [B,K,camera,latent_channels,h,w]")
        if not action_only:
            b, k, cameras, _, h, w = event.shape
            if k <= 0 or h % self.latent_patch_size or w % self.latent_patch_size:
                raise ValueError("invalid latent grid or horizon")
            if self.training_mode != "event_only" and (action is None or proprio is None or
                    action.shape != (b, k * self.event_temporal_compression, 7) or proprio.shape != (b, 8)):
                raise ValueError("invalid action or proprio shape")
            if obs_latent.shape != (b, cameras, self.latent_channels, h, w) or t.shape != (b,):
                raise ValueError("observation latent or time shape mismatch")
            gh, gw = h // self.latent_patch_size, w // self.latent_patch_size
            z = torch.cat((obs_latent[:, None], event), 1)
            e = self.event_input(z.reshape(-1, self.latent_channels, h, w)).flatten(2).transpose(1, 2)
            e = e.reshape(b, k + 1, cameras, gh * gw, -1)
            e = (e + self.camera.to(e)[None, None, :, None]).flatten(1, 3)
            ec, _ = event_action_coordinates(k, gh, gw, num_cameras=cameras, device=event.device)
            if self.event_temporal_compression == 4:
                ec = ec.float()
                ec[:, 0] = torch.where(ec[:, 0] == 0, 0, ec[:, 0] * 4 - 1.5)
        a = (torch.cat((self.state_mlp(proprio)[:, None], self.action_mlp(action)), 1)
             if self.training_mode != "event_only" else None)
        ac = (event_action_coordinates(action.shape[1], 1, 1, num_cameras=1, device=action.device)[1]
              if a is not None else None)
        if len(context) == 3:
            ctx, mask, visual_tokens = context
        elif len(context) == 2:
            # Explicit legacy context has no modality boundary: inject it everywhere.
            ctx, mask = context
            visual_tokens = 0
        else:
            raise ValueError("context must be (tokens, mask) or (tokens, mask, visual_token_count)")
        head_dim = self.blocks[0].action.qkv.out_features // (3 * self.blocks[0].heads)
        event_rope = multimodal_rope_frequencies(ec, head_dim, dtype=e.dtype) if e is not None else None
        action_rope = multimodal_rope_frequencies(ac, head_dim, dtype=a.dtype) if a is not None else None
        event_time = sinusoidal_embedding_1d(self.blocks[0].event.time.freq_dim, t) if not action_only else None
        action_time = (event_time if self.blocks[0].action.time.freq_dim == self.blocks[0].event.time.freq_dim
                       and not action_only else sinusoidal_embedding_1d(self.blocks[0].action.time.freq_dim, t))
        time_sinusoid = (event_time, action_time)
        if context_kv is not None and len(context_kv) != len(self.blocks):
            raise ValueError("context K/V cache must contain one entry per EWM block")
        if memory_kv is not None and len(memory_kv) != len(self.blocks):
            raise ValueError("memory K/V cache must contain one entry per EWM block")
        for index, block in enumerate(self.blocks):
            memory_options = {} if memory_kv is None else {"memory_kv": memory_kv[index]}
            # Text is always present. DINO is concatenated only every N blocks.
            if visual_tokens and index % self.dino_injection_interval:
                block_ctx, block_mask = ctx[:, visual_tokens:], mask[:, visual_tokens:]
            else:
                block_ctx, block_mask = ctx, mask
            block_args = (e, a, t, block_ctx, block_mask, ec, ac, event_rope, action_rope, time_sinusoid)
            if index >= self.gradient_checkpointing_skip_first_blocks:
                if context_kv is None:
                    e, a = self._checkpoint(block, *block_args, event_only=self.training_mode == "event_only", action_only=action_only, **memory_options)
                else:
                    event_context_kv, action_context_kv = context_kv[index]
                    e, a = block(
                        *block_args, event_context_kv=event_context_kv,
                        action_context_kv=action_context_kv, event_only=self.training_mode == "event_only", action_only=action_only, **memory_options)
            else:
                if context_kv is None:
                    e, a = block(*block_args, event_only=self.training_mode == "event_only", action_only=action_only, **memory_options)
                else:
                    event_context_kv, action_context_kv = context_kv[index]
                    e, a = block(
                        *block_args, event_context_kv=event_context_kv,
                        action_context_kv=action_context_kv, event_only=self.training_mode == "event_only", action_only=action_only, **memory_options)
        if action_only:
            return None, self.action_head(a[:, 1:])
        future = e.reshape(b, k + 1, cameras, gh, gw, -1)[:, 1:]
        patches = self.event_head(future).reshape(b, k, cameras, gh, gw, self.latent_channels, self.latent_patch_size, self.latent_patch_size)
        velocity = patches.permute(0, 1, 2, 5, 3, 6, 4, 7).reshape(b, k, cameras, self.latent_channels, h, w)
        return velocity, self.action_head(a[:, 1:]) if a is not None else None

    def forward(self, batch, *, context=None, t=None, event_noise=None, action_noise=None, diagnostics=False,
                profile_timing=False):
        rgb_mode = self.training_mode == "rgb_action"
        action_only = self.training_mode == "action_only"
        obs_key, future_key = (("action", None) if action_only else
                               ("obs_rgb", "future_rgb") if rgb_mode else ("obs_voxel", "future_voxel"))
        timings = {} if profile_timing else None
        if profile_timing:
            if batch[obs_key].is_cuda:
                torch.cuda.synchronize(batch[obs_key].device)
            clock = time.perf_counter()

        def mark(name):
            nonlocal clock
            if timings is not None:
                if batch[obs_key].is_cuda:
                    torch.cuda.synchronize(batch[obs_key].device)
                now = time.perf_counter()
                timings[name] = now - clock
                clock = now

        if action_only:
            a, at, t = rectified_action_flow_inputs(batch["action"], t=t, action_noise=action_noise)
            mark("flow_prepare")
            encoded_context = self._context(batch, context)
            mark("context_encode")
            _, av = self.predict_velocity(None, batch["proprio"], None, a, t, encoded_context)
            mark("dit_forward")
            loss = F.mse_loss(av.float(), at.float())
            result = {"loss": loss, "loss_action": loss}
            if diagnostics:
                with torch.no_grad():
                    x0 = a.float() - t[:, None, None].float() * av.float()
                    result["action_x0_mse"] = F.mse_loss(x0, batch["action"].float())
                    result["action_x0_mae"] = F.l1_loss(x0, batch["action"].float())
            mark("loss_and_diagnostics")
            if timings is not None:
                result["_profile"] = timings
            return result

        # Encode clean inputs once; never pass noisy voxels through the VAE.
        with torch.no_grad():
            if self.event_temporal_compression == 4:
                encoded = self.event_codec.encode(torch.cat((batch[obs_key][:, None], batch[future_key]), 1))
                observed, clean_event = encoded[:, 0], encoded[:, 1:]
            else:
                clean_event = self.event_codec.encode(batch[future_key])
                observed = self.event_codec.encode(batch[obs_key][:, None])[:, 0]
        mark("vae_encode")
        if clean_event.shape[2] != self.num_event_cameras:
            raise ValueError("dataloader and model event_camera_mode must match")
        if self.training_mode == "event_only":
            e, et, t = rectified_event_flow_inputs(clean_event, t=t, event_noise=event_noise)
            a = at = None
        else:
            e, a, et, at, t = rectified_flow_inputs(
                clean_event, batch["action"], t=t, event_noise=event_noise, action_noise=action_noise)
        mark("flow_prepare")
        encoded_context = self._context(batch, context)
        mark("context_encode")
        memory_kv = self.prepare_memory_kv(batch)
        if memory_kv is not None:
            mark("memory_encode")
        ev, av = self.predict_velocity(observed, batch.get("proprio"), e, a, t, encoded_context, **({} if memory_kv is None else {"memory_kv": memory_kv}))
        mark("dit_forward")
        valid = batch.get("valid_region_mask")
        latent_valid = None
        if valid is not None:
            if valid.shape != (ev.shape[0], 1, batch[obs_key].shape[-2], batch[obs_key].shape[-1]):
                raise ValueError("valid_region_mask must be [B,1,H,W]")
            latent_valid = F.interpolate(valid.float(), size=ev.shape[-2:], mode="area")[:, None, None]
        event_error = (ev.float() - et.float()).square()
        event_loss = (event_error * latent_valid).sum() / (latent_valid.sum() * ev.shape[1] * ev.shape[2] * ev.shape[3]) if latent_valid is not None else event_error.mean()
        action_loss = (F.mse_loss(av.float(), at.float()) if av is not None
                       else event_loss.new_zeros(()))
        with torch.no_grad():
            x0 = e.float() - t.reshape(-1, 1, 1, 1, 1, 1).float() * ev.float()
            latent_error = (x0 - clean_event.float()).square()
            latent_mse = (latent_error * latent_valid).sum() / (latent_valid.sum() * ev.shape[1] * ev.shape[2] * ev.shape[3]) if latent_valid is not None else latent_error.mean()
        target_loss_key = "loss_rgb" if rgb_mode else "loss_event"
        result = {"loss": event_loss + action_loss, target_loss_key: event_loss,
                  "loss_action": action_loss, "latent_flow_mse": event_loss.detach(),
                  "latent_x0_mse": latent_mse}
        if diagnostics:
            with torch.no_grad():
                n = min(self.diagnostic_samples, x0.shape[0])
                pred = self._decode_future(observed[:n], x0[:n])
                if rgb_mode:
                    target = batch[future_key][:n].float() / 255.0
                    error = (pred - target).square()
                    mse = error.mean()
                    result.update(rgb_x0_mse=mse,
                                  rgb_x0_mae=(pred - target).abs().mean(),
                                  rgb_x0_psnr=-10 * torch.log10(mse.clamp_min(1e-12)))
                else:
                    target = batch[future_key][:n].float()
                    pixel_valid = (valid[:n, None, None].expand_as(target) if valid is not None
                                   else torch.ones_like(target, dtype=torch.bool))
                    active = (target > 0) & pixel_valid
                    inactive = (target <= 0) & pixel_valid
                    predicted = (pred > self.diagnostic_occupancy_threshold) & pixel_valid
                    error = (pred - target).square()
                    tp = (active & predicted).sum().float()
                    precision = tp / predicted.sum().clamp_min(1)
                    recall = tp / active.sum().clamp_min(1)
                    total = pixel_valid.sum().clamp_min(1)
                    result.update(voxel_x0_mse=(error * pixel_valid).sum() / total,
                        voxel_x0_active_mse=(error * active).sum() / active.sum().clamp_min(1),
                        voxel_x0_inactive_mse=(error * inactive).sum() / inactive.sum().clamp_min(1),
                        event_precision=precision, event_recall=recall,
                        event_f1=2 * precision * recall / (precision + recall).clamp_min(1e-8),
                        pred_nonzero_ratio=predicted.sum() / total, target_nonzero_ratio=active.sum() / total)
        mark("loss_and_diagnostics")
        if timings is not None:
            result["_profile"] = timings
        return result

    def _decode_future(self, observed, future):
        if self.event_temporal_compression == 4:
            return self.event_codec.decode(torch.cat((observed[:, None], future), 1))[:, 1:]
        return self.event_codec.decode(future)

    compute_loss = forward

    @torch.no_grad()
    def sample_policy(self, batch, *, horizon, normalizer=None, num_steps=20, context=None,
                      generator=None, decode_future=True, cache_context_kv=True):
        """Backward Euler; single-expert modes return only their active modality."""
        from openwam.model.video_backbone.cosmos_predict25.scheduler import CosmosFlowSchedulerAdapter

        action_only = self.training_mode == "action_only"
        if horizon <= 0 or num_steps <= 0 or (not action_only and horizon % self.event_temporal_compression):
            raise ValueError("horizon and num_steps must be positive")
        was_training = self.training
        if was_training:
            self.eval()
        try:
            obs = batch["proprio"] if action_only else batch["obs_rgb"] if self.training_mode == "rgb_action" else batch["obs_voxel"]
            b = obs.shape[0]
            if getattr(self, "_inference_bf16_weights", False) and (
                    not torch.is_autocast_enabled(obs.device.type) or
                    torch.get_autocast_dtype(obs.device.type) != torch.bfloat16):
                raise ValueError("cached BF16 inference requires BF16 autocast")
            inference_velocity = getattr(self, "_inference_velocity", None)
            if inference_velocity is not None and obs.is_cuda:
                # Start one graph iteration per sampling request, not per flow step.
                torch.compiler.cudagraph_mark_step_begin()
            predict = self.predict_velocity if inference_velocity is None else inference_velocity
            ctx = self._context(batch, context)
            context_kv = self.prepare_context_kv(ctx) if cache_context_kv else None
            observed = None if action_only else self.event_codec.encode(obs[:, None])[:, 0]
            memory_kv = self.prepare_memory_kv(batch)
            e = (None if action_only else torch.randn(
                (b, horizon // self.event_temporal_compression, *observed.shape[1:]),
                device=obs.device, dtype=torch.float32, generator=generator))
            a = (torch.randn((b, horizon, 7), device=obs.device, dtype=torch.float32, generator=generator)
                 if self.training_mode != "event_only" else None)
            scheduler = CosmosFlowSchedulerAdapter(shift_video=1.0)
            scheduler.set_timesteps(num_steps)
            for sigma, timestep in zip(scheduler.sigmas, scheduler.timesteps):
                t = sigma.to(a if action_only else e).expand(b)
                ev, av = predict(
                    observed, batch.get("proprio"), e, a, t, ctx, context_kv=context_kv, **({} if memory_kv is None else {"memory_kv": memory_kv}))
                if e is not None:
                    e = scheduler.step(ev, timestep, e)
                if a is not None:
                    a = scheduler.step(av, timestep, a)
            result = {} if action_only else {"future_latent": e}
            if a is not None:
                if normalizer is None:
                    raise ValueError("action sampling requires an action normalizer")
                a = a.clamp(-1, 1)
                result.update(action=normalizer.denormalize_action(a), action_normalized=a)
            if decode_future and not action_only:
                key = "future_rgb" if self.training_mode == "rgb_action" else "future_voxel"
                result[key] = self._decode_future(observed, e)
            return result
        finally:
            if was_training:
                self.train(True)


__all__ = [
    "EWMJointBlock",
    "EWMContextEncoder",
    "EWMModel",
    "canonical_voxel",
    "event_action_coordinates",
    "multimodal_rope",
    "full_mixed_self_attention",
    "rectified_flow_inputs",
    "rectified_event_flow_inputs",
    "rectified_action_flow_inputs",
]
