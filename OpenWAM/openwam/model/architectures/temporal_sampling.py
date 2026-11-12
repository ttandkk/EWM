"""Input frame sampling around a frozen causal video codec; actions keep their rate."""
from dataclasses import dataclass

import torch

from .temporal_supervision import latent_mask


@dataclass(frozen=True)
class TemporalSampling:
    stride: int = 1
    compression: int = 4

    def __post_init__(self):
        if type(self.stride) is not int or self.stride not in (1, 2, 3):
            raise ValueError('event_temporal_downsample must be an integer in {1, 2, 3}')

    @property
    def group_size(self):
        return self.stride * self.compression

    def future_slices(self, horizon):
        if horizon <= 0 or horizon % self.group_size:
            raise ValueError('future horizon must be positive and divisible by '
                             'event_temporal_compression * event_temporal_downsample; no frame padding')
        return horizon // self.group_size

    def indices(self, horizon, *, device=None):
        """Zero-based future indices: s-1, 2s-1, ...; keep the endpoint."""
        self.future_slices(horizon)
        return torch.arange(self.stride - 1, horizon, self.stride, device=device)

    def sample(self, future):
        self.future_slices(future.shape[1])
        if self.stride == 1:
            return future
        return future.index_select(1, self.indices(future.shape[1], device=future.device))

    def mask(self, valid):
        # Dataset tail padding is not a real target: require complete source groups.
        return latent_mask(valid, self.group_size)

    def frame_mask(self, valid, horizon):
        return valid.repeat_interleave(self.group_size, dim=1)[:, :horizon]

    def coordinates(self, times):
        """Centers of the sampled frames in original action/frame units."""
        if self.compression == 1 and self.stride == 1:
            return times
        return torch.where(times == 0, 0,
                           (times * self.compression - (self.compression - 1) / 2) * self.stride)

    def restore(self, future):
        """Repeat sampled predictions over their source intervals; no invented motion."""
        return future if self.stride == 1 else future.repeat_interleave(self.stride, dim=1)
