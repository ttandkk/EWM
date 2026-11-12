"""Validity of padded future labels; never part of policy observations."""
import torch


def temporal_mask(batch, key, values):
    mask = batch.get(key)
    if mask is None:
        return None
    if mask.dtype != torch.bool or mask.shape != values.shape[:2]:
        raise ValueError(f'{key} must be boolean [B,K] matching its labels')
    if (mask[:, 1:] & ~mask[:, :-1]).any():
        raise ValueError(f'{key} must be a valid prefix followed by padding')
    return mask


def expand_time(mask, values):
    return mask.reshape(*mask.shape, *((1,) * (values.ndim - 2)))


def masked_mean(error, mask=None):
    if mask is None:
        return error.mean()
    weights = mask.to(error).expand_as(error)
    # where also avoids including NaN padding in a nominally zero-weight sum.
    return torch.where(weights > 0, error * weights, 0).sum() / weights.sum().clamp_min(1)


def pad_future_frames(observed, future, valid):
    """Canonical edge padding before the frozen temporal codec."""
    if valid is None:
        return future
    frames = torch.cat((observed[:, None], future), dim=1)
    last = valid.long().sum(1)
    edge = frames[torch.arange(len(frames), device=frames.device), last][:, None]
    return torch.where(expand_time(valid, future), future, edge)


def latent_mask(valid, compression):
    if valid is None:
        return None
    if valid.shape[1] % compression:
        raise ValueError('future mask horizon must divide by codec compression')
    # LeanVAE has causal history and groups of four new frames per latent.
    # A partial final group contains padding, so it is not a real target.
    return valid.reshape(valid.shape[0], -1, compression).all(-1)
