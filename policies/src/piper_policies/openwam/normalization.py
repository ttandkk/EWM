"""Adapt only the known Piper statistics schema; keep native normalization math."""
from contextlib import contextmanager
import hashlib
import io
from pathlib import Path
import tempfile

import numpy as np


def normalization_payload(path):
    source = Path(path).read_bytes()
    raw = np.load(io.BytesIO(source), allow_pickle=True).item()
    if not isinstance(raw, dict):
        raise ValueError('Piper OpenWAM normalization must be a stats mapping')
    if set(raw) == {'eef', 'state'}:
        state_key, schema, mapping = 'state', 'piper_legacy_state_v1', {'state': 'eef_state'}
    elif set(raw) == {'eef', 'eef_state'}:
        state_key, schema, mapping = 'eef_state', 'piper_native_eef_state_v1', {}
    else:
        raise ValueError('Unknown Piper OpenWAM normalization schema; expected eef plus state or eef_state')
    required, optional = {'min', 'max', 'mean', 'std'}, {'q01', 'q99'}
    for name, size in (('eef', 7), (state_key, 8)):
        block = raw[name]
        if (not isinstance(block, dict) or not required <= set(block)
                or set(block) - required - optional or bool('q01' in block) != bool('q99' in block)):
            raise ValueError(f'Unknown Piper statistics fields for {name}')
        for key, value in block.items():
            array = np.asarray(value)
            if array.shape != (size,) or array.dtype.kind not in 'fiu' or not np.isfinite(array).all():
                raise ValueError(f'{name}.{key} must contain finite {size}D statistics')
    effective = source
    if mapping:
        canonical = {'eef': raw['eef'], 'eef_state': raw['state']}
        buffer = io.BytesIO()
        np.save(buffer, canonical)
        effective = buffer.getvalue()
        restored = np.load(io.BytesIO(effective), allow_pickle=True).item()
        for key, output_key in (('eef', 'eef'), ('state', 'eef_state')):
            for statistic, original in raw[key].items():
                original, exported = np.asarray(original), np.asarray(restored[output_key][statistic])
                if (original.shape != exported.shape or original.dtype != exported.dtype
                        or not np.array_equal(original, exported)):
                    raise ValueError('Normalization statistics changed during schema translation')
    summary = dict(source_schema=schema, mapping=mapping, action_dim=7, state_dim=8,
                   source_sha256=hashlib.sha256(source).hexdigest(),
                   effective_sha256=hashlib.sha256(effective).hexdigest())
    return summary, effective


@contextmanager
def native_statistics_directory(path, expected):
    """Keep translated files alive until the native factory has read its arrays."""
    path = Path(path)
    summary, payload = normalization_payload(path)
    if summary != expected:
        raise ValueError('Normalization statistics differ from the described policy')
    if not summary['mapping']:
        yield path.parent
        return
    with tempfile.TemporaryDirectory(prefix='piper-openwam-stats-') as directory:
        directory = Path(directory)
        (directory / 'normalization_stats.npy').write_bytes(payload)
        yield directory


def build_compatible_normalizer(cfg, path, expected):
    from openwam.deploy.model_loader import _build_normalizer
    # The factory eagerly reads the NPY and owns ordinary arrays, not mmap/file
    # handles. Cleanup after construction therefore preserves inference lifetime.
    with native_statistics_directory(path, expected) as directory:
        return _build_normalizer(cfg, str(directory))
