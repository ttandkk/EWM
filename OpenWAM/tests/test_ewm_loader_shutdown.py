"""Early exit must finish prefetched worker results and report failures."""
import pytest
from accelerate.data_loader import DataLoaderShard
from torch.utils.data import DataLoader, Dataset

from scripts.train_ewm.train_ewm import _finish_persistent_loader


class ShutdownDataset(Dataset):
    def __init__(self, length=20, fail_at=None):
        self.length = length
        self.fail_at = fail_at

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        if index == self.fail_at:
            raise ValueError('prefetched sample failed')
        return index


@pytest.mark.parametrize('wrapped', [False, True])
@pytest.mark.parametrize('length', [2, 20])
def test_persistent_loader_finishes_prefetch(wrapped, length):
    cls = DataLoaderShard if wrapped else DataLoader
    loader = cls(ShutdownDataset(length), batch_size=1, num_workers=2,
                 persistent_workers=True, multiprocessing_context='spawn')
    iterator = iter(loader)
    assert next(iterator).item() == 0
    iterator.close() if wrapped else None
    _finish_persistent_loader(loader)
    base = getattr(loader, 'base_dataloader', loader)
    assert base._iterator is None


def test_persistent_loader_reports_prefetch_error():
    loader = DataLoader(ShutdownDataset(fail_at=2), batch_size=1, num_workers=2,
                        persistent_workers=True, multiprocessing_context='spawn')
    iterator = iter(loader)
    assert next(iterator).item() == 0
    with pytest.raises(ValueError, match='prefetched sample failed'):
        _finish_persistent_loader(loader)
    assert loader._iterator is None
