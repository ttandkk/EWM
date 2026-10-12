"""Source purity, rank alignment, epoch coverage and deterministic replay."""
from collections import Counter
import math
import pytest
from accelerate.data_loader import BatchSamplerShard
from openwam.dataloader.ewm_source_batch_sampler import SourceHomogeneousBatchSampler

RANGES = {"reassemble": [(0, 7), (14, 19)], "dvla": [(7, 14)]}

def source(index):
    return "dvla" if 7 <= index < 14 else "reassemble"

@pytest.mark.parametrize("world_size", [1, 2, 3, 8])
def test_all_ranks_share_source_with_full_epoch_coverage(world_size):
    sampler = SourceHomogeneousBatchSampler(RANGES, 2, num_processes=world_size, seed=7)
    batches = list(sampler)
    assert len(batches) == len(sampler)
    seen = Counter(i for batch in batches for i in batch)
    assert set(seen) == set(range(19))
    expected = {s: math.ceil(n / (2 * world_size)) for s, n in sampler.source_sizes.items()}
    actual = Counter()
    for begin in range(0, len(batches), world_size):
        group = batches[begin:begin + world_size]
        assert all(len(batch) == 2 for batch in group)
        sources = {source(i) for batch in group for i in batch}
        assert len(sources) == 1
        actual[next(iter(sources))] += 1
    assert actual == expected
    for s, n in sampler.source_sizes.items():
        repeats = sum(count - 1 for index, count in seen.items() if source(index) == s)
        assert repeats == expected[s] * 2 * world_size - n
        assert repeats < 2 * world_size
    shards = [list(BatchSamplerShard(sampler, world_size, rank, split_batches=False))
              for rank in range(world_size)]
    assert all(len(shard) == len(batches) // world_size for shard in shards)
    for step in range(len(shards[0])):
        assert len({source(i) for rank in shards for i in rank[step]}) == 1
        assert [shards[rank][step] for rank in range(world_size)] == batches[step*world_size:(step+1)*world_size]

def test_epoch_replay_and_single_source():
    sampler = SourceHomogeneousBatchSampler({"dvla": [(0, 31)]}, 4, num_processes=2, seed=42)
    first = list(sampler)
    assert first == list(sampler)
    sampler.set_epoch(3)
    changed = list(sampler)
    assert changed != first
    replay = SourceHomogeneousBatchSampler({"dvla": [(0, 31)]}, 4, num_processes=2, seed=42)
    replay.set_epoch(3)
    assert list(replay) == changed

@pytest.mark.parametrize("ranges,batch_size,world_size", [
    ({}, 2, 1), ({"a": [(0, 4)], "b": [(3, 6)]}, 2, 1),
    ({"a": [(0, 4)]}, 0, 1), ({"a": [(0, 4)]}, 2, 0),
])
def test_invalid_sampling_contract_is_rejected(ranges, batch_size, world_size):
    with pytest.raises(ValueError):
        SourceHomogeneousBatchSampler(ranges, batch_size, num_processes=world_size)
