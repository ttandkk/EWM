"""Source-homogeneous batches, including across Accelerate data-parallel ranks."""
import math

import torch
from torch.utils.data import Sampler


class SourceHomogeneousBatchSampler(Sampler):
    """Shuffle all windows once per epoch and interleave source batch groups.

    A group contains ``batch_size * num_processes`` windows from one source.
    Yield its rank-sized batches consecutively so Accelerate's ordinary
    non-split sharding gives every rank the same source at each micro-step.
    Pad each source's final group from that source's shuffled windows only.
    At most global_batch_size-1 windows per source are repeated per epoch.
    """

    def __init__(self, source_ranges, batch_size, *, num_processes=1, seed=42):
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.seed = int(seed)
        self.epoch = 0
        # All yielded batches are full; Accelerate must not add cross-source padding.
        self.drop_last = True
        if self.batch_size <= 0 or self.num_processes <= 0:
            raise ValueError("batch_size and num_processes must be positive")
        self.source_ranges = {
            source: [(int(start), int(stop)) for start, stop in ranges if stop > start]
            for source, ranges in sorted(source_ranges.items())
        }
        self.source_ranges = {s: rs for s, rs in self.source_ranges.items() if rs}
        intervals = sorted(r for rs in self.source_ranges.values() for r in rs)
        if not intervals or any(start < 0 or stop <= start for start, stop in intervals):
            raise ValueError("source ranges must contain nonempty positive index intervals")
        if any(right[0] < left[1] for left, right in zip(intervals, intervals[1:])):
            raise ValueError("source index ranges must not overlap")
        self.source_sizes = {s: sum(stop - start for start, stop in rs)
                             for s, rs in self.source_ranges.items()}
        global_size = self.batch_size * self.num_processes
        self.source_group_counts = {s: math.ceil(n / global_size)
                                    for s, n in self.source_sizes.items()}

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return sum(self.source_group_counts.values()) * self.num_processes

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        global_size = self.batch_size * self.num_processes
        shuffled = {}
        sources = list(self.source_ranges)
        schedule = []
        for source_id, source in enumerate(sources):
            indices = torch.cat([torch.arange(start, stop) for start, stop in self.source_ranges[source]])
            indices = indices[torch.randperm(len(indices), generator=generator)]
            required = self.source_group_counts[source] * global_size
            padding = required - len(indices)
            if padding:
                indices = torch.cat((indices, indices.repeat(math.ceil(padding / len(indices)))[:padding]))
            shuffled[source] = indices
            schedule.extend([source_id] * self.source_group_counts[source])
        order = torch.randperm(len(schedule), generator=generator).tolist()
        offsets = {source: 0 for source in sources}
        for group in order:
            source = sources[schedule[group]]
            start = offsets[source]
            offsets[source] += global_size
            for rank in range(self.num_processes):
                begin = start + rank * self.batch_size
                yield shuffled[source][begin:begin + self.batch_size].tolist()
