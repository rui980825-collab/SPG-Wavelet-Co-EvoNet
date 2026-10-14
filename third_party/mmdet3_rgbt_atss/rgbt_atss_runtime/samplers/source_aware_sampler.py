from __future__ import annotations

import math
from typing import Iterator, List

import torch
from mmengine.dataset import DefaultSampler
from mmdet.datasets.dataset_wrappers import ConcatDataset
from mmdet.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class SourceAwareSampler(DefaultSampler):
    """Sampler with explicit per-source exposure control for mixed datasets.

    The current QFDet smoke uses batch_size=1 and two-source ConcatDataset:
    repeated real data first, repeated synth data second. This sampler keeps
    the model and losses fixed, and only makes early training see a stable
    source cadence such as R,R,R,R,S.
    """

    def __init__(
        self,
        dataset,
        shuffle: bool = True,
        seed: int | None = None,
        round_up: bool = True,
        real_ratio: int = 4,
        synth_ratio: int = 1,
        deterministic_cycle: bool = True,
    ) -> None:
        super().__init__(dataset=dataset, shuffle=shuffle, seed=seed, round_up=round_up)
        if not isinstance(self.dataset, ConcatDataset):
            raise TypeError("SourceAwareSampler expects a ConcatDataset.")
        if len(self.dataset.datasets) != 2:
            raise ValueError("SourceAwareSampler currently supports exactly two sources.")
        if real_ratio <= 0 or synth_ratio <= 0:
            raise ValueError("real_ratio and synth_ratio must both be positive.")
        self.real_ratio = int(real_ratio)
        self.synth_ratio = int(synth_ratio)
        self.deterministic_cycle = bool(deterministic_cycle)
        self.real_len = len(self.dataset.datasets[0])
        self.synth_len = len(self.dataset.datasets[1])
        self.real_offset = 0
        self.synth_offset = self.real_len
        self._cycle = [0] * self.real_ratio + [1] * self.synth_ratio
        # Define one mixed epoch relative to a single effective pass over the
        # real dataset, then expose synth samples according to the requested
        # real:synth cadence. This keeps "0.25x" mixing aligned with the
        # intended 4:1 real-to-synth exposure ratio instead of inheriting the
        # raw ConcatDataset length.
        base_total = math.ceil(
            self.real_len * (self.real_ratio + self.synth_ratio) / self.real_ratio
        )
        if self.round_up:
            self.mixed_total_size = math.ceil(base_total / self.world_size) * self.world_size
        else:
            self.mixed_total_size = base_total
        self.mixed_num_samples = math.ceil(self.mixed_total_size / self.world_size)

    def _build_source_pool(self, source_len: int, seed_offset: int, refill_idx: int) -> List[int]:
        if not self.shuffle:
            return list(range(source_len))
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch + seed_offset + refill_idx * 9973)
        return torch.randperm(source_len, generator=generator).tolist()

    def __iter__(self) -> Iterator[int]:
        real_pool = self._build_source_pool(self.real_len, seed_offset=11, refill_idx=0)
        synth_pool = self._build_source_pool(self.synth_len, seed_offset=29, refill_idx=0)
        real_ptr = 0
        synth_ptr = 0
        real_refill = 0
        synth_refill = 0

        def next_from_real() -> int:
            nonlocal real_pool, real_ptr, real_refill
            if real_ptr >= len(real_pool):
                real_refill += 1
                real_pool = self._build_source_pool(self.real_len, seed_offset=11, refill_idx=real_refill)
                real_ptr = 0
            idx = self.real_offset + real_pool[real_ptr]
            real_ptr += 1
            return idx

        def next_from_synth() -> int:
            nonlocal synth_pool, synth_ptr, synth_refill
            if synth_ptr >= len(synth_pool):
                synth_refill += 1
                synth_pool = self._build_source_pool(self.synth_len, seed_offset=29, refill_idx=synth_refill)
                synth_ptr = 0
            idx = self.synth_offset + synth_pool[synth_ptr]
            synth_ptr += 1
            return idx

        indices: List[int] = []
        cycle_len = len(self._cycle)
        if self.deterministic_cycle:
            for pos in range(self.mixed_total_size):
                source_id = self._cycle[pos % cycle_len]
                indices.append(next_from_real() if source_id == 0 else next_from_synth())
        else:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch + 53)
            probs = torch.tensor(
                [self.real_ratio, self.synth_ratio], dtype=torch.float32
            )
            probs = probs / probs.sum()
            draws = torch.multinomial(
                probs, self.mixed_total_size, replacement=True, generator=generator
            ).tolist()
            for source_id in draws:
                indices.append(next_from_real() if source_id == 0 else next_from_synth())

        indices = indices[self.rank:self.mixed_total_size:self.world_size]
        return iter(indices)

    def __len__(self) -> int:
        return self.mixed_num_samples
