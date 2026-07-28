"""Uniform replay buffer over self-play samples.

Ring buffer over a plain list, NOT a deque: random access into a deque is
O(n) per index, so sampling 256 items from a 300k deque walked ~38M
pointer hops per batch — it silently dominated the learner step time at
scale (invisible in small-buffer tests). Same FIFO eviction, same uniform
sampling, O(1) access.
"""

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data = []  # grows to capacity, then overwritten ring-wise
        self._cursor = 0

    def add(self, samples):
        data, cap = self.data, self.capacity
        for s in samples:
            if len(data) < cap:
                data.append(s)
            else:
                data[self._cursor] = s
                self._cursor = (self._cursor + 1) % cap

    def sample(self, batch_size: int, rng: np.random.Generator):
        idx = rng.integers(len(self.data), size=batch_size)
        return [self.data[int(i)] for i in idx]

    def __len__(self):
        return len(self.data)
