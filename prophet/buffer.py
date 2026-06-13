"""Uniform replay buffer over self-play samples."""

from collections import deque

import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.data = deque(maxlen=capacity)

    def add(self, samples):
        self.data.extend(samples)

    def sample(self, batch_size: int, rng: np.random.Generator):
        idx = rng.integers(len(self.data), size=batch_size)
        return [self.data[int(i)] for i in idx]

    def __len__(self):
        return len(self.data)
