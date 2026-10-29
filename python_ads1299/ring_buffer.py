"""Fixed-capacity buffer for the live eight-channel plot."""

import numpy as np


class MultiChannelRingBuffer:
    def __init__(self, capacity, channels=8):
        self.capacity = int(capacity)
        self.channels = int(channels)
        if self.capacity <= 0 or self.channels <= 0:
            raise ValueError("capacity and channels must be positive")
        self.values = np.empty((self.capacity, self.channels), dtype=np.float64)
        self.clear()

    def clear(self):
        self._write = 0
        self._count = 0
        self.source_length = 0

    def append(self, values):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.channels:
            raise ValueError("plot data must have one column per channel")
        size = len(values)
        if size == 0:
            return
        recent = values[-self.capacity:]
        count = len(recent)
        first = min(count, self.capacity - self._write)
        self.values[self._write:self._write + first] = recent[:first]
        if count > first:
            self.values[:count - first] = recent[first:]
        self._write = (self._write + count) % self.capacity
        self._count = min(self.capacity, self._count + size)
        self.source_length += size

    def snapshot(self):
        start = (self._write - self._count) % self.capacity
        if not self._count:
            return np.empty((0, self.channels), dtype=np.float64)
        if start < self._write:
            return self.values[start:self._write].copy()
        return np.concatenate((self.values[start:], self.values[:self._write]))
