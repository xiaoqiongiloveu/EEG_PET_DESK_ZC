"""按 EEG 帧序号和设备时间戳统计丢包与断流。"""

MAX_OUTAGE_S = 10.0
MIN_TIMEBASE_PERIOD_MS = 4.0


class StreamLossTracker:
    def __init__(self, sample_rate=250.0):
        self.sample_rate = float(sample_rate)
        self.lost_by_seq = 0
        self.lost_by_time = 0
        self.gaps = 0
        self.outages = 0
        self._last_seq = None
        self._last_ms = None

    def loss_rate(self, received):
        total = int(received) + self.lost_by_seq
        return self.lost_by_seq / total if total > 0 else 0.0

    def update(self, sequence, device_ms):
        seq = int(sequence) & 0xFFFF
        ts = int(device_ms) & 0xFFFFFFFF
        max_gap = max(1, int(self.sample_rate * MAX_OUTAGE_S))
        outage = False

        if self._last_seq is not None:
            gap = (seq - self._last_seq - 1) & 0xFFFF
            if gap:
                if gap <= max_gap:
                    self.lost_by_seq += gap
                    self.gaps += 1
                else:
                    outage = True
        self._last_seq = seq

        if self._last_ms is not None and self._update_time(ts, max_gap):
            outage = True
        self._last_ms = ts
        if outage:
            self.outages += 1

    def _update_time(self, ts, max_gap):
        delta = (ts - self._last_ms) & 0xFFFFFFFF
        expected = 1000.0 / self.sample_rate if self.sample_rate > 0 else 0.0
        if delta >= 0x80000000:
            return True
        if expected <= 0.0 or expected < MIN_TIMEBASE_PERIOD_MS:
            return False
        if delta > max_gap * expected:
            return True
        missing = int((delta - 0.5 * expected) / expected)
        if missing > 0:
            self.lost_by_time += missing
        return False
