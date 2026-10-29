"""Streaming estimate of completed eye blinks from frontal EEG samples."""

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, find_peaks, sosfiltfilt


class EyeBlinkDetector:
    """Count frontal blink-shaped pulses; input values are microvolts."""

    PRIMARY = (5, 6, 7)
    FALLBACK = (0, 2, 4, 7)

    def __init__(self):
        self.reset()

    def reset(self):
        self._rate = None
        self._buffer = np.empty((0, 8), dtype=float)
        self._buffer_start = 0
        self._sample_count = 0
        self._last_counted = -10**9
        self._pending = np.empty((0, 8), dtype=float)

    def process(self, samples, sample_rate):
        rate = float(sample_rate)
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError("sample_rate must be positive")
        data = np.asarray(samples, dtype=float)
        if data.size == 0:
            return 0
        if data.ndim != 2 or data.shape[1] != 8:
            raise ValueError("samples must contain eight channels")
        if self._rate is not None and rate != self._rate:
            self.reset()
        self._rate = rate
        count = 0
        step = max(1, int(0.2 * rate))
        self._pending = np.concatenate((self._pending, data), axis=0)
        while len(self._pending) >= step:
            chunk = self._pending[:step]
            self._pending = self._pending[step:]
            self._buffer = np.concatenate((self._buffer, chunk), axis=0)
            self._sample_count += len(chunk)
            keep = max(int(7 * rate), 64)
            if len(self._buffer) > keep:
                discard = len(self._buffer) - keep
                self._buffer = self._buffer[discard:]
                self._buffer_start += discard
            count += self._count_ready_peaks()
        return count

    def _count_ready_peaks(self):
        rate = self._rate
        raw = self._buffer
        if len(raw) < max(int(rate), 32):
            return 0

        finite = np.isfinite(raw)
        centers = np.array([
            np.median(raw[finite[:, ch], ch]) if np.any(finite[:, ch]) else 0.0
            for ch in range(8)
        ])
        filled = np.where(finite, raw, centers)
        # A local baseline suppresses drift without the positive side lobes
        # that a zero-phase high-pass creates around large broad pulses.
        sos = butter(3, min(6.0, rate * 0.4), btype="lowpass",
                     fs=rate, output="sos")
        smoothed = sosfiltfilt(sos, filled, axis=0)
        baseline = uniform_filter1d(
            smoothed, size=max(3, int(2.0 * rate) | 1), axis=0,
            mode="nearest")
        filtered = smoothed - baseline

        # A short mains-frequency burst can leave a slow envelope that looks
        # like a blink after low-pass filtering. Keep its local energy separate
        # from the low-frequency candidate waveform.
        high_frequency_rms = None
        high_frequency_top = min(65.0, 0.45 * rate)
        if high_frequency_top > 40.0:
            high_frequency_sos = butter(
                3, (35.0, high_frequency_top), btype="bandpass",
                fs=rate, output="sos",
            )
            high_frequency = sosfiltfilt(high_frequency_sos, filled, axis=0)
            high_frequency_rms = np.sqrt(np.maximum(
                0.0,
                uniform_filter1d(
                    high_frequency ** 2,
                    size=max(3, int(0.1 * rate) | 1),
                    axis=0, mode="nearest",
                ),
            ))

        q10, q90 = np.percentile(filled, (10, 90), axis=0)
        spread = q90 - q10
        usable = (np.mean(finite, axis=0) > 0.9) & (spread > 5)
        primary = [ch for ch in self.PRIMARY if usable[ch]]
        if len(primary) >= 2:
            primary_spread = spread[primary]
            if np.max(primary_spread) <= 5 * np.min(primary_spread):
                leads = primary
            else:
                leads = [ch for ch in self.FALLBACK if usable[ch]]
        else:
            leads = [ch for ch in self.FALLBACK if usable[ch]]
        fallback = 7 in leads and set(leads).issubset(self.FALLBACK)
        if len(leads) < (3 if fallback else 2):
            return 0

        composite = np.median(filtered[:, leads], axis=1)
        distance = max(1, int(0.28 * rate))
        # Peaks need right-side context before count.
        peaks, properties = find_peaks(
            composite, prominence=18.0, distance=distance,
            width=(max(1, int(0.07 * rate)), int(1.3 * rate)),
        )
        count = 0
        for i, peak in enumerate(peaks):
            absolute = self._buffer_start + int(peak)
            if absolute <= self._last_counted + int(0.27 * rate):
                continue
            if peak > len(raw) - max(int(0.7 * rate), 4):
                continue
            height = float(composite[peak])
            if height < 40:
                continue
            prominence = float(properties["prominences"][i])
            if prominence < 20:
                continue
            if high_frequency_rms is not None and self._has_high_frequency_burst(
                high_frequency_rms, peak, leads, height, rate
            ):
                continue
            if not self._is_frontal_pulse(raw, filtered, finite, peak,
                                          leads, fallback, height, rate):
                continue
            self._last_counted = absolute
            count += 1
        return count

    @staticmethod
    def _has_high_frequency_burst(rms, peak, leads, height, rate):
        start = max(0, peak - int(0.44 * rate))
        stop = min(len(rms), peak + int(0.20 * rate) + 1)
        quiet = np.percentile(rms[:, leads], 25, axis=0)
        threshold = np.maximum(
            np.maximum(20.0, 5.0 * quiet), 0.8 * height
        )
        return bool(np.any(np.count_nonzero(
            rms[start:stop, leads] > threshold, axis=1
        ) >= 2))

    @staticmethod
    def _is_frontal_pulse(raw, filtered, finite, peak, leads,
                          fallback, height, rate):
        radius = max(1, int(0.035 * rate))
        lo, hi = max(0, peak - radius), min(len(raw), peak + radius + 1)
        local = np.max(filtered[lo:hi, leads], axis=0)
        supporting = local > max(12, 0.28 * height)
        if np.count_nonzero(supporting) < (3 if fallback else 2):
            return False
        if not np.all(finite[lo:hi, :][:, leads]):
            return False

        # All-lead motion has no frontal spatial contrast. With only four
        # connected fallback leads, equal amplitudes are likewise ambiguous.
        other = [ch for ch in range(8) if ch not in leads]
        other_active = np.max(filtered[lo:hi, other], axis=0)
        if np.count_nonzero(other_active > max(14, 0.55 * height)) >= 4:
            return False
        if fallback:
            active = local[supporting]
            if len(active) >= 4 and np.std(active) < 0.045 * np.mean(active):
                return False

        # A held DC step is not a blink even though its leading edge rises.
        before = raw[max(0, peak - int(0.7 * rate)):
                     max(0, peak - int(0.45 * rate)), leads]
        if len(before) > 2:
            early = np.median(before, axis=0)
            post_levels = []
            for start, end in ((0.18, 0.26), (0.5, 0.6), (0.8, 0.95)):
                after = raw[peak + int(start * rate):
                            peak + int(end * rate), leads]
                if len(after) > 2:
                    post_levels.append(np.median(after, axis=0))
            if post_levels and min(np.median(level - early)
                                   for level in post_levels) > max(28, 0.75 * height):
                return False
        return True
