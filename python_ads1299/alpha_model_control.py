"""A calibrated, one-dimensional model controller driven by 9-13 Hz power.

The input is raw microvolts, independently of the waveform display filters.
This measures a frequency band; it does not classify thoughts or eye states.
"""

from __future__ import annotations

import math

import numpy as np


class AlphaModelController:
    WINDOW_SECONDS = 2.0
    CALIBRATION_SECONDS = 5.0
    SMOOTHING_SECONDS = 0.7
    POSITION_RANGE_DB = 6.0
    MAX_PEAK_TO_PEAK_UV = 500.0
    MIN_STD_UV = 0.1

    def __init__(self):
        self.reset()

    def reset(self):
        """Discard the calibration and return the model to its center."""
        self.status = "waiting"
        self.band_db = None
        self.baseline_db = None
        self.offset_db = None
        self.position = 0.0
        self.calibration_progress = 0.0
        self.signal_valid = False
        self._sample_rate = None
        self._last_sample_count = None
        self._last_valid_sample_count = None
        self._enabled_channels = None
        self._calibration_samples = 0
        self._calibration_values = []
        self._smoothed_db = None
        self._resume_status = "waiting"

    def start_calibration(self):
        """Collect five seconds of fresh, valid band-power observations."""
        # Keep the source identity so a subsequent rate/channel change can
        # invalidate this calibration, including while the model is paused.
        rate = self._sample_rate
        count = self._last_sample_count
        enabled = self._enabled_channels
        self.reset()
        self._sample_rate = rate
        self._last_sample_count = count
        self._enabled_channels = enabled
        self.status = "calibrating"

    def pause(self):
        if self.status != "paused":
            self._resume_status = self.status
            self.status = "paused"
            self._last_valid_sample_count = None

    def resume(self):
        if self.status == "paused":
            self.status = self._resume_status
            self._last_valid_sample_count = None

    @staticmethod
    def _channel_indices(enabled_channels, channel_count):
        if enabled_channels is None:
            return tuple(range(channel_count))
        enabled = list(enabled_channels)
        if all(isinstance(value, (bool, np.bool_)) for value in enabled):
            return tuple(index for index, value in enumerate(enabled) if value)
        return tuple(sorted(set(int(index) for index in enabled)))

    def _invalidate_signal(self):
        self.signal_valid = False
        self.band_db = None
        self.offset_db = None
        self.position = 0.0
        self._last_valid_sample_count = None
        self._smoothed_db = None

    def _power_db(self, channels, rate, indices):
        window_size = int(round(self.WINDOW_SECONDS * rate))
        powers = []
        window = np.hanning(window_size)
        frequencies = np.fft.rfftfreq(window_size, 1.0 / rate)
        target = (frequencies >= 9.0) & (frequencies <= 13.0)
        for index in indices:
            if not 0 <= index < len(channels):
                continue
            values = np.asarray(channels[index], dtype=np.float64)
            if values.ndim != 1 or values.size < window_size:
                continue
            values = values[-window_size:]
            if not np.all(np.isfinite(values)):
                continue
            values = values - np.median(values)
            if np.ptp(values) > self.MAX_PEAK_TO_PEAK_UV:
                continue
            if np.std(values) < self.MIN_STD_UV:
                continue
            values = values - np.mean(values)
            spectrum = np.fft.rfft(values * window)
            psd = np.abs(spectrum) ** 2 / (rate * np.sum(window ** 2))
            # A real-valued signal has equal energy at positive and negative
            # frequencies. Keep DC and the even-length Nyquist bin undoubled.
            if window_size % 2 == 0:
                psd[1:-1] *= 2.0
            else:
                psd[1:] *= 2.0
            band_power = float(np.sum(psd[target]) * rate / window_size)
            powers.append(10.0 * math.log10(max(band_power, 1e-12)))

        # A majority of invalid selected channels invalidates the observation.
        if not indices or len(powers) * 2 < len(indices):
            return None
        return float(np.median(powers))

    @staticmethod
    def _weighted_median(observations):
        values = sorted(observations)
        midpoint = sum(weight for _value, weight in values) / 2.0
        cumulative = 0
        for value, weight in values:
            cumulative += weight
            if cumulative >= midpoint:
                return float(value)
        return float(values[-1][0])

    def update(self, channels, rate, sample_count, enabled_channels=None):
        """Process fresh samples and return whether the observable state changed.

        ``channels`` may contain only a rolling two-second buffer. The separate
        ``sample_count`` is the cumulative number of acquired samples, never the
        number of UI refreshes. ``enabled_channels`` accepts a boolean mask or
        channel indices. Source changes discard the baseline and require a new
        explicit calibration.
        """
        try:
            rate = float(rate)
            sample_count = int(sample_count)
            indices = self._channel_indices(enabled_channels, len(channels))
        except (TypeError, ValueError, OverflowError):
            self._invalidate_signal()
            return False
        if not np.isfinite(rate) or rate <= 26.0 or sample_count < 0:
            self._invalidate_signal()
            return False

        source_changed = (
            self._sample_rate is not None and rate != self._sample_rate
        ) or (
            self._last_sample_count is not None
            and sample_count < self._last_sample_count
        ) or (
            self._enabled_channels is not None
            and indices != self._enabled_channels
        )
        if source_changed:
            self.reset()
        self._sample_rate = rate
        self._enabled_channels = indices
        if not source_changed and sample_count == self._last_sample_count:
            return False
        self._last_sample_count = sample_count
        if source_changed or self.status == "paused":
            return source_changed

        try:
            power_db = self._power_db(channels, rate, indices)
        except (TypeError, ValueError, OverflowError):
            power_db = None
        if power_db is None:
            self._invalidate_signal()
            return True

        self.signal_valid = True
        self.band_db = power_db
        previous_count = self._last_valid_sample_count
        self._last_valid_sample_count = sample_count
        new_samples = 0 if previous_count is None else sample_count - previous_count
        # A rolling buffer cannot validate a gap larger than its own duration.
        # Anchor again instead of crediting unseen time to calibration or EMA.
        if new_samples > int(round(rate * self.WINDOW_SECONDS)):
            new_samples = 0

        if self.status == "calibrating":
            needed = max(1, int(round(rate * self.CALIBRATION_SECONDS)))
            weight = min(new_samples, needed - self._calibration_samples)
            if weight > 0:
                self._calibration_values.append((power_db, weight))
                self._calibration_samples += weight
            self.calibration_progress = min(1.0, self._calibration_samples / needed)
            if self._calibration_samples >= needed:
                self.baseline_db = self._weighted_median(self._calibration_values)
                self._smoothed_db = self.baseline_db
                self.offset_db = 0.0
                self.position = 0.0
                self.status = "running"
            return True

        if self.status == "running" and self.baseline_db is not None:
            if self._smoothed_db is None:
                self._smoothed_db = power_db
            elif new_samples > 0:
                alpha = -math.expm1(-new_samples / (rate * self.SMOOTHING_SECONDS))
                self._smoothed_db += alpha * (power_db - self._smoothed_db)
            self.offset_db = self._smoothed_db - self.baseline_db
            self.position = float(np.clip(
                self.offset_db / self.POSITION_RANGE_DB, -1.0, 1.0
            ))
        return True
