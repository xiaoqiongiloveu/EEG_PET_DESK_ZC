import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python_ads1299"))
from eye_blink import EyeBlinkDetector


RATE = 250


def samples(seconds=6.0, blink_times=(), amplitudes=None, noise=0.8):
    rng = np.random.default_rng(21)
    data = rng.normal(0.0, noise, (int(seconds * RATE), 8))
    if amplitudes is None:
        amplitudes = (0, 0, 0, 0, 0, 120, 150, 110)
    time = np.arange(len(data)) / RATE
    for blink_time in blink_times:
        pulse = np.exp(-0.5 * ((time - blink_time) / 0.075) ** 2)
        data += pulse[:, None] * np.asarray(amplitudes)
    return data


class EyeBlinkTests(unittest.TestCase):
    def test_counts_completed_front_pulses_once_across_arbitrary_chunks(self):
        data = samples(blink_times=(2.0, 3.5))
        detector = EyeBlinkDetector()
        sizes = (123, 11, 400, 1, 77, 200, 9, 679)
        offset = 0
        count = 0
        for size in sizes:
            count += detector.process(data[offset:offset + size], RATE)
            offset += size
        self.assertEqual(offset, len(data))
        self.assertEqual(count, 2)
        self.assertEqual(detector.process(np.empty((0, 8)), RATE), 0)
        self.assertEqual(detector.process([], RATE), 0)

    def test_rejects_global_and_single_channel_artifacts(self):
        for amplitudes in (
            (130,) * 8,
            (0, 0, 0, 0, 0, 0, 160, 0),
        ):
            with self.subTest(amplitudes=amplitudes):
                detector = EyeBlinkDetector()
                self.assertEqual(
                    detector.process(samples(blink_times=(2.0,), amplitudes=amplitudes), RATE),
                    0,
                )

    def test_counts_fallback_frontal_pulses_with_two_disconnected_leads(self):
        data = samples(
            blink_times=(2.0, 3.5),
            amplitudes=(100, 0, 130, 0, 110, 0, 0, 95),
        )
        data[:, 5:7] = 0.0
        data[2 * RATE, 5] = 4000.0
        data[int(3.5 * RATE), 6] = -4000.0
        data[:, [1, 3]] = 0.0
        detector = EyeBlinkDetector()
        count = sum(
            detector.process(data[index:index + 37], RATE)
            for index in range(0, len(data), 37)
        )
        self.assertEqual(count, 2)

    def test_counts_blink_shared_across_six_frontal_leads(self):
        data = samples(
            blink_times=(2.0,),
            amplitudes=(100, 0, 130, 0, 110, 120, 150, 95),
        )
        data[:, [1, 3]] = 0.0
        self.assertEqual(EyeBlinkDetector().process(data, RATE), 1)

    def test_rejects_uniform_pulse_with_only_four_usable_leads(self):
        data = samples(blink_times=(2.0,), amplitudes=(120, 0, 120, 0, 120, 0, 0, 120))
        data[:, [1, 3, 5, 6]] = 0.0
        self.assertEqual(EyeBlinkDetector().process(data, RATE), 0)

    def test_counts_separate_nearby_pulses_once_each(self):
        data = samples(blink_times=(2.0, 2.65, 3.3))
        self.assertEqual(EyeBlinkDetector().process(data, RATE), 3)

        broad = samples()
        time = np.arange(len(broad)) / RATE
        crest = np.exp(-0.5 * ((time - 2.0) / 0.16) ** 2)
        broad += crest[:, None] * np.asarray((0, 0, 0, 0, 0, 120, 150, 110))
        self.assertEqual(EyeBlinkDetector().process(broad, RATE), 1)

    def test_counts_with_persistent_missing_primary_leads(self):
        data = samples(
            blink_times=(2.0,),
            amplitudes=(90, 0, 120, 0, 100, 0, 0, 80),
        )
        data[:, 5:7] = np.nan
        self.assertEqual(EyeBlinkDetector().process(data, RATE), 1)

    def test_ignores_single_overloaded_lead_during_valid_blink(self):
        data = samples(blink_times=(2.0,))
        data[2 * RATE, 5] = 10_000.0
        self.assertEqual(EyeBlinkDetector().process(data, RATE), 1)

        data = samples(blink_times=(2.0,))
        data[2 * RATE, 5] = 187_500.0
        self.assertEqual(EyeBlinkDetector().process(data, RATE), 1)

    def test_large_smooth_fallback_pulses_remain_countable(self):
        data = samples(
            blink_times=(2.0, 2.65, 3.3),
            amplitudes=(3000, 0, 3900, 0, 3300, 0, 0, 2850),
        )
        data[:, [1, 3, 5, 6]] = 0.0
        self.assertEqual(EyeBlinkDetector().process(data, RATE), 3)

    def test_counts_broad_blinks_on_four_connected_leads_without_filter_tail_duplicates(self):
        for amplitude in (100, 300, 3000):
            for width in (0.25, 0.35, 0.45):
                with self.subTest(amplitude=amplitude, width=width):
                    data = samples(seconds=8.0, amplitudes=(0,) * 8)
                    data[:, [1, 3, 5, 6]] = 0.0
                    time = np.arange(len(data)) / RATE
                    pulse = np.exp(-0.5 * ((time - 3.0) / width) ** 2)
                    data += pulse[:, None] * np.asarray(
                        (amplitude, 0, 1.3 * amplitude, 0,
                         1.1 * amplitude, 0, 0, 0.95 * amplitude)
                    )
                    detector = EyeBlinkDetector()
                    count = sum(
                        detector.process(data[index:index + 73], RATE)
                        for index in range(0, len(data), 73)
                    )
                    self.assertEqual(count, 1)

    def test_counts_two_separate_broad_blinks_on_four_connected_leads(self):
        data = samples(seconds=8.0, amplitudes=(0,) * 8)
        data[:, [1, 3, 5, 6]] = 0.0
        time = np.arange(len(data)) / RATE
        for center in (2.5, 3.7):
            pulse = np.exp(-0.5 * ((time - center) / 0.25) ** 2)
            data += pulse[:, None] * np.asarray((300, 0, 390, 0, 330, 0, 0, 285))
        self.assertEqual(EyeBlinkDetector().process(data, RATE), 2)

    def test_rejects_simultaneous_one_sample_spikes_and_global_motion(self):
        spikes = samples()
        spikes[2 * RATE, [0, 2, 4, 7]] += [300, 350, 320, 280]
        self.assertEqual(EyeBlinkDetector().process(spikes, RATE), 0)

        movement = samples()
        movement[2 * RATE:3 * RATE] += np.asarray(
            (115, 120, 125, 118, 130, 110, 120, 125)
        )
        self.assertEqual(EyeBlinkDetector().process(movement, RATE), 0)

    def test_rejects_sustained_wave_and_nonfinite_sample(self):
        detector = EyeBlinkDetector()
        data = samples()
        time = np.arange(len(data)) / RATE
        data[:, 5:8] += (time >= 2.0)[:, None] * 100.0
        self.assertEqual(detector.process(data, RATE), 0)

        detector.reset()
        data = samples(blink_times=(2.0, 4.0))
        data[4 * RATE, 6] = np.nan
        self.assertEqual(detector.process(data, RATE), 1)

    def test_sustained_alpha_activity_is_not_a_blink(self):
        data = samples()
        time = np.arange(len(data)) / RATE
        data[:, 5:8] += 90 * np.sin(2 * np.pi * 10 * time)[:, None]
        detector = EyeBlinkDetector()
        self.assertEqual(detector.process(data, RATE), 0)

    def test_rejects_brief_frontal_50hz_bursts_with_envelope_shift(self):
        time = np.arange(8 * RATE) / RATE
        data = np.random.default_rng(22).normal(0, 4, (len(time), 8))
        data[:, [1, 3, 4, 5]] = 0.0
        leads = np.asarray((1, 0, 1.2, 0, 0, 0, 1.1, 1.15))
        for center in (2.0, 3.2, 4.4, 5.6):
            envelope = np.exp(-0.5 * ((time - center) / 0.15) ** 2)
            interference = envelope * (
                30 + 110 * np.sin(2 * np.pi * 50 * (time - center))
            )
            data += interference[:, None] * leads

        with_blink = data.copy()
        blink = np.exp(-0.5 * ((time - 6.7) / 0.08) ** 2)
        with_blink += blink[:, None] * np.asarray(
            (120, 0, 150, 0, 0, 0, 60, 110)
        )

        for chunk_size in (len(data), 37):
            with self.subTest(chunk_size=chunk_size):
                detector = EyeBlinkDetector()
                count = sum(
                    detector.process(data[index:index + chunk_size], RATE)
                    for index in range(0, len(data), chunk_size)
                )
                self.assertEqual(count, 0)

                detector = EyeBlinkDetector()
                count = sum(
                    detector.process(with_blink[index:index + chunk_size], RATE)
                    for index in range(0, len(with_blink), chunk_size)
                )
                self.assertEqual(count, 1)

    def test_rate_change_resets_partial_event(self):
        detector = EyeBlinkDetector()
        data = samples(blink_times=(2.0,))
        self.assertEqual(detector.process(data[:2 * RATE], RATE), 0)
        self.assertEqual(detector.process(np.zeros((250, 8)), 500), 0)
        self.assertEqual(detector.process(np.empty((0, 8)), 500), 0)

    def test_rejects_invalid_shape_and_rate(self):
        detector = EyeBlinkDetector()
        with self.assertRaises(ValueError):
            detector.process(np.zeros((100, 7)), RATE)
        with self.assertRaises(ValueError):
            detector.process(np.zeros((100, 8)), 0)


if __name__ == "__main__":
    unittest.main()
