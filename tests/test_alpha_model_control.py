"""Signal and streaming checks for the calibrated alpha model controller."""

from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python_ads1299"))

from alpha_model_control import AlphaModelController


class AlphaModelControllerTests(unittest.TestCase):
    RATE = 250

    def signal(self, frequency=12.0, amplitude=20.0, rate=None):
        rate = self.RATE if rate is None else rate
        time = np.arange(2 * rate) / rate
        return amplitude * np.sin(2.0 * np.pi * frequency * time)

    def calibrate(self, controller, signal=None, step=25):
        signal = self.signal() if signal is None else signal
        controller.start_calibration()
        for count in range(500, 1751, step):
            controller.update([signal], self.RATE, count, [True])
        self.assertEqual(controller.status, "running")

    def test_single_sided_psd_measures_alpha_power_in_microvolts_squared(self):
        controller = AlphaModelController()
        alpha = self.signal()
        controller.update([alpha], self.RATE, 500, [True])
        alpha_db = controller.band_db
        # A sine wave with 20 uV peak amplitude has 200 uV^2 mean power.
        self.assertAlmostEqual(alpha_db, 10.0 * np.log10(200.0), places=2)
        controller.update([self.signal(20.0)], self.RATE, 525, [True])
        self.assertGreater(alpha_db - controller.band_db, 50.0)
        controller.update([alpha + 4000.0], self.RATE, 550, [True])
        self.assertAlmostEqual(controller.band_db, alpha_db, places=8)

    def test_calibrated_amplitude_moves_model_right_and_left(self):
        controller = AlphaModelController()
        self.calibrate(controller)
        self.assertAlmostEqual(controller.position, 0.0)
        for count in range(1775, 2501, 25):
            controller.update([self.signal(amplitude=40.0)], self.RATE, count, [True])
        self.assertGreater(controller.position, 0.97)
        self.assertAlmostEqual(controller.offset_db, 6.02, delta=0.15)
        for count in range(2525, 3501, 25):
            controller.update([self.signal(amplitude=10.0)], self.RATE, count, [True])
        self.assertLess(controller.position, -0.97)

    def test_new_sample_time_controls_calibration_even_with_repeated_calls(self):
        signal = self.signal()
        controller = AlphaModelController()
        controller.start_calibration()
        controller.update([signal], self.RATE, 500, [True])
        for _ in range(100):
            self.assertFalse(controller.update([signal], self.RATE, 500, [True]))
        self.assertEqual(controller.calibration_progress, 0.0)
        controller.update([signal], self.RATE, 750, [True])
        self.assertAlmostEqual(controller.calibration_progress, 0.2)
        controller.update([signal], self.RATE, 1000, [True])
        controller.update([signal], self.RATE, 1250, [True])
        controller.update([signal], self.RATE, 1500, [True])
        self.assertEqual(controller.status, "calibrating")
        controller.update([signal], self.RATE, 1750, [True])
        self.assertEqual(controller.status, "running")
        self.assertEqual(controller.calibration_progress, 1.0)
        fine = AlphaModelController()
        self.calibrate(fine, signal)
        self.assertAlmostEqual(controller.baseline_db, fine.baseline_db)

    def test_pause_excludes_elapsed_samples_and_restores_previous_state(self):
        controller = AlphaModelController()
        signal = self.signal()
        controller.start_calibration()
        controller.update([signal], self.RATE, 500, [True])
        controller.update([signal], self.RATE, 750, [True])
        controller.pause()
        for count in range(775, 2501, 25):
            self.assertFalse(controller.update([signal], self.RATE, count, [True]))
        self.assertEqual(controller.calibration_progress, 0.2)
        controller.resume()
        self.assertEqual(controller.status, "calibrating")
        controller.update([signal], self.RATE, 2525, [True])
        self.assertEqual(controller.calibration_progress, 0.2)
        for count in range(2550, 3526, 25):
            controller.update([signal], self.RATE, count, [True])
        self.assertEqual(controller.status, "running")
        baseline = controller.baseline_db
        controller.pause()
        controller.update([signal], self.RATE, 5000, [True])
        controller.resume()
        self.assertEqual(controller.status, "running")
        self.assertEqual(controller.baseline_db, baseline)

    def test_rate_count_rollback_and_channel_change_discard_baseline(self):
        for change in ("rate", "count", "channels"):
            with self.subTest(change=change):
                controller = AlphaModelController()
                self.calibrate(controller)
                if change == "rate":
                    controller.update([self.signal(rate=500)], 500, 1775, [True])
                elif change == "count":
                    controller.update([self.signal()], self.RATE, 10, [True])
                else:
                    controller.update([self.signal(), self.signal()], self.RATE, 1775, [True, True])
                self.assertEqual(controller.status, "waiting")
                self.assertIsNone(controller.baseline_db)
                self.assertEqual(controller.position, 0.0)

    def test_invalid_majority_freezes_calibration_and_does_not_credit_gap(self):
        controller = AlphaModelController()
        signal = self.signal()
        controller.start_calibration()
        controller.update([signal] * 3, self.RATE, 500, [True] * 3)
        controller.update([signal] * 3, self.RATE, 750, [True] * 3)
        noisy = self.signal(amplitude=300.0)
        nonfinite = signal.copy()
        nonfinite[-1] = np.nan
        for count in range(775, 2501, 25):
            controller.update([signal, noisy, nonfinite], self.RATE, count, [True] * 3)
        self.assertFalse(controller.signal_valid)
        self.assertAlmostEqual(controller.calibration_progress, 0.2)
        controller.update([signal] * 3, self.RATE, 2525, [True] * 3)
        self.assertTrue(controller.signal_valid)
        self.assertAlmostEqual(controller.calibration_progress, 0.2)
        controller.update([signal] * 3, self.RATE, 2550, [True] * 3)
        self.assertAlmostEqual(controller.calibration_progress, 0.22)

    def test_valid_channel_median_ignores_single_bad_channel(self):
        controller = AlphaModelController()
        signal = self.signal()
        channels = [signal, signal * 2.0, np.zeros_like(signal)]
        controller.update(channels, self.RATE, 500, [True] * 3)
        self.assertTrue(controller.signal_valid)
        expected = (10.0 * np.log10(200.0) + 10.0 * np.log10(800.0)) / 2.0
        self.assertAlmostEqual(controller.band_db, expected, places=2)

    def test_empty_flat_nonfinite_and_unobserved_time_are_safe(self):
        controller = AlphaModelController()
        controller.start_calibration()
        for count, channels in enumerate(([], [[]], [np.zeros(500)], [np.full(500, np.inf)])):
            controller.update(channels, self.RATE, count, [True])
            self.assertFalse(controller.signal_valid)
            self.assertEqual(controller.position, 0.0)
        signal = self.signal()
        controller.update([signal], self.RATE, 500, [True])
        controller.update([signal], self.RATE, 10000, [True])
        self.assertEqual(controller.calibration_progress, 0.0)
        self.assertEqual(controller.status, "calibrating")


if __name__ == "__main__":
    unittest.main()
