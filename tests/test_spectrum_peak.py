"""Checks for the live peak readout on the EEG power spectrum."""

import importlib.util
import os
import re
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python_ads1299"))


class SpectrumPeakTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt5.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])
        module_path = ROOT / "python_ads1299" / "main_uart_v1.2.py"
        name = "main_uart_spectrum_peak_test_import"
        spec = importlib.util.spec_from_file_location(name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        cls.MainWindow = module.MainWindow

    @staticmethod
    def _read_peak_text(annotation):
        text = annotation.get_text()
        frequency = re.search(r"([+-]?\d+(?:\.\d+)?)\s*Hz", text, re.IGNORECASE)
        power = re.search(r"([+-]?\d+(?:\.\d+)?)\s*dB", text, re.IGNORECASE)
        if frequency is None or power is None:
            raise AssertionError(f"Peak label needs frequency in Hz and power in dB: {text!r}")
        return float(frequency.group(1)), float(power.group(1))

    def test_peak_follows_strongest_enabled_channel_and_clears_without_data(self):
        rate = 250
        time = np.arange(4 * rate) / rate
        frequencies = [5, 7, 12, 16, 20, 22, 27, 30]
        amplitudes = [2, 3, 32, 2, 2, 20, 2, 2]
        channels = [
            (amplitude * np.sin(2 * np.pi * frequency * time)).tolist()
            for frequency, amplitude in zip(frequencies, amplitudes)
        ]

        window = self.MainWindow()
        try:
            window._update_spectrum(channels, rate)
            annotation = window.spectrum_peak_annotation
            marker = window.spectrum_peak_marker
            self.assertTrue(annotation.get_visible())
            self.assertTrue(marker.get_visible())
            peak_frequency, peak_db = self._read_peak_text(annotation)
            self.assertAlmostEqual(peak_frequency, 12, delta=0.3)
            self.assertAlmostEqual(
                peak_db,
                np.max(window.spectrum_lines[2].get_ydata()),
                delta=0.2,
            )

            window.toggle_channel(False, 2)
            self.assertTrue(annotation.get_visible())
            self.assertTrue(marker.get_visible())
            peak_frequency, peak_db = self._read_peak_text(annotation)
            self.assertAlmostEqual(peak_frequency, 22, delta=0.3)
            self.assertAlmostEqual(
                peak_db,
                np.max(window.spectrum_lines[5].get_ydata()),
                delta=0.2,
            )

            window._update_spectrum([[] for _ in range(8)], rate)
            self.assertFalse(annotation.get_visible())
            self.assertFalse(marker.get_visible())
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
