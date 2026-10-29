"""Frequency selection and offline rhythm analysis from synthetic EEG."""

import importlib.util
import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python_ads1299"))

from history_replay import alpha_activity_mask
from rhythm_analysis import analyze_rhythms, parse_ranges, read_analysis_csv


NAMES = ["O1", "O2", "C3", "C4", "PZ", "F5", "F6", "AF7"]


class RhythmAnalysisTests(unittest.TestCase):
    def test_configurable_frequency_band_changes_detection(self):
        fs = 250
        t = np.arange(8 * fs) / fs
        ten_hz = 20 * np.sin(2 * np.pi * 10 * t)
        eighteen_hz = 20 * np.sin(2 * np.pi * 18 * t)
        self.assertGreater(np.mean(alpha_activity_mask(ten_hz, fs)), 0.7)
        self.assertFalse(np.any(alpha_activity_mask(ten_hz, fs, band=(17, 19))))
        self.assertGreater(np.mean(alpha_activity_mask(eighteen_hz, fs, band=(17, 19))), 0.7)
        with self.assertRaises(ValueError):
            alpha_activity_mask(ten_hz, fs, band=(13, 9))

    def test_alpha_berger_and_mu_erd_with_dc_offset_and_artifact(self):
        fs = 250
        t = np.arange(40 * fs) / fs
        data = np.zeros((8, len(t)))
        data[0] = 10000 + np.where(t < 20, 35, 5) * np.sin(2 * np.pi * 10 * t)
        data[2] = 15000 + np.where(t < 20, 30, 4) * np.sin(2 * np.pi * 11 * t)
        for index in (1, 3, 4, 5, 6, 7):
            data[index] = 8000 + 4 * np.sin(2 * np.pi * 18 * t)
        data[:, 25 * fs:26 * fs] += 500 * np.sin(2 * np.pi * 2 * t[:fs])

        result = analyze_rhythms(
            data, fs, NAMES,
            segments={"closed": "0-20", "open": "20-40", "rest": "0-20",
                      "move": "20-40", "post": "30-40", "eyesopen": "20-40"},
        )
        self.assertAlmostEqual(result["bad_fraction"], 1 / 40)
        self.assertGreater(result["good_seconds"], 35)
        self.assertEqual(result["alpha"][0]["verdict"], "检出 α")
        self.assertGreater(result["alpha"][0]["berger_pct"], 50)
        self.assertEqual(result["alpha"][0]["berger_verdict"], "典型阻断")
        self.assertLess(result["mu"][2]["erd_pct"], -40)
        self.assertEqual(result["mu"][2]["verdict"], "显著 ERD")
        self.assertTrue(np.all(np.isfinite(result["psds"])))

    def test_ranges_and_short_data_are_rejected(self):
        self.assertEqual(parse_ranges("0-5,4-7", 10), [(0.0, 7.0)])
        with self.assertRaises(ValueError):
            parse_ranges("7-12", 10)
        with self.assertRaises(ValueError):
            analyze_rhythms(np.zeros((8, 250)), 250, NAMES)

    def test_generic_csv_and_flat_signal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "custom_250sps_raw.csv"
            with path.open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.writer(stream)
                writer.writerow(("time", "O1", "C3", "event"))
                writer.writerows((index / 250, 0, 0, "rest") for index in range(250 * 5))
            names, data, linked = read_analysis_csv(path, NAMES)
            self.assertEqual(names, ["O1", "C3"])
            self.assertFalse(linked)
            result = analyze_rhythms(data, 250, names, mode="alpha")
            self.assertEqual(result["alpha"][0]["verdict"], "未检出")
            self.assertIsNone(result["spectrum"][0]["peak_hz"])


class RhythmDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt5.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])
        name = "main_uart_rhythm_test_import"
        spec = importlib.util.spec_from_file_location(
            name, ROOT / "python_ads1299" / "main_uart_v1.2.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        cls.MainWindow = module.MainWindow

    def test_window_runs_current_data_and_updates_channel_name(self):
        fs = 250
        t = np.arange(10 * fs) / fs
        window = self.MainWindow()
        try:
            for channel in window.AdsData.emg_data:
                channel.extend((15 * np.sin(2 * np.pi * 10 * t)).tolist())
            window.open_rhythm_analysis()
            dialog = window.rhythm_analysis_dialog
            dialog.run_analysis()
            self.assertIsNotNone(dialog.result)
            self.assertEqual(dialog.spectrum_table.rowCount(), 8)
            self.assertTrue(dialog.export_button.isEnabled())
            dialog.rename_channel(7, "NewAF7")
            self.assertEqual(dialog.spectrum_table.item(7, 0).text(), "NewAF7")
            with tempfile.TemporaryDirectory() as directory, patch(
                "rhythm_analysis_dialog.QFileDialog.getExistingDirectory", return_value=directory,
            ):
                dialog.export_report()
                self.assertEqual(len(list(Path(directory).glob("*_spectrum.csv"))), 1)
                self.assertEqual(len(list(Path(directory).glob("*_spectrum.png"))), 1)
            screenshot = os.environ.get("EEG_ANALYSIS_SCREENSHOT")
            if screenshot:
                dialog.tabs.setCurrentIndex(1)
                self.app.processEvents()
                self.assertTrue(dialog.grab().save(screenshot))
            dialog.threshold_spin.setValue(250)
            self.assertIsNone(dialog.result)
            self.assertFalse(dialog.export_button.isEnabled())
        finally:
            self.app.processEvents()
            window.close()
            self.app.processEvents()

    def test_main_frequency_control_refreshes_history_detection(self):
        fs = 250
        t = np.arange(8 * fs) / fs
        window = self.MainWindow()
        try:
            window.history_data = [20 * np.sin(2 * np.pi * 18 * t) for _ in range(8)]
            window.history_rate = fs
            window.is_history_mode = True
            window.history_slider.setRange(0, 80)
            window.history_slider.setValue(80)
            window._render_history()
            self.assertFalse(np.isfinite(window.highlight_lines[0].get_ydata()).any())
            window.band_low.setValue(17)
            self.assertEqual(window.detection_band, (17.0, 21.0))
            window.band_high.setValue(19)
            self.assertIn("17–19 Hz", window.alpha_status.text())
            self.assertTrue(np.isfinite(window.highlight_lines[0].get_ydata()).any())
        finally:
            self.app.processEvents()
            window.close()
            self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
