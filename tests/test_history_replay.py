import csv
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python_ads1299"))
from history_replay import alpha_activity_mask, read_history_csv, sample_rate_from_filename


CHANNELS = ["F5", "F3", "FZ", "F4", "F6", "AF8", "FPZ", "AF7"]


class HistoryDataTests(unittest.TestCase):
    def test_frequency_detection_and_window_alignment(self):
        rate = 250
        time = np.arange(rate * 12) / rate
        alpha = 25 * np.sin(2 * np.pi * 10 * time)
        beta = 25 * np.sin(2 * np.pi * 20 * time)
        self.assertGreater(np.mean(alpha_activity_mask(alpha, rate)), 0.9)
        self.assertFalse(np.any(alpha_activity_mask(beta, rate)))
        self.assertFalse(np.any(alpha_activity_mask(np.zeros_like(alpha), rate)))

        full = alpha_activity_mask(alpha, rate)
        clipped = alpha_activity_mask(alpha[500:2500], rate, sample_offset=500)
        self.assertTrue(np.array_equal(full[500:2000], clipped[:1500]))

    def test_csv_loads_export_format_and_rejects_missing_samples(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "20261016_120000_250sps_raw.csv"
            with path.open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.writer(stream)
                writer.writerow(list(reversed(CHANNELS)))
                writer.writerow(list(range(8)))
                writer.writerow(list(range(10, 18)))
            data = read_history_csv(path, CHANNELS)
            self.assertEqual(sample_rate_from_filename(path), 250)
            self.assertEqual(len(data), 8)
            self.assertEqual(data[0].tolist(), [7.0, 17.0])

            with path.open("a", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerow([""] * 7 + ["1"])
            with self.assertRaises(ValueError):
                read_history_csv(path, CHANNELS)


class HistoryWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt5.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])
        name = "main_uart_v1_2_test_import"
        spec = importlib.util.spec_from_file_location(
            name, ROOT / "python_ads1299" / "main_uart_v1.2.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)

    def test_history_navigation_annotation_and_return_to_live(self):
        from main_uart_v1_2_test_import import MainWindow

        rate = 250
        time = np.arange(rate * 12) / rate
        channels = [20 * np.sin(2 * np.pi * 10 * time)]
        channels.extend(20 * np.sin(2 * np.pi * 20 * time) for _ in range(7))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "20261016_120000_250sps_raw.csv"
            with path.open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.writer(stream)
                writer.writerow(CHANNELS)
                writer.writerows(zip(*channels))

            window = MainWindow()
            try:
                class SerialStub:
                    is_open = True
                    in_waiting = 0

                    def __init__(self):
                        self.writes = []

                    def write(self, payload):
                        self.writes.append(payload)

                    def flush(self):
                        pass

                    def close(self):
                        self.is_open = False

                window.serial_port = SerialStub()
                window.is_collecting = True
                with patch("main_uart_v1_2_test_import.QFileDialog.getOpenFileName", return_value=(str(path), "")):
                    window.open_history()
                self.assertTrue(window.is_history_mode)
                self.assertFalse(window.is_collecting)
                self.assertTrue(window.serial_port.writes)
                self.assertEqual(window.history_slider.value(), 100)
                self.assertTrue(np.isfinite(window.highlight_lines[0].get_ydata()).any())
                self.assertFalse(np.isfinite(window.highlight_lines[1].get_ydata()).any())
                screenshot_path = os.environ.get("EEG_TEST_SCREENSHOT")
                if screenshot_path:
                    window.show()
                    self.app.processEvents()
                    self.assertTrue(window.grab().save(screenshot_path))
                window.history_slider.setValue(110)
                self.assertAlmostEqual(window.axes[-1].get_xlim()[1], 11.0)
                window.toggle_history_playback()
                self.assertTrue(window.history_timer.isActive())
                window._advance_history()
                self.assertEqual(window.history_slider.value(), 111)
                window.toggle_history_playback()
                window.axes[0].set_ylim(-50, 50)
                window._on_plot_button_press(SimpleNamespace(
                    button=1, dblclick=True, inaxes=window.axes[0],
                ))
                self.assertEqual(window.axes[0].get_ylim(), (-100.0, 100.0))
                window.leave_history()
                self.assertFalse(window.is_history_mode)
                self.assertFalse(window.history_timer.isActive())
                self.assertTrue(window.btn_filter.isEnabled())
                self.assertTrue(window.btn_start.isEnabled())
            finally:
                window.close()

    def test_alpha_occurrences_survive_redraw_and_reset_after_three_minutes(self):
        from main_uart_v1_2_test_import import MainWindow

        window = MainWindow()
        try:
            first = [True] + [False] * 7
            window._track_alpha_occurrences(first, 500)
            window._update_alpha_status(1)
            self.assertEqual(window.alpha_occurrence_count, 1)
            self.assertEqual(window.alpha_reset_timer.interval(), 180_000)
            self.assertTrue(window.alpha_reset_timer.isActive())

            window._track_alpha_occurrences(first, 500)
            window._track_alpha_occurrences(first, 625)
            self.assertEqual(window.alpha_occurrence_count, 1)
            window._track_alpha_occurrences([False] * 8, 750)
            window._track_alpha_occurrences([True, True] + [False] * 6, 875)
            self.assertEqual(window.alpha_occurrence_count, 3)

            with patch.object(window, "update_graph", return_value=True), patch.object(
                window, "_update_spectrum"
            ):
                window._refresh_all_graphs()
            self.assertEqual(window.alpha_occurrence_count, 3)
            window._clear_waveform_lines()
            self.assertIn("累计3次", window.alpha_status.text())

            window.is_history_mode = True
            window._update_alpha_status(5)
            self.assertEqual(window.alpha_occurrence_count, 3)
            self.assertIn("回访", window.alpha_status.text())
            window.alpha_reset_timer.timeout.emit()
            self.assertEqual(window.alpha_occurrence_count, 0)
            window.is_history_mode = False
            window._update_alpha_status(None)
            self.assertIn("累计0次", window.alpha_status.text())
        finally:
            window.close()

    def test_live_serial_updates_count_only_new_eeg_samples(self):
        from main_uart_v1_2_test_import import MainWindow

        window = MainWindow()
        try:
            window.serial_port = SimpleNamespace(
                is_open=True, in_waiting=1, read=lambda _size: b"x", close=lambda: None,
            )
            window.is_collecting = True
            add_samples = [True]
            active = [True]

            def parse_data(_payload):
                if add_samples[0]:
                    for channel in window.AdsData.emg_data:
                        channel.append(0.0)

            def channel_active(index, *_args, **_kwargs):
                return index == 0 and active[0]

            with patch.object(window.AdsData, "parse_data", side_effect=parse_data), patch.object(
                window, "update_graph", side_effect=channel_active
            ), patch.object(window, "_update_spectrum"):
                window.update_show()
                self.assertEqual(window.alpha_occurrence_count, 1)
                add_samples[0] = False
                active[0] = False
                window.update_show()
                add_samples[0] = True
                active[0] = True
                window.update_show()
                self.assertEqual(window.alpha_occurrence_count, 1)
                active[0] = False
                window.update_show()
                active[0] = True
                window.update_show()
                self.assertEqual(window.alpha_occurrence_count, 2)
                window.is_collecting = False
                active[0] = False
                window.update_show()
                active[0] = True
                window.update_show()
                self.assertEqual(window.alpha_occurrence_count, 2)
        finally:
            window.is_collecting = False
            window.serial_port = None
            window.close()


if __name__ == "__main__":
    unittest.main()
