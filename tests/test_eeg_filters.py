import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python_ads1299"))
from eeg_filters import ConfigurableEEGFilter


RATE = 250


def tone_amplitude(signal, frequency, rate=RATE):
    time = np.arange(signal.size) / rate
    phase = 2 * np.pi * frequency * time
    return 2 * np.hypot(
        np.mean(signal * np.sin(phase)),
        np.mean(signal * np.cos(phase)),
    )


class ConfigurableEEGFilterTests(unittest.TestCase):
    def test_independent_dc_removal_preserves_eeg_and_high_frequencies(self):
        time = np.arange(24 * RATE) / RATE
        data = np.tile(
            100.0 + 30 * np.sin(2 * np.pi * 10 * time)
            + 12 * np.sin(2 * np.pi * 80 * time),
            (8, 1),
        )
        eeg_filter = ConfigurableEEGFilter()
        eeg_filter.configure(RATE, 0.1, 100.0, False, False, True)
        result = eeg_filter.data_filter(RATE, data)

        settled = result[0, -5 * RATE:]
        self.assertLess(abs(np.mean(settled)), 5.0)
        self.assertAlmostEqual(tone_amplitude(settled, 10), 30.0, delta=1.5)
        self.assertAlmostEqual(tone_amplitude(settled, 80), 12.0, delta=0.6)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_dc_only_with_constant_input_settles_to_zero(self):
        data = np.full((8, 24 * RATE), 125.0)
        eeg_filter = ConfigurableEEGFilter()
        eeg_filter.configure(RATE, 0.1, 100.0, False, False, True)
        result = eeg_filter.data_filter(RATE, data)

        self.assertLess(np.max(np.abs(result[:, -RATE:])), 5.0)

    def test_streaming_chunks_match_single_block_for_filter_combinations(self):
        rng = np.random.default_rng(42)
        time = np.arange(7 * RATE) / RATE
        data = rng.normal(0, 3, (8, time.size))
        data += 45 + 18 * np.sin(2 * np.pi * 10 * time)
        data += 9 * np.sin(2 * np.pi * 50 * time)
        chunk_sizes = (1, 37, 8, 250, 3, 415, 1, 1000)

        for bandpass, notch in ((False, False), (False, True), (True, True)):
            with self.subTest(bandpass=bandpass, notch=notch):
                full_filter = ConfigurableEEGFilter()
                full_filter.configure(RATE, 0.1, 100.0, bandpass, notch, True)
                expected = full_filter.data_filter(RATE, data)

                stream_filter = ConfigurableEEGFilter()
                stream_filter.configure(RATE, 0.1, 100.0, bandpass, notch, True)
                chunks = []
                offset = 0
                for size in chunk_sizes:
                    chunks.append(stream_filter.data_filter(RATE, data[:, offset:offset + size]))
                    offset += size
                    self.assertEqual(
                        stream_filter.data_filter(RATE, data[:, :0]).shape, (8, 0)
                    )
                chunks.append(stream_filter.data_filter(RATE, data[:, offset:]))
                actual = np.concatenate(chunks, axis=1)
                np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-8)

    def test_dc_setting_can_be_disabled_without_changing_legacy_filter_call(self):
        data = np.full((8, RATE), 90.0)
        eeg_filter = ConfigurableEEGFilter()
        eeg_filter.configure(RATE, 0.1, 100.0, False, False, True)
        eeg_filter.data_filter(RATE, data)

        eeg_filter.configure(RATE, 0.1, 100.0, False, False)
        np.testing.assert_array_equal(eeg_filter.data_filter(RATE, data), data)


class DcRemovalInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt5.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])
        name = "main_uart_dc_filter_test_import"
        module_path = Path(__file__).resolve().parents[1] / "python_ads1299" / "main_uart_v1.2.py"
        spec = importlib.util.spec_from_file_location(name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)

    def test_dc_notch_toggles_and_history_controls(self):
        from main_uart_dc_filter_test_import import MainWindow

        time = np.arange(12 * RATE) / RATE
        raw = 75 + 20 * np.sin(2 * np.pi * 10 * time)
        raw += 16 * np.sin(2 * np.pi * 50 * time)
        window = MainWindow()
        try:
            window.AdsData.emg_data = [raw.tolist() for _ in range(8)]
            with patch.object(window, "_refresh_all_graphs"):
                window.dc_checkbox.setChecked(True)
                self.assertTrue(window.filter_enabled)
                self.assertFalse(window.btn_filter.isChecked())
                self.assertFalse(window.notch_checkbox.isChecked())
                dc_only = np.asarray(window.filtered_emg_data[0][-5 * RATE:])
                self.assertLess(abs(np.mean(dc_only)), 5.0)
                self.assertAlmostEqual(tone_amplitude(dc_only, 10), 20.0, delta=1.0)

                window.notch_checkbox.setChecked(True)
                with_notch = np.asarray(window.filtered_emg_data[0][-5 * RATE:])
                self.assertLess(tone_amplitude(with_notch, 50), 4.0)

                window.dc_checkbox.setChecked(False)
                self.assertTrue(window.filter_enabled)
                self.assertGreater(np.mean(window.filtered_emg_data[0][-RATE:]), 70.0)

                window.notch_checkbox.setChecked(False)
                self.assertFalse(window.filter_enabled)
                self.assertEqual(window._current_data_source(), window.AdsData.emg_data)

                with patch(
                    "main_uart_dc_filter_test_import.QFileDialog.getOpenFileName",
                    return_value=("test_250sps_raw.csv", ""),
                ), patch(
                    "main_uart_dc_filter_test_import.read_history_csv",
                    return_value=window.AdsData.emg_data,
                ), patch.object(window, "_render_history"):
                    window.open_history()
                self.assertFalse(window.dc_checkbox.isEnabled())
                window.leave_history()
                self.assertTrue(window.dc_checkbox.isEnabled())
        finally:
            window.close()

    def test_first_live_samples_reconfigure_after_collection_reset(self):
        from main_uart_dc_filter_test_import import MainWindow

        time = np.arange(2 * RATE) / RATE
        data = np.tile(90 + 15 * np.sin(2 * np.pi * 10 * time), (8, 1))
        window = MainWindow()
        try:
            with patch.object(window, "_refresh_all_graphs"):
                window.dc_checkbox.setChecked(True)
            window._reset_filter_runtime()
            self.assertIsNone(window.eeg_filter.settings)

            window.AdsData.emg_data = [channel[:125].tolist() for channel in data]
            window._append_new_filtered_data()
            self.assertIsNotNone(window.eeg_filter.settings)
            self.assertEqual(window.filtered_source_length, 125)
            self.assertTrue(all(len(channel) == 125 for channel in window.filtered_emg_data))

            for channel, new_samples in zip(window.AdsData.emg_data, data[:, 125:]):
                channel.extend(new_samples.tolist())
            window._append_new_filtered_data()
            self.assertEqual(window.filtered_source_length, data.shape[1])

            reference = ConfigurableEEGFilter()
            reference.configure(RATE, 0.1, 100.0, False, False, True)
            expected = reference.data_filter(RATE, data)
            np.testing.assert_allclose(window.filtered_emg_data, expected, rtol=0, atol=1e-8)
        finally:
            window.close()

    def test_live_plot_uses_bounded_window_and_keeps_export_data(self):
        from main_uart_dc_filter_test_import import MainWindow

        window = MainWindow()
        try:
            full_length = 20 * RATE
            raw = np.arange(full_length, dtype=float)
            window.AdsData.emg_data = [raw.tolist() for _ in range(8)]
            plot, detection, offset = window._live_plot_sources(
                window.AdsData.emg_data, RATE,
            )
            self.assertEqual(offset, 7 * RATE)
            self.assertEqual(len(plot[0]), 13 * RATE)
            np.testing.assert_array_equal(plot[0], raw[offset:])
            np.testing.assert_array_equal(detection[0], raw[offset:])
            self.assertEqual(len(window.AdsData.emg_data[0]), full_length)

            window.update_graph(0, plot[0], RATE, detection_data=detection[0],
                                sample_offset=offset)
            x_values = window.lines[0].get_xdata()
            self.assertAlmostEqual(x_values[0], 10.0)
            self.assertAlmostEqual(x_values[-1], (full_length - 1) / RATE)
        finally:
            window.close()

    def test_stop_impedance_sends_dedicated_command(self):
        from ads1299 import ADS1298cmd
        from main_uart_dc_filter_test_import import MainWindow

        window = MainWindow()
        try:
            window.serial_port = SimpleNamespace(is_open=True)
            window.is_impedance_testing = True
            with patch.object(window, "send_binary_command", return_value=True) as send:
                window.stop_impedance_test()
            send.assert_called_once_with(
                ADS1298cmd.stop_impedance_cmd(), "停止阻抗检测",
            )
            self.assertFalse(window.is_impedance_testing)
        finally:
            window.serial_port = None
            window.close()


if __name__ == "__main__":
    unittest.main()
