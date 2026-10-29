"""Serial and Qt lifecycle checks for the live EEG model."""

from contextlib import ExitStack
import importlib.util
import os
from pathlib import Path
import struct
import sys
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python_ads1299"))

from ads1299 import ADC_COUNT_TO_UV, BP_TYPE_CMD_RESP, BP_TYPE_EEG, bp_pack


MODULE_NAME = "main_uart_alpha_model_test_import"


def eeg_frame(values, sequence=0):
    counts = np.rint(np.asarray(values) / ADC_COUNT_TO_UV).astype(np.int32)
    return bp_pack(BP_TYPE_EEG, sequence, b"\0" * 3 + struct.pack("<8i", *counts))


def alpha_samples(start, count, amplitude=20.0, rate=250):
    times = np.arange(start, start + count, dtype=np.float64) / rate
    signal = amplitude * np.sin(2.0 * np.pi * 12.0 * times)
    return np.repeat(signal[:, None], 8, axis=1)


class SerialStub:
    def __init__(self, chunks=()):
        self.chunks = list(chunks)
        self.is_open = True
        self.written = []

    @property
    def in_waiting(self):
        return len(self.chunks[0]) if self.chunks else 0

    def read(self, _size):
        return self.chunks.pop(0)

    def write(self, payload):
        self.written.append(bytes(payload))
        return len(payload)

    def flush(self):
        pass

    def close(self):
        self.is_open = False


class QtModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt5.QtWidgets import QApplication
        from alpha_model_dialog import AlphaModelDialog

        cls.app = QApplication.instance() or QApplication([])
        cls.Dialog = AlphaModelDialog
        if MODULE_NAME not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                MODULE_NAME, ROOT / "python_ads1299" / "main_uart_v1.2.py"
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[MODULE_NAME] = module
            spec.loader.exec_module(module)
        cls.MainWindow = sys.modules[MODULE_NAME].MainWindow

    def make_dialog(self):
        dialog = self.Dialog([f"CH{index + 1}" for index in range(8)])

        def cleanup():
            dialog.close()
            dialog.deleteLater()

        self.addCleanup(cleanup)
        dialog.show()
        dialog.timer.stop()
        dialog.set_stream_active(True)
        return dialog

    def make_window(self):
        window = self.MainWindow()
        window.update_show_timer.stop()

        def cleanup():
            if window.alpha_model_dialog is not None:
                window.alpha_model_dialog.close()
            window.is_collecting = False
            window.serial_port = None
            window.close()
            pyplot = sys.modules[MODULE_NAME].plt
            pyplot.close(window.fig)
            pyplot.close(window.spectrum_canvas.figure)
            window.deleteLater()

        self.addCleanup(cleanup)
        return window

    def open_live_model(self, window):
        window.serial_port = SerialStub()
        window.is_collecting = True
        window.open_alpha_model()
        dialog = window.alpha_model_dialog
        dialog.timer.stop()
        self.assertTrue(dialog.isVisible())
        self.assertTrue(dialog._stream_active)
        return dialog

    def feed(self, dialog, count, amplitude=20.0):
        start = dialog.sample_count
        for offset in range(0, count, 50):
            size = min(50, count - offset)
            dialog.feed_samples(alpha_samples(start + offset, size, amplitude), 250)

    def calibrate(self, dialog):
        dialog.start_calibration()
        self.feed(dialog, 1800)
        self.assertEqual(dialog.controller.status, "running")
        self.assertTrue(dialog.scene.active)

    def update_once(self, window):
        with patch.object(window, "update_graph", return_value=False), patch.object(
            window, "_update_spectrum"
        ), patch.object(window.canvas, "draw_idle"), patch.object(
            window.spectrum_canvas, "draw_idle"
        ):
            window.update_show()


class AlphaModelDialogTests(QtModelTests):
    def test_fresh_raw_samples_calibrate_control_and_pause_without_motion(self):
        with patch("alpha_model_dialog.time.monotonic", return_value=100.0):
            dialog = self.make_dialog()
            self.calibrate(dialog)
            self.assertAlmostEqual(dialog.controller.baseline_db, 10.0 * np.log10(200), places=2)
            self.feed(dialog, 1000, amplitude=40.0)
            self.assertGreater(dialog.controller.position, 0.9)
            dialog._last_tick_time = 99.9
            dialog._tick()
            self.assertGreater(dialog.scene.position, 0.0)

            dialog.toggle_pause()
            held_position = dialog.scene.position
            held_target = dialog.controller.position
            self.feed(dialog, 750, amplitude=10.0)
            dialog._last_tick_time = 99.9
            dialog._tick()
            self.assertEqual(dialog.controller.status, "paused")
            self.assertEqual(dialog.controller.position, held_target)
            self.assertEqual(dialog.scene.position, held_position)
            self.assertFalse(dialog.scene.active)

            dialog.toggle_pause()
            self.feed(dialog, 1000, amplitude=10.0)
            self.assertEqual(dialog.controller.status, "running")
            self.assertTrue(dialog.scene.active)
            self.assertLess(dialog.controller.position, -0.9)

    def test_watchdog_freezes_without_samples_and_recovers_with_fresh_window(self):
        with patch("alpha_model_dialog.time.monotonic", return_value=100.0) as now:
            dialog = self.make_dialog()
            self.calibrate(dialog)
            self.feed(dialog, 750, amplitude=40.0)
            dialog._last_tick_time = 99.9
            dialog._tick()
            held_position = dialog.scene.position
            count = dialog.sample_count
            baseline = dialog.controller.baseline_db

            now.return_value = 102.0
            dialog._tick()
            self.assertTrue(dialog._stale)
            self.assertFalse(dialog.scene.active)
            self.assertEqual(dialog.scene.position, held_position)
            self.assertEqual(dialog.sample_count, count)
            self.assertIn("信号中断", dialog.status_label.text())
            self.assertEqual(len(dialog._buffers[0]), 0)

            self.feed(dialog, 50)
            self.assertFalse(dialog._stale)
            self.assertFalse(dialog.scene.active)
            self.feed(dialog, 450)
            self.assertTrue(dialog.scene.active)
            self.assertEqual(dialog.controller.baseline_db, baseline)

    def test_timeout_during_calibration_does_not_credit_missing_sample_time(self):
        with patch("alpha_model_dialog.time.monotonic", return_value=100.0) as now:
            dialog = self.make_dialog()
            dialog.start_calibration()
            self.feed(dialog, 750)
            progress = dialog.controller.calibration_progress
            self.assertGreater(progress, 0.0)
            self.assertLess(progress, 1.0)
            now.return_value = 103.0
            dialog._check_stream_timeout()
            self.feed(dialog, 500)
            self.assertEqual(dialog.controller.calibration_progress, progress)
            self.feed(dialog, 50)
            self.assertGreater(dialog.controller.calibration_progress, progress)

    def test_closed_dialog_rejects_samples_and_stops_animation(self):
        dialog = self.make_dialog()
        self.feed(dialog, 100)
        count = dialog.sample_count
        dialog.close()
        dialog.feed_samples(alpha_samples(count, 100), 250)
        self.assertFalse(dialog.isVisible())
        self.assertFalse(dialog.timer.isActive())
        self.assertFalse(dialog._stream_active)
        self.assertEqual(dialog.sample_count, count)

    def test_sample_rate_change_discards_baseline_and_rebuilds_analysis_window(self):
        with patch("alpha_model_dialog.time.monotonic", return_value=100.0):
            dialog = self.make_dialog()
            self.calibrate(dialog)
            dialog.feed_samples(alpha_samples(0, 100, rate=500), 500)
            self.assertEqual(dialog.sample_count, 100)
            self.assertIsNone(dialog.controller.baseline_db)
            self.assertEqual(dialog.controller.status, "waiting")
            self.assertFalse(dialog.scene.active)
            self.assertEqual(dialog._buffers[0].maxlen, 1000)
            self.assertEqual(len(dialog._buffers[0]), 100)
            dialog.start_calibration()
            for start in range(0, 3600, 100):
                dialog.feed_samples(alpha_samples(start, 100, rate=500), 500)
            self.assertEqual(dialog.controller.status, "running")
            self.assertTrue(dialog.scene.active)

    def test_reject_hides_dialog_stops_timer_and_rejects_new_samples(self):
        dialog = self.make_dialog()
        dialog.timer.start()
        self.feed(dialog, 100)
        count = dialog.sample_count
        self.assertTrue(dialog.timer.isActive())
        dialog.reject()
        dialog.feed_samples(alpha_samples(count, 100), 250)
        self.assertFalse(dialog.isVisible())
        self.assertFalse(dialog.timer.isActive())
        self.assertFalse(dialog._stream_active)
        self.assertEqual(dialog.sample_count, count)


class AlphaModelMainWindowTests(QtModelTests):
    def test_binary_raw_microvolts_bypass_filters_and_ack_or_redraw_does_not_recount(self):
        window = self.make_window()
        dialog = self.open_live_model(window)
        values = alpha_samples(0, 50)
        window.serial_port.chunks = [
            b"".join(eeg_frame(row, index) for index, row in enumerate(values)),
            bp_pack(BP_TYPE_CMD_RESP, 51, b"\x01\x00"),
        ]
        window.filter_enabled = True
        display_values = [[8888.0] * len(values) for _ in range(8)]
        with ExitStack() as stack:
            stack.enter_context(patch.object(window, "_append_new_filtered_data"))
            stack.enter_context(patch.object(window, "_current_data_source", return_value=display_values))
            delivered = stack.enter_context(patch.object(dialog, "feed_samples", wraps=dialog.feed_samples))
            self.update_once(window)
            self.assertEqual(delivered.call_count, 1)
            raw_samples, rate = delivered.call_args.args
            np.testing.assert_allclose(raw_samples, values, atol=ADC_COUNT_TO_UV)
            self.assertEqual(rate, window.AdsData.sample_rate)
            self.assertEqual(dialog.sample_count, len(values))
            self.assertAlmostEqual(
                window.AdsData.emg_data[0][-1], values[-1, 0], delta=ADC_COUNT_TO_UV,
            )
            self.update_once(window)
            self.update_once(window)
            with patch.object(window, "update_graph", return_value=False), patch.object(
                window, "_update_spectrum"
            ), patch.object(window.canvas, "draw_idle"), patch.object(
                window.spectrum_canvas, "draw_idle"
            ):
                window._refresh_all_graphs()
            self.assertEqual(sum(bool(len(call.args[0])) for call in delivered.call_args_list), 1)
            self.assertEqual(dialog.sample_count, len(values))

    def test_stop_and_disconnect_freeze_and_new_collection_resets_session(self):
        window = self.make_window()
        dialog = self.open_live_model(window)
        self.feed(dialog, 100)
        window.stop_collection()
        self.assertFalse(dialog._stream_active)
        held_count = dialog.sample_count
        window.serial_port.chunks = [eeg_frame((10.0,) * 8)]
        self.update_once(window)
        self.assertEqual(dialog.sample_count, held_count)
        window.subject_info = {"kind": "person", "name": "测试者", "age": 30}
        with patch.object(window, "send_binary_command", return_value=True):
            window.start_collect()
        self.assertTrue(dialog._stream_active)
        self.assertEqual(dialog.sample_count, 0)
        self.assertIsNone(dialog.controller.baseline_db)
        window._handle_serial_disconnect()
        self.assertIsNone(window.serial_port)
        self.assertFalse(dialog._stream_active)
        self.assertFalse(dialog.scene.active)

    def test_history_freezes_model_and_returning_live_keeps_collection_stopped(self):
        window = self.make_window()
        dialog = self.open_live_model(window)
        self.feed(dialog, 100)
        with patch(f"{MODULE_NAME}.QFileDialog.getOpenFileName", return_value=("test_250sps_raw.csv", "")), patch(
            f"{MODULE_NAME}.read_history_csv", return_value=[[0.0] * 250 for _ in range(8)]
        ), patch.object(window, "_render_history"):
            window.open_history()
        self.assertTrue(window.is_history_mode)
        self.assertFalse(dialog._stream_active)
        count = dialog.sample_count
        window.serial_port.chunks = [eeg_frame((20.0,) * 8)]
        self.update_once(window)
        self.assertEqual(dialog.sample_count, count)
        with patch.object(window, "_refresh_all_graphs"):
            window.leave_history()
        self.assertFalse(window.is_collecting)
        self.assertFalse(dialog._stream_active)

    def test_close_and_reopen_sync_actual_collection_and_available_channels(self):
        window = self.make_window()
        dialog = self.open_live_model(window)
        with patch("alpha_model_dialog.time.monotonic", return_value=100.0):
            self.calibrate(dialog)
        dialog.close()
        count = dialog.sample_count
        window.serial_port.chunks = [eeg_frame((20.0,) * 8)]
        self.update_once(window)
        self.assertEqual(dialog.sample_count, count)
        window.open_alpha_model()
        self.assertIs(window.alpha_model_dialog, dialog)
        self.assertTrue(dialog._stream_active)
        dialog.timer.stop()
        window.toggle_channel(False, 2)
        self.assertFalse(dialog.channel_checks[2].isEnabled())
        self.assertIsNone(dialog.controller.baseline_db)
        self.assertEqual(dialog.controller.status, "waiting")
        dialog.close()
        window.stop_collection()
        window.open_alpha_model()
        dialog.timer.stop()
        self.assertFalse(dialog._stream_active)
        self.assertFalse(dialog.scene.active)


if __name__ == "__main__":
    unittest.main()
