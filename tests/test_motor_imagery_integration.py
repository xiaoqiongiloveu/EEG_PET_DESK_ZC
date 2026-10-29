"""Check the serial and collection boundaries of motor imagery control."""

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


MODULE_NAME = "main_uart_motor_imagery_test_import"


def eeg_frame(values, sequence=0):
    counts = np.rint(np.asarray(values) / ADC_COUNT_TO_UV).astype(np.int32)
    payload = b"\0" * 3 + struct.pack("<8i", *counts)
    return bp_pack(BP_TYPE_EEG, sequence, payload)


class SerialStub:
    def __init__(self, chunks=()):
        self.chunks = list(chunks)
        self.is_open = True

    @property
    def in_waiting(self):
        return len(self.chunks[0]) if self.chunks else 0

    def read(self, _size):
        return self.chunks.pop(0)

    def write(self, payload):
        return len(payload)

    def flush(self):
        pass

    def close(self):
        self.is_open = False


class MotorImageryMainWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt5.QtWidgets import QApplication, QDialog

        cls.app = QApplication.instance() or QApplication([])
        if MODULE_NAME not in sys.modules:
            spec = importlib.util.spec_from_file_location(
                MODULE_NAME, ROOT / "python_ads1299" / "main_uart_v1.2.py"
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[MODULE_NAME] = module
            spec.loader.exec_module(module)
        cls.module = sys.modules[MODULE_NAME]

        class DialogProbe(QDialog):
            def __init__(self, channel_names, parent=None):
                super().__init__(parent)
                self.channel_names = list(channel_names)
                self.channels = []
                self.active = False
                self.reason = ""
                self.sessions = 0
                self.samples = []

            def start_session(self):
                self.sessions += 1
                self.samples.clear()

            def set_available_channels(self, states):
                self.channels = list(states)

            def set_stream_active(self, active, reason=""):
                self.active = bool(active)
                self.reason = reason

            def feed_samples(self, samples, rate):
                if len(samples):
                    self.samples.append((np.asarray(samples).copy(), rate))

        cls.DialogProbe = DialogProbe

    def make_window(self):
        window = self.module.MainWindow()
        window.update_show_timer.stop()

        def cleanup():
            window.is_collecting = False
            window.serial_port = None
            window.close()
            self.module.plt.close(window.fig)
            self.module.plt.close(window.spectrum_canvas.figure)
            window.deleteLater()

        self.addCleanup(cleanup)
        factory = patch.object(self.module, "MotorImageryDialog", self.DialogProbe)
        factory.start()
        self.addCleanup(factory.stop)
        return window

    def open_live_model(self, window):
        window.serial_port = SerialStub()
        window.is_collecting = True
        window.open_motor_imagery()
        dialog = window.motor_imagery_dialog
        self.assertTrue(dialog.isVisible())
        self.assertTrue(dialog.active)
        return dialog

    def update_once(self, window):
        with patch.object(window, "update_graph", return_value=False), patch.object(
            window, "_update_spectrum"
        ), patch.object(window.canvas, "draw_idle"), patch.object(
            window.spectrum_canvas, "draw_idle"
        ):
            window.update_show()

    def test_button_opens_reuses_and_reopens_with_current_channel_availability(self):
        window = self.make_window()
        window.channel_states[3] = False
        window.btn_motor_imagery.click()
        dialog = window.motor_imagery_dialog
        self.assertIsNotNone(dialog)
        self.assertEqual(dialog.channel_names, window.channel_names)
        self.assertEqual(dialog.channels, window.channel_states)
        self.assertFalse(dialog.active)
        self.assertIn("采集", dialog.reason)
        self.assertEqual(dialog.sessions, 1)
        window.btn_motor_imagery.click()
        self.assertEqual(dialog.sessions, 1)
        dialog.close()
        window.serial_port = SerialStub()
        window.is_collecting = True
        window.open_motor_imagery()
        self.assertIs(window.motor_imagery_dialog, dialog)
        self.assertEqual(dialog.sessions, 2)
        self.assertTrue(dialog.active)

    def test_real_binary_microvolts_bypass_filters_without_recounting_ack_or_redraw(self):
        window = self.make_window()
        dialog = self.open_live_model(window)
        times = np.arange(50, dtype=float) / 250
        values = np.repeat((20 * np.sin(2 * np.pi * 12 * times))[:, None], 8, axis=1)
        window.serial_port.chunks = [
            b"".join(eeg_frame(row, index) for index, row in enumerate(values)),
            bp_pack(BP_TYPE_CMD_RESP, 51, b"\x01\x00"),
        ]
        display_values = [[8888.0] * len(values) for _ in range(8)]
        with ExitStack() as stack:
            stack.enter_context(patch.object(window, "_append_new_filtered_data"))
            stack.enter_context(patch.object(window, "_current_data_source", return_value=display_values))
            self.update_once(window)
            self.update_once(window)
            self.update_once(window)
        self.assertEqual(len(dialog.samples), 1)
        raw_samples, rate = dialog.samples[0]
        np.testing.assert_allclose(raw_samples, values, atol=ADC_COUNT_TO_UV)
        self.assertEqual(rate, 250)
        self.assertAlmostEqual(
            window.AdsData.emg_data[0][-1], values[-1, 0], delta=ADC_COUNT_TO_UV,
        )

    def test_collection_stop_new_start_and_disconnect_propagate_to_model(self):
        window = self.make_window()
        dialog = self.open_live_model(window)
        window.stop_collection()
        self.assertFalse(dialog.active)
        window.serial_port.chunks = [eeg_frame((10.0,) * 8)]
        self.update_once(window)
        self.assertEqual(dialog.samples, [])
        with patch.object(window, "send_binary_command", return_value=True):
            window.start_collect()
        self.assertTrue(dialog.active)
        self.assertEqual(dialog.sessions, 2)
        window.toggle_channel(False, 2)
        self.assertFalse(dialog.channels[2])
        window._handle_serial_disconnect()
        self.assertFalse(dialog.active)
        self.assertIsNone(window.serial_port)
        self.assertIn("断开", dialog.reason)

    def test_history_and_hidden_model_do_not_receive_live_eeg(self):
        window = self.make_window()
        dialog = self.open_live_model(window)
        dialog.close()
        window.serial_port.chunks = [eeg_frame((10.0,) * 8)]
        self.update_once(window)
        self.assertEqual(dialog.samples, [])
        window.open_motor_imagery()
        with patch.object(self.module.QFileDialog, "getOpenFileName", return_value=("test_250sps_raw.csv", "")), patch.object(
            self.module, "read_history_csv", return_value=[[0.0] * 250 for _ in range(8)]
        ), patch.object(window, "_render_history"):
            window.open_history()
        self.assertTrue(window.is_history_mode)
        self.assertFalse(dialog.active)
        self.assertIn("历史", dialog.reason)
        window.serial_port.chunks = [eeg_frame((20.0,) * 8)]
        self.update_once(window)
        self.assertEqual(dialog.samples, [])
        dialog.close()
        window.open_motor_imagery()
        self.assertFalse(dialog.active)
        self.assertIn("历史", dialog.reason)

    def test_failed_stream_start_does_not_start_or_reset_model(self):
        window = self.make_window()
        window.serial_port = SerialStub()
        window.open_motor_imagery()
        dialog = window.motor_imagery_dialog
        with patch.object(window, "send_binary_command", return_value=False):
            window.start_collect()
        self.assertFalse(window.is_collecting)
        self.assertFalse(dialog.active)
        self.assertEqual(dialog.sessions, 1)

    def test_closing_main_window_stops_and_closes_model(self):
        window = self.make_window()
        dialog = self.open_live_model(window)
        window.close()
        self.assertFalse(dialog.active)
        self.assertFalse(dialog.isVisible())


if __name__ == "__main__":
    unittest.main()
