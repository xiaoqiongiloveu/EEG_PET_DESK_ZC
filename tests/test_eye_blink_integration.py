import importlib.util
import os
import struct
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python_ads1299"))

from ads1299 import ADC_COUNT_TO_UV, ADS1298Data, BP_TYPE_CMD_RESP, BP_TYPE_EEG, bp_pack
from eye_blink import EyeBlinkDetector


def eeg_frame(values, sequence=0):
    counts = np.rint(np.asarray(values) / ADC_COUNT_TO_UV).astype(np.int32)
    return bp_pack(BP_TYPE_EEG, sequence, b"\0" * 3 + struct.pack("<8i", *counts))


class RawSampleDeliveryTests(unittest.TestCase):
    def test_fragmented_frame_delivers_one_sample_then_clears_on_non_eeg_input(self):
        decoder = ADS1298Data()
        values = (0, 1, 2, 3, 4, 120, 150, 110)
        frame = eeg_frame(values)

        decoder.parse_data(frame[:17])
        self.assertEqual(decoder.new_raw_samples, [])
        decoder.parse_data(frame[17:])
        self.assertEqual(len(decoder.new_raw_samples), 1)
        np.testing.assert_allclose(decoder.new_raw_samples[0], values, atol=ADC_COUNT_TO_UV)

        decoder.parse_data(b"")
        self.assertEqual(decoder.new_raw_samples, [])
        decoder.parse_data(bp_pack(BP_TYPE_CMD_RESP, 2, b"\x01\x00"))
        self.assertEqual(decoder.new_raw_samples, [])
        decoder.parse_data(eeg_frame(values, 3))
        self.assertEqual(len(decoder.new_raw_samples), 1)

    def test_binary_eeg_pulse_reaches_detector_as_raw_microvolts(self):
        rate = 250
        time = np.arange(int(3.5 * rate)) / rate
        rng = np.random.default_rng(21)
        values = rng.normal(0, 0.8, (len(time), 8))
        pulse = np.exp(-0.5 * ((time - 2.0) / 0.075) ** 2)
        values += pulse[:, None] * np.array((0, 0, 0, 0, 0, 120, 150, 110))

        decoder = ADS1298Data()
        detector = EyeBlinkDetector()
        count = 0
        raw_peak = 0.0
        for start in range(0, len(values), 125):
            frames = b"".join(
                eeg_frame(row, index)
                for index, row in enumerate(values[start:start + 125], start)
            )
            decoder.parse_data(frames)
            self.assertEqual(len(decoder.new_raw_samples), min(125, len(values) - start))
            raw_peak = max(raw_peak, max(row[6] for row in decoder.new_raw_samples))
            count += detector.process(decoder.new_raw_samples, rate)

        self.assertEqual(count, 1)
        self.assertAlmostEqual(raw_peak, 150.0, delta=5.0)


class EyeBlinkCounterWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt5.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])
        name = "main_uart_v1_2_blink_test_import"
        spec = importlib.util.spec_from_file_location(
            name, ROOT / "python_ads1299" / "main_uart_v1.2.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        cls.MainWindow = module.MainWindow

    def test_live_redraw_and_non_eeg_frame_do_not_recount_last_sample(self):
        window = self.MainWindow()
        try:
            class SerialStub:
                is_open = True

                def __init__(self, chunks):
                    self.chunks = list(chunks)

                @property
                def in_waiting(self):
                    return len(self.chunks[0]) if self.chunks else 0

                def read(self, _size):
                    return self.chunks.pop(0)

            window.serial_port = SerialStub([
                eeg_frame((0, 1, 2, 3, 4, 120, 150, 110)),
                bp_pack(BP_TYPE_CMD_RESP, 2, b"\x01\x00"),
            ])
            window.is_collecting = True
            with patch.object(window.eye_blink_detector, "process", return_value=1) as process, patch.object(
                window, "update_graph", return_value=False
            ), patch.object(window, "_update_spectrum"):
                window.update_show()
                window.update_show()
                window.update_show()
                window._refresh_all_graphs()
            self.assertEqual(process.call_count, 1)
            self.assertEqual(window.eye_blink_count, 1)
        finally:
            window.is_collecting = False
            window.serial_port = None
            window.close()

    def test_count_starts_one_timer_and_resets_after_three_minutes(self):
        window = self.MainWindow()
        try:
            window.is_collecting = True
            sample = [(0.0,) * 8]
            with patch.object(window.eye_blink_detector, "process", side_effect=(1, 2, 1)):
                window._track_eye_blinks(sample, 250)
                self.assertEqual(window.eye_blink_count, 1)
                self.assertEqual(window.eye_blink_reset_timer.interval(), 180_000)
                self.assertTrue(window.eye_blink_reset_timer.isActive())
                window._track_eye_blinks(sample, 250)
                self.assertEqual(window.eye_blink_count, 3)
                self.assertIn("累计3次", window.eye_blink_status.text())

                window.eye_blink_reset_timer.timeout.emit()
                self.assertEqual(window.eye_blink_count, 0)
                self.assertFalse(window.eye_blink_reset_timer.isActive())
                window._track_eye_blinks(sample, 250)
                self.assertEqual(window.eye_blink_count, 1)
                self.assertTrue(window.eye_blink_reset_timer.isActive())
        finally:
            window.close()

    def test_new_collection_keeps_unexpired_count_and_history_does_not_add(self):
        window = self.MainWindow()
        try:
            window.eye_blink_count = 2
            window._update_eye_blink_status()
            window.eye_blink_reset_timer.start(180_000)
            window.serial_port = type("SerialStub", (), {"is_open": True})()
            with patch.object(window, "send_binary_command", return_value=True), patch.object(
                window.eye_blink_detector, "reset"
            ) as reset:
                window.start_collect()
            reset.assert_called_once_with()
            self.assertTrue(window.is_collecting)
            self.assertEqual(window.eye_blink_count, 2)
            self.assertTrue(window.eye_blink_reset_timer.isActive())

            window.is_history_mode = True
            with patch.object(window.eye_blink_detector, "process", return_value=1) as process:
                window._track_eye_blinks([(0.0,) * 8], 250)
            process.assert_not_called()
            self.assertEqual(window.eye_blink_count, 2)
        finally:
            window.is_collecting = False
            window.serial_port = None
            window.close()


if __name__ == "__main__":
    unittest.main()
