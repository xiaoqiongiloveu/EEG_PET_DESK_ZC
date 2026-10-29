"""Regression tests for raw acquisition, live plot buffering, and commands."""

import struct
import sys
import unittest
import importlib.util
import logging
import os
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python_ads1299"))

from ads1299 import (  # noqa: E402
    ADC_COUNT_TO_UV, ADS1298Data, ADS1298cmd, BP_CMD_STOP_IMP,
    BP_CMD_ENABLE_DEVICE, BP_CMD_GET_STATE, BP_TYPE_STATUS,
    LINK_STATS_MAGIC, BP_TYPE_CMD, BP_TYPE_EEG, bp_crc16, bp_pack,
)
from ring_buffer import MultiChannelRingBuffer  # noqa: E402
from link_stats import StreamLossTracker  # noqa: E402


class AcquisitionPipelineTests(unittest.TestCase):
    def test_decoder_preserves_raw_microvolts_at_high_sample_rate(self):
        decoder = ADS1298Data()
        decoder.sample_rate = 1000
        counts = (1200, -900, 600, -300, 150, -75, 30, 0)
        frame = bp_pack(BP_TYPE_EEG, 1, b"\0" * 3 + struct.pack("<8i", *counts))

        decoder.parse_data(frame[:9])
        self.assertEqual(decoder.new_raw_samples, [])
        decoder.parse_data(frame[9:])

        expected = np.asarray(counts) * ADC_COUNT_TO_UV
        np.testing.assert_allclose(decoder.new_raw_samples[0], expected)
        np.testing.assert_allclose([channel[0] for channel in decoder.emg_data], expected)
        decoder.parse_data(b"")
        self.assertEqual(decoder.new_raw_samples, [])
        self.assertEqual(len(decoder.emg_data[0]), 1)

    def test_live_buffer_wraps_without_losing_absolute_sample_count(self):
        ring = MultiChannelRingBuffer(5)
        first = np.arange(24, dtype=float).reshape(3, 8)
        second = np.arange(24, 56, dtype=float).reshape(4, 8)
        ring.append(first)
        ring.append(second)

        self.assertEqual(ring.source_length, 7)
        np.testing.assert_array_equal(ring.snapshot(), np.vstack((first, second))[-5:])
        ring.clear()
        self.assertEqual(ring.source_length, 0)
        self.assertEqual(ring.snapshot().shape, (0, 8))

    def test_stop_impedance_uses_dedicated_command(self):
        frame = ADS1298cmd.stop_impedance_cmd()
        body_length = struct.unpack_from("<H", frame, 2)[0]
        body = frame[4:4 + body_length]
        self.assertEqual(body[0], BP_TYPE_CMD)
        self.assertEqual(body[7:], bytes([BP_CMD_STOP_IMP]))
        self.assertEqual(BP_CMD_STOP_IMP, 0x14)
        self.assertEqual(struct.unpack_from("<H", frame, 4 + body_length)[0], bp_crc16(body))


class WirelessProtocolTests(unittest.TestCase):
    def test_status_frame_parses_device_counters(self):
        payload = (
            bytes([1, 50, 1]) + (400).to_bytes(2, "little") + bytes([3])
            + (7).to_bytes(4, "little") + bytes([0])
            + b"1.0.0\x00SN123\x00" + LINK_STATS_MAGIC
            + (11).to_bytes(4, "little") + (2).to_bytes(4, "little")
            + (5).to_bytes(4, "little")
        )
        decoder = ADS1298Data()
        decoder.parse_data(bp_pack(BP_TYPE_STATUS, 3, payload))
        self.assertEqual(decoder.last_status_seq, 3)
        self.assertTrue(decoder.device_status["link_stats"])
        self.assertEqual(decoder.device_status["ads_drops"], 11)
        self.assertEqual(decoder.device_status["wireless_tx_overflow"], 2)
        self.assertEqual(decoder.device_status["wireless_rx_overflow"], 5)
        self.assertEqual(decoder.device_status["firmware"], "1.0.0")

        decoder.parse_data(bp_pack(BP_TYPE_STATUS, 4, payload[:-14]))
        self.assertFalse(decoder.device_status["link_stats"])

    def test_eeg_events_and_bad_crc(self):
        decoder = ADS1298Data()
        good = bp_pack(BP_TYPE_EEG, 10, b"\0" * 3 + struct.pack("<8i", *([0] * 8)))
        decoder.parse_data(good)
        self.assertEqual(decoder.eeg_frame_events, [(10, 0)])
        self.assertEqual(decoder.eeg_frames_received, 1)
        bad = bytearray(good)
        bad[-1] ^= 0xFF
        decoder.parse_data(bad)
        self.assertEqual(decoder.crc_errors, 1)
        self.assertEqual(decoder.eeg_frame_events, [])
        self.assertEqual(decoder.eeg_frames_received, 1)

    def test_wireless_command_ids(self):
        for command, expected in (
            (ADS1298cmd.enable_device_cmd(), BP_CMD_ENABLE_DEVICE),
            (ADS1298cmd.get_state_cmd(), BP_CMD_GET_STATE),
        ):
            length = struct.unpack_from("<H", command, 2)[0]
            self.assertEqual(command[4 + 7], expected)
            self.assertEqual(struct.unpack_from("<H", command, 4 + length)[0],
                             bp_crc16(command[4:4 + length]))


class LossTrackerTests(unittest.TestCase):
    def test_sequence_and_time_loss(self):
        tracker = StreamLossTracker(250)
        tracker.update(1, 1000)
        tracker.update(4, 1012)
        self.assertEqual(tracker.lost_by_seq, 2)
        self.assertEqual(tracker.lost_by_time, 2)
        self.assertAlmostEqual(tracker.loss_rate(2), 0.5)

    def test_outage_does_not_inflate_loss(self):
        tracker = StreamLossTracker(250)
        tracker.update(1, 1000)
        tracker.update(60000, 30000)
        self.assertEqual(tracker.outages, 1)
        self.assertEqual(tracker.lost_by_seq, 0)
        self.assertEqual(tracker.lost_by_time, 0)


class WirelessWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        from PyQt5.QtWidgets import QApplication
        cls.app = QApplication.instance() or QApplication([])
        path = Path(__file__).resolve().parents[1] / "python_ads1299" / "main_uart_v1.2.py"
        spec = importlib.util.spec_from_file_location("wireless_main_window", path)
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_wifi_port_selection_and_command_order(self):
        from PyQt5.QtTest import QTest

        class FakeSerial:
            def __init__(self, **kwargs):
                self.name = kwargs["port"]
                self.baudrate = kwargs["baudrate"]
                self.is_open = True
                self.in_waiting = 0
                self.writes = []

            def reset_input_buffer(self):
                pass

            def write(self, data):
                self.writes.append((data[11], time.monotonic()))
                return len(data)

            def flush(self):
                pass

            def close(self):
                self.is_open = False

        ports = [
            SimpleNamespace(device="COM7", description="WiFi", vid=None, pid=None),
            SimpleNamespace(device="COM3", description="采集设备", vid=0x0483, pid=0x5740),
        ]
        with patch.object(self.module.list_ports, "comports", return_value=ports), \
                patch.object(self.module.serial, "Serial", side_effect=FakeSerial):
            window = self.module.MainWindow()
            try:
                self.assertEqual(window.port_combo.currentData(), "COM3")
                window.wifi_check.setChecked(True)
                self.assertEqual(window.baud_combo.currentData(), 1000000)
                window.port_combo.setCurrentIndex(window.port_combo.findData("COM7"))
                window.refresh_ports()
                self.assertEqual(window.port_combo.currentData(), "COM7")
                window.open_serial()
                port = window.serial_port
                self.assertEqual((port.name, port.baudrate), ("COM7", 1000000))
                window.AdsData.parse_data(bp_pack(BP_TYPE_STATUS, 1))
                window._check_device_ready()
                window.subject_info = {"kind": "person", "name": "测试者", "age": 30}
                window.start_collect()
                deadline = time.monotonic() + 2.0
                while len(port.writes) < 5 and time.monotonic() < deadline:
                    QTest.qWait(50)
                commands = port.writes[-4:]
                self.assertEqual([item[0] for item in commands], [0x0B, 0x03, 0x04, 0x01])
                self.assertTrue(all(b[1] - a[1] >= 0.11 for a, b in zip(commands, commands[1:])))
                window.stop_collection()
                QTest.qWait(370)
                self.assertEqual(port.writes[-1][0], 0x02)
            finally:
                window.close()


if __name__ == "__main__":
    unittest.main()
