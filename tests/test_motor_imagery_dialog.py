"""Exercise visible motor-imagery cues, fresh EEG, and model-control boundaries."""

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python_ads1299"))

from motor_imagery import MotorImageryClassifier


def sensorimotor_samples(start, count, label="left", rate=250, seed=0):
    """Independent-noise mu/beta rhythms with opposite C3/C4 attenuation."""
    times = np.arange(start, start + count, dtype=float) / rate
    phase = np.random.default_rng(seed).uniform(0, 2 * np.pi, 4)
    gains = (5.0, 15.0) if label == "left" else (15.0, 5.0)
    values = np.zeros((count, 8), dtype=float)
    for channel, (mu, beta) in enumerate(((10.0, 20.0), (10.5, 21.0))):
        values[:, channel] = gains[channel] * (
            np.sin(2 * np.pi * mu * times + phase[channel])
            + 0.5 * np.sin(2 * np.pi * beta * times + phase[channel + 2])
        )
    noise = np.random.default_rng(seed + start + 10000).normal(0, 0.8, values.shape)
    return values + noise


class MotorImageryDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt5.QtWidgets import QApplication
        from motor_imagery_dialog import MotorImageryDialog

        cls.app = QApplication.instance() or QApplication([])
        cls.Dialog = MotorImageryDialog
        labels = ["left", "right"] * 12
        trials = [sensorimotor_samples(0, 500, label, seed=index) for index, label in enumerate(labels)]
        cls.trained_model = MotorImageryClassifier()
        report = cls.trained_model.fit(trials, labels, 250, channel_indices=(0, 1))
        if not report.reliable:
            raise AssertionError("Synthetic training fixture did not pass independent-trial validation")

    def setUp(self):
        self.now = [100.0]
        clock = patch("motor_imagery_dialog.time.monotonic", side_effect=lambda: self.now[0])
        clock.start()
        self.addCleanup(clock.stop)

    def make_dialog(self, configure=True):
        dialog = self.Dialog([f"CH{index + 1}" for index in range(8)])

        def cleanup():
            dialog.close()
            dialog.deleteLater()

        self.addCleanup(cleanup)
        dialog.show()
        dialog.timer.stop()
        dialog.set_stream_active(True)
        if configure:
            dialog.c3_combo.setCurrentIndex(1)
            dialog.c4_combo.setCurrentIndex(2)
        self.feed(dialog, 50)
        return dialog

    def feed(self, dialog, count, label="left", rate=250, seed=100, real_time=True):
        for offset in range(0, count, 50):
            size = min(50, count - offset)
            values = sensorimotor_samples(dialog.sample_count, size, label, rate, seed)
            if real_time:
                self.now[0] += size / rate
            dialog.feed_samples(values, rate)

    def start_control(self, dialog):
        dialog.model = self.trained_model
        dialog._update_readouts()
        dialog.toggle_control()
        self.assertTrue(dialog._controlling)

    def assert_moves(self, dialog, label):
        self.feed(dialog, 800, label)
        expected = -1 if label == "left" else 1
        self.assertEqual(dialog._command, expected)
        self.assertTrue(dialog.scene.active)
        previous = dialog.scene.position
        dialog._last_tick_time = self.now[0] - 0.1
        dialog._tick()
        self.assertGreater((dialog.scene.position - previous) * expected, 0.0)

    def test_training_requires_live_eeg_and_distinct_enabled_inputs(self):
        dialog = self.make_dialog(configure=False)
        self.assertFalse(dialog.train_button.isEnabled())
        dialog.toggle_training()
        self.assertFalse(dialog._training)
        dialog.c3_combo.setCurrentIndex(1)
        dialog.c4_combo.setCurrentIndex(1)
        dialog.toggle_training()
        self.assertFalse(dialog._training)
        dialog.c4_combo.setCurrentIndex(2)
        self.assertTrue(dialog.train_button.isEnabled())
        dialog.set_available_channels([True, False] + [True] * 6)
        self.assertFalse(dialog.train_button.isEnabled())
        dialog.toggle_training()
        self.assertFalse(dialog._training)
        dialog.set_available_channels([True] * 8)
        dialog.set_stream_active(False)
        self.assertFalse(dialog.train_button.isEnabled())
        dialog.toggle_training()
        self.assertFalse(dialog._training)

    def test_buffered_data_and_elapsed_time_cannot_skip_unseen_cues(self):
        dialog = self.make_dialog()
        dialog.toggle_training()
        self.assertEqual(dialog._phase, "prepare")
        # Five seconds received in one burst cannot be assigned to multiple
        # prompts which the person has not yet seen for their required time.
        self.feed(dialog, 1250, real_time=False)
        self.assertEqual(dialog._phase, "prepare")
        self.assertEqual(dialog.progress_bar.value(), 0)
        self.now[0] += 1.1
        dialog._tick()
        self.assertEqual(dialog._phase, "prepare")
        self.feed(dialog, 50, real_time=False)
        self.assertEqual(dialog._phase, "imagine")
        self.assertEqual(len(dialog._trial_buffer), 0)
        self.assertEqual(dialog.progress_bar.value(), 0)
        # A redraw without new samples cannot create a labelled trial.
        self.now[0] += 0.3
        dialog._tick()
        self.assertEqual(dialog._phase, "imagine")
        self.assertEqual(len(dialog._trials), 0)

    def test_personal_cue_training_validates_then_controls_both_directions(self):
        dialog = self.make_dialog()
        dialog.toggle_training()
        iterations = 0
        while dialog._training and iterations < 1600:
            label = "left" if dialog._current_label == -1 else "right"
            self.feed(dialog, 50, label, seed=dialog._attempts)
            iterations += 1
        self.assertFalse(dialog._training)
        self.assertLess(iterations, 1600)
        self.assertEqual(dialog._labels.count(-1), 10)
        self.assertEqual(dialog._labels.count(1), 10)
        self.assertEqual(dialog.progress_bar.value(), 20)
        self.assertTrue(dialog.model.ready)
        self.assertTrue(dialog.control_button.isEnabled())
        self.assertIn("需实际试控", dialog.status_label.text())
        dialog.toggle_control()
        self.assert_moves(dialog, "left")
        self.assert_moves(dialog, "right")

    def test_pause_and_resume_require_fresh_data_before_moving(self):
        dialog = self.make_dialog()
        self.start_control(dialog)
        self.assert_moves(dialog, "left")
        dialog.toggle_control()
        held = dialog.scene.position
        self.feed(dialog, 1000, "right")
        dialog._tick()
        self.assertFalse(dialog._controlling)
        self.assertFalse(dialog.scene.active)
        self.assertEqual(dialog.scene.position, held)
        dialog.toggle_control()
        self.feed(dialog, 450, "right")
        self.assertFalse(dialog.scene.active)
        self.assertEqual(dialog._command, 0)
        self.assert_moves(dialog, "right")

    def test_invalid_current_eeg_stops_motion_immediately(self):
        dialog = self.make_dialog()
        self.start_control(dialog)
        self.assert_moves(dialog, "left")
        held = dialog.scene.position
        values = sensorimotor_samples(dialog.sample_count, 100, "left")
        values[:, 0] = np.nan
        self.now[0] += 0.4
        dialog.feed_samples(values, 250)
        self.assertEqual(dialog._command, 0)
        self.assertFalse(dialog.scene.active)
        dialog._tick()
        self.assertEqual(dialog.scene.position, held)
        self.assertIn("信号无效", dialog.status_label.text())

    def test_signal_loss_aborts_training_and_requires_explicit_control_restart(self):
        dialog = self.make_dialog()
        dialog.toggle_training()
        self.feed(dialog, 100)
        self.now[0] += 2.0
        dialog._tick()
        self.assertTrue(dialog._stale)
        self.assertFalse(dialog._training)
        self.assertIsNone(dialog._phase)
        self.assertEqual(len(dialog._trial_buffer), 0)
        self.feed(dialog, 50)
        self.assertFalse(dialog._stale)
        self.assertFalse(dialog._training)
        self.start_control(dialog)
        self.assert_moves(dialog, "right")
        held = dialog.scene.position
        self.now[0] += 2.0
        dialog._tick()
        self.assertFalse(dialog._controlling)
        self.assertFalse(dialog.scene.active)
        self.assertEqual(dialog.scene.position, held)
        self.feed(dialog, 1000)
        dialog._tick()
        self.assertFalse(dialog._controlling)
        self.assertEqual(dialog.scene.position, held)

    def test_expired_decisions_require_fresh_votes_before_moving_again(self):
        dialog = self.make_dialog()
        self.start_control(dialog)
        self.assert_moves(dialog, "left")
        held = dialog.scene.position
        self.now[0] += 0.8
        dialog._tick()
        self.assertTrue(dialog._controlling)
        self.assertEqual(dialog._command, 0)
        self.assertEqual(len(dialog._votes), 0)
        self.assertEqual(dialog.scene.position, held)
        self.feed(dialog, 100, "left")
        self.assertEqual(dialog._command, 0)
        self.assertFalse(dialog.scene.active)
        self.assertEqual(len(dialog._votes), 1)
        self.feed(dialog, 200, "left")
        self.assertEqual(dialog._command, -1)
        self.assertTrue(dialog.scene.active)

    def test_oversized_buffered_delivery_stops_training_and_active_control(self):
        dialog = self.make_dialog()
        dialog.toggle_training()
        self.now[0] += 0.2
        dialog.feed_samples(sensorimotor_samples(dialog.sample_count, 250), 250)
        self.assertFalse(dialog._training)
        self.assertIsNone(dialog._phase)
        self.assertEqual(dialog.progress_bar.value(), 0)
        self.assertEqual(len(dialog._buffers[0]), 0)
        self.assertIn("积压数据", dialog.status_label.text())
        self.feed(dialog, 50)
        self.start_control(dialog)
        self.assert_moves(dialog, "right")
        held = dialog.scene.position
        self.now[0] += 0.2
        dialog.feed_samples(sensorimotor_samples(dialog.sample_count, 250), 250)
        self.assertFalse(dialog._controlling)
        self.assertFalse(dialog.scene.active)
        self.assertEqual(dialog._command, 0)
        self.feed(dialog, 1000)
        dialog._tick()
        self.assertFalse(dialog._controlling)
        self.assertEqual(dialog.scene.position, held)

    def test_model_loaded_before_first_eeg_is_discarded_for_a_different_rate(self):
        dialog = self.make_dialog()
        dialog.start_session()
        self.assertIsNone(dialog._rate)
        dialog.model = self.trained_model
        dialog._update_readouts()
        self.assertFalse(dialog.control_button.isEnabled())
        self.feed(dialog, 50, rate=500)
        self.assertIsNone(dialog.model)
        self.assertFalse(dialog._controlling)
        self.assertEqual(dialog._rate, 500)
        self.assertIn("模型采样率", dialog.status_label.text())

    def test_rate_channel_and_new_session_discard_the_previous_personal_model(self):
        dialog = self.make_dialog()
        self.start_control(dialog)
        self.feed(dialog, 100, rate=500)
        self.assertIsNone(dialog.model)
        self.assertFalse(dialog._controlling)
        self.assertEqual(dialog.sample_count, 100)
        self.assertEqual(dialog._rate, 500)
        self.assertEqual(dialog._buffers[0].maxlen, 1000)
        dialog.model = self.trained_model
        dialog.c4_combo.setCurrentIndex(3)
        self.assertIsNone(dialog.model)
        self.assertEqual(len(dialog._buffers[0]), 0)
        dialog.model = self.trained_model
        dialog.start_session()
        self.assertIsNone(dialog.model)
        self.assertEqual(dialog.sample_count, 0)
        self.assertIsNone(dialog._rate)

    def test_close_and_escape_stop_timers_and_reject_hidden_samples(self):
        for method in ("close", "reject"):
            with self.subTest(method=method):
                dialog = self.make_dialog()
                self.start_control(dialog)
                dialog.timer.start()
                count = dialog.sample_count
                getattr(dialog, method)()
                dialog.feed_samples(sensorimotor_samples(count, 500), 250)
                self.assertFalse(dialog.isVisible())
                self.assertFalse(dialog._stream_active)
                self.assertFalse(dialog.timer.isActive())
                self.assertFalse(dialog._controlling)
                self.assertEqual(dialog.sample_count, count)

    def test_saved_model_roundtrip_requires_matching_hardware_channels_and_rate(self):
        dialog = self.make_dialog()
        dialog.model = self.trained_model
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "personal_model.json")
            with patch("motor_imagery_dialog.QFileDialog.getSaveFileName", return_value=(path, "")):
                dialog.save_model()
            self.assertTrue(Path(path).is_file())
            with open(path, encoding="utf-8") as stream:
                self.assertIsInstance(json.load(stream), dict)
            dialog.model = None
            with patch("motor_imagery_dialog.QFileDialog.getOpenFileName", return_value=(path, "")):
                dialog.load_model()
            self.assertTrue(dialog.model.ready)
            self.assertEqual(dialog.model.predict(sensorimotor_samples(0, 500, "right")).label, "right")
            dialog.c4_combo.setCurrentIndex(3)
            with patch("motor_imagery_dialog.QFileDialog.getOpenFileName", return_value=(path, "")), patch(
                "motor_imagery_dialog.QMessageBox.warning"
            ) as warning:
                dialog.load_model()
            self.assertIsNone(dialog.model)
            warning.assert_called_once()
            self.assertIn("通道", warning.call_args.args[2])
            dialog.c4_combo.setCurrentIndex(2)
            self.feed(dialog, 100, rate=500)
            with patch("motor_imagery_dialog.QFileDialog.getOpenFileName", return_value=(path, "")), patch(
                "motor_imagery_dialog.QMessageBox.warning"
            ) as warning:
                dialog.load_model()
            self.assertIsNone(dialog.model)
            warning.assert_called_once()
            self.assertIn("采样率", warning.call_args.args[2])


if __name__ == "__main__":
    unittest.main()
