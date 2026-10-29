"""Independent-trial validation and conservative motor-imagery output."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python_ads1299"))

from motor_imagery import MotorImageryClassifier


class MotorImageryTests(unittest.TestCase):
    RATE = 250

    def trial(self, label, seed):
        rng = np.random.default_rng(seed)
        time = np.arange(2 * self.RATE) / self.RATE
        phase = rng.uniform(0, 2 * np.pi, 4)
        mu_left = np.sin(2 * np.pi * 10.0 * time + phase[0])
        mu_right = np.sin(2 * np.pi * 10.5 * time + phase[1])
        beta_left = np.sin(2 * np.pi * 20.0 * time + phase[2])
        beta_right = np.sin(2 * np.pi * 21.0 * time + phase[3])
        gains = (5.0, 15.0) if label == "left" else (15.0, 5.0)
        return np.column_stack((
            gains[0] * mu_left + 0.5 * gains[0] * beta_left,
            gains[1] * mu_right + 0.5 * gains[1] * beta_right,
        )) + rng.normal(0.0, 1.0, (len(time), 2))

    def training(self):
        labels = ["left", "right"] * 12
        return [self.trial(label, index) for index, label in enumerate(labels)], labels

    def trained(self):
        model = MotorImageryClassifier()
        epochs, labels = self.training()
        report = model.fit(epochs, labels, self.RATE)
        self.assertTrue(report.reliable)
        return model

    def test_held_out_spatial_patterns_classify_both_directions(self):
        model = self.trained()
        self.assertAlmostEqual(model.report.balanced_accuracy, 1.0)
        self.assertEqual(model.report.trials_per_class, (12, 12))
        for label in ("left", "right"):
            with self.subTest(label=label):
                prediction = model.predict(self.trial(label, 100))
                self.assertEqual(prediction.label, label)
                self.assertTrue(prediction.signal_valid)
                self.assertTrue(prediction.model_ready)
                self.assertGreater(prediction.confidence, 0.70)

    def test_each_validation_fold_fits_its_own_spatial_filters(self):
        model = MotorImageryClassifier()
        epochs, labels = self.training()
        with patch.object(model, "_fit_pipeline", wraps=model._fit_pipeline) as fit_pipeline:
            model.fit(epochs, labels, self.RATE)
        # Five held-out folds plus the final model. No held-out trial can
        # contribute to that fold's spatial filters or LDA parameters.
        self.assertEqual(fit_pipeline.call_count, 6)
        for args in fit_pipeline.call_args_list[:5]:
            self.assertLess(len(args.args[0]), len(epochs))
        self.assertEqual(len(fit_pipeline.call_args_list[-1].args[0]), len(epochs))

    def test_identical_classes_fail_validation_and_never_move(self):
        model = MotorImageryClassifier()
        values = self.trial("left", 9)
        report = model.fit([values.copy() for _ in range(20)], ["left", "right"] * 10, self.RATE)
        self.assertFalse(report.reliable)
        self.assertFalse(model.ready)
        prediction = model.predict(values)
        self.assertIsNone(prediction.label)
        self.assertEqual(prediction.reason, "validation_failed")

    def test_dc_offset_and_global_gain_do_not_reverse_direction(self):
        model = self.trained()
        for label in ("left", "right"):
            values = self.trial(label, 200)
            self.assertEqual(model.predict(values + 8000).label, label)
            self.assertEqual(model.predict(values * 2).label, label)

    def test_bad_signal_and_channel_geometry_freeze_output(self):
        model = self.trained()
        valid = self.trial("left", 20)
        invalid = [valid[:100], np.zeros_like(valid), valid[:, :1], valid * 100]
        nan = valid.copy()
        nan[10, 0] = np.nan
        invalid.append(nan)
        for values in invalid:
            prediction = model.predict(values)
            self.assertIsNone(prediction.label)
            self.assertFalse(prediction.signal_valid)
            self.assertEqual(prediction.reason, "invalid_signal")

    def test_training_requires_independent_trials_and_known_labels(self):
        epochs, labels = self.training()
        model = MotorImageryClassifier()
        for args in (
            (epochs[:10], labels[:10], 250),
            (epochs, ["up"] * len(labels), 250),
            (epochs, labels, 50),
        ):
            with self.assertRaises(ValueError):
                model.fit(*args)
        with self.assertRaises(ValueError):
            model.fit(epochs, labels, self.RATE, channel_indices=[0, 0])
        with self.assertRaises(ValueError):
            model.fit(epochs, labels, self.RATE, channel_indices=[0])

    def test_failed_refit_and_reset_discard_old_model(self):
        model = self.trained()
        epochs, labels = self.training()
        with self.assertRaises(ValueError):
            model.fit(epochs[:2], labels[:2], self.RATE)
        self.assertFalse(model.ready)
        self.assertEqual(model.predict(epochs[0]).reason, "not_trained")
        model = self.trained()
        model.reset()
        self.assertFalse(model.ready)
        self.assertIsNone(model.report)

    def test_only_selected_channels_enter_classifier(self):
        model = MotorImageryClassifier()
        epochs, labels = self.training()
        expanded = [np.column_stack((epoch[:, 0], np.zeros(len(epoch)), epoch[:, 1])) for epoch in epochs]
        model.fit(expanded, labels, self.RATE, channel_indices=[0, 2])
        self.assertTrue(model.ready)
        values = self.trial("right", 300)
        prediction = model.predict(np.column_stack((values[:, 0], np.zeros(len(values)), values[:, 1])))
        self.assertEqual(prediction.label, "right")


if __name__ == "__main__":
    unittest.main()
