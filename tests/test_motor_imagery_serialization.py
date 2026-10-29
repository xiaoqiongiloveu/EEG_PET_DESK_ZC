"""Personal model JSON persistence preserves predictions and validation gates."""

import copy
import json
from pathlib import Path
import sys
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python_ads1299"))

from motor_imagery import MotorImageryClassifier


class MotorImagerySerializationTests(unittest.TestCase):
    RATE = 250
    CHANNELS = (2, 5)

    @classmethod
    def epoch(cls, label, seed):
        rng = np.random.default_rng(seed)
        time = np.arange(2 * cls.RATE) / cls.RATE
        values = rng.normal(0.0, 1.0, (len(time), 8))
        gains = (5.0, 15.0) if label == "left" else (15.0, 5.0)
        for channel, gain, frequency in zip(cls.CHANNELS, gains, (10.0, 10.5)):
            phase = rng.uniform(0.0, 2.0 * np.pi)
            values[:, channel] += gain * np.sin(2.0 * np.pi * frequency * time + phase)
            values[:, channel] += gain * 0.5 * np.sin(2.0 * np.pi * 20.0 * time + phase)
        return values

    @classmethod
    def setUpClass(cls):
        cls.model = MotorImageryClassifier()
        labels = ["left", "right"] * 10
        cls.model.fit([cls.epoch(label, seed) for seed, label in enumerate(labels)],
                      labels, cls.RATE, cls.CHANNELS)
        if not cls.model.ready:
            raise AssertionError("Synthetic persistence fixture must pass trial validation")
        # Round-trip through strict JSON, as the UI's model file does. This also
        # catches parameters accidentally left as numpy arrays or scalar types.
        cls.saved = json.loads(json.dumps(cls.model.to_dict(), allow_nan=False))

    def test_json_roundtrip_preserves_layout_report_and_both_predictions(self):
        loaded = MotorImageryClassifier.from_dict(copy.deepcopy(self.saved))
        self.assertEqual(loaded.sample_rate, self.RATE)
        self.assertEqual(loaded.channel_count, 8)
        self.assertEqual(loaded.channel_indices, self.CHANNELS)
        self.assertEqual(loaded.report, self.model.report)
        for label in ("left", "right"):
            with self.subTest(label=label):
                values = self.epoch(label, 200)
                before = self.model.predict(values)
                after = loaded.predict(values)
                self.assertEqual(after.label, label)
                self.assertEqual(after, before)
        self.assertEqual(loaded.to_dict(), self.saved)

    def test_loaded_validation_is_recomputed_and_can_disable_control(self):
        saved = copy.deepcopy(self.saved)
        saved["validation"]["predictions"] = ["right"] * 20
        # Extra claimed metrics cannot turn a failed independent-trial result
        # into an authorized controller.
        saved["validation"].update(reliable=True, balanced_accuracy=1.0)
        loaded = MotorImageryClassifier.from_dict(saved)
        self.assertFalse(loaded.ready)
        self.assertAlmostEqual(loaded.report.balanced_accuracy, 0.5)
        self.assertEqual(loaded.report.left_recall, 0.0)
        self.assertEqual(loaded.report.right_recall, 1.0)
        prediction = loaded.predict(self.epoch("left", 201))
        self.assertIsNone(prediction.label)
        self.assertTrue(prediction.signal_valid)
        self.assertEqual(prediction.reason, "validation_failed")
        reloaded = MotorImageryClassifier.from_dict(json.loads(json.dumps(loaded.to_dict())))
        self.assertFalse(reloaded.ready)

    def test_untrained_model_cannot_be_saved(self):
        with self.assertRaises(ValueError):
            MotorImageryClassifier().to_dict()

    def test_unknown_formats_and_incompatible_channel_layout_are_rejected(self):
        edits = (
            ("format", "another_model"),
            ("version", 2),
            ("bands", [[9, 13], [13, 30]]),
            ("sample_rate", 50),
            ("sample_rate", float("nan")),
            ("sample_rate", float("inf")),
            ("channel_count", 1),
            ("channel_count", 8.0),
            ("channel_indices", [2, 2]),
            ("channel_indices", [2, 8]),
            ("channel_indices", [-1, 5]),
            ("channel_indices", [2]),
            ("channel_indices", [True, 5]),
        )
        for key, value in edits:
            with self.subTest(key=key, value=value):
                saved = copy.deepcopy(self.saved)
                saved[key] = value
                with self.assertRaises(ValueError):
                    MotorImageryClassifier.from_dict(saved)
        for saved in (None, [], {}, {"format": "cw_eeg_motor_imagery", "version": 1}):
            with self.subTest(incomplete=saved):
                with self.assertRaises(ValueError):
                    MotorImageryClassifier.from_dict(saved)

    def test_corrupt_numeric_parameters_cannot_be_loaded(self):
        malformed = (
            ("filters", []),
            ("filters", [[[1.0, 0.0]], [[0.0, 1.0]]]),
            ("center", [0.0]),
            ("scale", [0.0] * 4),
            ("scale", [float("nan")] * 4),
            ("weight", [[0.0] * 4]),
            ("weight", [float("inf")] * 4),
            ("weight", [1e100] * 4),
            ("intercept", float("nan")),
        )
        for key, value in malformed:
            with self.subTest(key=key, value=value):
                saved = copy.deepcopy(self.saved)
                saved["pipeline"][key] = value
                with self.assertRaises(ValueError):
                    MotorImageryClassifier.from_dict(saved)

    def test_missing_or_invalid_independent_trial_evidence_is_rejected(self):
        malformed = (
            {"labels": ["left", "right"] * 9, "predictions": ["left", "right"] * 9},
            {"labels": ["left"] * 20, "predictions": ["left"] * 20},
            {"labels": self.saved["validation"]["labels"], "predictions": ["left"]},
            {"labels": self.saved["validation"]["labels"], "predictions": ["rest"] * 20},
            {"labels": [], "predictions": []},
        )
        for validation in malformed:
            with self.subTest(validation=validation):
                saved = copy.deepcopy(self.saved)
                saved["validation"] = validation
                with self.assertRaises(ValueError):
                    MotorImageryClassifier.from_dict(saved)


if __name__ == "__main__":
    unittest.main()
