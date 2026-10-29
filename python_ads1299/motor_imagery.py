"""Trial-trained left/right motor-imagery classifier.

Use separate, labelled two-second trials from sensorimotor electrodes (for
example C3/C4). Predictions describe learned EEG patterns, not arbitrary
thoughts. Training accuracy is never used as evidence of reliable control.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.linalg import eigh
from scipy.signal import butter, sosfiltfilt
from scipy.stats import binomtest


@dataclass(frozen=True)
class MotorImageryReport:
    balanced_accuracy: float
    left_recall: float
    right_recall: float
    p_value: float
    reliable: bool
    trials_per_class: tuple[int, int]
    cv_predictions: tuple[str, ...]


@dataclass(frozen=True)
class MotorImageryPrediction:
    label: str | None
    confidence: float
    signal_valid: bool
    model_ready: bool
    reason: str = ""


class MotorImageryClassifier:
    """Filter-bank CSP plus shrinkage LDA, with whole-trial validation.

    Epochs are samples by channels, in raw microvolts. Every cross-validation
    fold independently fits its spatial filters and classifier. Callers must
    provide independent trials, never split overlapping windows into trials.
    """

    WINDOW_SECONDS = 2.0
    MIN_TRIALS_PER_CLASS = 10
    CONFIDENCE_THRESHOLD = 0.70
    MAX_PEAK_TO_PEAK_UV = 500.0
    MIN_STD_UV = 0.1
    BANDS = ((8.0, 13.0), (13.0, 30.0))
    LABELS = ("left", "right")

    def __init__(self):
        self.reset()

    def reset(self):
        self.sample_rate = None
        self.channel_indices = None
        self.channel_count = None
        self.report = None
        self._pipeline = None
        self._sos = None
        self._training_labels = ()

    @property
    def ready(self):
        return self.report is not None and self.report.reliable

    def _prepare_epoch(self, epoch):
        values = np.asarray(epoch, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.channel_count:
            raise ValueError("EEG channel layout changed")
        values = self.validate_epoch(values, self.sample_rate, self.channel_indices)
        # Filtering uses only this trial. No samples from another trial enter
        # either validation or the online feature window.
        return [sosfiltfilt(sos, values, axis=0).T for sos in self._sos]

    @classmethod
    def validate_epoch(cls, epoch, sample_rate, channel_indices):
        """Check raw microvolts before accepting a single independent trial."""
        rate = float(sample_rate)
        if not math.isfinite(rate) or not 80 <= rate <= 4000:
            raise ValueError("Unsupported EEG sampling rate")
        values = np.asarray(epoch, dtype=np.float64)
        indices = tuple(channel_indices)
        if (values.ndim != 2 or len(indices) < 2 or len(set(indices)) != len(indices)
                or any(not isinstance(index, (int, np.integer)) or index < 0 or index >= values.shape[1]
                       for index in indices)):
            raise ValueError("Select distinct C3/C4 input channels")
        window = int(round(cls.WINDOW_SECONDS * rate))
        if len(values) < window:
            raise ValueError("A full two-second EEG window is required")
        values = values[-window:, indices]
        if not np.all(np.isfinite(values)):
            raise ValueError("EEG contains non-finite values")
        values = values - np.median(values, axis=0)
        if np.any(np.ptp(values, axis=0) > cls.MAX_PEAK_TO_PEAK_UV):
            raise ValueError("EEG amplitude is too large")
        if np.any(np.std(values, axis=0) < cls.MIN_STD_UV):
            raise ValueError("EEG contains a flat channel")
        values = values - np.mean(values, axis=0)
        return values

    @staticmethod
    def _covariance(values):
        covariance = values @ values.T
        covariance /= max(float(np.trace(covariance)), 1e-12)
        size = len(covariance)
        return 0.9 * covariance + 0.1 * np.eye(size) / size

    @staticmethod
    def _features(prepared, filters):
        features = []
        for bands in prepared:
            trial_features = []
            for values, spatial in zip(bands, filters):
                variance = np.var(spatial @ values, axis=1)
                variance = np.maximum(variance, 1e-12)
                trial_features.extend(np.log(variance / np.sum(variance)))
            features.append(trial_features)
        return np.asarray(features, dtype=np.float64)

    def _fit_pipeline(self, prepared, labels):
        filters = []
        for band in range(len(self.BANDS)):
            class_covariances = [
                np.mean([
                    self._covariance(prepared[index][band])
                    for index in np.flatnonzero(labels == label)
                ], axis=0)
                for label in (0, 1)
            ]
            _eigenvalues, eigenvectors = eigh(
                class_covariances[0], sum(class_covariances)
            )
            pairs = min(2, len(eigenvectors) // 2)
            selected = list(range(pairs)) + list(range(len(eigenvectors) - pairs, len(eigenvectors)))
            filters.append(eigenvectors[:, selected].T)
        features = self._features(prepared, filters)
        center = np.mean(features, axis=0)
        scale = np.maximum(np.std(features, axis=0), 1e-6)
        features = (features - center) / scale
        means = [np.mean(features[labels == label], axis=0) for label in (0, 1)]
        residuals = features - np.asarray(means)[labels]
        covariance = residuals.T @ residuals / max(1, len(features) - 2)
        identity = np.eye(features.shape[1])
        regularized = 0.75 * covariance + 0.25 * np.trace(covariance) / len(identity) * identity
        regularized += identity * 1e-6
        weight = np.linalg.solve(regularized, means[1] - means[0])
        intercept = -0.5 * float((means[0] + means[1]) @ weight)
        return filters, center, scale, weight, intercept

    def _scores(self, prepared, pipeline):
        filters, center, scale, weight, intercept = pipeline
        return ((self._features(prepared, filters) - center) / scale) @ weight + intercept

    def fit(self, epochs, labels, sample_rate, channel_indices=None):
        """Train from independent trials; invalid/incomplete trials are refused.

        Re-fitting always discards the previous classifier before validation,
        so failed training cannot leave an earlier model silently controlling.
        ``balanced_accuracy`` and per-class recalls are fractions, not percent.
        """
        self.reset()
        rate = float(sample_rate)
        if not math.isfinite(rate) or not 80 <= rate <= 4000:
            raise ValueError("Unsupported EEG sampling rate")
        epochs = list(epochs)
        labels = list(labels)
        if not epochs or len(epochs) != len(labels):
            raise ValueError("Each EEG trial needs one left/right label")
        first = np.asarray(epochs[0])
        if first.ndim != 2:
            raise ValueError("EEG trial shape must be samples by channels")
        indices = tuple(range(first.shape[1])) if channel_indices is None else tuple(channel_indices)
        if (len(indices) < 2 or len(set(indices)) != len(indices)
                or any(not isinstance(index, (int, np.integer)) or index < 0 or index >= first.shape[1]
                       for index in indices)):
            raise ValueError("Select at least two distinct EEG channels")
        if any(label not in self.LABELS for label in labels):
            raise ValueError("Training labels must be left or right")
        encoded = np.asarray([self.LABELS.index(label) for label in labels], dtype=int)
        counts = tuple(int(np.count_nonzero(encoded == label)) for label in (0, 1))
        if min(counts) < self.MIN_TRIALS_PER_CLASS:
            raise ValueError("At least ten independent trials per direction are required")

        self.sample_rate = rate
        self.channel_count = first.shape[1]
        self.channel_indices = indices
        self._sos = [butter(4, band, btype="bandpass", fs=rate, output="sos") for band in self.BANDS]
        prepared = [self._prepare_epoch(epoch) for epoch in epochs]
        rng = np.random.default_rng(1729)
        folds = min(5, min(counts))
        fold_groups = [np.array_split(rng.permutation(np.flatnonzero(encoded == label)), folds)
                       for label in (0, 1)]
        predictions = np.empty(len(epochs), dtype=int)
        all_indices = np.arange(len(epochs))
        for fold in range(folds):
            test_indices = np.concatenate([groups[fold] for groups in fold_groups])
            training_indices = np.setdiff1d(all_indices, test_indices)
            pipeline = self._fit_pipeline([prepared[index] for index in training_indices], encoded[training_indices])
            scores = self._scores([prepared[index] for index in test_indices], pipeline)
            predictions[test_indices] = (scores >= 0.0).astype(int)
        recalls = tuple(float(np.mean(predictions[encoded == label] == label)) for label in (0, 1))
        balanced_accuracy = sum(recalls) / 2.0
        correct = int(np.count_nonzero(predictions == encoded))
        # For uneven classes, the stronger constant-class baseline is used.
        chance = max(counts) / len(epochs)
        p_value = float(binomtest(correct, len(epochs), chance, alternative="greater").pvalue)
        reliable = balanced_accuracy >= 0.70 and min(recalls) >= 0.60 and p_value <= 0.05
        self.report = MotorImageryReport(
            balanced_accuracy, recalls[0], recalls[1], p_value, bool(reliable), counts,
            tuple(self.LABELS[index] for index in predictions),
        )
        self._pipeline = self._fit_pipeline(prepared, encoded)
        self._training_labels = tuple(labels)
        return self.report

    def predict(self, epoch):
        """Return no movement when training or current signal is unsuitable.

        Confidence is a classifier score mapped to [0, 1], not a clinically
        calibrated probability that a person intended a given direction.
        """
        if self._pipeline is None:
            return MotorImageryPrediction(None, 0.0, False, False, "not_trained")
        try:
            prepared = self._prepare_epoch(epoch)
        except (TypeError, ValueError, OverflowError):
            return MotorImageryPrediction(None, 0.0, False, self.ready, "invalid_signal")
        score = float(self._scores([prepared], self._pipeline)[0])
        if not math.isfinite(score):
            return MotorImageryPrediction(None, 0.0, False, self.ready, "invalid_signal")
        confidence = 1.0 / (1.0 + math.exp(-min(abs(score), 50.0)))
        if not self.ready:
            return MotorImageryPrediction(None, confidence, True, False, "validation_failed")
        if confidence < self.CONFIDENCE_THRESHOLD:
            return MotorImageryPrediction(None, confidence, True, True, "uncertain")
        label = "right" if score >= 0.0 else "left"
        return MotorImageryPrediction(label, confidence, True, True)

    def to_dict(self):
        """Serialize numeric parameters as JSON data, without executable code."""
        if self._pipeline is None or self.report is None:
            raise ValueError("Train a model before saving")
        filters, center, scale, weight, intercept = self._pipeline
        return {
            "format": "cw_eeg_motor_imagery", "version": 1,
            "sample_rate": self.sample_rate, "channel_count": self.channel_count,
            "channel_indices": list(self.channel_indices),
            "bands": [list(band) for band in self.BANDS],
            "pipeline": {
                "filters": [values.tolist() for values in filters],
                "center": center.tolist(), "scale": scale.tolist(),
                "weight": weight.tolist(), "intercept": intercept,
            },
            "validation": {
                "labels": list(self._training_labels),
                "predictions": list(self.report.cv_predictions),
            },
        }

    @classmethod
    def from_dict(cls, data):
        """Load a checked layout; recompute validation gates from held-out labels."""
        try:
            if (not isinstance(data, dict) or data.get("format") != "cw_eeg_motor_imagery"
                    or data.get("version") != 1
                    or data.get("bands") != [list(band) for band in cls.BANDS]):
                raise ValueError("Unsupported model format or feature version")
            rate = float(data["sample_rate"])
            count = data["channel_count"]
            indices = tuple(data["channel_indices"])
            if (not math.isfinite(rate) or not 80 <= rate <= 4000
                    or type(count) is not int or not 2 <= count <= 64
                    or len(indices) < 2 or len(set(indices)) != len(indices)
                    or any(type(index) is not int or not 0 <= index < count for index in indices)):
                raise ValueError("Invalid model sampling rate or channels")
            spatial_count = 2 * min(2, len(indices) // 2)
            feature_count = spatial_count * len(cls.BANDS)
            stored = data["pipeline"]
            filters = [np.asarray(item, dtype=np.float64) for item in stored["filters"]]
            center = np.asarray(stored["center"], dtype=np.float64)
            scale = np.asarray(stored["scale"], dtype=np.float64)
            weight = np.asarray(stored["weight"], dtype=np.float64)
            intercept = float(stored["intercept"])
            if (len(filters) != len(cls.BANDS)
                    or any(item.shape != (spatial_count, len(indices)) for item in filters)
                    or any(item.shape != (feature_count,) for item in (center, scale, weight))
                    or any(not np.all(np.isfinite(item)) for item in (*filters, center, scale, weight))
                    or not math.isfinite(intercept) or np.any(scale < 1e-6)
                    or any(np.any(np.abs(item) > 1e12) for item in (*filters, center, scale, weight))
                    or abs(intercept) > 1e12):
                raise ValueError("Invalid model parameter dimensions or values")
            validation = data["validation"]
            labels, predictions = tuple(validation["labels"]), tuple(validation["predictions"])
            if (not 20 <= len(labels) <= 10000 or len(labels) != len(predictions)
                    or any(label not in cls.LABELS for label in labels + predictions)):
                raise ValueError("Missing independent-trial validation results")
            encoded = np.asarray([cls.LABELS.index(label) for label in labels])
            predicted = np.asarray([cls.LABELS.index(label) for label in predictions])
            counts = tuple(int(np.count_nonzero(encoded == label)) for label in (0, 1))
            if min(counts) < cls.MIN_TRIALS_PER_CLASS:
                raise ValueError("Too few training trials")
            recalls = tuple(float(np.mean(predicted[encoded == label] == label)) for label in (0, 1))
            accuracy = sum(recalls) / 2.0
            correct = int(np.count_nonzero(predicted == encoded))
            p_value = float(binomtest(correct, len(labels), max(counts) / len(labels), alternative="greater").pvalue)
            reliable = accuracy >= 0.70 and min(recalls) >= 0.60 and p_value <= 0.05
            model = cls()
            model.sample_rate, model.channel_count, model.channel_indices = rate, count, indices
            model._sos = [butter(4, band, btype="bandpass", fs=rate, output="sos") for band in cls.BANDS]
            model._pipeline = filters, center, scale, weight, intercept
            model._training_labels = labels
            model.report = MotorImageryReport(accuracy, recalls[0], recalls[1], p_value,
                                             bool(reliable), counts, predictions)
            return model
        except (KeyError, TypeError, OverflowError, IndexError) as exc:
            raise ValueError("Invalid motor-imagery model file") from exc
