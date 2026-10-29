"""Regression cases based on the blink timing in the annotated EEG capture."""

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python_ads1299"))
from eye_blink import EyeBlinkDetector


RATE = 250
ANNOTATED_TIMES = (15.724, 17.036, 18.356, 18.724,
                   20.040, 21.728, 23.832, 24.720)


def annotated_blinks_with_drifting_fpz():
    """Eight frontal pulses, including a close pair, while FPZ drifts."""
    rng = np.random.default_rng(309)
    time = np.arange(30 * RATE) / RATE
    data = rng.normal(0.0, 1.2, (len(time), 8))
    data[:, [1, 3, 5]] = 0.0  # Disconnected leads in the supplied capture.

    # The FPZ drift greatly exceeds the blink amplitude and crosses zero.
    data[:, 6] = 1700 * np.sin(2 * np.pi * 0.07 * time)
    data[:, 6] += 650 * ((time >= 14) & (time < 20))
    amplitudes = (180, 105, 120, 230, 135, 120, 260, 105)
    for center, amplitude in zip(ANNOTATED_TIMES, amplitudes):
        pulse = np.exp(-0.5 * ((time - center) / 0.075) ** 2)
        data += pulse[:, None] * np.asarray(
            (amplitude, 0, 1.1 * amplitude, 0,
             0.9 * amplitude, 0, 0, 0.85 * amplitude)
        )
    return data


class AnnotatedBlinkRegressionTests(unittest.TestCase):
    def test_drifting_fpz_does_not_merge_eight_annotated_blinks(self):
        data = annotated_blinks_with_drifting_fpz()
        for chunk_size in (25, 37, len(data)):
            with self.subTest(chunk_size=chunk_size):
                detector = EyeBlinkDetector()
                count = sum(
                    detector.process(data[index:index + chunk_size], RATE)
                    for index in range(0, len(data), chunk_size)
                )
                self.assertEqual(count, len(ANNOTATED_TIMES))


if __name__ == "__main__":
    unittest.main()
