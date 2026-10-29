"""Regression against marked non-blink bursts in an external recording."""

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python_ads1299"))
from eye_blink import EyeBlinkDetector


RATE = 250
CSV_PATH = Path(r"C:\Users\31726\Desktop\20261019_092035_250sps_filtered.csv")
CHANNELS = "F5,F3,FZ,F4,F6,AF8,FPZ,AF7"
NON_BLINK_WINDOWS = (
    (72.1, 72.8),
    (73.1, 73.8),
    (74.1, 74.5),
    (75.75, 76.15),
)


class MarkedNonBlinkRecordingTests(unittest.TestCase):
    @unittest.skipUnless(CSV_PATH.is_file(), "marked EEG CSV is not installed")
    def test_no_marked_oscillatory_burst_is_counted(self):
        with CSV_PATH.open(encoding="utf-8-sig") as stream:
            self.assertEqual(stream.readline().strip(), CHANNELS)
        data = np.loadtxt(
            CSV_PATH, delimiter=",", skiprows=1, max_rows=77 * RATE,
            encoding="utf-8-sig",
        )
        self.assertEqual(data.shape, (77 * RATE, 8))

        detector = EyeBlinkDetector()
        detected_peaks = []
        for start in range(0, len(data), 25):
            count = detector.process(data[start:start + 25], RATE)
            self.assertLessEqual(count, 1, f"multiple peaks in chunk at {start / RATE:.3f} s")
            if count:
                detected_peaks.append(detector._last_counted / RATE)

        for begin, end in NON_BLINK_WINDOWS:
            with self.subTest(window=(begin, end)):
                observed = [time for time in detected_peaks if begin <= time < end]
                self.assertEqual(observed, [], f"detected peaks: {observed}")


if __name__ == "__main__":
    unittest.main()
