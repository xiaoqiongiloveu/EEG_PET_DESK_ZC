"""Regression against manually marked blinks in an external recording."""

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python_ads1299"))
from eye_blink import EyeBlinkDetector


RATE = 250
START_SECONDS = 910
DURATION_SECONDS = 20
CSV_PATH = Path(r"C:\Users\31726\Desktop\20261019_094302_250sps_filtered.csv")
CHANNELS = "F5,F3,FZ,F4,F6,AF8,FPZ,AF7"
BLINK_WINDOWS = (
    (919.784, 919.584, 919.984),
    (921.000, 920.800, 921.200),
    (921.712, 921.512, 921.912),
    (922.476, 922.276, 922.676),
    (923.816, 923.616, 924.016),
)
NON_BLINK_WINDOW = (923.10, 923.45)


class MarkedBlinkRecordingTests(unittest.TestCase):
    @unittest.skipUnless(CSV_PATH.is_file(), "marked EEG CSV is not installed")
    def test_marked_blinks_and_nearby_non_blink(self):
        with CSV_PATH.open(encoding="utf-8-sig") as stream:
            self.assertEqual(stream.readline().strip(), CHANNELS)
        data = np.loadtxt(
            CSV_PATH, delimiter=",", skiprows=1 + START_SECONDS * RATE,
            max_rows=DURATION_SECONDS * RATE, encoding="utf-8-sig",
        )
        self.assertEqual(data.shape, (DURATION_SECONDS * RATE, 8))

        detector = EyeBlinkDetector()
        detected_peaks = []
        for start in range(0, len(data), 25):
            count = detector.process(data[start:start + 25], RATE)
            self.assertLessEqual(
                count, 1,
                f"multiple peaks in chunk at {START_SECONDS + start / RATE:.3f} s",
            )
            if count:
                # The detector's sample positions start at the loaded slice.
                peak_time = START_SECONDS + detector._last_counted / RATE
                detected_peaks.append(peak_time)

        for labeled_peak, begin, end in BLINK_WINDOWS:
            with self.subTest(peak_seconds=labeled_peak):
                observed = [time for time in detected_peaks if begin <= time < end]
                self.assertEqual(
                    len(observed), 1,
                    f"expected one peak near {labeled_peak:.3f} s; "
                    f"observed {observed}; all peaks: {detected_peaks}",
                )

        begin, end = NON_BLINK_WINDOW
        observed = [time for time in detected_peaks if begin <= time < end]
        self.assertEqual(observed, [], f"unexpected peak near 923.276 s: {observed}")


if __name__ == "__main__":
    unittest.main()
