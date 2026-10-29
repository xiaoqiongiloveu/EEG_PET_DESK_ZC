"""Regression against the eight manually labeled blinks in an external recording."""

import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python_ads1299"))
from eye_blink import EyeBlinkDetector


RATE = 250
CSV_PATH = Path(r"C:\Users\31726\Desktop\20261016_173958_250sps_filtered.csv")
CHANNELS = "F5,F3,FZ,F4,F6,AF8,FPZ,AF7"

# Peak windows are disjoint, including the pair only 0.368 s apart.
LABELED_WINDOWS = (
    (15.724, 15.574, 15.874),
    (17.036, 16.886, 17.186),
    (18.356, 18.206, 18.506),
    (18.724, 18.574, 18.874),
    (20.040, 19.890, 20.190),
    (21.728, 21.578, 21.878),
    (23.832, 23.682, 23.982),
    (24.720, 24.570, 24.870),
)


class LabeledRecordingTests(unittest.TestCase):
    @unittest.skipUnless(CSV_PATH.is_file(), "labeled EEG CSV is not installed")
    def test_each_labeled_blink_is_counted_once(self):
        with CSV_PATH.open(encoding="utf-8-sig") as stream:
            self.assertEqual(stream.readline().strip(), CHANNELS)
        data = np.loadtxt(
            CSV_PATH, delimiter=",", skiprows=1, max_rows=27 * RATE,
            encoding="utf-8-sig",
        )
        self.assertEqual(data.shape, (27 * RATE, 8))

        detector = EyeBlinkDetector()
        detected_peaks = []
        for start in range(0, len(data), 25):
            stop = min(start + 25, len(data))
            count = detector.process(data[start:stop], RATE)
            if count:
                # Confirmation occurs later than the EEG peak and its latency
                # may change. The detector records the peak's sample index.
                detected_peaks.extend([detector._last_counted / RATE] * count)

        for peak, begin, end in LABELED_WINDOWS:
            with self.subTest(peak_seconds=peak):
                observed = [time for time in detected_peaks if begin <= time < end]
                self.assertEqual(
                    len(observed), 1,
                    f"expected one detected peak near {peak:.3f} s; "
                    f"observed {observed}; all detected peaks: {detected_peaks}",
                )


if __name__ == "__main__":
    unittest.main()
