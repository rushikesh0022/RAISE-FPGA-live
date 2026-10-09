import unittest

import numpy as np

from scripts.collect_zcu104_training_telemetry import FEATURES
from scripts.run_live_zcu104_prediction import LiveHistory


class LiveHistoryTests(unittest.TestCase):
    def row(self, t, run='run1', valid=True):
        return {'timestamp_s': str(t), 'run_id': run, 'input_valid': str(valid),
                **{f: str(40 + t) for f in FEATURES},
                **{f + '__timestamp_s': str(t) for f in FEATURES}}

    def test_history_is_causal_and_requires_33_samples(self):
        history = LiveHistory(FEATURES)
        for i in range(32):
            ready, _ = history.add(self.row(i / 10))
            self.assertIsNone(ready)
        ready, _ = history.add(self.row(3.2))
        self.assertEqual(ready[1].shape, (33, 7))
        np.testing.assert_allclose(ready[1][:, 0], 40 + np.arange(33) / 10, atol=1e-5)

    def test_run_boundary_and_gap_require_new_history(self):
        history = LiveHistory(FEATURES)
        for i in range(40):
            history.add(self.row(i / 10))
        ready, reset = history.add(self.row(4., run='run2'))
        self.assertTrue(reset)
        self.assertIsNone(ready)
        ready, reset = history.add(self.row(5., run='run2'))
        self.assertTrue(reset)
        self.assertIsNone(ready)

    def test_invalid_row_clears_history(self):
        history = LiveHistory(FEATURES)
        for i in range(40):
            history.add(self.row(i / 10))
        ready, reset = history.add(self.row(4., valid=False))
        self.assertTrue(reset)
        self.assertIsNone(ready)
        self.assertIsNone(history.add(self.row(4.1))[0])

    def test_uses_sensor_time_not_future_row_values(self):
        history = LiveHistory(FEATURES)
        for i in range(40):
            row = self.row(i / 10 + .05)
            for f in FEATURES:
                row[f + '__timestamp_s'] = str(i / 10 + .02)
            ready, _ = history.add(row)
        self.assertIsNotNone(ready)
        # End grid is 3.9, whereas the latest sensor was sampled at 3.92.
        self.assertAlmostEqual(float(ready[1][-1, 0]), 43.85, places=4)


if __name__ == '__main__':
    unittest.main()
