import argparse
import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest

from scripts.collect_zcu104_training_telemetry import (
    ExcursionTracker, FEATURES, LEGACY_CHANNELS, collect, discover, read_number,
)


class BoardCollectorTests(unittest.TestCase):
    def make_sensors(self, root):
        device = root / 'bus/iio/devices/iio:device9'
        device.mkdir(parents=True)
        (device / 'name').write_text('xilinx-ams')
        for f, base in LEGACY_CHANNELS.items():
            temp = f.startswith('temp_')
            (device / (base + '_raw')).write_text('41760' if temp else '18500')
            (device / (base + '_scale')).write_text('7.771515' if temp else '0.045776')
            if temp:
                (device / (base + '_offset')).write_text('-36058')
        monitor = root / 'class/hwmon/hwmon8'
        monitor.mkdir(parents=True)
        (monitor / 'name').write_text('ina226')
        (monitor / 'power1_input').write_text('15100000')

    def test_units_and_device_discovery(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_sensors(root)
            mapping = discover(root)
            self.assertEqual(set(mapping), set(FEATURES))
            values = {f: (read_number(Path(s['path'])) + s['offset']) * s['scale'] for f, s in mapping.items()}
            self.assertAlmostEqual(values['temp_pl_temp_C'], 44.31317853)
            self.assertAlmostEqual(values['iio_vccint_V'], .846856)
            self.assertAlmostEqual(values['power_pin_W'], 15.1)

    def test_excursion_requires_persistence_and_prior_below(self):
        tracker = ExcursionTracker(49)
        self.assertFalse(tracker.update(0., 48.))
        self.assertFalse(tracker.update(.1, 49.1))
        self.assertFalse(tracker.update(.4, 49.1))
        self.assertTrue(tracker.update(.6, 49.1))
        self.assertEqual(tracker.count, 1)
        left_censored = ExcursionTracker(49)
        for t in (0., .1, .3, .5, .8):
            self.assertFalse(left_censored.update(t, 50.))
        self.assertEqual(left_censored.count, 0)

    def test_gap_breaks_persistence(self):
        tracker = ExcursionTracker(49)
        for t, value in ((0., 48.), (.1, 50.), (.9, 50.), (1.1, 50.), (1.4, 50.)):
            self.assertFalse(tracker.update(t, value))
        self.assertEqual(tracker.count, 0)

    def test_capture_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_sensors(root)
            args = argparse.Namespace(sensor_map=None, sys_root=root, check_only=False,
                                      output=root / 'readings.csv', workload='fixture',
                                      rep_id=1, duration_s=.035, hz=100., stage='idle', stage_file=None, stream_csv=False)
            with contextlib.redirect_stdout(io.StringIO()):
                collect(args)
            with args.output.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertGreater(len(rows), 0)
            self.assertTrue(all(float(r['power_pin_W']) == 15.1 for r in rows))
            self.assertTrue(all(f + '__timestamp_s' in rows[0] for f in FEATURES))
            self.assertTrue(args.output.with_suffix('.metadata.json').exists())
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(ValueError):
                    collect(args)


if __name__ == '__main__':
    unittest.main()
