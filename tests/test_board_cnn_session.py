import argparse
import contextlib
import csv
import io
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

from scripts.run_board_cnn_session import TinyCNN, convolution, max_pool, phase_at, guard_reason
from scripts.collect_zcu104_training_telemetry import collect, FEATURES


class CNNWorkloadTests(unittest.TestCase):
    def test_actual_convolution_and_relu(self):
        image = [[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]]
        kernel = [[[[1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]]]
        self.assertEqual(convolution(image, kernel), [[[45.]]])
        negative = [[[[-1.] * 3 for _ in range(3)]]]
        self.assertEqual(convolution(image, negative), [[[0.]]])

    def test_pool(self):
        self.assertEqual(max_pool([[[1, 2], [3, 4]]]), [[[4]]])

    def test_forward_deterministic_and_finite(self):
        a, b = TinyCNN(), TinyCNN()
        self.assertEqual(a.forward(0), b.forward(0))
        self.assertEqual(len(a.forward(0)), 3)
        self.assertNotEqual(a.forward(0), a.forward(1))

    def test_stages_and_abort(self):
        self.assertEqual([phase_at(t, 60) for t in (0, 14.9, 15, 44.9, 45, 59)],
                         ['idle', 'idle', 'cnn_arm_cpu', 'cnn_arm_cpu', 'cool_down', 'cool_down'])
        row = dict(input_valid=True, temp_pl_temp_C=46, temp_ps_temp_C=47, temp_remote_temp_C=48)
        self.assertIsNone(guard_reason(row, 50))
        self.assertEqual(guard_reason(dict(row, temp_remote_temp_C=50), 50), 'temperature_abort')
        self.assertEqual(guard_reason(dict(row, input_valid=False), 50), 'invalid_telemetry')

    def test_observer_stage_is_saved_and_streamed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / 'raw'
            raw.write_text('46')
            mapping = root / 'map.json'
            import json
            mapping.write_text(json.dumps({f: {'path': str(raw), 'scale': 1} for f in FEATURES}))
            args = argparse.Namespace(sensor_map=mapping, sys_root=root, check_only=False,
                                      output=root / 'readings.csv', workload='fixture', rep_id=1,
                                      duration_s=.03, hz=100, stage='idle', stage_file=None, stream_csv=True)
            def mark(row):
                row['stage'] = 'cnn_arm_cpu'
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(io.StringIO()):
                collect(args, row_observer=mark)
            with args.output.open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertTrue(rows)
            self.assertTrue(all(r['stage'] == 'cnn_arm_cpu' for r in rows))
            self.assertEqual(list(csv.DictReader(io.StringIO(capture.getvalue()))), rows)

    def test_complete_cpu_session_with_mock_sensors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / 'raw'
            raw.write_text('46')
            mapping = root / 'map.json'
            import json
            mapping.write_text(json.dumps({f: {'path': str(raw), 'scale': 1} for f in FEATURES}))
            output = root / 'telemetry.csv'
            result = subprocess.run([sys.executable, '-m', 'scripts.run_board_cnn_session',
                                     '--sensor-map', str(mapping), '--output', str(output),
                                     '--duration-s', '4', '--stream-csv'],
                                    capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            rows = list(csv.DictReader(io.StringIO(result.stdout)))
            self.assertEqual(set(r['stage'] for r in rows), {'idle', 'cnn_arm_cpu', 'cool_down'})
            self.assertTrue(output.with_suffix('.metadata.json').exists())
            entries = [json.loads(line) for line in output.with_suffix('.workload.jsonl').read_text().splitlines()]
            self.assertTrue(any(entry['kind'] == 'cnn_inference' for entry in entries))
            self.assertEqual(entries[-1]['kind'], 'workload_stopped')
            self.assertIsNone(entries[-1]['abort_reason'])


if __name__ == '__main__':
    unittest.main()
