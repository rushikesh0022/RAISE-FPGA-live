"""Run the saved physics CNN+GRU on a board SSH CSV stream or a replay CSV."""
import argparse
from collections import deque
import csv
import json
from pathlib import Path
import queue
import re
import shlex
import subprocess
import threading
import time

import numpy as np
import torch

from model.raise_fpga import PhysicsGuidedNestedRaiseFPGA


class LiveHistory:
    def __init__(self, features):
        self.features = features
        self.samples = {f: deque(maxlen=128) for f in features}
        self.run_id = None
        self.last_row = None
        self.last_anchor = None

    def add(self, row):
        t = float(row['timestamp_s'])
        run = row['run_id']
        reset = run != self.run_id or (self.last_row is not None and (t <= self.last_row or t - self.last_row > .5))
        if reset:
            for buf in self.samples.values():
                buf.clear()
            self.last_anchor = None
        self.run_id, self.last_row = run, t
        valid = str(row.get('input_valid', 'true')).lower() in ('true', '1')
        if not valid:
            for buf in self.samples.values():
                buf.clear()
            self.last_anchor = None
            return None, True
        for feature in self.features:
            value = float(row[feature])
            measured = float(row.get(feature + '__timestamp_s') or t)
            if not np.isfinite([value, measured]).all() or measured > t + 1e-6:
                raise ValueError('Invalid sensor reading or acquisition time: ' + feature)
            self.samples[feature].append((measured, value))
        anchor = np.floor((t + 1e-9) * 10) / 10
        if self.last_anchor is not None and anchor <= self.last_anchor:
            return None, reset
        grid = anchor - np.arange(32, -1, -1) / 10
        history = np.empty((33, len(self.features)), dtype=np.float32)
        for index, feature in enumerate(self.features):
            readings = np.asarray(self.samples[feature])
            positions = np.searchsorted(readings[:, 0], grid + 1e-9, side='right') - 1
            if (positions < 0).any():
                return None, reset
            stale = grid - readings[positions, 0]
            if (stale > .2 + 1e-9).any() or (stale < -1e-8).any():
                return None, reset
            if (np.diff(readings[positions[0]:positions[-1] + 1, 0]) > .5).any():
                return None, reset
            history[:, index] = readings[positions, 1]
        self.last_anchor = anchor
        return (anchor, history), reset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--ssh', help='Board login, e.g. xilinx@192.168.1.25')
    source.add_argument('--input', type=Path, help='Replay a recorded CSV on the PC')
    p.add_argument('--remote-script', default='collect_zcu104_training_telemetry.py')
    p.add_argument('--with-cnn-workload', action='store_true', help='Run the supplied ARM-CPU CNN session on the board; NOT FPGA fabric')
    p.add_argument('--checkpoint', type=Path, default=Path('outputs/models/combined_board_20261006_physics_cnn_gru.pt'))
    p.add_argument('--duration-s', type=float, default=1800)
    p.add_argument('--predict-every-s', type=float, default=1.)
    p.add_argument('--output-dir', type=Path, default=Path('results/live'))
    args = p.parse_args()
    if args.with_cnn_workload and not args.ssh:
        p.error('--with-cnn-workload requires --ssh')
    if args.predict_every_s < .1 or args.duration_s <= 0:
        p.error('Prediction interval must be at least 0.1s and duration positive.')
    torch.set_num_threads(2)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint.get('model_class') != 'PhysicsGuidedNestedRaiseFPGA':
        raise ValueError('This live runner expects the retrained physics CNN+GRU checkpoint.')
    features = checkpoint['features']
    model = PhysicsGuidedNestedRaiseFPGA(channels=len(features))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    mean, scale = np.asarray(checkpoint['scaler_mean']), np.asarray(checkpoint['scaler_scale'])
    cutoffs = np.asarray(checkpoint['probability_cutoffs'])
    stamp = time.strftime('%Y%m%d_%H%M%S')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path = args.output_dir / ('telemetry_' + stamp + '.csv')
    prediction_path = args.output_dir / ('predictions_' + stamp + '.jsonl')
    process = None
    remote_exit_code = 0
    if args.ssh:
        if not re.fullmatch(r'[A-Za-z0-9_.-]+@[A-Za-z0-9.-]+', args.ssh):
            p.error('Use a user@hostname or user@IPv4 SSH destination.')
        remote_script = str(Path(args.remote_script).parent / 'run_board_cnn_session.py') if args.with_cnn_workload else args.remote_script
        remote = ['python3', '-u', remote_script, '--stream-csv', '--output',
                  'recordings/live_' + stamp + '.csv', '--workload', 'tiny_cnn_arm_cpu' if args.with_cnn_workload else 'telemetry_only',
                  '--duration-s', str(args.duration_s), '--hz', '10']
        process = subprocess.Popen(['ssh', '-T', args.ssh, ' '.join(shlex.quote(item) for item in remote)],
                                   stdout=subprocess.PIPE, text=True, encoding='utf-8', bufsize=1)
        stream = process.stdout
    else:
        stream = args.input.open(newline='')
    messages = queue.Queue(maxsize=256)

    def read_stream():
        try:
            for row in csv.DictReader(stream):
                messages.put(('row', row))
            messages.put(('eof', None))
        except Exception as exc:
            messages.put(('error', str(exc)))

    threading.Thread(target=read_stream, daemon=True).start()
    history = LiveHistory(features)
    pending = []
    last_prediction = -float('inf')
    last_arrival = time.monotonic()
    stale_printed = False
    print('Model loaded. Waiting for a valid 3.2-second history...', flush=True)
    try:
        with telemetry_path.open('x', newline='') as raw_file, prediction_path.open('x') as pred_file:
            writer = None

            def emit(record):
                pred_file.write(json.dumps(record, allow_nan=False) + '\n')
                pred_file.flush()

            while True:
                try:
                    kind, row = messages.get(timeout=1)
                except queue.Empty:
                    if time.monotonic() - last_arrival > 2 and not stale_printed:
                        print('Telemetry stale: live predictions paused.', flush=True)
                        stale_printed = True
                    continue
                if kind == 'error':
                    raise RuntimeError(row)
                if kind == 'eof':
                    break
                if writer is None:
                    missing = {'timestamp_s', 'run_id', *features}.difference(row)
                    if missing:
                        raise ValueError('Stream missing columns: ' + str(sorted(missing)))
                    writer = csv.DictWriter(raw_file, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                raw_file.flush()
                if time.monotonic() - last_arrival > 2 and args.ssh:
                    history = LiveHistory(features)
                    pending.clear()
                    last_prediction = -float('inf')
                last_arrival, stale_printed = time.monotonic(), False
                ready, reset = history.add(row)
                if reset:
                    pending.clear()
                    last_prediction = -float('inf')
                if ready is None:
                    continue
                anchor, values = ready
                pl = np.asarray(history.samples['temp_pl_temp_C'])
                remaining = []
                for forecast in pending:
                    due = forecast['anchor_timestamp_s'] + forecast['horizon_s']
                    if pl[-1, 0] < due:
                        remaining.append(forecast)
                        continue
                    i = np.searchsorted(pl[:, 0], due, side='right') - 1
                    if i >= 0 and due - pl[i, 0] <= .2:
                        actual = float(pl[i, 1])
                        emit({'kind': 'temperature_followup', **forecast,
                              'actual_pl_temperature_c': actual,
                              'actual_timestamp_s': float(pl[i, 0]),
                              'absolute_error_c': abs(actual - forecast['predicted_pl_temperature_c'])})
                pending = remaining
                if anchor - last_prediction < args.predict_every_s - 1e-6:
                    continue
                scaled = ((values - mean) / scale).astype(np.float32)
                before = time.perf_counter()
                with torch.inference_mode():
                    output = model(torch.from_numpy(scaled).unsqueeze(0))
                latency = (time.perf_counter() - before) * 1000
                risk = output['cumulative_risk'][0].numpy()
                current = float(values[-1, features.index('temp_pl_temp_C')])
                future = current + output['future_temperature_delta'][0].numpy()
                record = {'kind': 'prediction', 'run_id': history.run_id,
                          'anchor_timestamp_s': float(anchor), 'current_pl_temperature_c': current,
                          'model_inference_ms': latency,
                          'risk_scores_percent': (100 * risk).round(3).tolist(),
                          'predicted_transition': (risk >= cutoffs).tolist(),
                          'thresholds_c': [49, 50, 51], 'horizons_s': [.5, 2., 5.],
                          'predicted_pl_temperature_c': future.tolist(),
                          'note': 'Risk scores are exploratory and not calibrated safety probabilities.'}
                emit(record)
                print('t={:.1f}s PL={:.2f} C | forecast 0.5/2/5s={} C | risk rows 49/50/51C={} % | inference={:.2f}ms'.format(
                    anchor, current, [round(float(v), 2) for v in future],
                    [[round(float(v), 2) for v in values] for values in 100 * risk], latency), flush=True)
                for h, temperature in zip((.5, 2., 5.), future):
                    pending.append({'run_id': history.run_id, 'anchor_timestamp_s': float(anchor),
                                    'horizon_s': h, 'predicted_pl_temperature_c': float(temperature)})
                last_prediction = anchor
    except KeyboardInterrupt:
        print('Stopped by user.', flush=True)
    finally:
        if process:
            if process.poll() is None:
                process.terminate()
            try:
                code = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                code = process.wait()
            if code:
                remote_exit_code = code
                print('SSH/collector exited with code {}. Check board messages above.'.format(code))
        else:
            stream.close()
    print('Telemetry saved: ' + str(telemetry_path))
    print('Predictions and temperature follow-ups saved: ' + str(prediction_path))
    if remote_exit_code:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
