"""ARM-CPU CNN workload plus telemetry. NOT FPGA-fabric inference.

Standard-library-only, deterministic untrained CNN, synthetic inputs. This is
a computational workload for testing telemetry/prediction, not a classifier.
"""
import argparse
import json
import math
import multiprocessing as mp
from pathlib import Path
import queue
import random
import signal
import sys
import time

if __package__:
    from .collect_zcu104_training_telemetry import collect
else:
    from collect_zcu104_training_telemetry import collect


def convolution(image, kernels):
    channels, height, width = len(image), len(image[0]), len(image[0][0])
    output = []
    for kernel in kernels:
        plane = []
        for y in range(height - 2):
            line = []
            for x in range(width - 2):
                value = 0.
                for c in range(channels):
                    for dy in range(3):
                        for dx in range(3):
                            value += image[c][y + dy][x + dx] * kernel[c][dy][dx]
                line.append(max(0., value))
            plane.append(line)
        output.append(plane)
    return output


def max_pool(image):
    return [[[max(plane[y + dy][x + dx] for dy in range(2) for dx in range(2))
              for x in range(0, len(plane[0]) - 1, 2)]
             for y in range(0, len(plane) - 1, 2)] for plane in image]


class TinyCNN:
    """16x16x1 -> conv(4,3x3)/ReLU -> pool -> conv(8,3x3)/ReLU -> GAP -> dense(3)."""
    def __init__(self):
        rng = random.Random(104)
        def kernels(outputs, inputs):
            return [[[[rng.uniform(-.2, .2) for _ in range(3)] for _ in range(3)]
                     for _ in range(inputs)] for _ in range(outputs)]
        self.first = kernels(4, 1)
        self.second = kernels(8, 4)
        self.dense = [[rng.uniform(-.2, .2) for _ in range(8)] for _ in range(3)]

    def forward(self, frame=0):
        image = [[[math.sin((x + y + frame % 17) / 8.) for x in range(16)] for y in range(16)]]
        features = convolution(max_pool(convolution(image, self.first)), self.second)
        pooled = [sum(map(sum, plane)) / (len(plane) * len(plane[0])) for plane in features]
        return [sum(a * b for a, b in zip(weights, pooled)) for weights in self.dense]


def phase_at(elapsed, duration):
    return 'idle' if elapsed < duration * .25 else 'cnn_arm_cpu' if elapsed < duration * .75 else 'cool_down'


def guard_reason(row, limit):
    if not row['input_valid']:
        return 'invalid_telemetry'
    for feature in ('temp_pl_temp_C', 'temp_ps_temp_C', 'temp_remote_temp_C'):
        try:
            value = float(row[feature])
        except (ValueError, TypeError, KeyError):
            return 'temperature_unavailable'
        if not math.isfinite(value):
            return 'temperature_unavailable'
        if value >= limit:
            return 'temperature_abort'
    return None


def cpu_worker(stop, messages, duty, duration):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    model = TinyCNN()
    frame = 0
    last_report = time.monotonic()
    deadline = last_report + duration + 2
    while not stop.is_set() and time.monotonic() < deadline:
        start = time.monotonic()
        while not stop.is_set() and time.monotonic() - start < .1 * duty:
            before = time.monotonic()
            output = model.forward(frame)
            latency = (time.monotonic() - before) * 1000
            frame += 1
            if time.monotonic() - last_report >= 1:
                try:
                    messages.put_nowait({'kind': 'cnn_inference', 'frames': frame,
                                         'last_inference_ms': latency, 'output': output})
                except queue.Full:
                    pass
                last_report = time.monotonic()
        stop.wait(max(0., .1 - (time.monotonic() - start)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('recordings/cnn_session.csv'))
    p.add_argument('--duration-s', type=float, default=60)
    p.add_argument('--hz', type=float, default=10)
    p.add_argument('--stream-csv', action='store_true')
    p.add_argument('--workload', default='tiny_cnn_arm_cpu')
    p.add_argument('--duty-percent', type=float, default=25)
    p.add_argument('--abort-temperature-c', type=float, default=50)
    p.add_argument('--sys-root', type=Path, default=Path('/sys'))
    p.add_argument('--sensor-map', type=Path)
    args = p.parse_args()
    if not 0 < args.duration_s <= 1800 or not 1 <= args.hz <= 20:
        p.error('Use duration 0–1800s and sampling rate 1–20Hz.')
    if not 1 <= args.duty_percent <= 50 or not 40 <= args.abort_temperature_c <= 50:
        p.error('Duty must be 1–50%; operational abort must be 40–50C.')
    args.rep_id, args.stage, args.stage_file, args.check_only = 1, 'idle', None, False
    workload_log = args.output.with_suffix('.workload.jsonl')
    if args.output.exists() or args.output.with_suffix('.metadata.json').exists() or workload_log.exists():
        p.error('Output exists: choose a new filename.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    context = mp.get_context('spawn')
    stop = context.Event()
    messages = context.Queue(maxsize=16)
    process = None
    last_phase = None
    aborted = None

    def stop_worker():
        stop.set()
        if process is not None:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join()

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    with workload_log.open('x') as log:
        def record(entry):
            log.write(json.dumps(entry, allow_nan=False) + '\n')
            log.flush()

        record({'kind': 'session', 'execution_backend': 'ARM_CPU_NOT_FPGA',
                'weights': 'fixed_seed_untrained', 'inputs': 'synthetic_16x16',
                'architecture': 'Conv4-ReLU-MaxPool-Conv8-ReLU-GAP-Dense3',
                'duty_percent_target': args.duty_percent,
                'abort_temperature_c': args.abort_temperature_c,
                'note': 'Operational abort, not a manufacturer safety limit; duty is approximate.'})
        print('CNN workload: ARM CPU, NOT FPGA fabric. Untrained weights, synthetic inputs.', file=sys.stderr, flush=True)

        def observe(row):
            nonlocal process, last_phase, aborted
            reason = guard_reason(row, args.abort_temperature_c)
            if reason and aborted is None:
                aborted = reason
                stop_worker()
            phase = 'aborted_' + aborted if aborted else phase_at(row['timestamp_s'], args.duration_s)
            if phase != last_phase:
                if phase == 'cnn_arm_cpu':
                    process = context.Process(target=cpu_worker, args=(stop, messages, args.duty_percent / 100., args.duration_s), daemon=True)
                    process.start()
                elif phase != 'idle':
                    stop_worker()
                record({'kind': 'stage', 'elapsed_s': row['timestamp_s'], 'stage': phase})
                print('Workload stage: ' + phase, file=sys.stderr, flush=True)
                last_phase = phase
            if phase == 'cnn_arm_cpu' and process is not None and process.exitcode is not None:
                aborted = 'cnn_worker_exited'
                stop_worker()
                phase = 'aborted_' + aborted
                record({'kind': 'stage', 'elapsed_s': row['timestamp_s'], 'stage': phase})
            row['stage'] = phase
            while True:
                try:
                    entry = messages.get_nowait()
                except queue.Empty:
                    break
                entry['elapsed_s'] = row['timestamp_s']
                record(entry)
                print('CNN forwards: {} | last inference {:.2f}ms (ARM CPU)'.format(
                    entry['frames'], entry['last_inference_ms']), file=sys.stderr, flush=True)

        try:
            collect(args, row_observer=observe)
        finally:
            stop_worker()
            record({'kind': 'workload_stopped', 'abort_reason': aborted})
            messages.close()
            messages.join_thread()
    if aborted:
        print('CNN stopped: ' + aborted + '. Recording retained; do not treat this as a successful load test.', file=sys.stderr)
        raise SystemExit(2)


if __name__ == '__main__':
    main()
