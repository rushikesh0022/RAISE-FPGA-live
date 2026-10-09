#!/usr/bin/env python3
"""Read-only ZCU104 telemetry capture; Python standard library only.

Run on the board while an existing workload runs separately. This script does
not change voltages, fan settings, clocks, or workload execution.
Live excursion counts are advisory; the project event detector supplies final
training labels, including censoring and refractory merging.
"""
import argparse
import csv
import datetime
import json
import math
from pathlib import Path
import sys
import time


FEATURES = (
    'temp_pl_temp_C', 'temp_ps_temp_C', 'temp_remote_temp_C',
    'iio_vccint_V', 'iio_vccaux_V', 'iio_vccbram_V', 'power_pin_W',
)
LEGACY_CHANNELS = {
    'temp_pl_temp_C': 'in_temp2_pl_temp',
    'temp_ps_temp_C': 'in_temp0_ps_temp',
    'temp_remote_temp_C': 'in_temp1_remote_temp',
    'iio_vccint_V': 'in_voltage18_vccint',
    'iio_vccaux_V': 'in_voltage19_vccaux',
    'iio_vccbram_V': 'in_voltage22_vccbram',
}


def read_number(path):
    value = float(path.read_text().strip())
    if not math.isfinite(value):
        raise ValueError('Nonfinite sensor value: ' + str(path))
    return value


def discover(sys_root):
    """Use the channel identities found in the user's October workbooks."""
    iio = sys_root / 'bus/iio/devices'
    devices = [p for p in sorted(iio.glob('iio:device*'))
               if (p / 'name').exists() and 'ams' in (p / 'name').read_text().lower()]
    if len(devices) != 1:
        raise ValueError('Expected one AMS device; use --sensor-map for a different Linux layout.')
    result = {}
    for feature, base in LEGACY_CHANNELS.items():
        device = devices[0]
        raw, scale, offset = device / (base + '_raw'), device / (base + '_scale'), device / (base + '_offset')
        if not raw.exists() or not scale.exists():
            raise ValueError('Missing expected channel ' + base + '; inspect channels and supply --sensor-map.')
        result[feature] = {'path': str(raw), 'scale': read_number(scale) / 1000,
                           'offset': read_number(offset) if offset.exists() else 0.,
                           'source_scale_path': str(scale),
                           'source_offset_path': str(offset) if offset.exists() else None}
    monitors = [p for p in sorted((sys_root / 'class/hwmon').glob('hwmon*'))
                if (p / 'name').exists() and 'ina226' in (p / 'name').read_text().lower()
                and (p / 'power1_input').exists()]
    if len(monitors) != 1:
        raise ValueError('Expected one INA226 power monitor; use --sensor-map to resolve power identity.')
    result['power_pin_W'] = {'path': str(monitors[0] / 'power1_input'), 'scale': 1e-6, 'offset': 0.}
    return result


class ExcursionTracker:
    """Provisional counts of sustained upward excursions (not final labels)."""
    def __init__(self, threshold):
        self.threshold = threshold
        self.candidate = None
        self.recovery = None
        self.active = False
        self.last = None
        self.seen_below = False
        self.count = 0

    def update(self, timestamp, temperature):
        if self.last is not None and timestamp - self.last > .5:
            self.candidate = self.recovery = None
            self.active = False
            self.seen_below = False
        self.last = timestamp
        if temperature < self.threshold:
            self.seen_below = True
        if not self.active:
            if temperature >= self.threshold:
                if self.candidate is None:
                    self.candidate = timestamp
                if timestamp - self.candidate >= .5 - 1e-9:
                    self.active = True
                    self.candidate = None
                    if self.seen_below:
                        self.count += 1
                        return True
            else:
                self.candidate = None
        elif temperature <= self.threshold - .5:
            if self.recovery is None:
                self.recovery = timestamp
            if timestamp - self.recovery >= 1. - 1e-9:
                self.active = False
                self.recovery = None
        else:
            self.recovery = None
        return False


def collect(args, row_observer=None):
    def log(*values, **kwargs):
        print(*values, file=sys.stderr if args.stream_csv else sys.stdout, **kwargs)
    mapping = json.loads(args.sensor_map.read_text()) if args.sensor_map else discover(args.sys_root)
    if set(mapping) != set(FEATURES):
        raise ValueError('Sensor map must contain exactly the seven model features.')
    for feature, spec in mapping.items():
        value = (read_number(Path(spec['path'])) + float(spec.get('offset', 0))) * float(spec['scale'])
        log(feature + ' = ' + format(value, '.5f'), flush=True)
    if args.check_only:
        log(json.dumps(mapping, indent=2))
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves previous recordings.
    if args.output.exists() or args.output.with_suffix('.metadata.json').exists():
        raise ValueError('Output already exists; choose a new filename.')
    start = time.monotonic()
    started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    run_id = args.workload + '/' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    trackers = {t: ExcursionTracker(t) for t in (49, 50, 51)}
    ranges = {f: [math.inf, -math.inf] for f in FEATURES}
    counts, invalid, missed = 0, 0, 0
    next_sample, next_print = start, start
    fields = ['run_id', 'workload_id', 'rep_id', 'timestamp_s', 'board_utc_timestamp',
              'stage', *FEATURES, 'read_span_ms', 'input_valid', 'read_error',
              *[f + '__timestamp_s' for f in FEATURES]]
    status = 'completed'
    log('Recording with integrated workload supervision.' if row_observer else
        'Recording. Run your workload separately. Ctrl+C closes the file safely.', flush=True)
    try:
        with args.output.open('x', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            stream = csv.DictWriter(sys.stdout, fieldnames=fields) if args.stream_csv else None
            if stream:
                stream.writeheader()
                sys.stdout.flush()
            while time.monotonic() - start < args.duration_s:
                delay = next_sample - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                row_start = time.monotonic()
                row = {'run_id': run_id, 'workload_id': args.workload, 'rep_id': args.rep_id,
                       'board_utc_timestamp': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                       'stage': args.stage, 'read_error': ''}
                if args.stage_file:
                    try:
                        row['stage'] = args.stage_file.read_text().strip()
                    except OSError:
                        row['stage'] = 'unknown'
                errors = []
                for feature in FEATURES:
                    before = time.monotonic()
                    try:
                        spec = mapping[feature]
                        value = (read_number(Path(spec['path'])) + float(spec.get('offset', 0))) * float(spec['scale'])
                        row[feature] = value
                        ranges[feature][0] = min(ranges[feature][0], value)
                        ranges[feature][1] = max(ranges[feature][1], value)
                    except (OSError, ValueError) as exc:
                        row[feature] = ''
                        errors.append(feature + ': ' + str(exc))
                    row[feature + '__timestamp_s'] = (before + time.monotonic()) / 2 - start
                end = time.monotonic()
                row['timestamp_s'] = end - start
                row['read_span_ms'] = (end - row_start) * 1000
                row['input_valid'] = not errors and end - row_start <= .2
                row['read_error'] = '; '.join(errors)
                if row_observer is not None:
                    row_observer(row)
                writer.writerow(row)
                if stream:
                    stream.writerow(row)
                    sys.stdout.flush()
                counts += 1
                invalid += not row['input_valid']
                # Use the actual PL acquisition time rather than the end-of-row time.
                if row['temp_pl_temp_C'] != '':
                    for threshold, tracker in trackers.items():
                        if tracker.update(row['temp_pl_temp_C__timestamp_s'], row['temp_pl_temp_C']):
                            log('Sustained PL excursion at {} C, t={:.2f}s'.format(threshold, row['timestamp_s']), flush=True)
                if end >= next_print:
                    handle.flush()
                    log('t={:.1f}s PL={} C | advisory excursions {} | invalid rows={}'.format(
                        end - start, row['temp_pl_temp_C'], {t: v.count for t, v in trackers.items()}, invalid), flush=True)
                    next_print = end + 5
                next_sample += 1 / args.hz
                if next_sample < end:
                    skipped = int((end - next_sample) * args.hz) + 1
                    missed += skipped
                    next_sample += skipped / args.hz
    except KeyboardInterrupt:
        status = 'interrupted_by_user'
    except BrokenPipeError:
        status = 'stream_disconnected'
    metadata = {'run_id': run_id, 'started_utc': started_utc, 'rows': counts,
                'invalid_rows': invalid, 'missed_schedule_slots': missed,
                'duration_s': time.monotonic() - start, 'requested_hz': args.hz,
                'mapping': mapping, 'status': status,
                'ranges': {f: [v if math.isfinite(v) else None for v in bounds] for f, bounds in ranges.items()},
                'advisory_sustained_excursions': {t: v.count for t, v in trackers.items()},
                'event_count_note': 'Advisory counts omit refractory merging; run the project event detector for training labels.',
                'power_identity': 'INA226 board input; equivalence to previous dataset power_pin_W is provisional.'}
    with args.output.with_suffix('.metadata.json').open('x') as handle:
        json.dump(metadata, handle, indent=2, allow_nan=False)
    log('Saved ' + str(args.output), flush=True)
    log(json.dumps(metadata, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('telemetry.csv'))
    p.add_argument('--workload', default='zcu104_session')
    p.add_argument('--rep-id', type=int, default=1)
    p.add_argument('--stage', default='unmarked')
    p.add_argument('--stage-file', type=Path)
    p.add_argument('--duration-s', type=float, default=1800)
    p.add_argument('--hz', type=float, default=10)
    p.add_argument('--sys-root', type=Path, default=Path('/sys'))
    p.add_argument('--sensor-map', type=Path, help='JSON feature -> path, scale, offset for other verified sensor layouts')
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--stream-csv', action='store_true', help='Stream CSV to stdout; progress goes to stderr')
    args = p.parse_args()
    if args.hz <= 0 or args.duration_s <= 0:
        p.error('duration and sampling rate must be positive')
    collect(args)


if __name__ == '__main__':
    main()
