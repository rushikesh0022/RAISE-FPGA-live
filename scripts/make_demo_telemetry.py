"""Create synthetic seven-channel readings for a deployment plumbing check only."""
import csv
import math
from pathlib import Path


def main():
    dest = Path('examples/replay_telemetry.csv')
    dest.parent.mkdir(parents=True, exist_ok=True)
    fields = ['timestamp_s', 'run_id', 'input_valid', 'temp_pl_temp_C',
              'temp_ps_temp_C', 'temp_remote_temp_C', 'iio_vccint_V',
              'iio_vccaux_V', 'iio_vccbram_V', 'power_pin_W']
    with dest.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i in range(400):
            t = i / 10
            temp = 46 + .02 * t + .15 * math.sin(t / 5)
            writer.writerow(dict(zip(fields, [t, 'synthetic_plumbing_check', True,
                                              temp, temp - .3, temp + 1.5,
                                              .85, 1.8, .85, 12.])))
    print(str(dest) + ' (SYNTHETIC; not an accuracy evaluation)')


if __name__ == '__main__':
    main()
