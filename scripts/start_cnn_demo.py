"""Copy the CPU CNN workload to ZCU104, check sensors, then run PC predictions."""
import argparse
from pathlib import Path
import re
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--board', required=True, help='Board account and Ethernet IP, e.g. root@192.168.1.25')
    parser.add_argument('--duration-s', type=float, default=60)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_.-]+@[A-Za-z0-9.-]+', args.board):
        parser.error('Use username@hostname or username@IPv4')
    if not 0 < args.duration_s <= 1800:
        parser.error('Duration must be positive and at most 1800 seconds')
    root = Path(__file__).resolve().parents[1]
    print('ARM-CPU CNN demo, NOT FPGA fabric. Keep normal cooling and protections enabled.', flush=True)
    print('SSH/SCP may request your board password more than once. Verify new host fingerprints.', flush=True)
    try:
        subprocess.run(['ssh', args.board, 'mkdir -p raise_fpga_live'], check=True)
        files = [str(root / 'scripts' / name) for name in
                 ('collect_zcu104_training_telemetry.py', 'run_board_cnn_session.py')]
        subprocess.run(['scp', *files, args.board + ':raise_fpga_live/'], check=True)
        subprocess.run(['ssh', args.board,
                        'python3 raise_fpga_live/collect_zcu104_training_telemetry.py --check-only'], check=True)
        command = [sys.executable, '-m', 'scripts.run_live_zcu104_prediction',
                   '--ssh', args.board, '--remote-script', 'raise_fpga_live/collect_zcu104_training_telemetry.py',
                   '--with-cnn-workload', '--duration-s', str(args.duration_s)]
        result = subprocess.run(command, cwd=str(root))
        raise SystemExit(result.returncode)
    except subprocess.CalledProcessError as exc:
        print('Setup stopped: SSH, copy or sensor check failed. Do not continue with invalid inputs.', file=sys.stderr)
        raise SystemExit(exc.returncode or 1)
    except FileNotFoundError:
        print('Missing ssh/scp executable: install Windows OpenSSH Client before continuing.', file=sys.stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        print('Stopped. Check the board for any remaining session before restarting.', file=sys.stderr)
        raise SystemExit(130)


if __name__ == '__main__':
    main()
