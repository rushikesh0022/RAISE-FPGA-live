"""Read-only board/runtime inventory. Does not load hardware or run a workload."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import platform


def report(search_root):
    compatible = Path('/proc/device-tree/compatible')
    identity = compatible.read_bytes().replace(b'\0', b',').decode(errors='replace') if compatible.exists() else ''
    modules = {}
    for name in ('pynq', 'pynq_dpu', 'vart', 'xir', 'numpy', 'torch'):
        try:
            modules[name] = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            modules[name] = False
    candidates = []
    visited = 0
    truncated = False
    for directory, folders, files in os.walk(str(search_root), followlinks=False):
        folders[:] = sorted(f for f in folders if not f.startswith('.') and f not in
                            ('node_modules', '__pycache__', 'venv', 'site-packages'))
        for name in sorted(files):
            visited += 1
            if name.endswith(('.bit', '.hwh', '.xmodel', '.xclbin')):
                candidates.append(str(Path(directory) / name))
            if visited >= 20000 or len(candidates) >= 100:
                truncated = True
                break
        if truncated:
            break
    return {'purpose': 'Inventory only; no hardware loaded and no workload executed.',
            'system': platform.system(), 'machine': platform.machine(),
            'python': platform.python_version(), 'device_tree_compatible': identity,
            'python_modules_found': modules, 'searched_directory': str(search_root),
            'hardware_model_files': candidates, 'scan_truncated': truncated,
            'note': 'File/module presence does not verify compatibility or accelerator execution. '
                    'Other folders and environments may contain additional installations.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--search-dir', type=Path, default=Path.home())
    args = parser.parse_args()
    if not args.search_dir.is_dir():
        parser.error('--search-dir must be an existing directory')
    print(json.dumps(report(args.search_dir), indent=2))


if __name__ == '__main__':
    main()
