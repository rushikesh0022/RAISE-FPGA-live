from pathlib import Path
import tempfile
import unittest

from scripts.check_board_workload_setup import report


class WorkloadInventoryTests(unittest.TestCase):
    def test_lists_assets_without_reading_or_modifying_them(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset = root / 'example.xmodel'
            asset.write_bytes(b'not an actual model')
            (root / '.private').mkdir()
            (root / '.private/hidden.bit').write_bytes(b'private')
            result = report(root)
            self.assertEqual(result['hardware_model_files'], [str(asset)])
            self.assertEqual(asset.read_bytes(), b'not an actual model')
            self.assertFalse(result['scan_truncated'])
            self.assertIn('no workload executed', result['purpose'])


if __name__ == '__main__':
    unittest.main()
