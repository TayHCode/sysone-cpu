"""Both ONNX graph and external weights must be complete and hash verified."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from scripts.download_onnx import validate_artifacts


class ONNXIntegrityTests(unittest.TestCase):
    def setUp(self):
        Path('.scratch').mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir='.scratch')
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.spec = {}
        for name, content in [('onnx/model.onnx', b'graph'), ('onnx/model.onnx_data', b'weights')]:
            path = self.folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            self.spec[name] = (len(content), hashlib.sha256(content).hexdigest())

    def test_graph_and_external_data_valid(self):
        validate_artifacts(self.folder, self.spec)

    def test_missing_external_data(self):
        (self.folder / 'onnx/model.onnx_data').unlink()
        with self.assertRaises(ValueError): validate_artifacts(self.folder, self.spec)

    def test_same_size_corruption(self):
        (self.folder / 'onnx/model.onnx_data').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'SHA-256'): validate_artifacts(self.folder, self.spec)

    def test_symlink_refused(self):
        target = self.folder / 'external'
        target.write_bytes(b'weights')
        path = self.folder / 'onnx/model.onnx_data'
        path.unlink()
        try:
            path.symlink_to(target)
        except (OSError, NotImplementedError):
            self.skipTest('Creating symlinks requires platform permission')
        with self.assertRaises(ValueError): validate_artifacts(self.folder, self.spec)
