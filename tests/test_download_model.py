"""Integrity checks use tiny fixtures, no model download or runtime dependency."""
import hashlib
import tempfile
import unittest
from pathlib import Path

from scripts.download_model import FILES, validate_model


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        # Keep scratch in the supplied checkout, including under release verification.
        Path(".scratch").mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=".scratch")
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        for name in FILES:
            path = self.folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture")
        self.sha = hashlib.sha256(b"fixture").hexdigest()

    def test_valid_files(self):
        validate_model(self.folder, 7, self.sha)

    def test_same_size_corruption(self):
        (self.folder / FILES[0]).write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            validate_model(self.folder, 7, self.sha)

    def test_wrong_size(self):
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            validate_model(self.folder, 8, self.sha)

    def test_missing_config(self):
        (self.folder / "encoder/config.json").unlink()
        with self.assertRaisesRegex(ValueError, "Missing or unsafe"):
            validate_model(self.folder, 7, self.sha)


if __name__ == "__main__":
    unittest.main()
