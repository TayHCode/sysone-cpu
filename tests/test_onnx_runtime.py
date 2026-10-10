"""The recipe binds to the tested native API and refuses invalid thread settings."""
import unittest
from unittest.mock import patch
from scripts.onnx_runtime import load


class NativeRuntimeTests(unittest.TestCase):
    def test_invalid_threads_refused_before_importing_model_dependencies(self):
        for threads in [0, -1]:
            with self.subTest(threads=threads):
                with self.assertRaisesRegex(ValueError, 'positive'):
                    load('unused', 'unused', threads)

    def test_untested_runtime_version_refused(self):
        with patch('scripts.onnx_runtime.importlib.metadata.version', return_value='0.3.4'):
            with self.assertRaisesRegex(ValueError, '0.4.2'):
                load('unused', 'unused', 4)
