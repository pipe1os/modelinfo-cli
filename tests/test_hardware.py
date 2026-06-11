import unittest
from unittest.mock import patch, MagicMock
from src.modelinfo.hardware import normalize_gpu_string, resolve_gpu, detect_local_gpu, KNOWN_GPUS

class TestHardware(unittest.TestCase):

    def test_normalize_gpu_string(self):
        self.assertEqual(normalize_gpu_string("NVIDIA GeForce RTX 3090"), "rtx3090")
        self.assertEqual(normalize_gpu_string("AMD Radeon RX 7900 XT"), "rx7900xt")
        self.assertEqual(normalize_gpu_string("Intel Arc A770"), "arca770")
        self.assertEqual(normalize_gpu_string("NVIDIA GeForce RTX 3090 - Edition"), "rtx3090")
        self.assertEqual(normalize_gpu_string("AMD Radeon RX 7900 XT - Graphics"), "rx7900xt")

    def test_resolve_gpu(self):
        self.assertEqual(resolve_gpu("rtx3090"), ("rtx3090", 24.0))
        self.assertEqual(resolve_gpu("rx7900xt"), ("rx7900xt", 20.0))
        self.assertEqual(resolve_gpu("arca770"), ("arca770", 16.0))
        self.assertIsNone(resolve_gpu("unknown_gpu"))

    def test_detect_local_gpu(self):
        with patch('subprocess.run') as mock_subprocess_run:
            mock_subprocess_run.return_value = MagicMock(stdout="RTX 3090, 1024\nRTX 3090, 1024", stderr="", returncode=0)
            gpu_info = detect_local_gpu()
            self.assertEqual(gpu_info[0], "Multi-GPU: 2x RTX 3090")
            self.assertEqual(gpu_info[1], 2048.0)
            self.assertEqual(gpu_info[2], 2)

        with patch('subprocess.run') as mock_subprocess_run:
            mock_subprocess_run.return_value = MagicMock(stdout="RTX 3090, 1024", stderr="", returncode=0)
            gpu_info = detect_local_gpu()
            self.assertEqual(gpu_info[0], "RTX 3090")
            self.assertEqual(gpu_info[1], 1024.0)
            self.assertEqual(gpu_info[2], 1)

        with patch('subprocess.run') as mock_subprocess_run:
            mock_subprocess_run.return_value = MagicMock(stdout="", stderr="", returncode=1)
            gpu_info = detect_local_gpu()
            self.assertIsNone(gpu_info)

if __name__ == '__main__':
    unittest.main()