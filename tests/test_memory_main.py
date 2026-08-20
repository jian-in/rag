from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from memory_hub.__main__ import _remove_own_pid, _write_pid, _write_status


class MemoryMainTests(unittest.TestCase):
    def test_pid_file_is_atomic_and_only_removed_by_its_owner(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            pid_path = Path(temp) / "server.pid"

            _write_pid(pid_path, 4123)
            self.assertEqual(pid_path.read_text(encoding="ascii"), "4123")

            _remove_own_pid(pid_path, 9999)
            self.assertTrue(pid_path.exists())

            _remove_own_pid(pid_path, 4123)
            self.assertFalse(pid_path.exists())

    def test_status_output_is_safe_when_pythonw_has_no_stderr(self) -> None:
        with patch("memory_hub.__main__.sys.stderr", None):
            _write_status("index ready")


if __name__ == "__main__":
    unittest.main()
