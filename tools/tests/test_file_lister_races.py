import ast
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch


class FileListerRaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[2] / "modules/util.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        nodes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MassFileListerCachedDir"]
        namespace = {"os": os}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)  # noqa: S102
        cls.CachedDir = namespace["MassFileListerCachedDir"]

    def scan(self, entries):
        scan = MagicMock()
        scan.__iter__.return_value = iter(entries)
        scan.__enter__.return_value = scan
        return scan

    def test_file_removed_between_enumeration_and_stat_is_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "keep.json").write_text("{}", encoding="utf-8")
            gone = root / "gone.json"
            gone.touch()
            with os.scandir(root) as scan:
                entries = list(scan)
            gone.unlink()
            # DirEntry may cache stat data on Windows: explicitly model the
            # disappearance that a fresh metadata query reports.
            entries = [Mock(name="removed", stat=Mock(side_effect=FileNotFoundError())) if e.name == "gone.json" else e for e in entries]
            handle = self.scan(entries)
            with patch("os.scandir", return_value=handle):
                cached = self.CachedDir(directory)
            self.assertEqual(set(cached.files_cased), {"keep.json"})
            self.assertEqual(set(cached.files), {"keep.json"})
            handle.__exit__.assert_called_once()

    def test_other_stat_errors_propagate_and_close_scan(self):
        entry = Mock(stat=Mock(side_effect=PermissionError("denied")))
        handle = self.scan([entry])
        with patch("os.scandir", return_value=handle), self.assertRaises(PermissionError):
            self.CachedDir("unused")
        handle.__exit__.assert_called_once()


if __name__ == "__main__":
    unittest.main()
