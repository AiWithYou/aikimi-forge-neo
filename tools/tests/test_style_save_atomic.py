from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from modules.styles import StyleDatabase


class StyleSaveAtomicTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.path = Path(folder.name) / "styles.csv"
        self.original = b'\xef\xbb\xbfname,prompt,negative_prompt\r\nOriginal,keep,negative\r\n'
        self.path.write_bytes(self.original)
        self.database = StyleDatabase([str(self.path)])

    def test_unencodable_prompt_keeps_existing_csv(self):
        self.database.styles["Original"] = self.database.styles["Original"]._replace(prompt="bad\ud800")
        with self.assertRaises(UnicodeEncodeError):
            self.database.save_styles()
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_replace_failure_keeps_original_and_removes_temp(self):
        self.database.styles["Original"] = self.database.styles["Original"]._replace(prompt="new")
        with patch("modules.atomic_file.os.replace", side_effect=OSError("disk write failed")):
            with self.assertRaises(OSError):
                self.database.save_styles()
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assertEqual(Path(str(self.path) + ".bak").read_bytes(), self.original)
        self.assertEqual({path.name for path in self.path.parent.iterdir()}, {"styles.csv", "styles.csv.bak"})

    def test_unicode_multiline_prompt_roundtrips_and_keeps_bom_and_backup(self):
        prompt = '日本語, "quoted"\nsecond line'
        self.database.styles["Original"] = self.database.styles["Original"]._replace(prompt=prompt)
        self.database.save_styles()
        self.assertTrue(self.path.read_bytes().startswith(b"\xef\xbb\xbf"))
        self.assertEqual(Path(str(self.path) + ".bak").read_bytes(), self.original)
        restored = StyleDatabase([str(self.path)])
        self.assertEqual(restored.styles["Original"].prompt, prompt)
        self.assertEqual(restored.styles["Original"].negative_prompt, "negative")


if __name__ == "__main__":
    unittest.main()
