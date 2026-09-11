import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from PIL import Image, ImageDraw

from tools.build_aikimi_motion_asset import encoded_indices, extract_components


ROOT = Path(__file__).resolve().parents[2]


class AikimiMotionBuilderTests(unittest.TestCase):
    def green_strip(self, centers):
        strip = Image.new("RGBA", (400, 160), (0, 255, 0, 255))
        draw = ImageDraw.Draw(strip)
        for center in centers:
            draw.rounded_rectangle(
                (center - 24, 28, center + 24, 142),
                radius=12,
                fill=(240, 244, 250, 255),
            )
        return strip

    def motion_fixture(self):
        strip = Image.new("RGBA", (640, 240), (0, 255, 0, 255))
        draw = ImageDraw.Draw(strip)
        for index, center in enumerate((80, 240, 400, 560)):
            draw.ellipse((center - 38, 28, center + 38, 112), fill=(244, 247, 252, 255))
            draw.rounded_rectangle(
                (center - 24 - index, 100, center + 24 + index, 204 - index * 3),
                radius=10,
                fill=(160, 185 + index * 5, 210, 255),
            )
            draw.rectangle(
                (center - 18, 196 - index * 3, center - 6, 226),
                fill=(248, 239, 232, 255),
            )
            draw.rectangle(
                (center + 6, 196 - index * 3, center + 18, 226),
                fill=(248, 239, 232, 255),
            )
            draw.ellipse((center - 17, 66, center - 10, 73), fill=(80, 105, 130, 255))
            draw.ellipse(
                (center + 10, 66 + index, center + 17, 73 + index),
                fill=(80, 105, 130, 255),
            )
        return strip

    def test_ping_pong_indices_do_not_duplicate_endpoints(self):
        self.assertEqual(encoded_indices(4, "ping-pong"), [0, 1, 2, 3, 2, 1])
        self.assertEqual(encoded_indices(5, "once"), [0, 1, 2, 3, 4])

    def test_component_extraction_accepts_one_pose_per_slot(self):
        entries, _, diagnostics = extract_components(
            self.green_strip((50, 150, 250, 350)),
            frame_count=4,
            alpha_cutoff=16 / 255,
        )

        self.assertEqual(len(entries), 4)
        self.assertEqual(len(diagnostics["principal_centers_x"]), 4)

    def test_component_extraction_rejects_missing_nominal_slot(self):
        with self.assertRaisesRegex(ValueError, "outside its nominal slot"):
            extract_components(
                self.green_strip((25, 75, 250, 350)),
                frame_count=4,
                alpha_cutoff=16 / 255,
            )

    def test_component_extraction_rejects_connected_poses(self):
        strip = self.green_strip((50, 150, 250, 350))
        draw = ImageDraw.Draw(strip)
        draw.rectangle((50, 80, 350, 90), fill=(240, 244, 250, 255))

        with self.assertRaisesRegex(ValueError, "foreground components"):
            extract_components(strip, frame_count=4, alpha_cutoff=16 / 255)

    def test_cli_fixture_builds_complete_apng_package(self):
        with tempfile.TemporaryDirectory(prefix="aikimi-builder-fixture-") as temporary:
            root = Path(temporary)
            source = root / "motion-strip.png"
            apng = root / "fixture.png"
            still = root / "fixture-still.webp"
            qa = root / "qa"
            self.motion_fixture().save(source)
            command = [
                sys.executable,
                str(ROOT / "tools" / "build_aikimi_motion_asset.py"),
                "--source",
                str(source),
                "--output-apng",
                str(apng),
                "--output-still",
                str(still),
                "--qa-dir",
                str(qa),
                "--frame-count",
                "4",
                "--durations-ms",
                "120,120,120,120",
                "--loop-mode",
                "ping-pong",
                "--anchor-mode",
                "baseline",
                "--width",
                "96",
                "--height",
                "120",
                "--padding",
                "6",
            ]
            completed = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )

            self.assertEqual(
                completed.returncode, 0, completed.stderr or completed.stdout
            )
            validation = json.loads(
                (qa / "validation.json").read_text(encoding="utf-8")
            )
            self.assertTrue(validation["ok"])
            self.assertEqual(validation["algorithm_version"], 2)
            self.assertEqual(validation["encoded_indices"], [0, 1, 2, 3, 2, 1])
            for output in (apng, still, qa / "preview.webp", qa / "contact-sheet.png"):
                self.assertTrue(output.is_file())
                self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
