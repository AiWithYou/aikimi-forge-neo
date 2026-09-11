import os
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from modules_forge.sensenova_u15_bridge import (
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_SOURCE_PATH,
    MODE_EDIT,
    MODE_TEXT,
    PROFILE_OFFICIAL_8STEP,
    PROFILE_QUALITY,
    QUANT_INT8_CONVROT,
    SenseNovaRequest,
    inspect_runtime,
    run_generation,
)


ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(
    os.getenv("SENSENOVA_U15_LIVE_TEST") == "1",
    "opt-in real SenseNova final INT8 ConvRot GPU test",
)
class SenseNovaU15LiveTest(unittest.TestCase):
    @staticmethod
    def _reference_images() -> tuple[Image.Image, Image.Image]:
        first = Image.new("RGB", (512, 512), (235, 242, 250))
        first_draw = ImageDraw.Draw(first)
        first_draw.ellipse((100, 80, 412, 392), fill=(30, 90, 190))
        second = Image.new("RGB", (512, 512), (250, 235, 220))
        second_draw = ImageDraw.Draw(second)
        second_draw.rectangle((112, 112, 400, 400), fill=(220, 70, 55))
        return first, second

    def _run_two_image_edit(
        self,
        output_size: int | None,
        *,
        vram_mode: str = "low",
        reference_images: tuple[Image.Image, Image.Image] | None = None,
        expected_size: tuple[int, int] | None = None,
    ) -> dict:
        status = inspect_runtime(DEFAULT_SOURCE_PATH, checkpoint=DEFAULT_CHECKPOINT_PATH)
        self.assertTrue(status.ready, " / ".join(status.messages))
        first, second = reference_images or self._reference_images()

        request = SenseNovaRequest(
            mode=MODE_EDIT,
            prompt=(
                "Use the centered blue circle from the first image as the main subject. "
                "Apply the warm red color palette from the second image while preserving a clean simple background."
            ),
            generation_profile=PROFILE_QUALITY,
            quantization=QUANT_INT8_CONVROT,
            checkpoint=str(DEFAULT_CHECKPOINT_PATH),
            source_path=str(DEFAULT_SOURCE_PATH),
            input_images=(first, second),
            width=output_size,
            height=output_size,
            input_max_pixels=str(512 * 512),
            steps=1,
            cfg_scale=4.0,
            img_cfg_scale=1.0,
            timestep_shift=3.0,
            seed=42,
            vram_mode=vram_mode,
            attn_backend="sdpa",
            dtype="bfloat16",
        )
        updates = list(
            run_generation(
                request,
                output_directory=ROOT / "outputs" / "sensenova_u15_live_smoke",
                cache_directory=ROOT / "cache" / "sensenova_u15_live_smoke",
                log_directory=ROOT / "logs" / "sensenova_u15_live_smoke",
            )
        )
        complete = [update for update in updates if update["stage"] == "complete"]
        self.assertEqual(len(complete), 1)
        output_path = Path(complete[0]["path"])
        self.assertTrue(output_path.is_file())
        with Image.open(output_path) as image:
            self.assertEqual(
                image.size,
                expected_size or (output_size, output_size),
            )
            values = np.asarray(image.convert("RGB"), dtype=np.float32)
        self.assertGreater(float(values.var()), 1.0)
        self.assertEqual(complete[0]["metadata"]["input_image_count"], 2)
        self.assertEqual(
            complete[0]["metadata"]["quantization"], QUANT_INT8_CONVROT
        )
        self.assertEqual(complete[0]["metadata"]["release_variant"], "final")
        self.assertEqual(complete[0]["metadata"]["loaded_int8_layers"], 588)
        self.assertEqual(complete[0]["metadata"]["schema_version"], 3)
        return complete[0]

    def test_convrot_two_image_edit_generates_nonuniform_png(self):
        self._run_two_image_edit(512)

    def test_low_vram_1024_profile_generates_two_image_edit(self):
        completed = self._run_two_image_edit(1024)
        self.assertEqual(completed["metadata"]["effective_input_max_pixels"], 512 * 512)

    def test_low_vram_2048_profile_with_downscaled_references(self):
        completed = self._run_two_image_edit(2048)
        self.assertEqual(completed["metadata"]["effective_input_max_pixels"], 512 * 512)

    def test_low_vram_auto_output_uses_original_reference_ratio(self):
        first = Image.new("RGB", (1600, 1200), (235, 242, 250))
        first_draw = ImageDraw.Draw(first)
        first_draw.ellipse((400, 200, 1200, 1000), fill=(30, 90, 190))
        second = Image.new("RGB", (1200, 1600), (250, 235, 220))
        second_draw = ImageDraw.Draw(second)
        second_draw.rectangle((200, 400, 1000, 1200), fill=(220, 70, 55))

        completed = self._run_two_image_edit(
            None,
            reference_images=(first, second),
            expected_size=(2368, 1792),
        )
        metadata = completed["metadata"]
        self.assertEqual(metadata["output_aspect_source"], "original_input_1")
        self.assertEqual(
            metadata["input_original_sizes"],
            [
                {"width": 1600, "height": 1200},
                {"width": 1200, "height": 1600},
            ],
        )
        self.assertEqual(
            metadata["input_prepared_sizes"],
            [
                {"width": 576, "height": 416},
                {"width": 416, "height": 576},
            ],
        )
        self.assertEqual(metadata["input_preprocessing"], "aspect_fit_edge_pad_32")

    def test_official_8step_t2i_generates_nonuniform_png(self):
        status = inspect_runtime(DEFAULT_SOURCE_PATH, checkpoint=DEFAULT_CHECKPOINT_PATH)
        self.assertTrue(status.ready, " / ".join(status.messages))
        self.assertTrue(status.lora_ready, " / ".join(status.messages))
        request = SenseNovaRequest(
            mode=MODE_TEXT,
            prompt=(
                "A quiet observatory on a snowy mountain at blue hour, one "
                "brass telescope pointing toward a bright comet, cinematic light."
            ),
            quantization=QUANT_INT8_CONVROT,
            checkpoint=str(DEFAULT_CHECKPOINT_PATH),
            width=512,
            height=512,
            generation_profile=PROFILE_OFFICIAL_8STEP,
            steps=8,
            cfg_scale=1.0,
            timestep_shift=3.0,
            seed=20260824,
            vram_mode="low",
            attn_backend="sdpa",
            dtype="bfloat16",
        )
        updates = list(
            run_generation(
                request,
                output_directory=ROOT / "outputs" / "sensenova_u15_live_smoke",
                cache_directory=ROOT / "cache" / "sensenova_u15_live_smoke",
                log_directory=ROOT / "logs" / "sensenova_u15_live_smoke",
            )
        )
        complete = [update for update in updates if update["stage"] == "complete"]
        self.assertEqual(len(complete), 1)
        output_path = Path(complete[0]["path"])
        with Image.open(output_path) as opened:
            image = opened.convert("RGB")
            values = np.asarray(image, dtype=np.float32)
        self.assertEqual(image.size, (512, 512))
        self.assertGreater(float(values.var()), 1.0)
        metadata = complete[0]["metadata"]
        self.assertEqual(metadata["generation_profile"], PROFILE_OFFICIAL_8STEP)
        self.assertEqual(metadata["steps"], 8)
        self.assertEqual(metadata["cfg_scale"], 1.0)
        self.assertEqual(metadata["official_8step_lora"]["targets"], 294)
        self.assertEqual(metadata["cuda_peak"]["ooms"], 0)


if __name__ == "__main__":
    unittest.main()
