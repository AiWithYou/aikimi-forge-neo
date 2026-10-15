import unittest

from PIL import Image

from modules_forge.flux3_klein import conditioning as c


class ConditioningTests(unittest.TestCase):
    def test_single_reference_padding_and_optional_crop(self):
        source = Image.new("RGB", (101, 75), "red")
        prepared = c.prepare_images([source])
        self.assertEqual((prepared.width, prepared.height), (112, 80))
        self.assertEqual(prepared.crop, (5, 2, 106, 77))
        self.assertEqual(prepared.references[0].getpixel((0, 0)), (255, 0, 0))
        explicit = c.prepare_images([source], width=256, height=128)
        self.assertIsNone(explicit.crop)
        self.assertEqual(source.size, (101, 75))

    def test_reference_limits_order_and_resolution(self):
        first = Image.new("RGB", (2400, 1200), "red")
        second = Image.new("RGB", (64, 80), "blue")
        prepared = c.prepare_images([first, second])
        self.assertLessEqual(prepared.references[0].width * prepared.references[0].height, 1024**2)
        self.assertEqual(prepared.references[0].width % 16, 0)
        self.assertEqual(prepared.references[1].getpixel((0, 0)), (0, 0, 255))
        with self.assertRaises(ValueError):
            c.prepare_images([second] * 11)
        with self.assertRaises(ValueError):
            c.prepare_images([], width=65, height=128)

    def test_exif_orientation_precedes_preparation(self):
        image = Image.new("RGB", (80, 64))
        image.getexif()[274] = 6
        prepared = c.prepare_images([image])
        self.assertEqual((prepared.width, prepared.height), (64, 80))

    def test_edit_boxes_preserve_null_provenance_and_ids(self):
        boxes = [
            {"id": "dog", "desc": "dog", "from": "ref_image_0", "src_bbox": [0.1, 0.2, 0.7, 0.8], "tgt_bbox": None}
        ]
        key, text, locked = c.plan("remove dog", boxes, 1)
        self.assertEqual(key, "i2i_bbox")
        self.assertEqual(locked[0]["src_bbox"], [100, 200, 700, 800])
        self.assertIsNone(locked[0]["tgt_bbox"])
        answer = '<dog> removed [{"id":"dog","from":"target","src_bbox":null,"tgt_bbox":[1,2,3,4],"desc":"brown dog"}]'
        caption, rows = c.split(c.reconcile(answer, locked))
        self.assertEqual(rows[0]["from"], "ref_image_0")
        self.assertIsNone(rows[0]["tgt_bbox"])
        self.assertEqual(rows[0]["desc"], "brown dog")
        self.assertIn("LOCKED BOUNDING-BOX", text)

    def test_reconcile_alias_and_reject_invalid_added_boxes(self):
        locked = [{"id": "obj_1", "bbox": [100, 100, 500, 500], "desc": "red"}]
        answer = '<balloon> and <bad> [{"id":"balloon","bbox":[100,100,500,500],"desc":"red balloon"},{"id":"bad","bbox":[0,0,0,0],"desc":"bad"}]'
        caption, rows = c.split(c.reconcile(answer, locked))
        self.assertIn("<obj_1>", caption)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], "obj_1")
        self.assertFalse(c.is_finished("poster [SALE]"))

    def test_normalized_boxes_do_not_accept_nan_or_reversed_coords(self):
        for bbox in ([0, 0, float("nan"), 1], [0.6, 0, 0.1, 1], [0, 0, 1.1, 1]):
            with self.assertRaises(ValueError):
                c.to_canvas([{"id": "obj", "bbox": bbox}])

    def test_pup_policy_and_request(self):
        image = Image.new("RGBA", (1300, 900), (0, 0, 0, 0))
        prepared = c.pup_reference(image, 2)
        self.assertLessEqual(prepared.width * prepared.height, 724**2)
        self.assertEqual(prepared.width % 32, 0)
        self.assertGreaterEqual(prepared.getpixel((0, 0))[0], 250)
        spec = c.pup_request(
            "move it", [image], [{"id": "obj", "from": "ref_image_0", "src_bbox": [0, 0, 1, 1], "tgt_bbox": None}]
        )
        self.assertEqual(spec.system_key, "i2i_bbox")
        self.assertIn("ref_image_0", spec.prompt)
        self.assertEqual(spec.presence_penalty, 0.0)
        self.assertEqual(c.pup_request("cat", []).presence_penalty, 1.5)

    def test_text_window_keeps_padding_and_caps_at_4096(self):
        tokens = {"qwen3vl_4b": [[(i, 1.0) for i in range(4081)] + [(151643, 1.0)] * 79]}
        capped = c.bound_text_tokens(tokens)
        self.assertEqual(len(capped["qwen3vl_4b"][0]), 4096)
        self.assertEqual(len(tokens["qwen3vl_4b"][0]), 4160)

    def test_pup_chat_matches_shipped_empty_thinking_template(self):
        spec = c.pup_request(" cat ", [])
        self.assertEqual(
            c.pup_chat_text(spec, " system "),
            "<|im_start|>system\nsystem<|im_end|>\n"
            "<|im_start|>user\ncat<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n",
        )


if __name__ == "__main__":
    unittest.main()
