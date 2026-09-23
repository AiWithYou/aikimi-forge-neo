import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from modules_forge import minimax_h3_handoff_store as store


def fixture():
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": "forge_h3/first.png"}},
        "2": {"class_type": "RandomNoise", "inputs": {"noise_seed": 731}},
        "3": {
            "class_type": "TestCondition",
            "inputs": {
                "image": ["1", 0],
                "prompt": "Keep forge_h3/first.png literally. 雨",
                "ref_images.ref_image_0": ["1", 0],
            },
        },
        "4": {
            "class_type": "BlockSparseAttention",
            "inputs": {"selection": "Sol-Attn (adaptive tau)", "selection.tau": 1.3},
        },
    }


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "forge_h3").mkdir()
        (self.root / "forge_h3/first.png").write_bytes(b"prepared-image-bytes")
        self.prepared = {"first_frame": "forge_h3/first.png", "images": [], "videos": [], "audios": []}

    def tearDown(self):
        self.tmp.cleanup()

    def snapshot(self, **kwargs):
        return store.create_snapshot(self.root, fixture(), self.prepared, {"seed": 731}, **kwargs)

    def test_snapshot_preserves_actual_and_editable_graph(self):
        source = fixture()
        record = store.create_snapshot(self.root, source, self.prepared, {"seed": 731})
        self.assertEqual(source, fixture())
        self.assertEqual(record["submitted_prompt"], fixture())
        self.assertEqual(record["prompt"]["3"], fixture()["3"])
        self.assertEqual(record["prompt"]["4"], fixture()["4"])
        self.assertEqual(record["prompt"]["2"]["inputs"]["noise_seed"], 731)
        ui_file = record["assets"][0]["ui_file"]
        self.assertEqual(record["prompt"]["1"]["inputs"]["image"], ui_file)
        self.assertTrue(ui_file.startswith(f"aikimi_h3_{record['token']}_"))
        self.assertTrue((self.root / ui_file).samefile(self.root / record["assets"][0]["file"]))

    def test_cleanup_and_original_changes_do_not_destroy_snapshot(self):
        record = self.snapshot()
        original = self.root / "forge_h3/first.png"
        original.write_bytes(b"changed!")
        original.unlink()
        loaded = store.read_snapshot(self.root, record["token"])
        self.assertEqual((self.root / loaded["assets"][0]["file"]).read_bytes(), b"prepared-image-bytes")

    def test_history_receipt_is_tied_to_this_output(self):
        a, b = self.snapshot(), self.snapshot()
        first, second = self.root / "first.mp4", self.root / "second.mp4"
        first.touch()
        second.touch()
        store.write_receipt(first, a)
        store.write_receipt(second, b)
        self.assertEqual(store.read_receipt(first)["token"], a["token"])
        self.assertEqual(store.read_receipt(second)["token"], b["token"])
        with self.assertRaises(store.HandoffError):
            store.write_receipt(first, b)
        store.write_receipt(first, a)

    def test_legacy_history_is_not_rebuilt_from_current_settings(self):
        video = self.root / "legacy.mp4"
        video.touch()
        with self.assertRaisesRegex(store.HandoffError, "旧履歴"):
            store.read_receipt(video)

    def test_missing_asset_blocks_open(self):
        record = self.snapshot()
        (self.root / record["assets"][0]["file"]).unlink()
        with self.assertRaises(store.HandoffError):
            store.read_snapshot(self.root, record["token"])

    def test_missing_editor_visible_asset_blocks_open(self):
        record = self.snapshot()
        (self.root / record["assets"][0]["ui_file"]).unlink()
        with self.assertRaises(store.HandoffError):
            store.read_snapshot(self.root, record["token"])

    def test_same_size_tampered_asset_blocks_open(self):
        record = self.snapshot()
        target = self.root / record["assets"][0]["file"]
        target.write_bytes(b"X" * target.stat().st_size)
        with self.assertRaisesRegex(store.HandoffError, "破損"):
            store.read_snapshot(self.root, record["token"])

    def test_tampered_prompt_blocks_open(self):
        record = self.snapshot()
        path = self.root / store.STORE / record["token"] / "manifest.json"
        record["prompt"]["2"]["inputs"]["noise_seed"] = 99
        path.write_bytes(store.json_bytes(record))
        with self.assertRaisesRegex(store.HandoffError, "破損"):
            store.read_snapshot(self.root, record["token"])

    def test_no_token_overwrite(self):
        record = self.snapshot(token="a" * 32)
        with self.assertRaises(store.HandoffError):
            self.snapshot(token=record["token"])

    def test_reject_traversal_and_windows_paths(self):
        for path in ("../private", "/etc/passwd", "C:/private", "a\\b", "a/../b", "a\x00b"):
            with self.subTest(path=path), self.assertRaises(store.HandoffError):
                store.checked_path(self.root, path, exists=False)

    def test_symlink_parent_and_media_rejected(self):
        outside = self.root / "other"
        outside.mkdir()
        (outside / "file.png").write_bytes(b"x")
        (self.root / "forge_h3/link").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(store.HandoffError):
            store.checked_path(self.root, "forge_h3/link/file.png")
        (self.root / store.STORE).symlink_to(outside, target_is_directory=True)
        with self.assertRaises(store.HandoffError):
            self.snapshot()

    def test_copy_failure_leaves_no_published_partial_snapshot(self):
        with mock.patch.object(store.shutil, "copyfile", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.snapshot()
        self.assertEqual(list((self.root / store.STORE).iterdir()), [])
        self.assertEqual(list(self.root.glob("aikimi_h3_[0-9a-f]*")), [])

    def test_media_mutation_during_copy_rejected(self):
        original_copy = store.shutil.copyfile

        def mutate(source, target):
            result = original_copy(source, target)
            Path(source).write_bytes(b"new content")
            return result

        with mock.patch.object(store.shutil, "copyfile", side_effect=mutate):
            with self.assertRaisesRegex(store.HandoffError, "コピー中"):
                self.snapshot()

    def test_unmapped_loader_or_unused_asset_rejected(self):
        prompt = fixture()
        prompt["1"]["inputs"]["image"] = "other.png"
        with self.assertRaises(store.HandoffError):
            store.create_snapshot(self.root, prompt, self.prepared, {})
        prompt = {"2": fixture()["2"]}
        with self.assertRaises(store.HandoffError):
            store.create_snapshot(self.root, prompt, self.prepared, {})

    def test_audio_video_control_and_multiple_images_preserved(self):
        for name in ("last.png", "ref.png", "ref.mp4", "ref.wav", "control.mp4"):
            (self.root / "forge_h3" / name).write_bytes(name.encode())
        prepared = {
            "first_frame": "forge_h3/first.png",
            "last_frame": "forge_h3/last.png",
            "images": ["forge_h3/ref.png"],
            "videos": [{"name": "forge_h3/ref.mp4", "has_audio": True}],
            "audios": ["forge_h3/ref.wav"],
            "control_video": "forge_h3/control.mp4",
        }
        graph = {}
        for index, name in enumerate(sorted(store.prepared_names(prepared))):
            kind, field = (
                ("LoadAudio", "audio")
                if name.endswith("wav")
                else ("LoadVideo", "file")
                if name.endswith("mp4")
                else ("LoadImage", "image")
            )
            graph[str(index)] = {"class_type": kind, "inputs": {field: name}}
        result = store.create_snapshot(self.root, graph, prepared, {})
        self.assertEqual(len(result["assets"]), 6)
        self.assertEqual(store.read_snapshot(self.root, result["token"]), result)

    def test_snapshot_json_size_guard(self):
        with mock.patch.object(store, "MAX_JSON_BYTES", 1024):
            with self.assertRaises(store.HandoffError):
                self.snapshot()

    def test_nonfinite_input_and_bad_connections_rejected(self):
        for value in (float("nan"), float("inf")):
            graph = fixture()
            graph["2"]["inputs"]["noise_seed"] = value
            with self.assertRaises(store.HandoffError):
                store.validate_prompt(graph)
        graph = fixture()
        graph["3"]["inputs"]["image"] = ["999", 0]
        with self.assertRaises(store.HandoffError):
            store.validate_prompt(graph)

    def test_loopback_url_has_only_opaque_token(self):
        self.assertEqual(
            store.handoff_url("http://127.0.0.1:8188", "a" * 32), "http://127.0.0.1:8188/#aikimi-h3=" + "a" * 32
        )
        self.assertIn("localhost:8188", store.handoff_url("http://localhost:8188/", "b" * 32))
        for url in (
            "https://evil.test:8188",
            "javascript:alert(1)",
            "http://127.0.0.1:8188@evil.test",
            "http://localhost:8188/path",
            "http://localhost:8188/?token=abc",
            "http://localhost",
            "http://127.0.0.1:8188/#other",
            "http://localhost:99999",
        ):
            with self.subTest(url=url), self.assertRaises(store.HandoffError):
                store.handoff_url(url, "a" * 32)

    def test_token_validation(self):
        for token in (None, "A" * 32, "a" * 31, "../manifest", "a" * 32 + "/x"):
            with self.subTest(token=token), self.assertRaises(store.HandoffError):
                store.token_value(token)


if __name__ == "__main__":
    unittest.main()
