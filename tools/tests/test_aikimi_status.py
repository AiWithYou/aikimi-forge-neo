import ast
import hashlib
import json
import struct
import unittest
import zlib
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from modules import aikimi_status

ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = ROOT / "assets" / "aikimi"
AIKIMI_UI_ROOT = ROOT / "extensions-builtin" / "aikimi-ui"
ASSISTANT_SOURCE = AIKIMI_UI_ROOT / "javascript" / "aikimiStatus.js"
ASSISTANT_CSS = AIKIMI_UI_ROOT / "style.css"
EXPECTED_STATES = {
    "idle",
    "loading_model",
    "generating",
    "completed",
    "queued",
    "warning",
    "error",
    "out_of_memory",
    "updating",
}


def visible_digest(image):
    rgba = bytearray(image.convert("RGBA").tobytes())
    for offset in range(0, len(rgba), 4):
        if rgba[offset + 3] == 0:
            rgba[offset : offset + 3] = b"\0\0\0"
    return hashlib.sha256(rgba).hexdigest()


def parse_apng(path):
    raw = path.read_bytes()
    if raw[:8] != b"\x89PNG\r\n\x1a\n":
        raise AssertionError(f"{path} does not have a PNG signature")

    offset = 8
    ihdr = None
    animation_control = []
    frame_controls = []
    chunk_types = []
    while offset < len(raw):
        length = struct.unpack(">I", raw[offset : offset + 4])[0]
        chunk_type = raw[offset + 4 : offset + 8]
        data_start = offset + 8
        data_end = data_start + length
        crc_end = data_end + 4
        if crc_end > len(raw):
            raise AssertionError(f"{path} contains a truncated {chunk_type!r} chunk")

        data = raw[data_start:data_end]
        stored_crc = struct.unpack(">I", raw[data_end:crc_end])[0]
        computed_crc = zlib.crc32(data, zlib.crc32(chunk_type)) & 0xFFFFFFFF
        if stored_crc != computed_crc:
            raise AssertionError(f"{path} contains an invalid {chunk_type!r} CRC")

        chunk_types.append(chunk_type)
        if chunk_type == b"IHDR":
            ihdr = struct.unpack(">IIBBBBB", data)
        elif chunk_type == b"acTL":
            animation_control.append(struct.unpack(">II", data))
        elif chunk_type == b"fcTL":
            frame_controls.append(struct.unpack(">IIIIIHHBB", data))

        offset = crc_end
        if chunk_type == b"IEND":
            break

    if offset != len(raw) or not chunk_types or chunk_types[-1] != b"IEND":
        raise AssertionError(f"{path} has bytes outside its PNG chunk stream")
    if ihdr is None or len(animation_control) != 1:
        raise AssertionError(f"{path} does not contain one APNG animation control chunk")
    if b"IDAT" not in chunk_types or b"fdAT" not in chunk_types:
        raise AssertionError(f"{path} is missing APNG image data")
    return ihdr, animation_control[0], frame_controls


class AikimiStatusTests(unittest.TestCase):
    def test_manifest_covers_states_and_references_local_assets(self):
        manifest = json.loads((ASSET_ROOT / "manifest.json").read_text(encoding="utf-8"))

        self.assertGreaterEqual(manifest["version"], 2)
        self.assertEqual(set(manifest["states"]), EXPECTED_STATES)
        self.assertIn(manifest["default_state"], manifest["states"])
        self.assertTrue(set(manifest["preload"]).issubset(EXPECTED_STATES))
        asset_keys = [state["asset"] for state in manifest["states"].values()]
        self.assertEqual(len(set(asset_keys)), len(EXPECTED_STATES))
        for state in manifest["states"].values():
            asset_key = state["asset"]
            self.assertIn(asset_key, manifest["assets"])
            asset = manifest["assets"][asset_key]
            if asset["loop"] == "once":
                self.assertNotIn(asset_key, manifest["preload"])
            for field in ("animated", "still"):
                filename = asset[field]
                self.assertEqual(Path(filename).name, filename)
                self.assertTrue((ASSET_ROOT / filename).is_file())

    def test_legacy_character_sources_remain_available(self):
        for name in ("idle", "working", "happy", "troubled"):
            with self.subTest(name=name):
                with Image.open(ASSET_ROOT / f"{name}.webp") as image:
                    self.assertEqual(image.size, (512, 640))
                    self.assertEqual(image.mode, "RGBA")
                    self.assertEqual(image.getchannel("A").getextrema(), (0, 255))

        with Image.open(ASSET_ROOT / "favicon.png") as favicon:
            self.assertEqual(favicon.size, (128, 128))
            self.assertEqual(favicon.mode, "RGBA")

    def test_apng_assets_match_declared_motion_contract(self):
        manifest = json.loads((ASSET_ROOT / "manifest.json").read_text(encoding="utf-8"))
        animation_hashes = set()
        still_hashes = set()
        preload_bytes = 0
        total_asset_bytes = 0

        for state, state_config in manifest["states"].items():
            with self.subTest(state=state):
                asset = manifest["assets"][state_config["asset"]]
                animated_path = ASSET_ROOT / asset["animated"]
                still_path = ASSET_ROOT / asset["still"]
                animation_hashes.add(hashlib.sha256(animated_path.read_bytes()).hexdigest())

                ihdr, animation_control, frame_controls = parse_apng(animated_path)
                width, height, bit_depth, color_type, *_ = ihdr
                self.assertEqual((width, height, bit_depth, color_type), (384, 480, 8, 6))
                self.assertEqual(animation_control, (asset["frames"], 1 if asset["loop"] == "once" else 0))
                self.assertEqual(len(frame_controls), asset["frames"])
                raw_durations = []
                for control in frame_controls:
                    _, frame_width, frame_height, x_offset, y_offset, numerator, denominator, disposal, blend = control
                    self.assertLessEqual(x_offset + frame_width, width)
                    self.assertLessEqual(y_offset + frame_height, height)
                    self.assertIn(disposal, (0, 1, 2))
                    self.assertIn(blend, (0, 1))
                    duration = Fraction(numerator * 1000, denominator or 100)
                    self.assertEqual(duration.denominator, 1)
                    raw_durations.append(int(duration))
                self.assertEqual(raw_durations, asset["durations_ms"])

                with Image.open(animated_path) as animation:
                    self.assertEqual(animation.format, "PNG")
                    self.assertTrue(animation.is_animated)
                    self.assertEqual(animation.size, (384, 480))
                    self.assertEqual(animation.n_frames, asset["frames"])
                    self.assertEqual(
                        animation.info.get("loop"),
                        1 if asset["loop"] == "once" else 0,
                    )

                    durations = []
                    frame_hashes = []
                    for index in range(animation.n_frames):
                        animation.seek(index)
                        animation.load()
                        durations.append(int(round(animation.info.get("duration", 0))))
                        frame = animation.convert("RGBA")
                        alpha = frame.getchannel("A")
                        self.assertEqual(alpha.getextrema(), (0, 255))
                        bbox = alpha.getbbox()
                        self.assertIsNotNone(bbox)
                        left, top, right, bottom = bbox
                        self.assertGreaterEqual(left, 18)
                        self.assertGreaterEqual(top, 18)
                        self.assertLessEqual(right, width - 18)
                        self.assertLessEqual(bottom, height - 18)
                        width, height = frame.size
                        self.assertIsNone(alpha.crop((0, 0, width, 1)).getbbox())
                        self.assertIsNone(alpha.crop((0, height - 1, width, height)).getbbox())
                        self.assertIsNone(alpha.crop((0, 0, 1, height)).getbbox())
                        self.assertIsNone(alpha.crop((width - 1, 0, width, height)).getbbox())
                        frame_hashes.append(visible_digest(frame))

                    self.assertEqual(durations, asset["durations_ms"])
                    self.assertGreater(len(set(frame_hashes)), 1)
                    if asset["loop"] == "ping-pong":
                        source_count = (asset["frames"] + 2) // 2
                        indices = [*range(source_count), *range(source_count - 2, 0, -1)]
                        self.assertEqual(len(indices), asset["frames"])
                        for frame_index, source_index in enumerate(indices):
                            self.assertEqual(frame_hashes[frame_index], frame_hashes[source_index])
                        still_frame_hash = frame_hashes[0]
                    else:
                        still_frame_hash = frame_hashes[-1]

                with Image.open(still_path) as still:
                    self.assertEqual(still.format, "WEBP")
                    self.assertEqual(still.size, (384, 480))
                    self.assertEqual(still.n_frames, 1)
                    still_rgba = still.convert("RGBA")
                    self.assertEqual(still_rgba.getchannel("A").getextrema(), (0, 255))
                    still_bbox = still_rgba.getchannel("A").getbbox()
                    self.assertIsNotNone(still_bbox)
                    left, top, right, bottom = still_bbox
                    self.assertGreaterEqual(left, 18)
                    self.assertGreaterEqual(top, 18)
                    self.assertLessEqual(right, 384 - 18)
                    self.assertLessEqual(bottom, 480 - 18)
                    still_hash = visible_digest(still_rgba)
                    self.assertEqual(still_hash, still_frame_hash)
                    still_hashes.add(still_hash)

                self.assertLessEqual(animated_path.stat().st_size, 4 * 1024 * 1024)
                self.assertLessEqual(still_path.stat().st_size, 512 * 1024)
                total_asset_bytes += animated_path.stat().st_size + still_path.stat().st_size
                if state in manifest["preload"]:
                    preload_bytes += animated_path.stat().st_size

        self.assertEqual(len(animation_hashes), len(EXPECTED_STATES))
        self.assertEqual(len(still_hashes), len(EXPECTED_STATES))
        self.assertLessEqual(preload_bytes, 12 * 1024 * 1024)
        self.assertLessEqual(total_asset_bytes, 30 * 1024 * 1024)

    def test_model_snapshot_uses_shallow_loader_state(self):
        model_data = SimpleNamespace(
            sd_model=SimpleNamespace(
                forge_objects=object(),
                filename=r"C:\models\loaded-aikimi.safetensors",
                sd_checkpoint_info=SimpleNamespace(name="folder-a/loaded-aikimi.safetensors"),
            ),
            forge_loading_parameters={
                "checkpoint_info": SimpleNamespace(
                    filename=r"C:\models\folder-b\aikimi.safetensors",
                    name="folder-b/aikimi.safetensors",
                )
            },
            forge_hash="old-hash",
            forge_loading=True,
            last_load_seconds=8.21,
            last_load_error=None,
        )

        status = aikimi_status._model_snapshot(model_data)

        self.assertTrue(status["loaded"])
        self.assertTrue(status["loading"])
        self.assertEqual(status["selected_name"], "folder-b/aikimi.safetensors")
        self.assertEqual(status["loaded_name"], "folder-a/loaded-aikimi.safetensors")
        self.assertTrue(status["reload_pending"])
        self.assertNotIn("path", status)
        self.assertEqual(status["last_load_seconds"], 8.21)

    def test_status_auth_matches_gradio_cookie_boundary(self):
        open_app = SimpleNamespace(auth=None, auth_dependency=None)
        request = SimpleNamespace(cookies={})
        self.assertTrue(aikimi_status.request_is_authorized(open_app, request))

        protected_app = SimpleNamespace(
            auth={"user": "hash"},
            auth_dependency=None,
            cookie_id="cookie",
            tokens={"valid-token": "user"},
        )
        self.assertFalse(aikimi_status.request_is_authorized(protected_app, request))
        request.cookies["access-token-unsecure-cookie"] = "valid-token"
        self.assertTrue(aikimi_status.request_is_authorized(protected_app, request))

    def test_generation_snapshot_reports_pending_queue_separately(self):
        state = SimpleNamespace(
            job_count=2,
            job_no=1,
            sampling_steps=10,
            sampling_step=5,
            time_start=None,
            job="task(test)",
            textinfo="Sampling",
        )

        status = aikimi_status._generation_snapshot(
            state,
            pending_tasks={"task(waiting-1)": 1.0, "task(waiting-2)": 2.0},
        )

        self.assertTrue(status["active"])
        self.assertEqual(status["progress"], 0.75)
        self.assertEqual(status["queue_size"], 2)
        self.assertNotIn("task", status)
        self.assertNotIn("job", status)
        self.assertNotIn("queue_tasks", status)

    def test_loader_status_defaults_do_not_change_model_contract(self):
        tree = ast.parse((ROOT / "modules" / "sd_models.py").read_text(encoding="utf-8"))
        model_data_class = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SdModelData"
        )
        constructor = next(
            node for node in model_data_class.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        assigned_attributes = {
            target.attr
            for node in ast.walk(constructor)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "self"
        }

        self.assertIn("forge_loading", assigned_attributes)
        self.assertIn("last_load_seconds", assigned_attributes)
        self.assertIn("last_load_error", assigned_attributes)

    def test_model_loading_flag_is_cleared_by_outer_finally(self):
        tree = ast.parse((ROOT / "modules" / "sd_models.py").read_text(encoding="utf-8"))
        reload_function = next(
            node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "forge_model_reload"
        )
        tracking_try = next(node for node in reload_function.body if isinstance(node, ast.Try))
        final_assignments = [
            node
            for node in ast.walk(ast.Module(body=tracking_try.finalbody, type_ignores=[]))
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Attribute) and target.attr == "forge_loading" for target in node.targets)
        ]

        self.assertEqual(len(final_assignments), 1)
        self.assertIs(final_assignments[0].value.value, False)

    def test_frontend_subscribes_without_replacing_progress_api(self):
        progress_source = (ROOT / "javascript" / "progressbar.js").read_text(encoding="utf-8")
        assistant_source = ASSISTANT_SOURCE.read_text(encoding="utf-8")

        for event_name in (
            "webui:task-start",
            "webui:task-progress",
            "webui:task-error",
            "webui:task-end",
        ):
            self.assertIn(event_name, progress_source)
            self.assertIn(event_name, assistant_source)
        self.assertIn("window.AikimiStatus", assistant_source)
        self.assertIn('matchMedia("(prefers-reduced-motion: reduce)")', assistant_source)
        self.assertIn("failedAssetUrls", assistant_source)
        self.assertIn("preloadConfiguredAssets", assistant_source)
        self.assertIn("let enabled = false", assistant_source)
        self.assertIn("const ACTIVE_POLL_MS = 1500", assistant_source)
        self.assertIn("const MAX_BACKOFF_MS = 60000", assistant_source)
        self.assertIn("const FETCH_TIMEOUT_MS = 8000", assistant_source)
        self.assertIn("pollingController === controller", assistant_source)
        self.assertIn("pollingGeneration", assistant_source)
        self.assertIn("window.__aikimiStatusInitialized", assistant_source)
        self.assertIn("publicTechnicalDetail", assistant_source)
        self.assertIn("window.AikimiTabs", assistant_source)
        self.assertIn('"aikimi:feature-tab-change"', assistant_source)
        self.assertIn("navigationIssue", assistant_source)
        self.assertIn("message: navigationIssue", assistant_source)
        self.assertIn("errorDetails: navigationIssue", assistant_source)
        self.assertIn("statusIsActive()", assistant_source)
        self.assertIn("if (statusIsActive()) scanOutputErrors();", assistant_source)
        self.assertIn("activeContainer", assistant_source)
        self.assertNotIn("createBrandHeader", assistant_source)
        self.assertNotIn("aikimi_assistant_position", assistant_source)
        self.assertNotIn("aikimi-working", (ROOT / "style.css").read_text(encoding="utf-8"))

        manifest = json.loads((ASSET_ROOT / "manifest.json").read_text(encoding="utf-8"))
        for asset in manifest["assets"].values():
            self.assertNotIn(asset["animated"], assistant_source)
            self.assertNotIn(asset["still"], assistant_source)

    def test_assistant_preferences_are_registered_and_manifest_driven(self):
        options_source = (ROOT / "modules" / "shared_options.py").read_text(encoding="utf-8")
        assistant_source = ASSISTANT_SOURCE.read_text(encoding="utf-8")
        css_source = ASSISTANT_CSS.read_text(encoding="utf-8")
        root_css_source = (ROOT / "style.css").read_text(encoding="utf-8")

        for option in (
            "aikimi_assistant_size",
            "aikimi_assistant_dialogue_enabled",
            "aikimi_assistant_animation_enabled",
        ):
            self.assertIn(option, options_source)
            self.assertIn(option, assistant_source)
        for value in ("small", "medium", "large"):
            self.assertIn(f'[data-size="{value}"]', css_source)
        self.assertIn('"aikimi_assistant_position"', options_source)
        self.assertIn('{"visible": False}', options_source)
        self.assertNotIn("data-position", css_source)
        self.assertNotIn("position: fixed", css_source)
        self.assertNotIn("#aikimi-status", root_css_source)
        self.assertNotIn(".aikimi-diagnostics", root_css_source)
        self.assertNotIn(".aikimi-about", root_css_source)
        self.assertIn("--aikimi-character-size: 40px", css_source)
        self.assertIn("--aikimi-character-size: 52px", css_source)
        self.assertIn("--aikimi-character-size: 64px", css_source)
        self.assertIn("max-height: 64px", css_source)
        self.assertIn("#aikimi-status .aikimi-status__summary", css_source)
        self.assertIn("var(--body-text-color) 72%", css_source)
        self.assertNotIn('content: "  ▾"', css_source)
        self.assertNotIn('content: "  ▴"', css_source)
        self.assertIn("@media (forced-colors: active)", css_source)
        self.assertIn("border-color: CanvasText", css_source)
        self.assertIn("outline-color: Highlight", css_source)
        self.assertIn('warning: "Warning"', assistant_source)
        self.assertIn('error: "Error"', assistant_source)
        self.assertIn('out_of_memory: "Out of memory"', assistant_source)
        self.assertIn("stillMode", assistant_source)
        self.assertIn("message.hidden = !dialogueEnabled", assistant_source)


if __name__ == "__main__":
    unittest.main()
