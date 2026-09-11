import copy
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from modules_forge import minimax_h3_bridge as bridge
from modules_forge import minimax_h3_clipcache as cache
from modules_forge.minimax_h3_acceleration import H3Acceleration
from modules_forge.minimax_h3_negpip import H3NegPiP
from tools.install_minimax_h3_clipcache import install


def cached_nodes():
    result = {}
    for name in cache.CLIP_CACHE_NODES:
        required = {
            "clip_name": [[bridge.H3_TEXT_ENCODER]],
            "vae": ["VAE"],
            "prompt": ["STRING"],
            "width": ["INT"],
            "height": ["INT"],
            "length": ["INT"],
        }
        optional = {"cache_mode": [["auto", "refresh"]]}
        if name == cache.CLIP_CACHE_REF2VA:
            required.update(audio_vae=["VAE"], ref_image_size=[["match", "max"]])
            for kind, count in cache.REFERENCE_LIMITS.items():
                optional.update(
                    {f"{kind}_{index}": ["AUDIO" if "audio" in kind else "IMAGE"] for index in range(count)}
                )
        else:
            optional.update(first_frame=["IMAGE"], last_frame=["IMAGE"])
        result[name] = {"input": {"required": required, "optional": optional}, "output": ["CONDITIONING", "LATENT"]}
    return result


class ClipCachePolicyTests(unittest.TestCase):
    def test_settings_round_trip_preserves_existing_positions_and_old_histories(self):
        option = H3Acceleration(clip_cache="auto", tile_batch_size=2)
        self.assertEqual(H3Acceleration.from_values(option.values()), option)
        self.assertEqual(H3Acceleration.from_dict(option.to_dict()), option)
        old = option.to_dict()
        del old["clip_cache"]
        self.assertEqual(H3Acceleration.from_dict(old).clip_cache, "off")
        self.assertEqual(H3Acceleration.from_values(option.values()[:-6]).clip_cache, "off")
        self.assertEqual(option.values()[3], 2)

    def test_invalid_modes_and_modified_clip_fail_explicitly(self):
        for value in (None, True, "enabled"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                H3Acceleration(clip_cache=value).validate()
        H3Acceleration(clip_cache="auto", negpip=H3NegPiP(enabled=True)).validate()

    def test_off_leaves_native_graph_unchanged(self):
        graph = bridge.build_workflow(bridge.H3Request(mode="text", prompt="A quiet observatory"), {}, seed=42)
        before = copy.deepcopy(graph)
        cache.apply_clipcache(graph, "off")
        self.assertEqual(graph, before)
        self.assertEqual(H3Acceleration().runtime_packs(), ())

    def test_cached_keyframes_keep_all_generation_inputs(self):
        request = bridge.H3Request(
            mode="keyframes",
            prompt="gentle wind",
            first_frame="first.png",
            last_frame="last.png",
            acceleration=H3Acceleration(clip_cache="refresh"),
        )
        graph = bridge.build_workflow(request, {"first_frame": "first.png", "last_frame": "last.png"}, seed=42)
        self.assertNotIn("2", graph)
        self.assertEqual(graph["5"]["class_type"], cache.CLIP_CACHE_FL2VA)
        self.assertEqual(graph["5"]["inputs"]["clip_name"], bridge.H3_TEXT_ENCODER)
        self.assertEqual(graph["5"]["inputs"]["cache_mode"], "refresh")
        self.assertIn("first_frame", graph["5"]["inputs"])
        self.assertIn("last_frame", graph["5"]["inputs"])
        self.assertEqual(graph["8"]["inputs"]["steps"], 20)
        self.assertEqual(graph["6"]["inputs"]["noise_seed"], 42)

    def test_reference_slots_keep_media_kind_and_order(self):
        graph = bridge.build_workflow(bridge.H3Request(mode="text", prompt="test"), {}, seed=1)
        graph["5"]["class_type"] = "MiniMaxH3ReferenceToVideo"
        inputs = graph["5"]["inputs"]
        inputs.update(
            {
                "ref_images.ref_image_0": ["20", 0],
                "ref_images.ref_image_8": ["28", 0],
                "ref_videos.ref_video_2": ["31", 0],
                "ref_video_audios.ref_video_audio_2": ["31", 1],
                "ref_audios.ref_audio_2": ["32", 0],
            }
        )
        cache.apply_clipcache(graph, "auto")
        self.assertEqual(graph["5"]["class_type"], cache.CLIP_CACHE_REF2VA)
        for key, link in {
            "ref_image_0": ["20", 0],
            "ref_image_8": ["28", 0],
            "ref_video_2": ["31", 0],
            "ref_video_audio_2": ["31", 1],
            "ref_audio_2": ["32", 0],
        }.items():
            self.assertEqual(graph["5"]["inputs"][key], link)
        self.assertFalse(any("." in key for key in graph["5"]["inputs"]))

    def test_selected_cache_pack_is_the_only_added_permission(self):
        option = H3Acceleration(clip_cache="auto")
        command = bridge._runtime_command(Path("python"), 8199, acceleration=option)
        self.assertIn("--disable-all-custom-nodes", command)
        self.assertEqual(command[command.index("--whitelist-custom-nodes") + 1 :], [cache.CLIP_CACHE_PACK])
        self.assertTrue(bridge._runtime_arguments_are_allowed(command[1:]))
        self.assertFalse(bridge._runtime_arguments_are_allowed(command[1:] + ["ComfyUI-Manager"]))

    def test_reference_workflow_retains_video_soundtrack_and_separate_audio(self):
        request = bridge.H3Request(
            mode="references",
            prompt="<Picture 1> <Audio 1> <Video 1> <Audio 2>",
            reference_images=("photo.png",),
            reference_videos=("clip.mp4",),
            reference_audios=("sound.wav",),
            acceleration=H3Acceleration(clip_cache="auto"),
        )
        graph = bridge.build_workflow(
            request,
            {"images": ["photo.png"], "videos": [{"name": "clip.mp4", "has_audio": True}], "audios": ["sound.wav"]},
        )
        inputs = graph["5"]["inputs"]
        self.assertEqual(inputs["ref_image_0"], ["20", 0])
        self.assertEqual(inputs["ref_video_0"], ["22", 0])
        self.assertEqual(inputs["ref_video_audio_0"], ["22", 1])
        self.assertEqual(inputs["ref_audio_0"], ["23", 0])
        self.assertEqual(inputs["prompt"], request.prompt)

    def test_schema_checks_required_fields_reference_slots_and_outputs(self):
        nodes = cached_nodes()
        cache.validate_clipcache_nodes(nodes)
        for mutate in (
            lambda n: n[cache.CLIP_CACHE_FL2VA].update(output=["LATENT"]),
            lambda n: n[cache.CLIP_CACHE_REF2VA]["input"]["optional"].pop("ref_audio_2"),
            lambda n: n[cache.CLIP_CACHE_REF2VA]["input"]["required"].update(new_required=["INT"]),
            lambda n: n[cache.CLIP_CACHE_FL2VA].update(output=None),
            lambda n: n[cache.CLIP_CACHE_FL2VA]["input"]["optional"].update(cache_mode={"unexpected": "object"}),
        ):
            bad = copy.deepcopy(nodes)
            mutate(bad)
            with self.assertRaises(ValueError):
                cache.validate_clipcache_nodes(bad)

    def test_missing_installation_fails_before_process_start(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "main.py").touch()
            (root / "models").mkdir()
            with mock.patch.object(bridge.subprocess, "Popen") as popen:
                with self.assertRaisesRegex(bridge.H3BridgeError, "未導入"):
                    bridge.start_runtime(
                        root, "http://127.0.0.1:8199", root / "logs", acceleration=H3Acceleration(clip_cache="auto")
                    )
                popen.assert_not_called()


class ClipCacheInstallTests(unittest.TestCase):
    def manifest(self, files):
        return {
            "repository": "Mu5hr00moO/ComfyUI-MiniMaxH3-CLIPCached",
            "revision": cache.CLIP_CACHE_REVISION,
            "files": [
                {
                    "path": name,
                    "bytes": len(content),
                    "blob": hashlib.sha1(
                        b"blob " + str(len(content)).encode() + b"\0" + content, usedforsecurity=False
                    ).hexdigest(),
                }
                for name, content in files.items()
            ],
        }

    def test_install_is_verified_idempotent_and_preserves_existing_edits_and_cache(self):
        files = {"__init__.py": b"# init", "nodes.py": b"# nodes", "LICENSE": b"MIT"}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "main.py").touch()
            (root / "models").mkdir()

            def download(request, **kwargs):
                return io.BytesIO(files[request.full_url.rsplit("/", 1)[1]])

            with (
                mock.patch.object(cache, "clipcache_manifest", return_value=self.manifest(files)),
                mock.patch("tools.install_minimax_h3_clipcache.urlopen", side_effect=download) as fetch,
            ):
                target = install(root)
                self.assertEqual(fetch.call_count, 3)
                self.assertEqual(install(root), target)
                self.assertEqual(fetch.call_count, 3)
                (target / "cache").mkdir()
                (target / "cache/user.safetensors").write_bytes(b"keep")
                (target / "nodes.py").write_bytes(b"user edit")
                with self.assertRaisesRegex(ValueError, "固定版と異なり"):
                    install(root)
                self.assertEqual((target / "nodes.py").read_bytes(), b"user edit")
                self.assertEqual((target / "cache/user.safetensors").read_bytes(), b"keep")

    def test_manifest_path_escape_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(self.manifest({"../escape.py": b"bad"})), encoding="utf-8")
            with mock.patch.object(cache, "MANIFEST", path), self.assertRaises(ValueError):
                cache.clipcache_manifest()
