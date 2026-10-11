"""Orbit applies the pinned accepted bytes and reports the same immutable identity."""

import hashlib
import importlib.util
import json
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from safetensors.torch import save

from modules_forge import local_assets
from modules_forge import minimax_h3_orbit_assets as orbit

ROOT = Path(__file__).resolve().parents[2]
NODE_FILE = ROOT / "extensions-builtin/minimax-h3-studio/comfyui_nodes/Aikimi-MiniMax-H3-Orbit/__init__.py"
TARGET = "diffusion_model.blocks.0.attn.out_proj.weight"


class FakePatcher:
    def __init__(self, weights=None):
        self.weights = {TARGET: torch.tensor([[10.0]])} if weights is None else weights
        self.model = types.SimpleNamespace(state_dict=lambda: self.weights)
        self.patches = {}

    def clone(self):
        copied = FakePatcher(self.weights)
        copied.patches = {key: entries[:] for key, entries in self.patches.items()}
        return copied

    def value(self):
        value = float(self.weights[TARGET].sum())
        for strength, adapter, *_ in self.patches.get(TARGET, []):
            value += strength * float((adapter.weights[0] @ adapter.weights[1]).sum())
        return value


class OrbitFixture(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.path = self.root / "loras" / orbit.MODEL_NAME
        self.path.parent.mkdir()
        self.data = save(
            {
                "diffusion_model.blocks.0.attn.out_proj.lora_A.weight": torch.tensor([[2.0]]),
                "diffusion_model.blocks.0.attn.out_proj.lora_B.weight": torch.tensor([[3.0]]),
            }
        )
        self.path.write_bytes(self.data)
        self.digest = hashlib.sha256(self.data).hexdigest()
        self.enterContext(patch.object(orbit, "MODEL_BYTES", len(self.data)))
        self.enterContext(patch.object(orbit, "MODEL_SHA256", self.digest))
        self.enterContext(patch.object(orbit, "model_root", return_value=self.root))
        self.enterContext(patch.object(local_assets, "HASH_CACHE", self.root / "hashes.json"))


class OrbitIdentityTests(OrbitFixture):
    def request(self):
        from modules_forge.minimax_h3_bridge import H3Request

        return H3Request(mode="orbit", prompt="A camera orbit", first_frame="photo.png", duration_seconds=3)

    def generation_fakes(self, stack):
        from modules_forge import minimax_h3_bridge as bridge

        readiness = bridge.RuntimeReadiness(self.root, "http://127.0.0.1:8189", connected=True)
        client = Mock()
        client.job.return_value = {"status": "completed"}
        ownership = Mock()
        ownership.acquire.return_value = True
        stack.enter_context(patch.object(bridge, "GPUOwnership", return_value=ownership))
        stack.enter_context(patch.object(bridge, "_GPU_OWNERSHIPS", {}))
        stack.enter_context(patch.object(bridge, "pending_jobs", Mock()))
        stack.enter_context(patch.object(bridge, "ComfyH3Client", return_value=client))
        for name in (
            "release_forge_vram",
            "_validate_request_runtime_constraints",
            "cleanup_stale_prepared_media",
            "cleanup_prepared_media",
            "_schedule_deferred_cleanup",
            "_clear_cancelled_job",
        ):
            stack.enter_context(patch.object(bridge, name))
        for name, value in {
            "ensure_ready": readiness,
            "prepare_media": {},
            "build_workflow": {},
            "_is_cancelled_job": False,
            "extract_history_video": self.root / "source.mp4",
            "mirror_result": self.root / "saved.mp4",
        }.items():
            stack.enter_context(patch.object(bridge, name, return_value=value))
        stack.enter_context(patch.object(bridge.workflow_handoff, "capture_generation", return_value=(None, "")))
        stack.enter_context(patch.object(bridge.workflow_handoff, "finish_generation", return_value=""))
        stack.enter_context(patch.object(bridge, "_loopback_server_process", return_value=Mock(pid=1)))
        stack.enter_context(patch("modules_forge.gpu_residency.register"))
        return bridge, client, ownership

    def test_waiting_replacement_stops_before_runtime_or_forge_unload(self):
        with ExitStack() as stack:
            bridge, client, ownership = self.generation_fakes(stack)
            ownership.acquire.side_effect = [False, True]
            stack.enter_context(patch.object(bridge.time, "sleep"))
            events = bridge.run_generation(self.request(), self.root, "http://127.0.0.1:8189", self.root, self.root)
            self.assertEqual(next(events)["stage"], "queued")
            self.path.write_bytes(b"!" * len(self.data))
            with self.assertRaisesRegex(ValueError, "Orbit.*変更"):
                next(events)
            client.submit.assert_not_called()
            bridge.ensure_ready.assert_not_called()
            bridge.release_forge_vram.assert_not_called()
            ownership.release.assert_called_once()

    def test_last_guard_failure_releases_lease_without_pending_submission(self):
        with ExitStack() as stack:
            bridge, client, ownership = self.generation_fakes(stack)

            def changed(_):
                self.path.write_bytes(b"!" * len(self.data))
                return Mock(pid=1)

            bridge._loopback_server_process.side_effect = changed
            with self.assertRaisesRegex(ValueError, "Orbit.*変更"):
                list(bridge.run_generation(self.request(), self.root, "http://127.0.0.1:8189", self.root, self.root))
            client.submit.assert_not_called()
            bridge.pending_jobs.write.assert_not_called()
            bridge._schedule_deferred_cleanup.assert_not_called()
            self.assertEqual(bridge._GPU_OWNERSHIPS, {})
            ownership.release.assert_called_once()

    def test_completed_job_requires_exact_receipt_and_uses_it_for_metadata(self):
        for present in (False, True):
            with self.subTest(receipt=present), ExitStack() as stack:
                bridge, client, _ = self.generation_fakes(stack)
                snapshot = orbit.snapshot(self.root)
                expected = dict(snapshot["identity"], applied_targets=[TARGET], strength_model=1.0)
                receipt = dict(expected, version=snapshot["versions"][""])
                client.history.side_effect = lambda job, receipt=receipt, present=present: (
                    {job: {"outputs": {"40": {"orbit_receipt": [receipt]}}}} if present else {}
                )
                events = bridge.run_generation(self.request(), self.root, "http://127.0.0.1:8189", self.root, self.root)
                if present:
                    self.assertEqual(list(events)[-1]["stage"], "complete")
                    self.assertEqual(bridge.mirror_result.call_args.kwargs["orbit_identity"], expected)
                else:
                    with self.assertRaisesRegex(bridge.H3BridgeError, "Orbit.*照合"):
                        list(events)
                    bridge.mirror_result.assert_not_called()
                bridge._schedule_deferred_cleanup.assert_not_called()

    def test_replacement_before_publication_leaves_no_result_or_metadata(self):
        from modules_forge import minimax_h3_bridge as bridge

        snapshot = orbit.snapshot(self.root)
        source = self.root / "source.mp4"
        source.write_bytes(b"fixture video")
        output = self.root / "output"

        def changed(*_):
            self.path.write_bytes(b"!" * len(self.data))
            return {}

        with (
            patch.object(bridge, "_validate_video_result", side_effect=changed),
            self.assertRaisesRegex(ValueError, "Orbit.*変更"),
        ):
            bridge.mirror_result(
                source,
                output,
                self.request(),
                "job",
                42,
                bridge.RuntimeReadiness(self.root, "http://127.0.0.1:8189", connected=True),
                orbit_snapshot=snapshot,
                orbit_identity=dict(snapshot["identity"], applied_targets=[TARGET], strength_model=1.0),
            )
        self.assertEqual(list(output.iterdir()), [])

    def test_acceptance_snapshot_rejects_wrong_pin_and_detects_replacement(self):
        snapshot = orbit.snapshot(self.root)
        self.assertEqual(
            snapshot["identity"], {"path": str(self.path.resolve()), "size": len(self.data), "sha256": self.digest}
        )
        self.path.write_bytes(b"!" * len(self.data))
        with self.assertRaisesRegex(ValueError, "変更"):
            orbit.verify(snapshot)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            orbit.snapshot(self.root)

    def test_receipt_is_required_and_cannot_claim_a_different_file_or_version(self):
        snapshot = orbit.snapshot(self.root)
        expected = dict(snapshot["identity"], applied_targets=[TARGET], strength_model=1.0)
        receipt = dict(expected, version=snapshot["versions"][""])
        history = {"job": {"outputs": {"40": {"orbit_receipt": [receipt]}}}}
        self.assertEqual(orbit.receipt_identity(history, "job", snapshot), expected)
        for value in (
            {},
            {"job": {"outputs": {}}},
            {"job": {"outputs": {"40": {"orbit_receipt": [dict(receipt, path=str(self.root / "other"))]}}}},
            {"job": {"outputs": {"40": {"orbit_receipt": [dict(receipt, version=[0, 0, None, 0])]}}}},
            {"job": {"outputs": {"40": {"orbit_receipt": [dict(receipt, applied_targets=[])]}}}},
            {"job": {"outputs": {"40": {"orbit_receipt": [dict(receipt, strength_model=0.5)]}}}},
        ):
            with self.subTest(history=value), self.assertRaisesRegex(ValueError, "Orbit"):
                orbit.receipt_identity(value, "job", snapshot)

    def test_bytes_only_receipt_cannot_prove_application(self):
        snapshot = orbit.snapshot(self.root)
        receipt = dict(snapshot["identity"], version=snapshot["versions"][""])
        with self.assertRaisesRegex(ValueError, "照合"):
            orbit.receipt_identity({"job": {"outputs": {"40": {"orbit_receipt": [receipt]}}}}, "job", snapshot)

    def test_missing_external_node_does_not_install_restart_or_modify_runtime(self):
        from modules_forge import minimax_h3_bridge as bridge

        readiness = bridge.RuntimeReadiness(self.root, "http://127.0.0.1:8189", connected=True)
        with (
            patch.object(bridge, "ComfyH3Client") as client,
            patch.object(bridge, "_stop_managed_runtime") as stop,
            patch.object(bridge, "start_runtime") as start,
        ):
            client.return_value.object_info.return_value = {}
            with self.assertRaisesRegex(bridge.H3BridgeError, "Orbit.*ノード"):
                bridge.validate_orbit_runtime(readiness)
            stop.assert_not_called()
            start.assert_not_called()
            self.assertFalse((self.root / "custom_nodes").exists())

    def test_owned_node_install_is_exact_and_does_not_overwrite_existing_code(self):
        (self.root / "main.py").touch()
        (self.root / "models").mkdir()
        target = orbit.install_node(self.root)
        self.assertEqual({p.name for p in target.iterdir()}, {"__init__.py", "orbit_bytes.py"})
        for path in target.iterdir():
            self.assertEqual(path.read_bytes(), (orbit.BUNDLE / path.name).read_bytes())
        self.assertEqual(orbit.install_node(self.root), target)
        (target / "__init__.py").write_bytes(b"unrelated existing code")
        with self.assertRaisesRegex(ValueError, "既存ファイル"):
            orbit.install_node(self.root)
        self.assertEqual((target / "__init__.py").read_bytes(), b"unrelated existing code")

    def test_owned_start_installs_node_before_spawn_and_enables_it(self):
        from modules_forge import minimax_h3_bridge as bridge

        (self.root / "main.py").touch()
        (self.root / "models").mkdir()
        disconnected = bridge.RuntimeReadiness(self.root, bridge.H3_SERVER_URL, connected=False)
        connected = bridge.RuntimeReadiness(self.root, bridge.H3_SERVER_URL, connected=True)
        with (
            patch.object(bridge, "resolve_runtime_root", side_effect=lambda root: root),
            patch.object(bridge, "inspect_readiness", return_value=connected),
            patch.object(bridge, "server_runtime_root", return_value=None),
            patch.object(bridge, "_python_for_runtime", return_value=Path(sys.executable)),
            patch.object(bridge, "_MANAGED_PROCESS", None),
            patch.object(bridge, "_MANAGED_PROCESS_IDENTITY", None),
            patch.object(bridge.time, "sleep"),
            patch.object(bridge, "_server_api_responding", return_value=True),
            patch.object(bridge.workflow_handoff, "install_bundle"),
            patch("modules_forge.nanosaur2_studio.source_ready", return_value=False),
            patch("modules_forge.ming_local.source_ready", return_value=False),
            patch.object(bridge.subprocess, "Popen") as spawn,
            patch.object(orbit, "install_node", wraps=orbit.install_node) as install,
        ):
            spawn.return_value.poll.return_value = None

            def installed_before_spawn(*_, **__):
                self.assertTrue((self.root / "custom_nodes" / orbit.PACK / "orbit_bytes.py").is_file())
                return spawn.return_value

            spawn.side_effect = installed_before_spawn
            self.assertIs(
                bridge._start_runtime_locked(
                    self.root, bridge.H3_SERVER_URL, self.root, initial_readiness=disconnected
                ),
                connected,
            )
            install.assert_called_once_with(self.root)
            self.assertIn(orbit.PACK, spawn.call_args.args[0])


class OrbitWorkerByteTests(OrbitFixture):
    def setUp(self):
        super().setUp()
        name = "_aikimi_orbit_test_node"
        spec = importlib.util.spec_from_file_location(
            name, NODE_FILE, submodule_search_locations=[str(NODE_FILE.parent)]
        )
        self.node = importlib.util.module_from_spec(spec)
        self.enterContext(patch.dict(sys.modules, {name: self.node}))
        spec.loader.exec_module(self.node)
        self.addCleanup(lambda: sys.modules.pop(name + ".orbit_bytes", None))
        self.enterContext(patch.object(self.node.orbit_bytes, "MODEL_BYTES", len(self.data)))
        self.enterContext(patch.object(self.node.orbit_bytes, "MODEL_SHA256", self.digest))
        paths = types.ModuleType("folder_paths")
        paths.get_folder_paths = lambda _: [str(self.root / "wrong"), str(self.path.parent)]
        comfy = types.ModuleType("comfy")
        comfy.__path__ = []
        sd = types.ModuleType("comfy.sd")

        def apply(model, clip, tensors, strength_model, strength_clip):
            self.assertIsNone(clip)
            self.assertEqual(strength_clip, 0)
            down = tensors["diffusion_model.blocks.0.attn.out_proj.lora_A.weight"]
            up = tensors["diffusion_model.blocks.0.attn.out_proj.lora_B.weight"]
            copied = model.clone()
            adapter = types.SimpleNamespace(name="lora", weights=(up, down, None, None, None, None))
            copied.patches.setdefault(TARGET, []).append((strength_model, adapter, 1.0, None, None))
            return copied, None

        self.apply = Mock(side_effect=apply)
        sd.load_lora_for_models = self.apply
        self.enterContext(patch.dict(sys.modules, {"folder_paths": paths, "comfy": comfy, "comfy.sd": sd}))

    def execute(self, **changes):
        values = {
            "model": FakePatcher(),
            "canonical_path": str(self.path.resolve()),
            "size": len(self.data),
            "sha256": self.digest,
            "version": json.dumps(list(local_assets.file_version(self.path))),
        }
        values.update(changes)
        return self.node.MiniMaxH3OrbitLora().apply(**values)

    def test_same_basename_in_another_registered_root_cannot_shadow_accepted_bytes(self):
        wrong = self.root / "wrong" / self.path.name
        wrong.parent.mkdir()
        wrong.write_bytes(b"wrong file with same name")
        result = self.execute()
        self.assertEqual(result["result"][0].value(), 16.0)
        self.assertEqual(result["ui"]["orbit_receipt"][0]["path"], str(self.path.resolve()))
        self.assertEqual(result["ui"]["orbit_receipt"][0]["sha256"], self.digest)

    def test_each_job_deserializes_verified_bytes_without_path_only_cache(self):
        from safetensors.torch import load

        instance = self.node.MiniMaxH3OrbitLora()
        values = {
            "model": FakePatcher(),
            "canonical_path": str(self.path.resolve()),
            "size": len(self.data),
            "sha256": self.digest,
            "version": json.dumps(list(local_assets.file_version(self.path))),
        }
        with patch("safetensors.torch.load", wraps=load) as decode:
            self.assertEqual(instance.apply(**values)["result"][0].value(), 16.0)
            self.assertEqual(instance.apply(**values)["result"][0].value(), 16.0)
        self.assertEqual(decode.call_count, 2)
        self.assertTrue(all(call.args == (self.data,) for call in decode.call_args_list))
        self.assertEqual(self.apply.call_count, 2)
        self.assertTrue(torch.isnan(torch.tensor(instance.IS_CHANGED(**values))))

    def test_change_during_deserialization_is_rejected_before_patching(self):
        from safetensors.torch import load

        def changed(data):
            tensors = load(data)
            self.path.write_bytes(b"!" * len(self.data))
            return tensors

        with patch("safetensors.torch.load", side_effect=changed), self.assertRaisesRegex(ValueError, "変更"):
            self.execute()
        self.apply.assert_not_called()

    def test_change_after_read_or_during_patch_cannot_return_a_receipt(self):
        original_hash = hashlib.sha256

        def changed_hash(data):
            digest = original_hash(data)
            self.path.write_bytes(b"!" * len(self.data))
            return digest

        with (
            patch.object(self.node.orbit_bytes.hashlib, "sha256", side_effect=changed_hash),
            self.assertRaisesRegex(ValueError, "変更"),
        ):
            self.execute()
        self.apply.assert_not_called()
        self.path.write_bytes(self.data)

        original_apply = self.apply.side_effect

        def changed_patch(*args):
            self.path.write_bytes(b"!" * len(self.data))
            return original_apply(*args)

        self.apply.side_effect = changed_patch
        with self.assertRaisesRegex(ValueError, "変更"):
            self.execute()

    def test_unreliable_version_falls_back_to_content_before_patching(self):
        from safetensors.torch import load

        version = list(local_assets.file_version(self.path))
        version[2] = None

        def changed(data):
            tensors = load(data)
            self.path.write_bytes(b"!" * len(self.data))
            return tensors

        with (
            patch.object(self.node.orbit_bytes, "_handle_version", return_value=version),
            patch("safetensors.torch.load", side_effect=changed),
            self.assertRaisesRegex(ValueError, "変更"),
        ):
            self.execute(version=json.dumps(version))
        self.apply.assert_not_called()

    def test_target_name_or_shape_mismatch_stops_before_comfy_loader(self):
        for weights in ({"diffusion_model.unrelated.weight": torch.ones(1, 1)}, {TARGET: torch.ones(2, 1)}):
            with self.subTest(weights=tuple(weights)), self.assertRaisesRegex(ValueError, "対象|形状"):
                self.execute(model=FakePatcher(weights))
        self.apply.assert_not_called()

    def test_no_new_patch_cannot_claim_success_even_with_existing_patches(self):
        existing = FakePatcher()
        existing.patches[TARGET] = [
            (1.0, types.SimpleNamespace(weights=(torch.ones(1, 1), torch.ones(1, 1))), 1.0, None, None)
        ]
        self.apply.side_effect = lambda model, *_: (model.clone(), None)
        with self.assertRaisesRegex(ValueError, "適用"):
            self.execute(model=existing)

    def test_new_patch_must_use_accepted_tensors_and_requested_strength(self):
        original_apply = self.apply.side_effect
        for broken in ("strength", "tensors"):

            def altered(*args, broken=broken):
                copied, _ = original_apply(*args)
                strength, adapter, *rest = copied.patches[TARGET][-1]
                if broken == "strength":
                    strength = 0
                else:
                    adapter.weights = (torch.zeros(1, 1), *adapter.weights[1:])
                copied.patches[TARGET][-1] = (strength, adapter, *rest)
                return copied, None

            self.apply.side_effect = altered
            with self.subTest(broken=broken), self.assertRaisesRegex(ValueError, "適用"):
                self.execute()

    def test_receipt_proves_all_applied_targets_and_requested_strength(self):
        receipt = self.execute(strength_model=0.5)["ui"]["orbit_receipt"][0]
        self.assertEqual(receipt.get("applied_targets"), [TARGET])
        self.assertEqual(receipt.get("strength_model"), 0.5)

    def test_logical_layer_shape_handles_packed_storage_without_false_rejection(self):
        model = FakePatcher({TARGET: torch.empty(1, 0)})
        model.model.get_submodule = lambda _: types.SimpleNamespace(in_features=1, out_features=1)
        self.assertEqual(self.execute(model=model)["ui"]["orbit_receipt"][0]["applied_targets"], [TARGET])

    def test_partial_application_is_rejected_when_other_pairs_are_registered(self):
        from safetensors.torch import load

        tensors = load(self.data)
        first = "diffusion_model.blocks.0.attn.out_proj"
        second = "diffusion_model.blocks.1.attn.out_proj"
        tensors.update({name.replace(first, second): value.clone() for name, value in list(tensors.items())})
        self.data = save(tensors)
        self.digest = hashlib.sha256(self.data).hexdigest()
        self.path.write_bytes(self.data)
        model = FakePatcher({TARGET: torch.ones(1, 1), second + ".weight": torch.ones(1, 1)})
        with (
            patch.object(self.node.orbit_bytes, "MODEL_BYTES", len(self.data)),
            patch.object(self.node.orbit_bytes, "MODEL_SHA256", self.digest),
            self.assertRaisesRegex(ValueError, "適用"),
        ):
            self.execute(model=model)

    def test_unpaired_tensor_structure_is_rejected_before_loader(self):
        with (
            patch(
                "safetensors.torch.load",
                return_value={"diffusion_model.blocks.0.attn.out_proj.lora_A.weight": torch.ones(1, 1)},
            ),
            self.assertRaisesRegex(ValueError, "構成"),
        ):
            self.execute()
        self.apply.assert_not_called()

    def test_unregistered_path_wrong_pin_and_stale_version_are_rejected(self):
        for values in (
            {"canonical_path": str(self.root / "outside" / self.path.name)},
            {"sha256": "0" * 64},
            {"version": "[0,0,null,0]"},
        ):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.execute(**values)
        self.apply.assert_not_called()


if __name__ == "__main__":
    unittest.main()
