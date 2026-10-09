"""Offline boundaries for publishing and importing official Turbo quantization."""

import copy
import hashlib
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from safetensors import safe_open

from modules_forge.qwen_image21.capabilities import quantization_precision
from modules_forge.qwen_image21.core import MODEL_REVISION
from modules_forge.qwen_image21.quantized_cache import INT8_SKIP_MODULES, cache_path, manifest
from modules_forge.qwen_image21.turbo import OFFICIAL_ID, OFFICIAL_REVISION
from tools import qwen21_hub_release as release

PROFILES = ("turbo_official_int8", "turbo_official_w4a8")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def tensor_file(profile, component):
    value = float(len(profile) + len(component))
    payload = struct.pack("<ff", value, -value)
    header = {
        "__metadata__": {"format": "pt", "original": "preserved"},
        "weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, len(payload)]},
    }
    return pack_tensor_file(header, payload)


def pack_tensor_file(header, payload):
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    return struct.pack("<Q", len(encoded)) + encoded + payload


def unpack_tensor_file(content):
    header_size = struct.unpack("<Q", content[:8])[0]
    return json.loads(content[8 : 8 + header_size]), content[8 + header_size :]


class Qwen21TurboHubReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.model_root("source")
        self.checkout = self.root / "checkout"
        for profile in PROFILES:
            docs = self.checkout / "docs" / "hf-qwen-release" / profile
            docs.mkdir(parents=True)
            for name in ("README.md", "NOTICE.md"):
                (docs / name).write_text(f"Official Turbo {profile}\n", encoding="utf-8")
            for component in release.COMPONENTS:
                self.write_cache(self.source, component, profile)
        for patcher in (
            patch.object(release, "ROOT", self.checkout),
            patch.object(release, "_identity", side_effect=self.identity, create=True),
            patch("builtins.print"),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def model_root(self, name):
        root = self.root / name
        (root / "model").mkdir(parents=True)
        (root / "model" / "LICENSE").write_text("Synthetic research license\n", encoding="utf-8")
        write_json(root / "model-files.json", {"revision": MODEL_REVISION})
        return root

    def identity(self, model_root, component, profile):
        root = Path(model_root)
        precision = quantization_precision(profile)
        source_file = {"path": f"{component}/weights.safetensors", "size": 16, "sha256": "a" * 64}
        identity = {
            "schema": 1,
            "revision": MODEL_REVISION,
            "component": component,
            "precision": precision,
            "recipe": {
                "version": 1,
                "dtype": "bfloat16",
                "skip_modules": list(INT8_SKIP_MODULES) if precision == "int8" and component == "transformer" else [],
            },
            "versions": {"torch": "test-1", "diffusers": "test-pin", "comfy-kitchen": "test-1"},
            "source_inventory": [source_file],
            "source_files": [["config.json", 16, (root / "model").stat().st_mtime_ns, "b" * 64]],
        }
        if component == "transformer":
            identity.update(
                revision=OFFICIAL_REVISION,
                source_model=OFFICIAL_ID,
                source_path=str(root / "turbo" / "official" / "transformer"),
                source_inventory=[],
                source_record={
                    "model": OFFICIAL_ID,
                    "revision": OFFICIAL_REVISION,
                    "files": {"official/transformer/weights.safetensors": {"size": 16, "sha256": "a" * 64}},
                },
            )
        return identity

    def write_cache(self, root, component, profile, identity=None):
        identity = identity or self.identity(root, component, profile)
        folder = cache_path(root / "model", identity)
        folder.mkdir(parents=True)
        config = {"model_type": "synthetic"}
        if quantization_precision(profile) == "int8":
            config["_name_or_path"] = str(root / "private" / component)
        write_json(folder / "config.json", config)
        (folder / "model.safetensors").write_bytes(tensor_file(profile, component))
        entries = [
            {
                "path": path.name,
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
                "sha256": release.sha256(path),
            }
            for path in sorted(folder.iterdir())
        ]
        write_json(folder / "complete.json", {"identity": identity, "files": entries})
        return folder

    def expected_source(self, identity):
        files = (
            {name.removeprefix("official/"): item for name, item in identity["source_record"]["files"].items()}
            if "source_record" in identity
            else {
                item["path"]: {"size": item["size"], "sha256": item["sha256"]} for item in identity["source_inventory"]
            }
        )
        return {
            "model": identity.get("source_model", release.BASE_MODEL),
            "revision": identity["revision"],
            "files": files,
        }

    def release_fixture(self, profile="turbo_official_int8", name="release"):
        directory = self.root / name
        directory.mkdir()
        record = {
            "schema": 2,
            "base_model": OFFICIAL_ID,
            "base_revision": OFFICIAL_REVISION,
            "precision": quantization_precision(profile),
            "profile": profile,
            "shared_base_model": release.BASE_MODEL,
            "shared_base_revision": MODEL_REVISION,
            "components": {},
        }
        for component in release.COMPONENTS:
            identity = self.identity(self.source, component, profile)
            path = directory / component / "model.safetensors"
            path.parent.mkdir()
            path.write_bytes(tensor_file(profile, component))
            record["components"][component] = {
                "recipe": identity["recipe"],
                "versions": identity["versions"],
                "source": self.expected_source(identity),
                "source_inventory": identity["source_inventory"],
                "files": [
                    {"path": f"{component}/{path.name}", "size": path.stat().st_size, "sha256": release.sha256(path)}
                ],
            }
        write_json(directory / "release_manifest.json", record)
        return directory, record

    def test_cli_accepts_official_quantized_profiles(self):
        for profile in PROFILES:
            with self.subTest(profile=profile), patch.object(release, "stage") as stage:
                output = self.root / profile
                self.assertEqual(
                    release.main(
                        ["stage", "--precision", profile, "--model-root", str(self.source), "--output", str(output)]
                    ),
                    0,
                )
                stage.assert_called_once_with(self.source.resolve(), profile, output.resolve())

    def test_stage_selects_exact_cache_and_exports_portable_source_identity(self):
        for profile in PROFILES:
            with self.subTest(profile=profile):
                for component in release.COMPONENTS:
                    decoy = self.identity(self.source, component, profile)
                    decoy["versions"]["torch"] = "stale-version"
                    self.write_cache(self.source, component, profile, decoy)
                base = self.identity(self.source, "transformer", profile)
                base["revision"], base["source_model"] = MODEL_REVISION, release.BASE_MODEL
                base.pop("source_record")
                self.write_cache(self.source, "transformer", profile, base)
                output = self.root / f"staged-{profile}"
                release.stage(self.source, profile, output)
                record = json.loads((output / "release_manifest.json").read_text(encoding="utf-8"))
                self.assertEqual((record["base_model"], record["base_revision"]), (OFFICIAL_ID, OFFICIAL_REVISION))
                self.assertEqual((record["precision"], record["profile"]), (quantization_precision(profile), profile))
                self.assertEqual(
                    (record["shared_base_model"], record["shared_base_revision"]), (release.BASE_MODEL, MODEL_REVISION)
                )
                for component in release.COMPONENTS:
                    identity = self.identity(self.source, component, profile)
                    self.assertEqual(record["components"][component]["source"], self.expected_source(identity))
                    self.assertEqual(record["components"][component]["recipe"], identity["recipe"])
                    self.assertEqual(record["components"][component]["versions"], identity["versions"])
                    _, payload = unpack_tensor_file((output / component / "model.safetensors").read_bytes())
                    self.assertEqual(payload, unpack_tensor_file(tensor_file(profile, component))[1])
                    if quantization_precision(profile) == "int8":
                        config = json.loads((output / component / "config.json").read_text(encoding="utf-8"))
                        self.assertEqual(
                            config["_name_or_path"], OFFICIAL_ID if component == "transformer" else release.BASE_MODEL
                        )
                self.assertFalse(list(output.rglob("complete.json")))
                self.assertIn(profile, (output / "README.md").read_text(encoding="utf-8"))
                for path in output.rglob("*.json"):
                    content = path.read_text(encoding="utf-8")
                    for private in (str(self.root), "source_path", "source_files", "mtime_ns"):
                        self.assertNotIn(private, content)

    def test_stage_rejects_invalid_exact_cache_without_falling_back(self):
        profile = "turbo_official_int8"
        folder = cache_path(self.source / "model", self.identity(self.source, "transformer", profile))
        original = json.loads((folder / "complete.json").read_text(encoding="utf-8"))
        for fault in ("identity", "hash", "weights-hash", "traversal", "duplicate"):
            with self.subTest(fault=fault):
                record = copy.deepcopy(original)
                if fault == "identity":
                    record["identity"]["versions"]["torch"] = "wrong-version"
                elif fault == "hash":
                    record["files"][0]["sha256"] = "0" * 64
                elif fault == "weights-hash":
                    next(item for item in record["files"] if item["path"] == "model.safetensors")["sha256"] = "0" * 64
                elif fault == "duplicate":
                    record["files"].append(copy.deepcopy(record["files"][0]))
                else:
                    record["files"][0]["path"] = "../outside.safetensors"
                write_json(folder / "complete.json", record)
                with self.assertRaises(ValueError):
                    release.stage(self.source, profile, self.root / f"invalid-{fault}")

    def test_stage_adds_per_file_notices_preserving_tensors_and_source_cache(self):
        for profile in PROFILES:
            with self.subTest(profile=profile):
                originals = {}
                for component in release.COMPONENTS:
                    folder = cache_path(self.source / "model", self.identity(self.source, component, profile))
                    originals[component] = {path.name: path.read_bytes() for path in folder.iterdir()}
                    with safe_open(folder / "model.safetensors", framework="np") as tensors:
                        self.assertEqual(tensors.get_tensor("weight").shape, (2,))
                output = self.root / f"notices-{profile}"
                release.stage(self.source, profile, output)
                record = json.loads((output / "release_manifest.json").read_text(encoding="utf-8"))
                for component in release.COMPONENTS:
                    identity = self.identity(self.source, component, profile)
                    model = identity.get("source_model", release.BASE_MODEL)
                    old_header, old_payload = unpack_tensor_file(originals[component]["model.safetensors"])
                    path = output / component / "model.safetensors"
                    new_header, new_payload = unpack_tensor_file(path.read_bytes())
                    self.assertEqual(new_payload, old_payload)
                    self.assertEqual(new_header["weight"], old_header["weight"])
                    metadata = new_header["__metadata__"]
                    for key, value in old_header["__metadata__"].items():
                        self.assertEqual(metadata[key], value)
                    notice = metadata["aikimi_modification_notice"]
                    for required in (
                        "Built with Qwen",
                        "Aikimi",
                        model,
                        identity["revision"],
                        quantization_precision(profile).upper(),
                    ):
                        self.assertIn(required, notice)
                    with safe_open(path, framework="np") as tensors:
                        self.assertEqual(tensors.get_tensor("weight").tobytes(), old_payload)
                        self.assertEqual(tensors.metadata(), metadata)
                    entries = {item["path"]: item for item in record["components"][component]["files"]}
                    exported = entries[f"{component}/model.safetensors"]
                    self.assertEqual(
                        (exported["size"], exported["sha256"]), (path.stat().st_size, release.sha256(path))
                    )
                    original_hash = hashlib.sha256(originals[component]["model.safetensors"]).hexdigest()
                    self.assertNotEqual(exported["sha256"], original_hash)
                    self.assertEqual(exported["quantized_source_sha256"], original_hash)
                    self.assertEqual(exported["tensor_payload_sha256"], hashlib.sha256(old_payload).hexdigest())
                    config = json.loads((output / component / "config.json").read_text(encoding="utf-8"))
                    json_notice = config["_aikimi_modification_notice"]
                    for required in ("Built with Qwen", "Aikimi", model, identity["revision"]):
                        self.assertIn(required, json_notice)
                    exported_config = entries[f"{component}/config.json"]
                    self.assertEqual(
                        exported_config["quantized_source_sha256"],
                        hashlib.sha256(originals[component]["config.json"]).hexdigest(),
                    )
                    folder = cache_path(self.source / "model", identity)
                    self.assertEqual({path.name: path.read_bytes() for path in folder.iterdir()}, originals[component])

    def test_stage_rejects_malformed_safetensors_before_exporting_payload(self):
        profile = "turbo_official_int8"
        folder = cache_path(self.source / "model", self.identity(self.source, "transformer", profile))
        header, payload = unpack_tensor_file(tensor_file(profile, "transformer"))
        outside = copy.deepcopy(header)
        outside["weight"]["data_offsets"][1] += 4
        overlap = copy.deepcopy(header)
        overlap["second_weight"] = copy.deepcopy(overlap["weight"])
        hole = copy.deepcopy(header)
        hole["weight"].update(shape=[1], data_offsets=[4, 8])
        size_mismatch = copy.deepcopy(header)
        size_mismatch["weight"]["shape"] = [3]
        private = copy.deepcopy(header)
        private["__metadata__"]["cache_path"] = "/mnt/neo-private"
        malformed = {
            "zero-header": struct.pack("<Q", 0),
            "oversized-header": struct.pack("<Q", 64 * 1024 * 1024 + 1),
            "truncated-header": struct.pack("<Q", 32) + b"{}",
            "offset-outside-file": pack_tensor_file(outside, payload),
            "overlapping-tensors": pack_tensor_file(overlap, payload),
            "payload-hole": pack_tensor_file(hole, payload),
            "descriptor-size-mismatch": pack_tensor_file(size_mismatch, payload),
            "extra-payload": pack_tensor_file(header, payload + b"tail"),
            "private-metadata": pack_tensor_file(private, payload),
        }
        complete = json.loads((folder / "complete.json").read_text(encoding="utf-8"))
        for fault, content in malformed.items():
            with self.subTest(fault=fault):
                path = folder / "model.safetensors"
                path.write_bytes(content)
                entry = next(item for item in complete["files"] if item["path"] == path.name)
                entry.update(size=len(content), mtime_ns=path.stat().st_mtime_ns, sha256=release.sha256(path))
                write_json(folder / "complete.json", complete)
                with self.assertRaises(ValueError):
                    release.stage(self.source, profile, self.root / f"malformed-{fault}")
                self.assertEqual(path.read_bytes(), content)

    def test_stage_rejects_private_absolute_path_outside_name_or_path(self):
        profile = "turbo_official_int8"
        folder = cache_path(self.source / "model", self.identity(self.source, "transformer", profile))
        private_path = "/tmp/neo-private"  # noqa: S108 - metadata only, never accessed.
        write_json(folder / "config.json", {"model_type": "synthetic", "artifact_cache": private_path})
        record = json.loads((folder / "complete.json").read_text(encoding="utf-8"))
        for entry in record["files"]:
            path = folder / entry["path"]
            entry.update(size=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns, sha256=release.sha256(path))
        write_json(folder / "complete.json", record)
        with self.assertRaisesRegex(ValueError, "[Ll]ocal.*path|[Pp]rivate.*path|[Aa]bsolute.*path"):
            release.stage(self.source, profile, self.root / "private-config")

    def test_install_rebinds_identity_and_atomically_completes_each_profile(self):
        for profile in PROFILES:
            with self.subTest(profile=profile):
                directory, _ = self.release_fixture(profile, f"release-{profile}")
                target = self.model_root(f"target-{profile}")
                release.install(target, directory, profile)
                release.install(target, directory, profile)
                for component in release.COMPONENTS:
                    identity = self.identity(target, component, profile)
                    destination = cache_path(target / "model", identity)
                    record = manifest(destination, identity, verify_hashes=True)
                    self.assertEqual(record["identity"], identity)
                    self.assertTrue(all("mtime_ns" in item for item in record["files"]))
                self.assertFalse(list((target / "quantized").rglob("*.building-import-*")))

    def test_install_rejects_wrong_source_pin_recipe_or_versions_before_publication(self):
        profile = "turbo_official_int8"
        directory, original = self.release_fixture(profile)
        for fault in (
            "base-model",
            "base-revision",
            "profile",
            "shared-model",
            "shared-revision",
            "source-model",
            "source-revision",
            "source-hash",
            "recipe",
            "versions",
            "encoder-source-revision",
            "encoder-versions",
        ):
            with self.subTest(fault=fault):
                record = copy.deepcopy(original)
                transformer = record["components"]["transformer"]
                if fault == "base-model":
                    record["base_model"] = release.BASE_MODEL
                elif fault == "base-revision":
                    record["base_revision"] = "0" * 40
                elif fault == "profile":
                    record["profile"] = "turbo_official_w4a8"
                elif fault == "shared-model":
                    record["shared_base_model"] = OFFICIAL_ID
                elif fault == "shared-revision":
                    record["shared_base_revision"] = "0" * 40
                elif fault == "source-model":
                    transformer["source"]["model"] = release.BASE_MODEL
                elif fault == "source-revision":
                    transformer["source"]["revision"] = MODEL_REVISION
                elif fault == "source-hash":
                    transformer["source"]["files"]["transformer/weights.safetensors"]["sha256"] = "0" * 64
                elif fault == "recipe":
                    transformer["recipe"]["dtype"] = "float32"
                elif fault == "versions":
                    transformer["versions"]["torch"] = "wrong-version"
                elif fault == "encoder-source-revision":
                    record["components"]["text_encoder"]["source"]["revision"] = OFFICIAL_REVISION
                else:
                    record["components"]["text_encoder"]["versions"]["torch"] = "wrong-version"
                write_json(directory / "release_manifest.json", record)
                target = self.model_root(f"invalid-{fault}")
                with self.assertRaises(ValueError):
                    release.install(target, directory, profile)
                self.assertFalse(list(target.rglob("complete.json")))

    def test_install_rejects_corrupt_hash_and_traversal_without_complete_cache(self):
        profile = "turbo_official_int8"
        directory, original = self.release_fixture(profile)
        for fault in ("hash", "traversal"):
            with self.subTest(fault=fault):
                record = copy.deepcopy(original)
                entry = record["components"]["transformer"]["files"][0]
                entry["sha256"] = "0" * 64 if fault == "hash" else entry["sha256"]
                if fault == "traversal":
                    entry["path"] = "transformer/../../outside.safetensors"
                    (self.root / "outside.safetensors").write_bytes(b"outside")
                write_json(directory / "release_manifest.json", record)
                target = self.model_root(f"invalid-{fault}")
                with self.assertRaisesRegex(ValueError, "[Cc]orrupt" if fault == "hash" else "[Pp]ath"):
                    release.install(target, directory, profile)
                self.assertFalse(list(target.rglob("complete.json")))

    def test_install_rejects_corrupt_existing_cache_instead_of_trusting_same_shape(self):
        profile = "turbo_official_int8"
        directory, _ = self.release_fixture(profile)
        target = self.model_root("corrupt-target")
        folder = self.write_cache(target, "transformer", profile)
        weights = folder / "model.safetensors"
        before = weights.read_bytes()
        weights.write_bytes(b"x" * len(before))
        with self.assertRaisesRegex(ValueError, "[Cc]orrupt"):
            release.install(target, directory, profile)
        self.assertEqual(weights.read_bytes(), b"x" * len(before))
        self.assertFalse(list((target / "quantized").glob("*/**/text_encoder/complete.json")))


if __name__ == "__main__":
    unittest.main()
