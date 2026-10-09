"""Official Turbo provenance, schedule, and atomic install boundaries."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from modules_forge.qwen_image21 import turbo
from modules_forge.qwen_image21.core import QwenImage21Error, atomic_json

MODEL_ID = "Qwen/Qwen-Image-2.1-Turbo"
REVISION = "d65dbc9a7e8f6b5479e33dee6030eaab2a906509"
PRECISIONS = ("turbo_official_int8", "turbo_official_w4a8", "turbo_official_bf16")
SIGMAS = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]
SHARDS = (
    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
)
PINNED = {
    "model_index.json": (578, "97e2febeb19d93cb41fa44807cde8bd42757950b5dffeacdb0331fb2d390cf20"),
    "scheduler/scheduler_config.json": (486, "16c948c71f9152a34cce6e3a2f309e97ad2dca32beecbe5e2fd3a7925b7eb4bb"),
    "transformer/config.json": (370, "2c567038ca190824728844b8d06d94ae02360e668707afd9734e789a7eb58ce3"),
    "transformer/diffusion_pytorch_model.safetensors.index.json": (
        30283,
        "17987f6623b1c814d0ef55a137d99142b7b3b040eb1bf241b5575dd35af803a2",
    ),
    SHARDS[0]: (9968332504, "6cccd922767f01694461bdcf1f34ea7b771f3442e4d5e338ea9aa55277431cc0"),
    SHARDS[1]: (4261951904, "69f53ebb063d2f606bdaff7e22e0e2144f28d978d73dc75e6e4293fc5982eab5"),
}


def fixture_files(mutator=None):
    documents = {
        "model_index.json": {"_class_name": "QwenImage21Pipeline", "sample_sigmas": SIGMAS},
        "scheduler/scheduler_config.json": {
            "_class_name": "FlowMatchEulerDiscreteScheduler",
            "num_train_timesteps": 1000,
            "shift": 1.0,
            "shift_terminal": None,
            "use_dynamic_shifting": False,
            "invert_sigmas": False,
            "use_karras_sigmas": False,
            "use_exponential_sigmas": False,
            "use_beta_sigmas": False,
        },
        "transformer/config.json": {"_class_name": "QwenImage21Transformer2DModel", "num_layers": 32},
        "transformer/diffusion_pytorch_model.safetensors.index.json": {
            "weight_map": {"img_in.weight": Path(SHARDS[0]).name, "norm_out.linear.weight": Path(SHARDS[1]).name}
        },
    }
    if mutator:
        mutator(documents)
    blobs = {name: json.dumps(value).encode() for name, value in documents.items()}
    blobs.update({SHARDS[0]: b"verified-shard-one", SHARDS[1]: b"verified-shard-two"})
    return blobs


class OfficialTurboAssetTests(unittest.TestCase):
    def installed(self, root, *, mutator=None):
        blobs = fixture_files(mutator)
        specs = {name: {"size": len(blob), "sha256": hashlib.sha256(blob).hexdigest()} for name, blob in blobs.items()}
        files = {f"official/{name}": entry.copy() for name, entry in specs.items()}
        for name, blob in blobs.items():
            path = root / "turbo/official" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(blob)
        inventory = {precision: {"model": MODEL_ID, "revision": REVISION, "files": files} for precision in PRECISIONS}
        atomic_json(root / "turbo-files.json", inventory)
        return specs, inventory

    def test_official_profiles_and_all_six_file_pins_are_fixed(self):
        self.assertEqual(turbo.OFFICIAL_ID, MODEL_ID)
        self.assertEqual(turbo.OFFICIAL_REVISION, REVISION)
        self.assertEqual(tuple(turbo.OFFICIAL_PRECISIONS), PRECISIONS)
        self.assertEqual(list(turbo.OFFICIAL_SIGMAS), SIGMAS)
        self.assertEqual(
            {name: (value["size"], value["sha256"]) for name, value in turbo.OFFICIAL_FILES.items()}, PINNED
        )
        for precision in PRECISIONS:
            self.assertEqual(turbo.PROFILES[precision], (MODEL_ID, REVISION, tuple(PINNED)))
            self.assertEqual(turbo.FOLDERS[precision], "official")

    def test_shared_official_install_selects_official_transformer_scheduler_and_eight_sigmas(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            specs, _ = self.installed(root)
            with mock.patch.object(turbo, "OFFICIAL_FILES", specs, create=True):
                for precision in PRECISIONS:
                    self.assertEqual(turbo.turbo_manifest(root, precision, verify_hashes=True)["model"], MODEL_ID)
                    self.assertEqual(turbo.transformer_directory(root, precision), root / "turbo/official/transformer")
                    self.assertEqual(turbo.scheduler_directory(root, precision), root / "turbo/official/scheduler")
                    self.assertEqual(turbo.sampling_sigmas(root, precision), SIGMAS)
                self.assertEqual(turbo.transformer_directory(root, "turbo_bf16"), root / "turbo/bf16/transformer")
                self.assertEqual(turbo.scheduler_directory(root, "turbo_bf16"), root / "turbo/scheduler")
                self.assertIsNone(turbo.sampling_sigmas(root, "turbo_bf16"))

    def test_manifest_rejects_false_source_missing_shard_and_forged_hash(self):
        for defect in ("model", "revision", "missing", "forged-hash"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                specs, inventory = self.installed(root)
                record = inventory[PRECISIONS[0]]
                if defect in ("model", "revision"):
                    record[defect] = "not-the-official-source"
                elif defect == "missing":
                    (root / "turbo/official" / SHARDS[1]).unlink()
                else:
                    record["files"][f"official/{SHARDS[0]}"]["sha256"] = "0" * 64
                atomic_json(root / "turbo-files.json", inventory)
                with mock.patch.object(turbo, "OFFICIAL_FILES", specs, create=True):
                    with self.assertRaises(QwenImage21Error):
                        turbo.turbo_manifest(root, PRECISIONS[0], verify_hashes=True)

    def test_manifest_hash_verification_rejects_same_size_modified_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            specs, _ = self.installed(root)
            path = root / "turbo/official" / SHARDS[0]
            path.write_bytes(b"x" * path.stat().st_size)
            with mock.patch.object(turbo, "OFFICIAL_FILES", specs, create=True):
                with self.assertRaisesRegex(QwenImage21Error, "SHA-256"):
                    turbo.turbo_manifest(root, PRECISIONS[0], verify_hashes=True)

    def test_small_configuration_hashes_are_checked_even_without_weight_hash_verification(self):
        for name in (name for name in PINNED if name.endswith(".json")):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                specs, _ = self.installed(root)
                path = root / "turbo/official" / name
                blob = path.read_bytes()
                # Alter a JSON key while retaining valid JSON and the exact file size.
                position = blob.index(b'"') + 1
                path.write_bytes(blob[:position] + b"x" + blob[position + 1 :])
                json.loads(path.read_text(encoding="utf-8"))
                with mock.patch.object(turbo, "OFFICIAL_FILES", specs, create=True):
                    with self.assertRaisesRegex(QwenImage21Error, "SHA-256"):
                        turbo.turbo_manifest(root, PRECISIONS[0], verify_hashes=False)

    def test_manifest_rejects_semantically_wrong_scheduler_or_sigma_even_with_matching_hash(self):
        mutations = (
            lambda docs: docs["scheduler/scheduler_config.json"].update(use_dynamic_shifting=True),
            lambda docs: docs["scheduler/scheduler_config.json"].update(shift_terminal=0.02),
            lambda docs: docs["model_index.json"].update(sample_sigmas=SIGMAS[:-1]),
            lambda docs: docs["model_index.json"].update(sample_sigmas=[True, *SIGMAS[1:]]),
            lambda docs: docs["transformer/config.json"].update(_class_name="QwenImageTransformer2DModel"),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                specs, _ = self.installed(root, mutator=mutation)
                with mock.patch.object(turbo, "OFFICIAL_FILES", specs, create=True):
                    with self.assertRaises(QwenImage21Error):
                        turbo.turbo_manifest(root, PRECISIONS[0], verify_hashes=True)

    def test_download_reuses_verified_files_and_publishes_all_aliases_without_erasing_viggle(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            specs, _ = self.installed(root)
            atomic_json(root / "turbo-files.json", {"turbo_bf16": {"keep": True}})
            info = SimpleNamespace(
                sha=REVISION,
                siblings=[
                    SimpleNamespace(rfilename=name, size=entry["size"], lfs=SimpleNamespace(sha256=entry["sha256"]))
                    for name, entry in specs.items()
                ],
            )
            download = mock.Mock(side_effect=AssertionError("already verified files must be reused"))
            hub = SimpleNamespace(
                HfApi=lambda: SimpleNamespace(model_info=lambda *a, **k: info), hf_hub_download=download
            )
            with (
                mock.patch.object(turbo, "OFFICIAL_FILES", specs, create=True),
                mock.patch.dict("sys.modules", {"huggingface_hub": hub}),
            ):
                turbo.download_turbo(root, PRECISIONS[0])
                for precision in PRECISIONS:
                    turbo.turbo_manifest(root, precision, verify_hashes=True)
            inventory = json.loads((root / "turbo-files.json").read_text(encoding="utf-8"))
            self.assertEqual(inventory["turbo_bf16"], {"keep": True})
            self.assertTrue(set(PRECISIONS).issubset(inventory))
            download.assert_not_called()

    def test_failed_download_does_not_publish_official_inventory(self):
        for defect in ("revision", "hash", "scheduler"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                blobs = (
                    fixture_files(
                        lambda docs: docs["scheduler/scheduler_config.json"].update(use_dynamic_shifting=True)
                    )
                    if defect == "scheduler"
                    else fixture_files()
                )
                specs = {
                    name: {"size": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}
                    for name, blob in blobs.items()
                }
                info = SimpleNamespace(
                    sha="wrong" if defect == "revision" else REVISION,
                    siblings=[
                        SimpleNamespace(rfilename=name, size=entry["size"], lfs=SimpleNamespace(sha256=entry["sha256"]))
                        for name, entry in specs.items()
                    ],
                )

                def download(repo, *, filename, revision, local_dir, blobs=blobs, defect=defect):
                    self.assertEqual((repo, revision), (MODEL_ID, REVISION))
                    path = Path(local_dir) / filename
                    path.parent.mkdir(parents=True, exist_ok=True)
                    blob = blobs[filename]
                    path.write_bytes(b"x" * len(blob) if defect == "hash" else blob)
                    return str(path)

                hub = SimpleNamespace(
                    HfApi=lambda info=info: SimpleNamespace(model_info=lambda *a, **k: info), hf_hub_download=download
                )
                with (
                    mock.patch.object(turbo, "OFFICIAL_FILES", specs, create=True),
                    mock.patch.dict("sys.modules", {"huggingface_hub": hub}),
                ):
                    with self.assertRaises((RuntimeError, QwenImage21Error, ValueError)):
                        turbo.download_turbo(root, PRECISIONS[0])
                self.assertFalse((root / "turbo-files.json").exists())


if __name__ == "__main__":
    unittest.main()
