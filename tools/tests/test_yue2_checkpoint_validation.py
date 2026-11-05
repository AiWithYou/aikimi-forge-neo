"""YuE2 custom checkpoints must load the exact selected tensor set."""

import json
import os
import struct
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from modules_forge import local_assets
from modules_forge.yue2_studio import core, service


def write_weights(path, tensors):
    header, payload = {}, bytearray()
    for name, (dtype, shape) in tensors.items():
        size = 1
        for dimension in shape:
            size *= dimension
        size *= {"F16": 2, "BF16": 2, "F32": 4, "I8": 1}[dtype]
        start = len(payload)
        payload.extend(bytes(size))
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [start, len(payload)]}
    header["__metadata__"] = {"format": "pt"}
    encoded = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


class YuE2CheckpointValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.folder = self.root / "model"
        self.folder.mkdir()
        self.expected = {"first.weight": (2, 3), "second.weight": (3,)}

    def validator(self):
        from modules_forge.yue2_studio import checkpoint_validation

        return checkpoint_validation

    def index(self, mapping):
        (self.folder / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": mapping}), encoding="utf-8"
        )

    def test_single_file_accepts_only_complete_float_tensor_schema(self):
        validate = self.validator().validate_tensor_headers
        path = self.folder / "model.safetensors"
        write_weights(path, {"first.weight": ("BF16", (2, 3)), "second.weight": ("F32", (3,))})
        validate(self.folder, self.expected)
        cases = (
            {"first.weight": ("BF16", (2, 3))},
            {"first.weight": ("BF16", (2, 3)), "second.weight": ("F32", (3,)), "extra": ("F32", (1,))},
            {"first.weight": ("BF16", (3, 2)), "second.weight": ("F32", (3,))},
            {"first.weight": ("I8", (2, 3)), "second.weight": ("F32", (3,))},
        )
        for tensors in cases:
            with self.subTest(tensors=tensors):
                write_weights(path, tensors)
                with self.assertRaisesRegex(ValueError, "YuE2"):
                    validate(self.folder, self.expected)

    def test_shards_require_index_to_match_actual_names_and_membership(self):
        validate = self.validator().validate_tensor_headers
        a, b = "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
        write_weights(self.folder / a, {"first.weight": ("F16", (2, 3))})
        write_weights(self.folder / b, {"second.weight": ("BF16", (3,))})
        self.index({"first.weight": a, "second.weight": b})
        validate(self.folder, self.expected)
        for mapping in ({"first.weight": b, "second.weight": a}, {"first.weight": a},
                        {"first.weight": a, "second.weight": b, "extra": b}):
            with self.subTest(mapping=mapping):
                self.index(mapping)
                with self.assertRaisesRegex(ValueError, "YuE2"):
                    validate(self.folder, self.expected)
        self.index({"first.weight": a, "second.weight": b})
        write_weights(self.folder / b, {"first.weight": ("F16", (2, 3)), "second.weight": ("BF16", (3,))})
        with self.assertRaisesRegex(ValueError, "YuE2"):
            validate(self.folder, self.expected)

    def test_repeated_json_tensor_names_are_rejected(self):
        raw = b'{"first.weight":{"dtype":"F16","shape":[1],"data_offsets":[0,2]},"first.weight":{"dtype":"F16","shape":[1],"data_offsets":[2,4]}}'
        (self.folder / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + bytes(4))
        with self.assertRaisesRegex(ValueError, "YuE2"):
            self.validator().validate_tensor_headers(self.folder, {"first.weight": (1,)})

    def test_unbounded_or_unsupported_config_is_rejected_before_meta_construction(self):
        validate = self.validator().validate_config
        validate({"model_type": "yue2"})
        for config in ({"num_hidden_layers": 99999999}, {"hidden_size": 0}, {"num_attention_heads": True},
                       {"num_attention_heads": 3, "num_key_value_heads": 2},
                       {"tie_word_embeddings": True}, {"rope_theta": float("inf")},
                       {"hidden_size": 10**10}):
            with self.subTest(config=config), self.assertRaisesRegex(ValueError, "YuE2"):
                validate({"model_type": "yue2", **config})

    def test_custom_revision_does_not_claim_official_model(self):
        (self.folder / "config.json").write_text('{"model_type":"yue2"}', encoding="utf-8")
        (self.folder / "qwen.tiktoken").write_bytes(b"tokenizer")
        (self.folder / "model.safetensors").write_bytes(b"custom")
        record = {"schema": 1, "official": {"model": "official", "vae": "shared", "python": "python",
                   "model_revision": "official-model", "source_revision": "code", "vae_revision": "vae"}}
        (self.root / "runtime.json").write_text(json.dumps(record), encoding="utf-8")
        with patch.object(local_assets, "HASH_CACHE", self.root / "hashes.json"):
            entry = core.runtime_manifest(self.root, "official", str(self.folder))
        self.assertNotIn("model_revision", entry)
        self.assertEqual((entry["source_revision"], entry["vae_revision"]), ("code", "vae"))

    def test_preflight_finishes_before_gpu_lease_and_snapshot_is_checked_again(self):
        selected = self.folder / "model.safetensors"
        selected.write_bytes(b"before")
        request = core.Request(style="piano", lyrics="lyrics", seed=1, checkpoint=str(self.folder))
        lease = Mock()
        studio = service.Studio(self.root / "runtime", self.root / "outputs", ownership_factory=lambda: lease)
        self.addCleanup(studio.shutdown)

        def mutate(*_args):
            selected.write_bytes(b"after")

        with (patch.object(local_assets, "HASH_CACHE", self.root / "hashes.json"),
              patch.object(service, "runtime_manifest", return_value={"model": str(self.folder), "python": "dedicated"}),
              patch.object(service, "validate_custom_checkpoint", side_effect=mutate)):
            with self.assertRaisesRegex(ValueError, "変更"):
                studio.start(request, "owner")
        lease.acquire.assert_not_called()

    def test_preflight_uses_only_pinned_cpu_interpreter(self):
        runtime = self.root / "runtime"
        python = runtime / "official" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        python.parent.mkdir(parents=True)
        python.touch()
        process, tree = Mock(), Mock()
        process.poll.return_value = 0
        process.wait.return_value = 0
        tree.handshake.return_value = b"GO\n"
        with (patch.object(service.subprocess, "Popen", return_value=process) as spawn,
              patch.object(service, "ProcessTree", return_value=tree)):
            service.validate_custom_checkpoint(runtime, {"python": str(python)}, str(self.folder))
            args, kwargs = spawn.call_args
            self.assertEqual(args[0][:3], [str(python.absolute()), "-m", "modules_forge.yue2_studio.checkpoint_validation"])
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "-1")
            self.assertEqual(kwargs["env"]["PYTHONDONTWRITEBYTECODE"], "1")
            self.assertEqual(Path(kwargs["cwd"]), Path(service.__file__).resolve().parents[2])
            process.stdin.write.assert_called_once_with(b"GO\n")
            process.wait.assert_any_call(timeout=60)
            tree.close.assert_called_once()
            with self.assertRaisesRegex(ValueError, "専用Python"):
                service.validate_custom_checkpoint(runtime, {"python": "untrusted"}, str(self.folder))
            self.assertEqual(spawn.call_count, 1)

    def test_preflight_timeout_terminates_owned_process_tree(self):
        runtime = self.root / "runtime"
        python = runtime / "official" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        python.parent.mkdir(parents=True)
        python.touch()
        process, tree = Mock(), Mock()
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("validation", 60), 0]
        tree.handshake.return_value = b"GO\n"
        with (patch.object(service.subprocess, "Popen", return_value=process),
              patch.object(service, "ProcessTree", return_value=tree)):
            with self.assertRaisesRegex(ValueError, "制限時間"):
                service.validate_custom_checkpoint(runtime, {"python": str(python)}, str(self.folder))
        tree.terminate.assert_called_once()
        tree.close.assert_called_once()
        process.wait.assert_any_call(timeout=5)


if __name__ == "__main__":
    unittest.main()
