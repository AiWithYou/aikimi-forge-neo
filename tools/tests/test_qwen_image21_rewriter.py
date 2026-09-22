"""Optional rewriter integrity, parsing, install isolation, and UI contracts."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from modules_forge.qwen_image21 import prompt_rewriter as rewriter
from modules_forge.qwen_image21.core import QwenImage21Error, atomic_json
from tools import setup_qwen_image21 as setup


def installed_rewriter(root, *, editing=False):
    model_id, revision, directory = rewriter.profile(editing)
    model = root / directory
    model.mkdir()
    atomic_json(
        model / "config.json",
        {
            "model_type": "qwen3_5" if editing else "qwen3_5_text",
            "quantization_config": {
                "quant_method": "bitsandbytes",
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_use_double_quant": True,
            },
        },
    )
    for name in ("model.safetensors", "tokenizer.json", "system_prompt.txt"):
        (model / name).write_text("fixture", encoding="utf-8")
    if editing:
        atomic_json(
            model / "processor_config.json", {"image_processor": {"image_processor_type": "Qwen2VLImageProcessor"}}
        )
    record = {
        "schema": 1,
        "model": model_id,
        "revision": revision,
        "quantization": "nf4-double",
        "files": [
            {"path": path.name, "size": path.stat().st_size, "sha256": rewriter.file_hash(path)}
            for path in model.iterdir()
        ],
    }
    atomic_json(model / "rewriter-files.json", record)
    return model, record


class RewriterTests(unittest.TestCase):
    def test_chat_turn_end_stops_generation_in_addition_to_document_end(self):
        tokenizer = SimpleNamespace(eos_token_id=248046)
        model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=248044))
        self.assertEqual(rewriter.generation_eos_ids(tokenizer, model), [248046, 248044])
        model.generation_config.eos_token_id = [248044, 248046]
        self.assertEqual(rewriter.generation_eos_ids(tokenizer, model), [248046, 248044])
        tokenizer.eos_token_id = None
        model.generation_config.eos_token_id = None
        with self.assertRaises(QwenImage21Error):
            rewriter.generation_eos_ids(tokenizer, model)

    def test_edit_parser_requires_one_valid_ratio_source(self):
        for ratio, follow in (("", "<image2>"), ("16:9", ""), ("18:39", ""), ("2.39:1", "")):
            answer = {"rewritten_prompt": "Edit <image2>.", "wh_ratio": ratio, "ratio_follow": follow}
            self.assertEqual(rewriter.parse_rewrite(json.dumps(answer), editing=True), answer)
        for ratio, follow in (
            ("", ""),
            ("1:1", "<image1>"),
            ("", "<image0>"),
            ("", "<image11>"),
            ("0:1", ""),
            ("NaN:1", ""),
        ):
            answer = {"rewritten_prompt": "Edit it.", "wh_ratio": ratio, "ratio_follow": follow}
            with self.subTest(answer=answer), self.assertRaises(QwenImage21Error):
                rewriter.parse_rewrite(json.dumps(answer), editing=True)
        with self.assertRaises(QwenImage21Error):
            rewriter.parse_rewrite('{"rewritten_prompt":"Edit it.","wh_ratio":"1:1"}', editing=True)

    def test_edit_rewrite_preserves_literals_references_and_original_constraints(self):
        prompt = "<image1>の看板を「夏祭り」に変更。背景は変えない。"
        answer = {"rewritten_prompt": 'In <image1>, write "夏祭り" on the sign.', "ratio_follow": "<image1>"}
        result = rewriter.validate_edit_rewrite(answer, prompt, 2)
        self.assertTrue(result["rewritten_prompt"].endswith(prompt))
        self.assertNotIn("authoritative", answer["rewritten_prompt"])
        for changes in (
            {"rewritten_prompt": 'In <image1>, write "festival" on the sign.'},
            {"rewritten_prompt": 'Write "夏祭り" on the sign.'},
            {"rewritten_prompt": 'Copy <image3> to <image1> and write "夏祭り".'},
            {"ratio_follow": "<image3>"},
        ):
            with self.subTest(changes=changes), self.assertRaises(QwenImage21Error):
                rewriter.validate_edit_rewrite({**answer, **changes}, prompt, 2)

    def test_edit_model_manifest_is_separate_and_requires_recorded_image_processor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            installed_rewriter(root)
            self.assertIn("--edit-prompt-rewriter-only", rewriter.rewriter_status(root, editing=True))
            model, record = installed_rewriter(root, editing=True)
            self.assertEqual(rewriter.rewriter_manifest(root, editing=True, verify_hashes=True)["path"], str(model))
            record["files"] = [item for item in record["files"] if item["path"] != "processor_config.json"]
            atomic_json(model / "rewriter-files.json", record)
            with self.assertRaisesRegex(QwenImage21Error, "画像入力"):
                rewriter.rewriter_manifest(root, editing=True)

    def test_natural_image_references_are_checked_but_quoted_image_text_is_not(self):
        prompt = "Image 1 contains the marked region. 画像2のカップを1枚目へ移し、看板に「画像9」と書く。"
        answer = {
            "rewritten_prompt": 'Move the cup from <image2> into <image1> and write "画像9" on the sign.',
            "ratio_follow": "<image1>",
        }
        self.assertTrue(rewriter.validate_edit_rewrite(answer, prompt, 2)["rewritten_prompt"].endswith(prompt))
        for text in (
            'Move the cup from Image 3 into Image 1 and write "画像9".',
            'Move the cup into 画像1 and write "画像9".',
        ):
            with self.subTest(text=text), self.assertRaises(QwenImage21Error):
                rewriter.validate_edit_rewrite({**answer, "rewritten_prompt": text}, prompt, 2)
        # Unadorned quantities are never interpreted as image IDs.
        result = {"rewritten_prompt": 'Add 12 stars and write "Image 9".', "ratio_follow": "<image1>"}
        rewriter.validate_edit_rewrite(result, 'Add 12 stars and write "Image 9".', 1)
        dimensions = "Make the image 16:9. 画像1920x1080で12個の星を追加。"
        rewriter.validate_edit_rewrite({"rewritten_prompt": dimensions}, dimensions, 1)
        with self.assertRaisesRegex(QwenImage21Error, "参照番号に対応する画像"):
            rewriter.rewrite_prompt(
                Path("."), "Image 3を編集。", 1024, 1024, 0, mock.Mock(), mock.Mock(), images=[object()]
            )

    def test_only_unambiguous_single_image_references_can_be_made_explicit(self):
        answer = {"rewritten_prompt": "Recolor the cup blue.", "ratio_follow": "<image1>"}
        result = rewriter.validate_edit_rewrite(answer, "Image 1のカップを青く。", 1)
        self.assertTrue(result["rewritten_prompt"].startswith("<image1>: Recolor"))
        for prompt, count, changes in (
            ("Image 1のカップを青く。", 2, {}),
            ("Image 2のカップを青く。", 1, {}),
            ("Image 1のカップを青く。", 1, {"ratio_follow": ""}),
            ("Image 1のカップを青く。", 1, {"rewritten_prompt": "Recolor Image 2 blue."}),
        ):
            with self.subTest(prompt=prompt, count=count, changes=changes), self.assertRaises(QwenImage21Error):
                rewriter.validate_edit_rewrite({**answer, **changes}, prompt, count)

    def test_parser_supports_official_thinking_and_direct_non_thinking_json(self):
        answer = {"rewritten_prompt": 'A poster reading "夏祭り".', "wh_ratio": "2:3"}
        text = json.dumps(answer, ensure_ascii=False)
        for value in (text, f"<think>Ignore this reasoning.</think>{text}", f"```json\n{text}\n```"):
            with self.subTest(value=value):
                self.assertEqual(rewriter.parse_rewrite(value), answer)

    def test_invalid_or_truncated_response_never_becomes_an_image_prompt(self):
        for value in (
            "<think>unfinished",
            "[]",
            "null",
            "{}",
            '{"rewritten_prompt": "x"}',
            json.dumps({"rewritten_prompt": "", "wh_ratio": "1:1"}),
            json.dumps({"rewritten_prompt": "x", "wh_ratio": "unknown"}),
            json.dumps({"rewritten_prompt": 42, "wh_ratio": "1:1"}),
            json.dumps({"rewritten_prompt": "x" * 12001, "wh_ratio": "1:1"}),
        ):
            with self.subTest(value=value[:80]), self.assertRaises(QwenImage21Error):
                rewriter.parse_rewrite(value)

    def test_optional_install_checks_real_quantization_files_and_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, _ = installed_rewriter(root)
            self.assertEqual(rewriter.rewriter_manifest(root, verify_hashes=True)["path"], str(model))
            (model / "model.safetensors").write_text("changed", encoding="utf-8")
            # Equal-sized changes are caught by explicit verification.
            with self.assertRaisesRegex(QwenImage21Error, "SHA-256"):
                rewriter.rewriter_manifest(root, verify_hashes=True)
            (model / "model.safetensors").unlink()
            with self.assertRaisesRegex(QwenImage21Error, "OFF"):
                rewriter.rewriter_manifest(root)

    def test_registration_rejects_escape_duplicate_files_and_wrong_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, record = installed_rewriter(root)
            for changes in (
                {"revision": "wrong"},
                {"files": [{"path": "../../outside", "size": 1}]},
                {"files": [*record["files"], record["files"][0]]},
                {"files": []},
            ):
                atomic_json(model / "rewriter-files.json", {**record, **changes})
                with self.subTest(changes=changes), self.assertRaises(QwenImage21Error):
                    rewriter.rewriter_manifest(root)

    def test_bf16_checkpoint_cannot_masquerade_as_nf4(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model, record = installed_rewriter(root)
            config = json.loads((model / "config.json").read_text(encoding="utf-8"))
            config["quantization_config"]["bnb_4bit_use_double_quant"] = False
            atomic_json(model / "config.json", config)
            next(item for item in record["files"] if item["path"] == "config.json")["size"] = (
                (model / "config.json").stat().st_size
            )
            atomic_json(model / "rewriter-files.json", record)
            with self.assertRaisesRegex(QwenImage21Error, "NF4"):
                rewriter.rewriter_manifest(root)

    def test_missing_rewriter_is_actionable_without_model_imports_or_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertIn("--prompt-rewriter-only", rewriter.rewriter_status(root))
            self.assertEqual(list(root.iterdir()), [])

    def test_quantization_includes_large_output_head_and_double_quant(self):
        capture = mock.Mock(side_effect=lambda **kwargs: kwargs)
        with mock.patch.dict(
            "sys.modules",
            {
                "torch": SimpleNamespace(bfloat16="bfloat16"),
                "transformers": SimpleNamespace(BitsAndBytesConfig=capture),
            },
        ):
            config = rewriter.quantization_config()
            edit_config = rewriter.quantization_config(editing=True)
        self.assertIs(config["load_in_4bit"], True)
        self.assertEqual(config["llm_int8_skip_modules"], [])
        self.assertIs(config["bnb_4bit_use_double_quant"], True)
        self.assertEqual(config["bnb_4bit_quant_type"], "nf4")
        self.assertEqual(edit_config["llm_int8_skip_modules"], ["model.visual"])
        self.assertIs(edit_config["bnb_4bit_use_double_quant"], True)

    def test_rewriter_only_reuses_environment_and_never_downloads_image_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = mock.Mock()
            with (
                mock.patch("modules_forge.qwen_image21.core.runtime_lock", return_value=lock),
                mock.patch("modules_forge.qwen_image21_environment.environment_status", return_value=(True, "OK")),
                mock.patch.object(setup, "install_environment") as install,
                mock.patch.object(setup, "download_model") as download,
                mock.patch.object(setup, "execute") as execute,
            ):
                self.assertEqual(setup.main(["--root", str(root), "--prompt-rewriter-only"]), 0)
            install.assert_not_called()
            download.assert_not_called()
            self.assertIn("--prepare-rewriter", execute.call_args.args[0])
            self.assertFalse((root / "runtime.json").exists())
            lock.close.assert_called_once()

    def test_edit_rewriter_install_and_verify_do_not_touch_text_or_image_models(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = mock.Mock()
            with (
                mock.patch("modules_forge.qwen_image21.core.runtime_lock", return_value=lock),
                mock.patch("modules_forge.qwen_image21_environment.environment_status", return_value=(True, "OK")),
                mock.patch.object(setup, "install_environment") as install,
                mock.patch.object(setup, "download_model") as download,
                mock.patch.object(setup, "execute") as execute,
            ):
                self.assertEqual(setup.main(["--root", str(root), "--edit-prompt-rewriter-only"]), 0)
            install.assert_not_called()
            download.assert_not_called()
            self.assertIn("--prepare-edit-rewriter", execute.call_args.args[0])
            with (
                mock.patch("modules_forge.qwen_image21.core.runtime_lock", return_value=lock),
                mock.patch.object(rewriter, "rewriter_manifest") as manifest,
            ):
                self.assertEqual(setup.main(["--root", str(root), "--edit-prompt-rewriter-only", "--verify"]), 0)
            manifest.assert_called_once_with(root, verify_hashes=True, editing=True)
            with mock.patch("sys.stdout", new_callable=io.StringIO) as output:
                self.assertEqual(setup.main(["--root", str(root), "--edit-prompt-rewriter-only", "--dry-run"]), 0)
            plan = json.loads(output.getvalue())
            self.assertEqual(plan["model"], rewriter.EDIT_MODEL_ID)
            self.assertIsNone(plan["prompt_rewriter"])
            self.assertIs(plan["edit_prompt_rewriter"]["text_only"], False)


@unittest.skipUnless(os.environ.get("QWEN_IMAGE21_REWRITER_GPU_TEST") == "1", "Opt-in dedicated CUDA environment test")
class RewriterCudaRoundtripTests(unittest.TestCase):
    def test_real_qwen35_text_nf4_save_reload_and_forward(self):
        import gc

        import torch
        from transformers import AutoModelForCausalLM, Qwen3_5Config, Qwen3_5ForConditionalGeneration, Qwen3_5TextConfig

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required")
        config = Qwen3_5TextConfig(
            vocab_size=128,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            linear_num_key_heads=2,
            linear_num_value_heads=4,
            linear_key_head_dim=16,
            linear_value_head_dim=16,
            layer_types=["linear_attention", "full_attention"],
            max_position_embeddings=128,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000,
                "partial_rotary_factor": 0.5,
                "mrope_section": [1, 1, 2],
                "mrope_interleaved": True,
            },
        )
        with tempfile.TemporaryDirectory() as temporary:
            source, quantized = Path(temporary) / "source", Path(temporary) / "nf4"
            # Match the official PE checkpoint's multimodal architecture. Loading
            # through AutoModelForCausalLM must drop vision and remap text keys.
            full_config = Qwen3_5Config(
                text_config=config.to_dict(),
                vision_config={
                    "depth": 1,
                    "hidden_size": 32,
                    "intermediate_size": 64,
                    "num_heads": 4,
                    "patch_size": 4,
                    "temporal_patch_size": 2,
                    "out_hidden_size": 64,
                    "num_position_embeddings": 16,
                },
            )
            original = Qwen3_5ForConditionalGeneration(full_config).to(torch.bfloat16)
            embedding = original.model.language_model.embed_tokens.weight.detach().clone()
            original.save_pretrained(source)
            original = None
            model = AutoModelForCausalLM.from_pretrained(
                source,
                dtype=torch.bfloat16,
                device_map={"": "cuda:0"},
                quantization_config=rewriter.quantization_config(),
                local_files_only=True,
            ).eval()
            layers = rewriter.check_quantized_model(model)
            torch.testing.assert_close(model.model.embed_tokens.weight.cpu(), embedding, rtol=0, atol=0)
            with torch.inference_mode():
                expected = model(torch.tensor([[1, 7, 9]], device="cuda:0")).logits.cpu()
            self.assertTrue(torch.isfinite(expected).all())
            model.save_pretrained(quantized)
            model = None
            gc.collect()
            torch.cuda.empty_cache()
            restored = AutoModelForCausalLM.from_pretrained(
                quantized,
                dtype=torch.bfloat16,
                device_map={"": "cuda:0"},
                local_files_only=True,
            ).eval()
            self.assertEqual(rewriter.check_quantized_model(restored), layers)
            with torch.inference_mode():
                actual = restored(torch.tensor([[1, 7, 9]], device="cuda:0")).logits.cpu()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            restored = None
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
