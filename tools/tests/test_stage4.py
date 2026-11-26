"""Regression contracts for Studio output, generation controls and shared workspace state."""

import io
import multiprocessing
import sys
import tempfile
import time
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tools.tests.test_stage2_integration import ROOT, load


def concurrent_update(root, key, barrier):
    from modules_forge import aikimi_gallery, prompt_library

    for module, update in (
        (prompt_library, lambda: prompt_library.PromptLibrary(root).save({"id": key, "title": key})),
        (aikimi_gallery, lambda: aikimi_gallery.Gallery(root).favorite("outputs/" + key + ".wav", True)),
    ):
        original = module.write_json

        def slow_write(path, value, original=original):
            time.sleep(0.25)
            original(path, value)

        with patch.object(module, "write_json", slow_write):
            barrier.wait(timeout=30)
            update()


class WorkspaceRegressionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        (self.root / "outputs").mkdir()

    def test_initialized_app_contracts(self):
        import subprocess

        result = subprocess.run(  # noqa: S603 - isolated CPU Forge initialization
            [sys.executable, "-m", "tools.tests.stage4_app_probe"],
            cwd=ROOT,
            capture_output=True,
            encoding="utf8",
            errors="replace",
            timeout=240,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_two_processes_preserve_updates(self):
        from modules_forge.aikimi_gallery import Gallery
        from modules_forge.prompt_library import PromptLibrary

        library = PromptLibrary(self.root)
        library.entries()
        ctx = multiprocessing.get_context("spawn")
        barrier = ctx.Barrier(2)
        children = []
        for key in ("sky", "sea"):
            (self.root / f"outputs/{key}.wav").touch()
            child = ctx.Process(target=concurrent_update, args=(self.root, key, barrier))
            child.start()
            children.append(child)
        for child in children:
            child.join(45)
            if child.is_alive():
                child.terminate()
                child.join()
            self.assertEqual(child.exitcode, 0)
        with self.subTest(store="templates"):
            self.assertEqual({"sky", "sea"}, {v["id"] for v in library.entries()} - {"landscape"})
        with self.subTest(store="favorites"):
            items = Gallery(self.root).list_items(favorites=True)["items"]
            self.assertEqual({item["id"] for item in items}, {"outputs/sky.wav", "outputs/sea.wav"})

    def test_workspace_state_outside_backup_list(self):
        from modules import config_states
        from modules_forge.aikimi_gallery import Gallery
        from modules_forge.prompt_library import PromptLibrary

        gallery = Gallery(self.root)
        (self.root / "outputs/test.wav").touch()
        gallery.favorite("outputs/test.wav", True)
        library = PromptLibrary(self.root)
        library.entries()
        output = io.StringIO()
        with (
            patch.object(config_states, "config_states_dir", self.root / "config_states"),
            patch.object(config_states, "all_config_states", {}),
            redirect_stdout(output),
        ):
            self.assertFalse(config_states.list_config_states())
        self.assertEqual(output.getvalue(), "")
        for path in (gallery.favorites_path, library.path):
            self.assertEqual(path.parent, self.root / "config_states/aikimi_workspace")

    def test_h3_image_studio_filter(self):
        from modules_forge.aikimi_gallery import Gallery

        for name in ("images", "p_h3_images", "official_h3_images"):
            directory = self.root / "outputs/minimax_h3" / name
            directory.mkdir(parents=True)
            (directory / "frame.png").touch()
        gallery = Gallery(self.root)
        self.assertEqual(gallery.list_items(studio="minimax_h3_image")["total"], 3)
        self.assertEqual(gallery.list_items(studio="minimax_h3")["total"], 0)

    def test_studio_data_directory(self):
        from modules_forge import aikimi_gallery
        from modules_forge.flux3_klein import service as flux
        from modules_forge.irodori import service as irodori
        from modules_forge.moss_image import service as moss

        with patch("modules.paths_internal.data_path", str(self.root)):
            for module, owner in ((flux, "flux3"), (irodori, "irodori"), (moss, "moss_image")):
                with self.subTest(studio=owner):
                    path = module.Service().outputs
                    self.assertTrue(path.is_relative_to(self.root / "outputs"))
                    path.mkdir()
                    (path / "output.wav").touch()
                    self.assertEqual(aikimi_gallery.Gallery(self.root).list_items(studio=owner)["total"], 1)

    def test_relative_data_directory_keeps_absolute_worker_output_paths(self):
        import os

        from modules_forge.flux3_klein import service as flux
        from modules_forge.flux3_klein.history import History
        from modules_forge.irodori import service as irodori
        from modules_forge.moss_image import service as moss

        with (
            tempfile.TemporaryDirectory(dir=ROOT) as directory,
            patch("modules.paths_internal.data_path", os.path.relpath(directory)),
        ):
            for module in (flux, irodori, moss):
                path = module.Service().outputs
                self.assertTrue(path.is_relative_to(Path(directory) / "outputs"))
            self.assertEqual(History().root, flux.Service().outputs)

    def test_bounded_scan_and_pagination(self):
        from modules_forge import aikimi_gallery as gallery

        for i in range(100):
            (self.root / f"outputs/{i}.wav").touch()
        instance = gallery.Gallery(self.root)
        with (
            patch.object(gallery, "MAX_SCAN_ENTRIES", 24, create=True),
            patch.object(instance, "_metadata", return_value={}) as read,
        ):
            first = instance.list_items(limit=5)
            second = instance.list_items(offset=5, limit=5)
            self.assertTrue(first["truncated"])
            self.assertEqual(first["total"], 24)
            self.assertEqual(read.call_count, 10)
            self.assertEqual(len({v["id"] for v in first["items"] + second["items"]}), 10)
            read.reset_mock()
            instance.list_items(query="absent")
            self.assertLessEqual(read.call_count, 24)

    def test_gallery_searches_filename_and_cached_sidecars_together(self):
        from modules_forge import aikimi_gallery

        image = self.root / "outputs/sea.wav"
        image.touch()
        sidecar = image.with_suffix(".json")
        sidecar.write_text('{"prompt":"sky"}')
        gallery = aikimi_gallery.Gallery(self.root)
        with patch.object(aikimi_gallery, "read_json", wraps=aikimi_gallery.read_json) as read:
            for _ in range(2):
                self.assertEqual(gallery.list_items(query="sea.wav sky")["total"], 1)
            self.assertEqual(sum(call.args[0] == sidecar for call in read.call_args_list), 1)
            sidecar.write_text('{"prompt":"sunrise"}')
            self.assertEqual(gallery.list_items(query="sky")["total"], 0)
            self.assertEqual(gallery.list_items(query="sunrise")["total"], 1)

    def test_moss_slg_label(self):
        from tools.tests.test_moss_image_ui import UITests

        module = UITests().load_screen()
        blocks = module.on_ui_tabs()[0][0]
        self.addCleanup(blocks.close)
        skip = next(c for c in blocks.blocks.values() if c.elem_id == "moss-image-skip-layer")
        for cfg in (1, 3):
            self.assertTrue(module.cfg_controls(cfg)[1]["label"].startswith(skip.label))

    def test_single_cpu_updates(self):
        from modules import ui_extensions as ext

        for cores in (1, None):
            with (
                self.subTest(cores=cores),
                patch.object(ext.os, "cpu_count", return_value=cores),
                patch.object(ext.shared, "state", Mock()),
                patch.object(ext, "extension_table", return_value=""),
                patch.object(ext.extensions, "extensions", []),
            ):
                self.assertEqual(ext.check_updates("", "[]"), ("", ""))

    def test_sparse_ranges_validated_before_expansion(self):
        mod = load("extensions-builtin/sd_forge_sparse/scripts/sparse.py")
        parse = mod.SparseAttentionForForge._parse_block_list
        self.assertEqual(parse("0, 1-3"), {0, 1, 2, 3})
        for invalid in ("1-99999999", "0-256,0-256", "3-1", "1,x", "-1", "256,257", ",".join(["0"] * 300)):
            with (
                self.subTest(invalid=invalid),
                patch.object(mod, "range", side_effect=AssertionError("expanded"), create=True),
                self.assertRaises(ValueError),
            ):
                parse(invalid)

    def test_euler_four_model_evaluations(self):
        import torch

        from modules_forge.packages.k_diffusion.sampling import sample_euler

        with patch.dict(sys.modules, networks=SimpleNamespace(available_network_aliases={})):
            mod = load("extensions-builtin/sd_forge_lora_schedule/scripts/control.py")
        wrapper = mod.OnlineLoRAPatch("w", [0.25, None], name="a")
        unet = SimpleNamespace(weight_wrapper_patches={"w": [wrapper]})
        fake = SimpleNamespace(sd_model=SimpleNamespace(forge_objects=SimpleNamespace(unet=unet)))
        mod.LoRAControl.mapping["a"] = [(0, 0), (1, 1)]
        self.addCleanup(mod.LoRAControl.restore)
        denoiser = SimpleNamespace(step=0, total_steps=4)
        params = SimpleNamespace(sampling_step=0, total_sampling_steps=4, denoiser=denoiser)
        weights = []

        def evaluate(x, sigma):
            mod.LoRAControl.adjust_lora(params)
            weights.append(wrapper.patch[0][0])
            denoiser.step += 1
            return x

        with patch.object(mod, "shared", fake):
            sample_euler(
                evaluate,
                torch.ones(1, 1),
                torch.arange(4, -1, -1),
                disable=True,
                callback=lambda d: setattr(params, "sampling_step", d["i"]),
            )
        self.assertEqual(weights, [0, 1 / 3, 2 / 3, 1])

    def test_spectrum_uses_sampler_pass_steps(self):
        extension = ROOT / "extensions-builtin/sd_forge_spectrum"
        with (
            patch.object(sys, "path", [str(extension), *sys.path]),
            patch.dict(sys.modules, {"modules.infotext_utils": SimpleNamespace(PasteField=Mock())}),
        ):
            mod = load("extensions-builtin/sd_forge_spectrum/scripts/spectrum.py")
        p = Mock(steps=20, txt2img_upscale=False, extra_generation_params={})
        with patch.object(mod.SpectrumNode, "patch") as apply:
            for p.is_hr_pass, p.hr_second_pass_steps, expected in ((False, 8, 20), (True, 8, 8), (True, 0, 20)):
                with self.subTest(hires=p.is_hr_pass, steps=p.hr_second_pass_steps):
                    mod.SpectrumForForge().process_before_every_sampling(p, True, *range(7))
                    self.assertEqual(apply.call_args.args[1], expected)
