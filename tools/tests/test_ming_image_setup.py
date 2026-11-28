import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tools import setup_ming_image as setup


class SetupTests(unittest.TestCase):
    def test_verified_plan(self):
        files = setup.download_plan(profile="comfy_org_w4a8")
        self.assertEqual(len(files), 3)
        self.assertTrue(all(len(item["sha256"]) == 64 and item["size"] > 0 for item in files))

    def test_pending(self):
        with patch.object(
            setup.core, "manifest", return_value={"profiles": {"vram24g_hq": {"status": "pending_publish"}}}
        ):
            with self.assertRaisesRegex(ValueError, "sync-hf-manifest"):
                setup.download_plan(profile="vram24g_hq")

    def test_sync_requires_matching_hub_and_manifest(self):
        data = setup.manifest()
        names = {f["path"] for p in ("vram24g_hq", "vram16g_hq") for f in data["profiles"][p]["files"]}
        self.assertEqual(len(names), 5)
        rows = {name: dict(bytes=20, sha256="a" * 64) for name in sorted(names)}
        envelope = dict(
            source=data["repo_id"],
            revision=data["revision"],
            comfyui=data["runtime"]["comfy_revision"],
            comfy_kitchen="0.2.42",
        )
        siblings = [
            SimpleNamespace(rfilename=name, size=20, lfs=SimpleNamespace(size=20, sha256="a" * 64)) for name in rows
        ]
        revision = "b" * 40
        api = SimpleNamespace(model_info=lambda *a, **k: SimpleNamespace(sha=revision, siblings=siblings))
        with tempfile.TemporaryDirectory() as folder:
            source, local = Path(folder) / "source.json", Path(folder) / "local.json"
            local.write_text(json.dumps(data))

            def download(**kwargs):
                self.assertEqual(kwargs["revision"], revision)
                return source

            source.write_text(json.dumps({**envelope, "files": dict(list(rows.items())[:-1])}))
            with self.assertRaises(ValueError):
                setup.sync_hf_manifest(path=local, api=api, download=download)
            self.assertEqual(json.loads(local.read_text()), data)
            original = local.read_bytes()

            def failed_download(**kwargs):
                raise RuntimeError("offline Hub failure")

            for invalid in (
                None,
                "[",
                "[]",
                json.dumps({**envelope, "files": {**rows, "extra": {}}}),
                json.dumps({**envelope, "files": dict.fromkeys(names, {"bytes": 20, "sha256": "f" * 64})}),
            ):
                source.unlink(missing_ok=True)
                if invalid is not None:
                    source.write_text(invalid)
                with self.subTest(invalid=invalid), self.assertRaises(setup.MingImageSetupError):
                    setup.sync_hf_manifest(path=local, api=api, download=download)
                self.assertEqual(local.read_bytes(), original)
            with self.assertRaises(setup.MingImageSetupError):
                setup.sync_hf_manifest(path=local, api=api, download=failed_download)
            self.assertEqual(local.read_bytes(), original)
            publication = {**rows, "README.md": {}, "LICENSE": {}, "comparisons/eval_01.png": {}}
            source.write_text(json.dumps({**envelope, "files": publication}))
            setup.sync_hf_manifest(path=local, api=api, download=download)
            synced = json.loads(local.read_text())
            self.assertEqual(synced["profiles"]["vram24g_hq"]["revision"], revision)
            self.assertEqual(synced["profiles"]["vram16g_hq"]["status"], "published")

    def test_existing_checkout_rejects_untracked_code(self):
        with tempfile.TemporaryDirectory() as folder:
            checkout = Path(folder)
            (checkout / ".git").mkdir()
            (checkout / ".git/index").touch()
            (checkout / "extra.py").touch()

            def command(args, cwd):
                if "status" in args and "--untracked-files=all" in args:
                    return "?? extra.py\0"
                return setup.manifest()["runtime"]["comfy_revision"] if "rev-parse" in args else ""

            with self.assertRaisesRegex(setup.MingImageSetupError, "変更|未追跡"):
                setup.prepare_comfy_checkout(checkout, command, "git")

    def test_checkout_junctions_are_rejected_before_git(self):
        import os
        from unittest.mock import Mock

        for name in ("ComfyUI-0.2", "ComfyUI-0.2/.git", "ComfyUI-0.2/comfy"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                base = Path(folder)
                checkout = base / "ComfyUI-0.2"
                link = base / name
                link.parent.mkdir(parents=True, exist_ok=True)
                if link.name == "comfy":
                    (checkout / ".git").mkdir()
                outside = base / "outside"
                outside.mkdir()
                if os.name == "nt":
                    import _winapi

                    _winapi.CreateJunction(str(outside), str(link))
                else:
                    link.symlink_to(outside, target_is_directory=True)
                command = Mock()
                with self.assertRaisesRegex(setup.MingImageSetupError, "リンク|ジャンクション"):
                    setup.prepare_comfy_checkout(checkout, command, "git")
                command.assert_not_called()
                with self.assertRaises(setup.core.MingImageError):
                    setup.core.check_checkout_paths(checkout)

    def test_runtime_preflight_rejects_extra_code_including_ignored_files(self):
        from unittest.mock import Mock

        core = setup.core
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / "models/Ming-Image"
            paths = core.runtime_paths(root)
            checkout = paths["comfy"]
            (checkout / ".git").mkdir(parents=True)
            (checkout / "comfy").mkdir()
            (checkout / "comfy/sd.py").touch()
            paths["python"].parent.mkdir(parents=True)
            paths["python"].touch()
            revision = core.manifest()["runtime"]["comfy_revision"]
            core.atomic_json(
                paths["base"] / ".runtime.json",
                {
                    "schema_version": 1,
                    "fingerprint": core.runtime_fingerprint(),
                    "comfy_revision": revision,
                },
            )
            for entry, allowed in (
                ("", True),
                ("!! comfy/__pycache__/sd.cpython-313.pyc", True),
                ("?? extra.py", False),
                ("!! ignored/helper.py", False),
            ):

                def git_or_python(args, output_entry=entry, **kwargs):
                    if "status" in args:
                        self.assertIn("--ignored", args)
                        self.assertIn("--untracked-files=all", args)
                        output = output_entry + "\0"
                    else:
                        output = revision if "rev-parse" in args else "(3, 13) 0.2.42"
                    return Mock(returncode=0, stdout=output)

                with self.subTest(entry=entry), patch.object(core.subprocess, "run", side_effect=git_or_python):
                    self.assertEqual(core.runtime_ready(root), allowed)
