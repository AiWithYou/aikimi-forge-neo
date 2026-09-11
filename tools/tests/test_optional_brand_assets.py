"""ブランド画像を含めない配布での起動処理を確認する。"""

import ast
import os
import tempfile
import unittest
from pathlib import Path

import fastapi
import fastapi.staticfiles
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]


class OptionalBrandAssetsTests(unittest.TestCase):
    def test_asset_mount_with_and_without_brand_directory(self):
        tree = ast.parse((ROOT / "modules/ui.py").read_text(encoding="utf-8"))
        guard = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.If) and ast.unparse(node.test) == "os.path.isdir(aikimi_assets)"
        )
        code = compile(ast.Module(body=[guard], type_ignores=[]), "asset_mount", "exec")
        with tempfile.TemporaryDirectory() as directory:
            assets = Path(directory) / "aikimi"
            for present in (False, True):
                with self.subTest(present=present):
                    if present:
                        assets.mkdir()
                        (assets / "manifest.json").write_text("{}", encoding="utf-8")
                    app = fastapi.FastAPI()
                    exec(code, {"app": app, "fastapi": fastapi, "os": os, "aikimi_assets": str(assets)})  # noqa: S102 - 検査対象のローカルコードのみ実行
                    with TestClient(app) as client:
                        response = client.get("/aikimi-assets/manifest.json")
                        self.assertEqual(response.status_code, 200 if present else 404)

    def test_launch_omits_missing_favicon(self):
        tree = ast.parse((ROOT / "webui.py").read_text(encoding="utf-8"))
        value = next(
            node.value for node in ast.walk(tree) if isinstance(node, ast.keyword) and node.arg == "favicon_path"
        )
        code = compile(ast.Expression(body=value), "favicon_path", "eval")
        with tempfile.TemporaryDirectory() as directory:
            favicon = Path(directory) / "favicon.png"
            self.assertIsNone(eval(code, {"os": os, "aikimi_favicon": str(favicon)}))  # noqa: S307 - 検査対象のローカル式のみ評価
            favicon.write_bytes(b"fixture")
            self.assertEqual(eval(code, {"os": os, "aikimi_favicon": str(favicon)}), str(favicon))  # noqa: S307 - 同上
