import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from modules_forge import minimax_h3_handoff as handoff
from modules_forge import minimax_h3_handoff_store as store


class RuntimePackTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "main.py").touch()
        (self.root / "models").mkdir()
        (self.root / "input").mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def test_pack_installs_only_audited_files_and_is_idempotent(self):
        target = handoff.install_bundle(self.root)
        self.assertEqual(
            {p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()}, set(handoff.PACK_HASHES)
        )
        self.assertEqual(handoff.install_bundle(self.root), target)

    def test_modified_installation_and_unknown_files_not_overwritten(self):
        target = handoff.install_bundle(self.root)
        (target / "web/handoff.js").write_text("user changes")
        with self.assertRaises(store.HandoffError):
            handoff.install_bundle(self.root)
        self.assertEqual((target / "web/handoff.js").read_text(), "user changes")

    def test_unknown_installation_file_rejected(self):
        target = handoff.install_bundle(self.root)
        (target / "user.py").touch()
        with self.assertRaises(store.HandoffError):
            handoff.install_bundle(self.root)

    def test_linked_custom_node_directory_rejected(self):
        outside = self.root / "outside"
        outside.mkdir()
        (self.root / "custom_nodes").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(store.HandoffError):
            handoff.install_bundle(self.root)

    def test_pack_has_no_model_nodes_and_restricts_origin(self):
        from aiohttp import web

        target = handoff.install_bundle(self.root)
        routes = web.RouteTableDef()
        server = types.ModuleType("server")
        server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(routes=routes))
        folder = types.ModuleType("folder_paths")
        folder.get_input_directory = lambda: str(self.root / "input")
        spec = importlib.util.spec_from_file_location(
            "h3_test_pack", target / "__init__.py", submodule_search_locations=[str(target)]
        )
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, {"server": server, "folder_paths": folder, "h3_test_pack": module}):
            spec.loader.exec_module(module)
        self.assertEqual(module.NODE_CLASS_MAPPINGS, {})
        self.assertEqual(len(list(routes)), 1)

        def req(remote="127.0.0.1", host="127.0.0.1:8188", headers=None):
            return types.SimpleNamespace(remote=remote, host=host, scheme="http", headers=headers or {})

        self.assertTrue(module._local_request(req()))
        self.assertFalse(module._local_request(req(remote="203.0.113.1")))
        self.assertFalse(module._local_request(req(host="attacker.invalid:8188")))
        self.assertFalse(module._local_request(req(headers={"Origin": "https://attacker.invalid"})))
        self.assertFalse(module._local_request(req(headers={"Sec-Fetch-Site": "cross-site"})))


if __name__ == "__main__":
    unittest.main()


class RouteHandlerTests(unittest.IsolatedAsyncioTestCase):
    async def test_readonly_handler_success_missing_tamper_and_cross_origin(self):
        import json

        from aiohttp import web

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "main.py").touch()
            (root / "models").mkdir()
            (root / "input/forge_h3").mkdir(parents=True)
            (root / "input/forge_h3/a.png").write_bytes(b"asset")
            graph = {"1": {"class_type": "LoadImage", "inputs": {"image": "forge_h3/a.png"}}}
            record = store.create_snapshot(root / "input", graph, {"images": ["forge_h3/a.png"]}, {"seed": 1})
            (root / "input/forge_h3/a.png").unlink()
            target = handoff.install_bundle(root)
            routes = web.RouteTableDef()
            server = types.ModuleType("server")
            server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(routes=routes))
            folder = types.ModuleType("folder_paths")
            folder.get_input_directory = lambda: str(root / "input")
            spec = importlib.util.spec_from_file_location(
                "h3_handler_pack", target / "__init__.py", submodule_search_locations=[str(target)]
            )
            module = importlib.util.module_from_spec(spec)
            with mock.patch.dict(sys.modules, {"server": server, "folder_paths": folder, "h3_handler_pack": module}):
                spec.loader.exec_module(module)

                def req(token=record["token"], headers=None):
                    return types.SimpleNamespace(
                        remote="127.0.0.1",
                        host="127.0.0.1:8188",
                        scheme="http",
                        headers=headers or {},
                        match_info={"token": token},
                    )

                result = await module.get_h3_workflow(req())
                self.assertEqual(result.status, 200)
                self.assertEqual(json.loads(result.body)["metadata"]["seed"], 1)
                self.assertEqual(result.headers["Cache-Control"], "no-store")
                self.assertEqual(
                    (await module.get_h3_workflow(req(headers={"Origin": "https://evil.invalid"}))).status, 403
                )
                self.assertEqual((await module.get_h3_workflow(req("f" * 32))).status, 404)
                self.assertEqual((await module.get_h3_workflow(req("../private"))).status, 404)
                (root / "input" / record["assets"][0]["file"]).write_bytes(b"xxxxx")
                result = await module.get_h3_workflow(req())
                self.assertEqual(result.status, 404)
                self.assertIn("破損", json.loads(result.body)["error"])
