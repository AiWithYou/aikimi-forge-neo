"""RAM profile flags stay bounded and cannot silently select another profile."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from modules_forge import minimax_h3_bridge as bridge


class H3RamProfileTests(unittest.TestCase):
    def test_ram_profile_requires_encoder_release(self):
        request = bridge.H3Request(mode="text", prompt="test")
        with self.assertRaisesRegex(bridge.H3BridgeError, "CLIP"):
            bridge._validate_request_runtime_constraints(
                request, SimpleNamespace(acceleration=request.acceleration), bridge.RUNTIME_PROFILE_RAM
            )

    def test_sparse_workflow_can_reuse_cached_conditioning(self):
        from modules_forge.jev_sparse import h3_integration
        from modules_forge.minimax_h3_acceleration import H3Acceleration

        h3_integration.install()
        try:
            options = H3Acceleration(model_variant="fused_turbo", attention="h3_fixed5", clip_cache="auto")
            request = bridge.H3Request(mode="text", prompt="a red paper boat", steps=4, acceleration=options)
            graph = bridge.build_workflow(request, {}, seed=42)
            self.assertEqual(graph["5"]["class_type"], "MiniMaxH3CLIPCachedFL2VA")
            self.assertNotIn("2", graph)
            guider = next(node for node in graph.values() if node["class_type"] == "BasicGuider")
            self.assertEqual(guider["inputs"]["model"], ["aikimi_h3_sparse", 0])
        finally:
            h3_integration.uninstall()

    def test_profile_roundtrip_and_other_flags_remain_rejected(self):
        command = bridge._runtime_command(Path("python"), 8192, bridge.RUNTIME_PROFILE_RAM)
        self.assertEqual(bridge.runtime_profile_from_args(command[1:], 8192), bridge.RUNTIME_PROFILE_RAM)
        self.assertIsNone(bridge.runtime_profile_from_args(command[1:] + ["--cpu"], 8192))
        fast = bridge._runtime_command(Path("python"), 8192, bridge.RUNTIME_PROFILE_FAST)
        self.assertIsNone(bridge.runtime_profile_from_args(fast[1:] + ["--disable-dynamic-vram"], 8192))
        self.assertIsNone(bridge.runtime_profile_from_args(fast[1:] + ["--cache-classic"], 8192))

    def test_low_memory_is_rejected_before_launch(self):
        with patch("psutil.virtual_memory", return_value=SimpleNamespace(total=32 * 1024**3)):
            with self.assertRaisesRegex(bridge.H3BridgeError, "64GB"):
                bridge.start_runtime(
                    Path("unused"), "http://127.0.0.1:8192", Path("unused"), runtime_profile=bridge.RUNTIME_PROFILE_RAM
                )
