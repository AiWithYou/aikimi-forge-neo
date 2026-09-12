"""GPU非対応のPyTorchを、一般的なデバイス検出失敗と区別する。"""

import unittest
from unittest.mock import patch

from modules import launch_utils
from tools.tests import test_launch_utils_environment


class ReleaseGpuProbeTests(unittest.TestCase):
    def test_unsupported_kernel_reports_the_pytorch_remedy(self):
        with (
            patch.object(
                launch_utils,
                "args",
                test_launch_utils_environment.PrepareEnvironmentTests._args(skip_torch_cuda_test=False),
            ),
            patch.object(launch_utils, "is_installed", return_value=True),
            patch.object(
                launch_utils,
                "check_run_python",
                return_value=(False, "CUDA error: no kernel image is available for execution on the device"),
            ),
            patch.object(launch_utils, "git_tag", return_value="test"),
            patch.object(launch_utils.os, "remove", side_effect=OSError),
            patch.object(launch_utils.startup_timer, "record"),
        ):
            with self.assertRaisesRegex(SystemError, "manually install older version of PyTorch"):
                launch_utils.prepare_environment()
