"""Real local IPC with a deterministic substitute for the paid SDK worker."""

import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from modules_forge.jev_sparse import common


@pytest.fixture
def mock_sdk(monkeypatch):
    original = subprocess.Popen
    children, launches = [], []

    def install(choice=50, confidence=0.9, *, blocked=False):
        selected = "request['allowed'][key][0]" if choice is None else json.dumps(choice)
        script = (
            "import json,sys,time\n"
            "for line in sys.stdin:\n"
            " request=json.loads(line)\n"
            + (" time.sleep(30)\n" if blocked else "")
            + f" answer={{'decisions':{{key:{{'choice':{selected},'confidence':{confidence!r}}} for key in request['allowed']}}}}\n"
            " print(json.dumps(answer),flush=True)\n"
        )

        def start(argv, **kwargs):
            launches.append((argv, kwargs))
            child = original([sys.executable, "-I", "-u", "-c", script], **kwargs)  # noqa: S603
            children.append(child)
            return child

        monkeypatch.setattr(common.subprocess, "Popen", start)
        return SimpleNamespace(children=children, launches=launches)

    yield install
    for child in children:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=2)
