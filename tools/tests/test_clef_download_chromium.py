"""Result downloads prepare a fresh partial snapshot before the first browser click."""

import json
import os
import subprocess
import time
from urllib.request import Request, urlopen

import gradio as gr
import pytest
from websockets.sync.client import connect

from modules_forge.clef.core import atomic_json
from tools.tests.chromium_helpers import close_chromium, find_chromium, reserve_local_port


def test_json_and_csv_download_the_latest_partial_result_on_each_click(tmp_path, monkeypatch):
    import modules_forge.clef.ui as ui
    import modules_forge.clef.workspace_ui as workspace

    chromium = find_chromium()
    if not chromium:
        pytest.skip("Chromium is not installed")
    outputs = tmp_path / "outputs"
    identifier = "20261011T120000-12345678"
    directory = outputs / identifier
    result = {
        "id": identifier,
        "status": "running",
        "request": {"questions": {"q": {"type": "noul", "instructions": "条件"}}},
        "items": [{"id": str(index), "name": f"{index}.txt", "status": "pending"} for index in range(2)],
    }
    atomic_json(directory / "result.json", result)

    def append(index, probability):
        item = {**result["items"][index], "status": "done", "answers": {"q": {"type": "noul", "noul": probability}}}
        with (directory / "results.jsonl").open("a", encoding="utf-8", newline="\n") as journal:
            journal.write(json.dumps({"item": item, "seconds": index + 1}) + "\n")

    append(0, 0.8)
    monkeypatch.setattr(ui, "OUTPUTS", outputs)
    monkeypatch.setattr(workspace, "history_for", lambda: [])
    monkeypatch.setattr(workspace, "environment_status", lambda: "test")
    with gr.Blocks() as app:
        workspace.build(outputs=outputs)
        export = next(
            function for function in app.fns.values() if function.fn and function.fn.__name__ == "export_json"
        )
        buttons = {
            component.label: component
            for component in app.blocks.values()
            if isinstance(component, gr.DownloadButton)
            and component.label in {"表示中の結果 JSONを保存", "表示中の一覧 CSVを保存"}
        }
        app.load(
            lambda: (result, identifier + " · test", gr.update(interactive=True), gr.update(interactive=True)),
            outputs=[*export.inputs, *buttons.values()],
            api_visibility="private",
        )
    port = reserve_local_port()
    app.launch(
        server_name="127.0.0.1", server_port=port, prevent_thread_lock=True, quiet=True, allowed_paths=[str(outputs)]
    )
    profile = tmp_path / "chrome-profile"
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    debugging_port = reserve_local_port()
    process = subprocess.Popen(  # noqa: S603
        [
            chromium,
            "--headless=new",
            "--disable-gpu",
            "--disable-extensions",
            "--no-first-run",
            "--no-default-browser-check",
            f"--user-data-dir={profile}",
            f"--remote-debugging-port={debugging_port}",
            "--remote-allow-origins=*",
            "about:blank",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        start_new_session=os.name != "nt",
    )
    websocket = None
    send = None
    try:
        deadline = time.monotonic() + 20
        while True:
            try:
                request = Request(f"http://127.0.0.1:{debugging_port}/json/new?about:blank", method="PUT")  # noqa: S310
                with urlopen(request, timeout=1) as response:  # noqa: S310
                    target = json.load(response)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.05)
        websocket = connect(target["webSocketDebuggerUrl"], open_timeout=5, close_timeout=2, max_size=None)
        command_id = 0

        def send(method, params=None):
            nonlocal command_id
            command_id += 1
            websocket.send(json.dumps({"id": command_id, "method": method, "params": params or {}}))
            while True:
                response = json.loads(websocket.recv(timeout=10))
                if response.get("id") == command_id:
                    assert "error" not in response, response
                    return response.get("result", {})

        def evaluate(expression):
            response = send("Runtime.evaluate", {"expression": expression, "returnByValue": True})
            assert "exceptionDetails" not in response, response
            return response.get("result", {}).get("value")

        send("Browser.setDownloadBehavior", {"behavior": "allow", "downloadPath": str(downloads)})
        send("Page.navigate", {"url": f"http://127.0.0.1:{port}"})
        deadline = time.monotonic() + 30
        while not evaluate(
            "Array.from(document.querySelectorAll('button')).some(button => "
            "button.textContent.trim() === '表示中の結果 JSONを保存' && !button.disabled)"
        ):
            assert time.monotonic() < deadline, "Download buttons did not become ready"
            time.sleep(0.05)

        def download(label, suffix):
            before = {path: path.read_bytes() for path in downloads.glob("*" + suffix)}
            evaluate(
                "Array.from(document.querySelectorAll('button')).find(button => "
                f"button.textContent.trim() === {json.dumps(label)}).click()"
            )
            deadline = time.monotonic() + 15
            while True:
                saved = [path for path in downloads.glob("*" + suffix) if before.get(path) != path.read_bytes()]
                if saved:
                    return next(iter(saved))
                assert time.monotonic() < deadline, f"The first click did not download {label}"
                time.sleep(0.05)

        first = json.loads(download("表示中の結果 JSONを保存", ".json").read_text(encoding="utf-8"))
        assert [item["status"] for item in first["items"]] == ["done", "pending"]
        append(1, 0.3)
        second = json.loads(download("表示中の結果 JSONを保存", ".json").read_text(encoding="utf-8"))
        assert [item["status"] for item in second["items"]] == ["done", "done"]
        csv = download("表示中の一覧 CSVを保存", ".csv").read_text(encoding="utf-8-sig")
        assert "0.8" in csv and "0.3" in csv
        append(0, 0.6)
        second_csv = download("表示中の一覧 CSVを保存", ".csv").read_text(encoding="utf-8-sig")
        assert "0.6" in second_csv and "0.8" not in second_csv
        snapshots = list((directory / "downloads").glob("*/result.json"))
        assert len(snapshots) == 2 and snapshots[0].parent != snapshots[1].parent
    finally:
        close_chromium(
            process,
            websocket=websocket,
            close_browser=(lambda: send("Browser.close")) if send else None,
            owned_port=debugging_port,
        )
        app.close()
