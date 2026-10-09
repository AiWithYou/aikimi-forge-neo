"""Cancelable, resumable setup owned by the server rather than the browser."""

import atexit
import os
import subprocess
import sys
import threading
import time
import uuid

from .core import ROOT, IrisError


class Preparation:
    def __init__(self):
        self.jobs = {}
        self.guard = threading.Lock()
        atexit.register(self.shutdown)

    def start(self, task, precision):
        with self.guard:
            if any(job["status"] == "running" for job in self.jobs.values()):
                raise IrisError("Irisのモデルを準備中です。終了後に再実行してください。")
            identifier = uuid.uuid4().hex
            directory = ROOT / "cache/iris-setup" / identifier
            directory.mkdir(parents=True)
            job = {
                "id": identifier,
                "status": "running",
                "directory": str(directory),
                "cancel": False,
                "process": None,
                "tree": None,
            }
            self.jobs[identifier] = job
            threading.Thread(target=self.execute, args=(job, task, precision), daemon=True).start()
            return identifier

    def execute(self, job, task, precision):
        from pathlib import Path

        from modules_forge.yue2_studio.service import ProcessTree

        status = "error"
        try:
            with (Path(job["directory"]) / "setup.log").open("w", encoding="utf-8") as log:
                process = subprocess.Popen(  # noqa: S603 -- Fixed setup CLI with enumerated UI choices, shell=False.
                    [
                        sys.executable,
                        "-u",
                        str(ROOT / "tools/setup_iris.py"),
                        "--task",
                        task,
                        "--precision",
                        precision,
                        "--parent-guard",
                    ],
                    cwd=ROOT,
                    stdin=subprocess.PIPE,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8"),
                    start_new_session=os.name != "nt",
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                job["process"] = process
                tree = ProcessTree(process)
                job["tree"] = tree
                process.stdin.write(tree.handshake())
                process.stdin.flush()
                while process.poll() is None:
                    if job["cancel"]:
                        tree.terminate()
                    time.sleep(0.1)
                status = "cancelled" if job["cancel"] else "complete" if process.returncode == 0 else "error"
        except Exception as exc:
            job["error"] = str(exc)
        finally:
            process, tree = job["process"], job["tree"]
            while process is not None and process.poll() is None:
                try:
                    tree.terminate() if tree else process.kill()
                except OSError:
                    pass
                time.sleep(0.1)
            if tree:
                tree.close()
            if process and process.stdin:
                process.stdin.close()
            job["status"] = status

    def poll(self, identifier):
        from pathlib import Path

        job = self.jobs.get(identifier, {"status": "idle"})
        message = job.get("error", "モデルを準備中…")
        if job.get("directory"):
            path = Path(job["directory"]) / "setup.log"
            if path.is_file():
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
                if lines:
                    message = lines[-1][-400:]
        return {"status": job["status"], "message": message}

    def cancel(self, identifier):
        job = self.jobs.get(identifier)
        if job:
            job["cancel"] = True

    def shutdown(self):
        for job in self.jobs.values():
            if job["status"] == "running":
                job["cancel"] = True


PREPARATION = Preparation()
