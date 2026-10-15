"""Own test Chrome descendants before they can start, without app dependencies."""

from __future__ import annotations

import ctypes as c
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from contextlib import ExitStack
from pathlib import Path


class _BasicLimit(c.Structure):
    _fields_ = [
        ("process_time", c.c_int64),
        ("job_time", c.c_int64),
        ("flags", c.c_uint32),
        ("min_working", c.c_size_t),
        ("max_working", c.c_size_t),
        ("active", c.c_uint32),
        ("affinity", c.c_size_t),
        ("priority", c.c_uint32),
        ("scheduling", c.c_uint32),
    ]


class _ExtendedLimit(c.Structure):
    _fields_ = [
        ("basic", _BasicLimit),
        ("io", c.c_uint64 * 6),
        ("process_memory", c.c_size_t),
        ("job_memory", c.c_size_t),
        ("peak_process", c.c_size_t),
        ("peak_job", c.c_size_t),
    ]


class _Accounting(c.Structure):
    _fields_ = [
        ("user_time", c.c_int64),
        ("kernel_time", c.c_int64),
        ("period_user_time", c.c_int64),
        ("period_kernel_time", c.c_int64),
        ("page_faults", c.c_uint32),
        ("total", c.c_uint32),
        ("active", c.c_uint32),
        ("terminated", c.c_uint32),
    ]


def _windows():
    return os.name == "nt"


def _kernel32():
    from ctypes import wintypes as w

    kernel = c.WinDLL("kernel32", use_last_error=True)
    signatures = {
        "CreateJobObjectW": ([c.c_void_p, w.LPCWSTR], w.HANDLE),
        "OpenJobObjectW": ([w.DWORD, w.BOOL, w.LPCWSTR], w.HANDLE),
        "SetInformationJobObject": ([w.HANDLE, c.c_int, c.c_void_p, w.DWORD], w.BOOL),
        "QueryInformationJobObject": ([w.HANDLE, c.c_int, c.c_void_p, w.DWORD, c.c_void_p], w.BOOL),
        "TerminateJobObject": ([w.HANDLE, w.UINT], w.BOOL),
        "AssignProcessToJobObject": ([w.HANDLE, w.HANDLE], w.BOOL),
        "GetCurrentProcess": ([], w.HANDLE),
        "OpenProcess": ([w.DWORD, w.BOOL, w.DWORD], w.HANDLE),
        "IsProcessInJob": ([w.HANDLE, w.HANDLE, c.POINTER(w.BOOL)], w.BOOL),
        "WaitForSingleObject": ([w.HANDLE, w.DWORD], w.DWORD),
        "CloseHandle": ([w.HANDLE], w.BOOL),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(kernel, name)
        function.argtypes, function.restype = arguments, result
    return kernel


def _native_error(operation):
    return OSError(c.get_last_error(), f"Windows Chromium Job {operation} failed")


class _WindowsJob:
    def __init__(self):
        self.kernel = _kernel32()
        self.name = "Local\\ForgeChromium-" + uuid.uuid4().hex
        self.handle = self.kernel.CreateJobObjectW(None, self.name)
        if not self.handle:
            raise _native_error("creation")
        limits = _ExtendedLimit()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE; no breakaway.
        if not self.kernel.SetInformationJobObject(self.handle, 9, c.byref(limits), c.sizeof(limits)):
            self.kernel.CloseHandle(self.handle)
            self.handle = None
            raise _native_error("protection")

    def _freeze_admission(self):
        limits = _ExtendedLimit()
        # Existing members survive; new child creation cannot run user code.
        limits.basic.flags = 0x2000 | 0x8  # KILL_ON_JOB_CLOSE | ACTIVE_PROCESS
        limits.basic.active = 1
        if not self.kernel.SetInformationJobObject(self.handle, 9, c.byref(limits), c.sizeof(limits)):
            raise _native_error("admission barrier")

    def _retain_processes(self, processes, deadline):
        capacity = 32
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError("Windows Chromium Job snapshot deadline elapsed")
            data = c.create_string_buffer(8 + capacity * c.sizeof(c.c_size_t))
            success = self.kernel.QueryInformationJobObject(self.handle, 3, data, c.sizeof(data), None)
            assigned, listed = c.cast(data, c.POINTER(c.c_uint32))[:2]
            if not success and c.get_last_error() != 234:  # ERROR_MORE_DATA
                raise _native_error("process snapshot")
            if not success or assigned > capacity or listed < assigned:
                capacity = max(capacity * 2, assigned)
                continue
            pids = c.cast(c.addressof(data) + 8, c.POINTER(c.c_size_t))[:listed]
            break
        for pid in pids:
            if time.monotonic() >= deadline:
                raise TimeoutError("Windows Chromium Job handle deadline elapsed")
            # Hold the SAME noninherited handle through membership check and wait.
            process = self.kernel.OpenProcess(0x101000, False, pid)  # SYNCHRONIZE | QUERY_LIMITED_INFORMATION
            if not process:
                raise _native_error("process open")
            try:
                member = c.c_int()
                if not self.kernel.IsProcessInJob(process, self.handle, c.byref(member)):
                    raise _native_error("membership query")
                if not member.value:
                    raise RuntimeError("Windows Chromium Job process membership changed")
            except BaseException:
                self.kernel.CloseHandle(process)
                raise
            processes.append(process)

    def close(self, timeout=5):
        if self.handle is None:
            return
        handle, processes = self.handle, []
        deadline = time.monotonic() + timeout
        try:
            try:
                self._freeze_admission()
                self._retain_processes(processes, deadline)
            finally:
                # Even a failed snapshot must terminate only this owned Job.
                if not self.kernel.TerminateJobObject(handle, 1):
                    raise _native_error("termination")
                for process in processes:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Windows Chromium Job process rundown deadline elapsed")
                    result = self.kernel.WaitForSingleObject(process, max(1, int(remaining * 1000)))
                    if result == 258 or time.monotonic() >= deadline:  # WAIT_TIMEOUT
                        raise TimeoutError("Windows Chromium Job process rundown deadline elapsed")
                    if result != 0:  # WAIT_OBJECT_0
                        raise _native_error("process rundown wait")
                failed_close = False
                while processes:
                    if not self.kernel.CloseHandle(processes.pop()):
                        failed_close = True
                if failed_close:
                    raise _native_error("process handle close")
                while True:
                    accounting = _Accounting()
                    if not self.kernel.QueryInformationJobObject(
                        handle, 1, c.byref(accounting), c.sizeof(accounting), None
                    ):
                        raise _native_error("process accounting")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Windows Chromium Job accounting deadline elapsed")
                    if accounting.active == 0:
                        break
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Windows Chromium Job still has active processes")
                    time.sleep(min(0.02, max(0, deadline - time.monotonic())))
        finally:
            self.handle = None
            failed_close = False
            for owned in [*processes, handle]:
                if not self.kernel.CloseHandle(owned):
                    failed_close = True
            if failed_close:
                raise _native_error("handle close")


def _join_windows_job(name):
    if re.fullmatch(r"Local\\ForgeChromium-[0-9a-f]{32}", name) is None:
        raise ValueError("Invalid Chromium Job name")
    kernel = _kernel32()
    handle = kernel.OpenJobObjectW(0x0001, False, name)  # JOB_OBJECT_ASSIGN_PROCESS
    if not handle:
        raise _native_error("open")
    try:
        if not kernel.AssignProcessToJobObject(handle, kernel.GetCurrentProcess()):
            raise _native_error("assignment")
    finally:
        if not kernel.CloseHandle(handle):
            raise _native_error("gate handle close")


def _gate(job_name, status, command):
    _join_windows_job(job_name)
    child = subprocess.Popen(  # noqa: S603 - trusted executable/arguments from owning test process
        command,
        stdin=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    temporary = status.with_suffix(".tmp")
    temporary.write_text(json.dumps({"pid": child.pid}), encoding="ascii")
    temporary.replace(status)
    return child.wait()


def _read_status(path):
    with path.open("rb") as source:
        return source.read(129)


def _read_pid(gate, status, deadline):
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Chromium gate handshake deadline elapsed")
        try:
            raw = _read_status(status)
        except (FileNotFoundError, PermissionError):
            if gate.poll() is not None:
                raise RuntimeError(f"Chromium gate exited before browser startup: {gate.returncode}") from None
            time.sleep(min(0.02, remaining))
            continue
        if time.monotonic() >= deadline:
            raise TimeoutError("Chromium gate handshake arrived after deadline")
        if len(raw) > 128:
            raise ValueError("Oversized Chromium gate handshake")
        result = json.loads(raw)
        if (
            not isinstance(result, dict)
            or set(result) != {"pid"}
            or type(result["pid"]) is not int
            or not 0 < result["pid"] < 2**32
        ):
            raise ValueError("Invalid Chromium gate handshake")
        return result["pid"]


def _stop_gate(gate, timeout=5):
    if gate.poll() is None:
        gate.kill()
    gate.wait(timeout=timeout)


class _WindowsProcess:
    def __init__(self, gate, job, pid, command):
        self._gate, self._job = gate, job
        self.pid, self.args = pid, command
        self._closed = False

    @property
    def returncode(self):
        return self._gate.returncode

    def poll(self):
        return self._gate.poll()

    def wait(self, timeout=None):
        return self._gate.wait(timeout=timeout)

    def close_tree(self, timeout=5):
        if self._closed:
            return
        deadline = time.monotonic() + timeout
        try:
            self._job.close(timeout=timeout)
        finally:
            _stop_gate(self._gate, timeout=max(0, deadline - time.monotonic()))
        self._closed = True

    def kill(self):
        self.close_tree()

    terminate = kill


def spawn_chromium(command, *, deadline=None, **popen_kwargs):
    """Windows joins a Job before browser creation; POSIX keeps its process group."""
    if not _windows():
        return subprocess.Popen(command, **popen_kwargs)  # noqa: S603 - trusted local browser command
    deadline = deadline if deadline is not None else time.monotonic() + 10
    with ExitStack() as failed_start:
        directory = tempfile.TemporaryDirectory(prefix="forge-chrome-gate-")
        failed_start.callback(directory.cleanup)
        job = _WindowsJob()
        failed_start.callback(job.close)
        status = Path(directory.name) / "pid.json"
        # Bypass a venv's launcher: until assignment, this direct Python has no children.
        python = sys._base_executable
        gate_command = [
            python,
            "-I",
            "-S",
            str(Path(__file__).resolve()),
            "--join-job",
            job.name,
            str(status),
            "--",
            *command,
        ]
        options = {"stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL, **popen_kwargs}
        options["creationflags"] = options.get("creationflags", 0) | getattr(subprocess, "CREATE_NO_WINDOW", 0)
        gate = subprocess.Popen(gate_command, **options)  # noqa: S603 - fixed stdlib gate plus trusted browser args
        failed_start.callback(_stop_gate, gate)
        pid = _read_pid(gate, status, deadline)
        directory.cleanup()
        failed_start.pop_all()
        return _WindowsProcess(gate, job, pid, command)


if __name__ == "__main__":
    if len(sys.argv) < 6 or sys.argv[1] != "--join-job" or sys.argv[4] != "--":
        raise SystemExit(2)
    raise SystemExit(_gate(sys.argv[2], Path(sys.argv[3]), sys.argv[5:]))
