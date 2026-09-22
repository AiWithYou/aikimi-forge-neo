"""One bounded Jev session per outer image job, not per nested upscale tile."""

from __future__ import annotations

import contextvars
import functools
import hashlib
import json
import math
import struct
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

from .common import (
    REPLAY_ENV,
    ExperimentCancelled,
    JevBudget,
    ReplayError,
    RunLog,
    client_available,
    close_client,
    create_client,
    replay_paths_from_environment,
)
from .credentials import cloud_source
from .krea2 import Options

SCRIPT_TITLE = "Krea2 Jev / Sparse"
_JOB = contextvars.ContextVar("aikimi_krea2_job", default=None)


def parse_options(
    mode="off",
    keep=10,
    minimum=4096,
    tile_mode="off",
    timeout=10,
    decision_cadence="once",
    interval=2,
    tile_cadence="once",
    job_max_calls=0,
    job_max_wait_seconds=0,
    replay_logs="",
):
    if isinstance(keep, bool) or isinstance(timeout, bool):
        raise ValueError("保持率と待ち時間は数値で指定してください。")
    if isinstance(minimum, bool) or int(minimum) != minimum:
        raise ValueError("最小token数は整数で指定してください。")
    if isinstance(interval, bool) or int(interval) != interval:
        raise ValueError("再判定の間隔は整数で指定してください。")
    if isinstance(job_max_calls, bool) or int(job_max_calls) != job_max_calls:
        raise ValueError("Jevの生成全体上限は整数で指定してください。")
    options = Options(
        mode=mode,
        keep_percent=float(keep),
        min_tokens=int(minimum),
        tile_mode=tile_mode,
        timeout=float(timeout),
        decision_cadence=decision_cadence,
        update_interval=int(interval),
        tile_cadence=tile_cadence,
        job_max_calls=int(job_max_calls),
        job_max_wait_seconds=job_max_wait_seconds,
        replay_logs=replay_logs,
    )
    options.validate()
    return options


def processing_options(p):
    for script in getattr(getattr(p, "scripts", None), "alwayson_scripts", None) or ():
        if script.title() == SCRIPT_TITLE:
            args = (getattr(p, "script_args", None) or ())[script.args_from : script.args_to]
            return parse_options(*args) if args else Options()
    return Options()


def cancelled():
    from modules import shared

    return bool(shared.state.interrupted or shared.state.skipped)


@functools.lru_cache(maxsize=32)
def _checkpoint_is_krea(filename, modified_ns, size):
    # Inspect the requested checkpoint, not the previously loaded model. The
    # selectable upscale script runs before process_images reloads its model.
    try:
        with Path(filename).open("rb") as stream:
            header_length = struct.unpack("<Q", stream.read(8))[0]
            if not 0 < header_length <= min(16 * 1024 * 1024, size - 8):
                return False
            header = json.loads(stream.read(header_length))
        return isinstance(header, dict) and any(key.endswith("txtfusion.projector.weight") for key in header)
    except (OSError, ValueError, struct.error):
        return False


def krea_selected(p):
    from modules import sd_models, shared

    selection = (getattr(p, "override_settings", None) or {}).get(
        "sd_model_checkpoint", shared.opts.sd_model_checkpoint
    )
    info = sd_models.get_closet_checkpoint_match(selection)
    if info is None:
        return False
    try:
        path = Path(info.filename)
        stat = path.stat()
        return _checkpoint_is_krea(str(path), stat.st_mtime_ns, stat.st_size)
    except OSError:
        return False


class TileAllocator:
    def __init__(
        self, mode, log_root, timeout=10, *, client=None, is_cancelled=None, cadence="once", owns_client=True, prompt=""
    ):
        if mode not in {"rules", "jev"}:
            raise ValueError("Invalid tile allocation mode")
        if cadence not in {"once", "stage"}:
            raise ValueError("Invalid tile allocation cadence")
        self.mode, self.timeout = mode, timeout
        self.cadence, self.stage = cadence, 0
        self.log = RunLog(
            log_root,
            "krea2-tiles",
            {"mode": mode, "cadence": cadence, "api_call_limit": 1 if cadence == "once" else "upscale_stages"},
            prompt,
        )
        self.client = client
        self.owns_client = owns_client
        self.decisions = 0
        self.cancelled = is_cancelled or cancelled
        self.decided = False
        self.choices = {}
        self.bounds = None
        self.counts = Counter()
        self.allocation_sources = Counter()
        self.source = "rules"
        self.circuit_open = False

    @staticmethod
    def importance(score, knee):
        if not math.isfinite(score) or score < 0 or not math.isfinite(knee) or knee <= 0:
            raise ValueError("Invalid tile detail statistics")
        return score / (score + knee)

    @classmethod
    def bucket(cls, score, knee):
        return min(7, int(cls.importance(score, knee) * 8))

    def prepare(self, scores, minimum, maximum, knee, *, layout=None):
        if minimum < 1 or maximum < minimum:
            raise ValueError("Invalid tile step limits")
        if self.cancelled():
            raise ExperimentCancelled("Generation cancelled")
        self.stage += 1
        if self.circuit_open or (self.decided and self.cadence == "once"):
            return
        if self.mode == "jev" and self.client is not None and not client_available(self.client):
            return
        self.decided = True
        self.choices = {}
        self.bounds = (minimum, maximum)
        groups = defaultdict(list)
        for score in scores:
            groups[str(self.bucket(score, knee))].append(float(score))
        if not groups or self.mode != "jev":
            return
        try:
            if self.client is None:
                self.client = create_client(
                    timeout=self.timeout,
                    cancelled=self.cancelled,
                    environment=None if replay_paths_from_environment() else cloud_source(),
                )
            self.choices = self.client.decide(
                {
                    "decision_kind": "tile_steps",
                    "target": "Krea2 high-resolution tile compute allocation",
                    "constraints": "SPEED-FIRST. Choose 0 or a supplied positive step count for each observed detail group. Zero retains the enlarged base and adds no generated detail. Groups are ordered from weak (0) to strong (7) measured texture/edge detail. Prefer skipping very weak, flat groups, minimum positive steps for typical groups, maximum for unusually strong detail. These are aggregate numerical proxies, not semantic image analysis or measured quality. Do not invent subjects, faces or text. No forced quota. Choices apply until the next scheduled stage decision. Respect supplied positive step bounds.",
                    "stage": self.stage,
                    "decision_cadence": self.cadence,
                    "tile_layout": {
                        "count": len(scores),
                        "detail_group_order": [str(self.bucket(score, knee)) for score in scores],
                        "geometry": layout,
                        "detail_knee": knee,
                    },
                    "groups": {
                        key: {
                            "tiles": len(values),
                            "min_detail": min(values),
                            "max_detail": max(values),
                            "mean_detail": sum(values) / len(values),
                        }
                        for key, values in groups.items()
                    },
                    "detail_knee": knee,
                    "positive_step_bounds": [minimum, maximum],
                },
                {key: tuple(float(v) for v in sorted({0, minimum, maximum})) for key in groups},
                float(maximum),
            )
            self.source = "jev"
            self.decisions += 1
            diagnostics = {
                key: {"steps": int(value), "confidence": self.client.last_diagnostics.get(key, {}).get("confidence")}
                for key, value in self.choices.items()
            }
            self.log.write(
                "decision",
                source=self.source,
                stage=self.stage,
                steps_by_detail_group=self.choices,
                diagnostics=diagnostics,
                api_calls=self.client.calls,
                api_wait_seconds=self.client.wait_seconds,
                replay=getattr(self.client, "last_replay", None),
                replay_calls=getattr(self.client, "replay_calls", 0),
            )
        except (ExperimentCancelled, ReplayError):
            raise
        except Exception as exc:
            self.source = "rules_fallback"
            self.circuit_open = True
            self.log.write("decision", source=self.source, error_type=type(exc).__name__)

    def steps(self, score, minimum, maximum, knee):
        importance = self.importance(score, knee)
        group = str(self.bucket(score, knee))
        if self.bounds == (minimum, maximum) and group in self.choices:
            steps = int(self.choices[group])
            source = "jev"
        else:
            steps = 0 if importance < 0.20 else minimum if importance < 0.75 else maximum
            source = self.source if self.source != "jev" else "rules_unseen_group"
        self.counts[str(steps)] += 1
        self.allocation_sources[source] += 1
        return steps

    def close(self, status):
        if self.owns_client:
            try:
                close_client(self.client, status)
            except ReplayError:
                self.log.finish("failed", error_type="ReplayError")
                raise
        self.log.finish(
            status,
            selected_step_counts=dict(self.counts),
            allocation_sources=dict(self.allocation_sources),
            source=self.source,
            api_calls=self.client.calls if self.client else 0,
            api_wait_seconds=self.client.wait_seconds if self.client else 0,
            replay_calls=getattr(self.client, "replay_calls", 0),
            decisions=self.decisions,
        )


class Session:
    def __init__(self, options, log_root=None, *, prompt=""):
        options.validate()
        self.options = options
        self.log_root = Path(log_root or Path(__file__).resolve().parents[2] / "outputs/jev-sparse")
        self.run = None
        self.allocator = None
        self.closed = False
        self.prompt = prompt
        self.client = None
        self.budget = JevBudget(options.job_max_calls, options.job_max_wait_seconds)
        self.replay_paths = (
            replay_paths_from_environment({REPLAY_ENV: options.replay_logs}) if options.replay_logs else None
        )

    def get_client(self):
        if self.closed:
            raise RuntimeError("Krea2 generation session is closed")
        if self.client is None and "jev" in {self.options.mode, self.options.tile_mode}:
            paths = self.replay_paths if self.replay_paths is not None else replay_paths_from_environment()
            self.client = create_client(
                timeout=self.options.timeout,
                cancelled=cancelled,
                environment=None if paths else cloud_source(),
                budget=self.budget,
                replay_paths=paths,
                prompt_sha256=hashlib.sha256(self.prompt.encode()).hexdigest(),
            )
        return self.client

    def prepare_tiles(self, scores, minimum, maximum, knee, *, layout=None):
        if self.closed:
            raise RuntimeError("Krea2 generation session is closed")
        if self.options.tile_mode == "off":
            return None
        if self.allocator is None:
            self.allocator = TileAllocator(
                self.options.tile_mode,
                self.log_root,
                self.options.timeout,
                cadence=self.options.tile_cadence,
                client=self.get_client() if self.options.tile_mode == "jev" else None,
                owns_client=False,
                prompt=self.prompt,
            )
        self.allocator.prepare(scores, minimum, maximum, knee, layout=layout)
        return self.allocator

    def close(self, status):
        if self.closed:
            return
        self.closed = True
        try:
            try:
                close_client(self.client, status)
            except ReplayError:
                status = "failed"
                raise
        finally:
            try:
                if self.run is not None:
                    self.run.close(status)
            finally:
                if self.allocator is not None:
                    self.allocator.close(status)


def current_session():
    return _JOB.get()


@contextmanager
def generation_scope(p):
    existing = current_session()
    if existing is not None:
        yield existing
        return
    options = processing_options(p)
    if options.mode == "off" and options.tile_mode == "off":
        yield None
        return
    if not krea_selected(p):
        yield None
        return
    session = Session(options, prompt=str(getattr(p, "prompt", "")))
    token = _JOB.set(session)
    status = "failed"
    try:
        yield session
        status = "cancelled" if cancelled() else "completed"
    except ExperimentCancelled:
        status = "cancelled"
        raise
    except BaseException:
        status = "cancelled" if cancelled() else "failed"
        raise
    finally:
        try:
            session.close(status)
        finally:
            _JOB.reset(token)


def whole_image_job(function):
    @functools.wraps(function)
    def wrapped(self, p, *args, **kwargs):
        with generation_scope(p):
            return function(self, p, *args, **kwargs)

    return wrapped


def prepare_tile_allocation(p, scores, minimum, maximum, knee, *, layout=None):
    session = current_session()
    if session is None:
        return None
    allocator = session.prepare_tiles(scores, minimum, maximum, knee, layout=layout)
    if allocator is not None:
        p.extra_generation_params.update(
            {
                "Krea2 tile allocation": session.options.tile_mode,
                "Krea2 tile cadence": session.options.tile_cadence,
                "Krea2 tile allocation log": str(allocator.log.path),
            }
        )
    return allocator
