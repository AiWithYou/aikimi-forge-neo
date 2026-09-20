"""GPU使用権の内側で、待機中のモデルだけを保持・回収する。"""

import contextvars
import logging
import sys
import threading
import time
from contextlib import contextmanager

AUTO = "auto"
KEEP = "keep"
RELEASE = "release"
IDLE_SECONDS = 300
_engine = contextvars.ContextVar("gpu_engine", default="forge")
_resources = {}
_timer = None
_idle_since = None
_active = False
_log = logging.getLogger(__name__)


def policy():
    shared = sys.modules.get("modules.shared")
    value = getattr(getattr(shared, "opts", None), "aikimi_model_retention", AUTO)
    if value not in {AUTO, KEEP, RELEASE}:
        raise ValueError("生成後のモデル保持設定が不正です。")
    return value


@contextmanager
def engine_scope(engine):
    token = _engine.set(engine)
    try:
        yield
    finally:
        _engine.reset(token)


def release_resource(name):
    entry = _resources.get(name)
    if entry is not None:
        entry[1]()  # 終了・退避の確認ができない限り、登録を消さない。
        _resources.pop(name, None)


def register(name, cleanup, engine):
    """呼び出し側はqueue_lockを保持し、cleanupはGPU待機をしない。"""
    _resources[name] = (engine, cleanup)


def begin(engine):
    global _active, _timer
    policy()
    if _timer is not None:
        _timer.cancel()
        _timer = None
    for name, (owner, _) in list(_resources.items()):
        if owner != engine:
            release_resource(name)
    if engine == "forge":
        from modules_forge.gpu_ownership import release_forge_vram

        register("forge", release_forge_vram, "forge")
    _active = True


def finish():
    global _active, _idle_since, _timer
    if _timer is not None:
        _timer.cancel()
        _timer = None
    _active = False
    _idle_since = time.monotonic()
    if policy() == RELEASE:
        for name in list(_resources):
            release_resource(name)
    elif policy() == AUTO and _resources:
        _timer = threading.Timer(IDLE_SECONDS, expire_idle)
        _timer.daemon = True
        _timer.start()


def expire_idle():
    from modules_forge.gpu_ownership import queue_lock

    with engine_scope(None):
        if not queue_lock.acquire(False):
            return
        try:
            if policy() == AUTO and not _active and _idle_since is not None:
                if time.monotonic() - _idle_since >= IDLE_SECONDS:
                    for name in list(_resources):
                        release_resource(name)
        except Exception:
            _log.exception("待機モデルの自動解放に失敗しました")
        finally:
            queue_lock.release()


def release_idle():
    from modules_forge.gpu_ownership import queue_lock, release_forge_vram

    with engine_scope(None):
        if not queue_lock.acquire(False):
            return "生成中です。終了後に解放してください。"
        try:
            forge_registered = "forge" in _resources
            for name in list(_resources):
                release_resource(name)
            if not forge_registered:
                release_forge_vram()
            return "待機中のモデルを解放しました。"
        finally:
            queue_lock.release()


def apply_policy():
    from modules_forge.gpu_ownership import queue_lock

    with engine_scope(None):
        if queue_lock.acquire(False):
            try:
                finish()
            finally:
                queue_lock.release()
