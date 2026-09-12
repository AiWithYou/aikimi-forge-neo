"""UI/APIと外部workerが共有するGPUの排他。停止確認後にだけ返却する。"""

import sys
import threading

from modules.fifo_lock import FIFOLock

queue_lock = FIFOLock()


def release_forge_vram():
    """呼出側がGPU所有権を保持した状態でForgeモデルを退避する。"""
    sd_models = sys.modules.get("modules.sd_models")
    if sd_models is not None:
        from modules import devices

        sd_models.unload_model_weights()
        devices.torch_gc()


class GPUOwnership:
    """HTTPやgeneratorの寿命から切り離して保持できる、同じqueue_lockの所有権。"""

    def __init__(self):
        self._lock = queue_lock
        self._guard = threading.Lock()
        self._owned = False

    def acquire(self, blocking=False):
        with self._guard:
            if not self._owned:
                self._owned = self._lock.acquire(blocking)
            return self._owned

    def release(self):
        with self._guard:
            if self._owned:
                self._owned = False
                self._lock.release()
