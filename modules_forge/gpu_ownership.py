"""UI/APIと外部workerが共有するGPUの排他。停止確認後にだけ返却する。"""

import sys
import threading

from modules.fifo_lock import FIFOLock


class GPUQueueLock(FIFOLock):
    def __init__(self):
        super().__init__()
        self._startup_lock = threading.Lock()
        self._restored = False

    def acquire(self, blocking=True):
        with self._startup_lock:
            if not self._restored:
                from modules_forge import minimax_h3_pending

                records = minimax_h3_pending.read_all()
                if records:
                    from modules_forge.minimax_h3_bridge import recover_pending_generations

                    super().acquire()
                    # 再起動直後も、Forge/APIの最初のGPU操作より先に所有権を復元する。
                    worker = threading.Thread(
                        target=recover_pending_generations,
                        args=(records, self.release),
                        daemon=True,
                    )
                    try:
                        worker.start()
                    except BaseException:
                        super().release()
                        raise
                self._restored = True
        return super().acquire(blocking)

    __enter__ = acquire


queue_lock = GPUQueueLock()


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
