"""FIFO lock with explicit ownership handoff, including cross-thread release."""

import collections
import threading


# reference: https://gist.github.com/vitaliyp/6d54dd76ca2c3cdfc1149d33007dc34a
class FIFOLock:
    def __init__(self):
        self._lock = threading.Lock()
        self._inner_lock = threading.Lock()
        self._pending_threads = collections.deque()

    def acquire(self, blocking=True):
        with self._inner_lock:
            if self._lock.acquire(False):
                return True
            if not blocking:
                return False

            release_event = threading.Event()
            self._pending_threads.append(release_event)

        try:
            release_event.wait()
            # release() transfers the still-locked mutex to this waiter. A new
            # caller must not acquire it while the selected waiter is waking up.
            return True
        except BaseException:
            with self._inner_lock:
                try:
                    self._pending_threads.remove(release_event)
                except ValueError:
                    # Ownership was already transferred. Do not strand either
                    # the mutex or the next waiter when this wait is interrupted.
                    self._release_next()
            raise

    def _release_next(self):
        """Transfer or release ownership while holding _inner_lock."""
        if self._pending_threads:
            self._pending_threads.popleft().set()
        else:
            self._lock.release()

    def release(self):
        with self._inner_lock:
            self._release_next()

    __enter__ = acquire

    def __exit__(self, t, v, tb):
        self.release()
