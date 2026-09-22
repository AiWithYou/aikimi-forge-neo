"""Deterministic queue handoff and interrupted-wait regression tests; no GPU."""

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from modules import fifo_lock


class WaitInterrupted(BaseException):
    """Simulate a signal interrupting Event.wait without process-wide signals."""


class GatedEvent:
    """Pause a selected waiter until the test explicitly resumes it."""

    def __init__(self, *, interrupt=False):
        self.entered = threading.Event()
        self.signaled = threading.Event()
        self.resume = threading.Event()
        self.interrupt = interrupt

    def set(self):
        self.signaled.set()

    def wait(self):
        self.entered.set()
        if not self.signaled.wait(3) or not self.resume.wait(3):
            raise AssertionError("test waiter did not receive its handoff")
        if self.interrupt:
            raise WaitInterrupted()
        return True


class FIFOLockHandoffTests(unittest.TestCase):
    def setUp(self):
        self.lock = fifo_lock.FIFOLock()
        self.threads = []
        self.events = []
        self.errors = []
        self.order = []

    def tearDown(self):
        for event in self.events:
            event.resume.set()
        for thread in self.threads:
            thread.join(4)
        self.assertFalse(any(thread.is_alive() for thread in self.threads))
        self.assertEqual(self.errors, [])

    def enqueue(self, index, *, interrupt=False):
        event = GatedEvent(interrupt=interrupt)
        self.events.append(event)

        def run():
            try:
                acquired = self.lock.acquire()
                if acquired:
                    try:
                        self.order.append(index)
                    finally:
                        self.lock.release()
            except WaitInterrupted:
                if not interrupt:
                    self.errors.append("unexpected interrupted waiter")
            except BaseException as exc:
                self.errors.append(exc)

        # Replace only the module binding, not threading.Event globally: Thread
        # itself uses Event during startup.
        with patch.object(fifo_lock, "threading", SimpleNamespace(Event=lambda: event, Lock=threading.Lock)):
            thread = threading.Thread(target=run, daemon=True)
            self.threads.append(thread)
            thread.start()
            self.assertTrue(event.entered.wait(2))
        return event

    def test_nonblocking_acquire_and_reuse(self):
        self.assertTrue(self.lock.acquire(False))
        self.assertFalse(self.lock.acquire(False))
        self.lock.release()
        self.assertTrue(self.lock.acquire(False))
        self.lock.release()

    def test_release_without_acquisition_is_an_error(self):
        with self.assertRaises(RuntimeError):
            self.lock.release()

    def test_context_manager_releases_after_error(self):
        with self.assertRaisesRegex(ValueError, "test"):
            with self.lock:
                raise ValueError("test")
        self.assertTrue(self.lock.acquire(False))
        self.lock.release()

    def test_other_thread_can_release_a_transferred_gpu_lease(self):
        self.lock.acquire()
        thread = threading.Thread(target=self.lock.release)
        thread.start()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertTrue(self.lock.acquire(False))
        self.lock.release()

    def test_newcomer_cannot_barge_during_handoff(self):
        self.lock.acquire()
        first = self.enqueue(0)
        self.lock.release()
        self.assertTrue(first.signaled.wait(2))
        barged = self.lock.acquire(False)
        if barged:
            self.lock.release()
        first.resume.set()
        self.assertFalse(barged, "a signaled waiter still owns the next acquisition")

    def test_waiters_receive_ownership_in_enqueue_order(self):
        self.lock.acquire()
        events = [self.enqueue(index) for index in range(3)]
        self.lock.release()
        for index, event in enumerate(events):
            self.assertTrue(event.signaled.wait(2))
            self.assertEqual(self.order, list(range(index)))
            event.resume.set()
            self.threads[index].join(2)
        self.assertEqual(self.order, [0, 1, 2])

    def test_interrupted_wait_is_removed_before_handoff(self):
        self.lock.acquire()

        class InterruptedEvent:
            def wait(self):
                raise WaitInterrupted()

        with patch.object(fifo_lock, "threading", SimpleNamespace(Event=InterruptedEvent)):
            with self.assertRaises(WaitInterrupted):
                self.lock.acquire()
        # An aborted waiter must not consume the next real waiter's wakeup.
        self.assertEqual(len(self.lock._pending_threads), 0)
        self.assertFalse(self.lock.acquire(False))
        self.lock.release()

    def test_interrupted_selected_waiter_hands_off_to_successor(self):
        self.lock.acquire()
        first = self.enqueue(0, interrupt=True)
        second = self.enqueue(1)
        self.lock.release()
        self.assertTrue(first.signaled.wait(2))
        first.resume.set()
        self.threads[0].join(2)
        successor_signaled = second.signaled.wait(0.5)
        # Unstick the original implementation after recording the failure.
        if not successor_signaled and self.lock.acquire(False):
            self.lock.release()
        second.resume.set()
        self.assertTrue(successor_signaled, "an interrupted grantee must pass ownership onward")

    def test_interrupted_selected_waiter_without_successor_unlocks(self):
        self.lock.acquire()
        first = self.enqueue(0, interrupt=True)
        self.lock.release()
        first.resume.set()
        self.threads[0].join(2)
        self.assertTrue(self.lock.acquire(False))
        self.lock.release()

    def test_many_contenders_remain_mutually_exclusive(self):
        start = threading.Barrier(13)
        counter = [0]

        def increment():
            try:
                start.wait(3)
                for _ in range(100):
                    with self.lock:
                        value = counter[0]
                        # Yield inside the critical section without timing sleeps.
                        threading.Event().wait(0.00001)
                        counter[0] = value + 1
            except BaseException as exc:
                self.errors.append(exc)

        for _ in range(12):
            thread = threading.Thread(target=increment, daemon=True)
            self.threads.append(thread)
            thread.start()
        start.wait(3)
        for thread in self.threads:
            thread.join(4)
        self.assertEqual(counter[0], 1200)


class GPUQueueLockHandoffTests(FIFOLockHandoffTests):
    """The production GPU queue must preserve the base lock's handoff contract."""

    def setUp(self):
        super().setUp()
        from modules_forge import gpu_ownership, gpu_residency

        self.lock = gpu_ownership.GPUQueueLock()
        # Recovery is tested separately. Never inspect a user's pending jobs.
        self.lock._restored = True
        self.begin = self.enterContext(patch.object(gpu_residency, "begin"))
        self.finish = self.enterContext(patch.object(gpu_residency, "finish"))

    def test_failed_model_preparation_returns_queue_ownership(self):
        self.begin.side_effect = RuntimeError("prepare failed")
        with self.assertRaisesRegex(RuntimeError, "prepare failed"):
            self.lock.acquire(False)
        self.begin.side_effect = None
        self.assertTrue(self.lock.acquire(False))
        self.lock.release()

    def test_failed_cleanup_returns_queue_ownership(self):
        self.assertTrue(self.lock.acquire(False))
        self.finish.side_effect = RuntimeError("cleanup failed")
        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            self.lock.release()
        self.finish.side_effect = None
        self.assertTrue(self.lock.acquire(False))
        self.lock.release()


if __name__ == "__main__":
    unittest.main()
