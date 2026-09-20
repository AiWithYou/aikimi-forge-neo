"""Failure cleanup keeps queued generations isolated without using a GPU."""

import unittest
from contextlib import closing
from functools import wraps
from threading import Lock
from types import SimpleNamespace
from unittest.mock import Mock

from tools.tests.test_gpu_ownership import load_function


class QueueCleanupTests(unittest.TestCase):
    def wrapper(self, lock, state, progress, outer):
        return load_function(
            "modules/call_queue.py",
            "wrap_gradio_gpu_call",
            {
                "wraps": wraps,
                "queue_lock": lock,
                "shared": SimpleNamespace(state=state),
                "progress": progress,
                "wrap_gradio_call": outer,
            },
        )

    def test_state_and_wrapper_cleanup_finish_before_releasing_queue(self):
        lock = Lock()
        events = []
        state, progress = Mock(), Mock()
        state.end.side_effect = lambda: events.append(("state end", lock.locked()))

        def outer(callback, **_kwargs):
            def wrapped(*args):
                events.append(("monitor start", lock.locked()))
                try:
                    return callback(*args)
                finally:
                    events.append(("monitor stop", lock.locked()))

            return wrapped

        work = Mock(side_effect=RuntimeError("generation failed"))
        generate = self.wrapper(lock, state, progress, outer)(work)
        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            generate("task(first)")

        self.assertEqual(events, [("monitor start", True), ("state end", True), ("monitor stop", True)])
        self.assertFalse(lock.locked())
        progress.finish_task.assert_called_once_with("task(first)")
        progress.record_results.assert_not_called()

    def test_failed_state_start_still_finishes_task(self):
        state, progress = Mock(), Mock()
        state.begin.side_effect = RuntimeError("state failed")
        work = Mock()
        generate = self.wrapper(Lock(), state, progress, lambda callback, **_kwargs: callback)(work)
        with self.assertRaisesRegex(RuntimeError, "state failed"):
            generate("task(first)")

        progress.finish_task.assert_called_once_with("task(first)")
        state.end.assert_called_once()
        work.assert_not_called()

    def test_cancelled_lock_wait_removes_pending_task_without_resetting_active_state(self):
        class CancelledWait:
            def __enter__(self):
                raise InterruptedError("wait cancelled")

            def __exit__(self, *_args):
                return False

        state, progress = Mock(), Mock()
        work = Mock()
        generate = self.wrapper(CancelledWait(), state, progress, lambda callback, **_kwargs: callback)(work)
        with self.assertRaisesRegex(InterruptedError, "wait cancelled"):
            generate("task(waiting)")

        progress.add_task_to_queue.assert_called_once_with("task(waiting)")
        progress.finish_task.assert_called_once_with("task(waiting)")
        state.begin.assert_not_called()
        state.end.assert_not_called()
        work.assert_not_called()


class ApiCleanupTests(unittest.TestCase):
    def prepare(self, method, failure):
        lock = Lock()
        events = []
        state = Mock()
        state.end.side_effect = lambda: events.append(("state end", lock.locked()))
        pending = []
        finish = Mock(side_effect=lambda task: (pending.remove(task), events.append(("task end", lock.locked()))))
        processing = Mock()
        constructor = Mock(return_value=processing)
        run = Mock(side_effect=RuntimeError("processing failed"))
        if failure == "constructor":
            constructor.side_effect = RuntimeError("constructor failed")
        elif failure == "begin":
            state.begin.side_effect = RuntimeError("begin failed")

        namespace = {
            "models": SimpleNamespace(
                StableDiffusionTxt2ImgProcessingAPI=object, StableDiffusionImg2ImgProcessingAPI=object
            ),
            "scripts": SimpleNamespace(scripts_txt2img=object(), scripts_img2img=object()),
            "sd_samplers": SimpleNamespace(get_sampler_and_scheduler=lambda *_args: ("Euler", "Automatic")),
            "validate_sampler_name": lambda name: name,
            "add_task_to_queue": pending.append,
            "start_task": Mock(),
            "finish_task": finish,
            "shared": SimpleNamespace(state=state, sd_model=object(), total_tqdm=Mock()),
            "opts": SimpleNamespace(
                outdir_txt2img_grids="", outdir_txt2img_samples="", outdir_img2img_grids="", outdir_img2img_samples=""
            ),
            "StableDiffusionProcessingTxt2Img": constructor,
            "StableDiffusionProcessingImg2Img": constructor,
            "closing": closing,
            "_run_api_processing": run,
            "process_extra_images": Mock(),
            "decode_base64_to_image": lambda _value: object(),
        }
        api = SimpleNamespace(
            queue_lock=lock,
            apply_infotext=Mock(),
            get_selectable_script=Mock(return_value=(None, None)),
            init_script_args=Mock(return_value=[]),
            default_script_arg_txt2img=[],
            default_script_arg_img2img=[],
        )
        request = SimpleNamespace(
            force_task_id="task(api)",
            script_name=None,
            sampler_name="Euler",
            sampler_index=None,
            scheduler="Automatic",
            save_images=False,
            init_images=["image"],
            mask=None,
        )
        request.copy = lambda update: SimpleNamespace(**(vars(request) | update))
        callback = load_function("modules/api/api.py", method, namespace)
        return callback, api, request, namespace, pending, events, processing

    def test_api_failures_finalize_tasks_and_state_while_holding_queue(self):
        for method in ("text2imgapi", "img2imgapi"):
            for failure in ("constructor", "begin", "processing"):
                with self.subTest(method=method, failure=failure):
                    callback, api, request, namespace, pending, events, processing = self.prepare(method, failure)
                    with self.assertRaisesRegex(RuntimeError, f"{failure} failed"):
                        callback(api, request)

                    self.assertEqual(pending, [])
                    self.assertEqual(events, [("task end", True), ("state end", True)])
                    namespace["shared"].total_tqdm.clear.assert_called_once()
                    self.assertFalse(api.queue_lock.locked())
                    self.assertEqual(processing.close.call_count, int(failure == "processing"))


if __name__ == "__main__":
    unittest.main()
