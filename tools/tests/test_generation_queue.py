import ast
from collections import OrderedDict
from pathlib import Path
import sys
import unittest


original_argv = sys.argv
try:
    sys.argv = [sys.argv[0]]
    from modules import progress
finally:
    sys.argv = original_argv


ROOT = Path(__file__).resolve().parents[2]


class GenerationQueueTests(unittest.TestCase):
    def setUp(self):
        self.original_pending = progress.pending_tasks
        self.original_current = progress.current_task
        self.original_finished = progress.finished_tasks
        progress.pending_tasks = OrderedDict()
        progress.current_task = None
        progress.finished_tasks = []

    def tearDown(self):
        progress.pending_tasks = self.original_pending
        progress.current_task = self.original_current
        progress.finished_tasks = self.original_finished

    def test_pending_tasks_keep_fifo_order_and_position_text(self):
        progress.add_task_to_queue("task(first)")
        progress.add_task_to_queue("task(second)")

        pending = progress.get_pending_tasks()
        second = progress.progressapi(progress.ProgressRequest(id_task="task(second)"))

        self.assertEqual(pending.size, 2)
        self.assertEqual(pending.tasks, ["task(first)", "task(second)"])
        self.assertTrue(second.queued)
        self.assertEqual(second.textinfo, "In queue: 2/2")

        progress.start_task("task(first)")
        self.assertEqual(progress.current_task, "task(first)")
        self.assertEqual(list(progress.pending_tasks), ["task(second)"])

    def test_queue_buttons_use_existing_multi_submit_wiring(self):
        toprow_source = (ROOT / "modules" / "ui_toprow.py").read_text(encoding="utf-8")
        ui_source = (ROOT / "modules" / "ui.py").read_text(encoding="utf-8")
        javascript_source = (ROOT / "javascript" / "ui.js").read_text(encoding="utf-8")
        webui_source = (ROOT / "webui.py").read_text(encoding="utf-8")

        self.assertIn('self.id_part in {"txt2img", "img2img"}', toprow_source)
        self.assertIn('"Add to Queue"', toprow_source)
        self.assertIn('elem_id=f"{self.id_part}_queue"', toprow_source)
        self.assertIn('"_js": "submit_txt2img_queue"', ui_source)
        self.assertIn('"_js": "submit_img2img_queue"', ui_source)
        self.assertGreaterEqual(ui_source.count('"trigger_mode": "multiple"'), 2)
        self.assertIn("function submitQueued(tabname, args)", javascript_source)
        self.assertIn('return submitQueued("txt2img", arguments)', javascript_source)
        self.assertIn('return submitQueued("img2img", arguments)', javascript_source)
        self.assertIn("const submitTasksByTab = new Map()", javascript_source)
        self.assertIn("startSubmitTask(tabname, id)", javascript_source)
        self.assertIn("finishSubmitTask(tabname, id)", javascript_source)
        self.assertIn("shared.demo.queue(default_concurrency_limit=32)", webui_source)

        queue_function = javascript_source.split("function submitQueued", 1)[1].split(
            "function submit_txt2img_queue", 1
        )[0]
        self.assertNotIn("localSet", queue_function)

    def test_interrupt_wiring_remains_common_to_all_toprows(self):
        tree = ast.parse(
            (ROOT / "modules" / "ui_toprow.py").read_text(encoding="utf-8")
        )
        toprow = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Toprow"
        )
        create_submit_box = next(
            node
            for node in toprow.body
            if isinstance(node, ast.FunctionDef) and node.name == "create_submit_box"
        )
        interrupt_function = next(
            node
            for node in create_submit_box.body
            if isinstance(node, ast.FunctionDef) and node.name == "interrupt_function"
        )

        self.assertIsNotNone(interrupt_function)
        queue_condition = next(
            node for node in create_submit_box.body if isinstance(node, ast.If)
        )
        self.assertFalse(
            any(isinstance(node, ast.FunctionDef) for node in queue_condition.body)
        )


if __name__ == "__main__":
    unittest.main()
