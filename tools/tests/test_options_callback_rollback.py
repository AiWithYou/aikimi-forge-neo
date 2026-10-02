"""Failed option callbacks must restore the effective value and saved presence."""

import unittest
from unittest.mock import Mock

from tools.tests.test_api_public_state import load_isolated_options_module


class OptionsCallbackRollbackTests(unittest.TestCase):
    def setUp(self):
        module = load_isolated_options_module()
        self.callback = Mock(side_effect=ValueError("invalid setting"))
        self.options = module.Options({"example": module.OptionInfo(7, onchange=self.callback)}, restricted_opts=set())

    def test_failed_callback_keeps_default_when_not_in_saved_config(self):
        self.options.data.clear()
        self.assertFalse(self.options.set("example", 9))
        self.assertEqual(self.options.example, 7)
        self.assertNotIn("example", self.options.data)
        self.callback.assert_called_once_with()

    def test_setting_to_effective_default_does_not_invoke_callback(self):
        self.options.data.clear()
        self.assertFalse(self.options.set("example", 7))
        self.callback.assert_not_called()
        self.assertNotIn("example", self.options.data)

    def test_explicit_saved_value_is_restored_after_callback_failure(self):
        self.options.data["example"] = 3
        self.assertFalse(self.options.set("example", 9))
        self.assertEqual(self.options.example, 3)
        self.assertEqual(self.options.data, {"example": 3})

    def test_explicit_none_is_restored_as_explicit_none(self):
        self.options.data["example"] = None
        self.assertFalse(self.options.set("example", 9))
        self.assertIsNone(self.options.example)
        self.assertEqual(self.options.data, {"example": None})

    def test_successful_callback_stores_new_value(self):
        self.options.data.clear()
        self.callback.side_effect = None
        self.assertTrue(self.options.set("example", 9))
        self.assertEqual(self.options.example, 9)
        self.callback.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
