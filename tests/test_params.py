"""The PARAM_* substitution rules a workflow file relies on."""

from __future__ import annotations

from support import unittest

from sillytalk_plugin.params import find_placeholders, substitute


class SubstituteTest(unittest.TestCase):
    def test_whole_value_only(self):
        graph = {"1": {"inputs": {"text": "PARAM_PROMPT", "note": "say PARAM_PROMPT now"}}}
        out = substitute(graph, {"PARAM_PROMPT": "a red apple"})
        self.assertEqual(out["1"]["inputs"]["text"], "a red apple")
        self.assertEqual(out["1"]["inputs"]["note"], "say PARAM_PROMPT now")

    def test_numbers_reach_widgets_as_numbers(self):
        graph = {"10": {"inputs": {"seed": "PARAM_SEED", "width": "PARAM_INT_OUTPUT_IMAGE_WIDTH"}}}
        out = substitute(graph, {"PARAM_SEED": 7, "PARAM_INT_OUTPUT_IMAGE_WIDTH": 1328})
        self.assertEqual(out["10"]["inputs"]["seed"], 7)
        self.assertEqual(out["10"]["inputs"]["width"], 1328)

    def test_lists_and_nesting(self):
        graph = {"1": {"inputs": {"anything": ["PARAM_PROMPT", ["PARAM_SEED"], {"deep": "PARAM_SEED"}]}}}
        out = substitute(graph, {"PARAM_PROMPT": "p", "PARAM_SEED": 3})
        self.assertEqual(out["1"]["inputs"]["anything"], ["p", [3], {"deep": 3}])

    def test_input_is_never_modified(self):
        graph = {"1": {"inputs": {"text": "PARAM_PROMPT"}}}
        substitute(graph, {"PARAM_PROMPT": "changed"})
        self.assertEqual(graph["1"]["inputs"]["text"], "PARAM_PROMPT")

    def test_missing_value_stays_and_is_reported(self):
        graph = {"1": {"inputs": {"text": "PARAM_PROMPT", "other": "PARAM_FOO"}}}
        out = substitute(graph, {"PARAM_PROMPT": "p"})
        self.assertEqual(out["1"]["inputs"]["other"], "PARAM_FOO")
        self.assertEqual(find_placeholders(out), {"PARAM_FOO"})


if __name__ == "__main__":
    unittest.main()
