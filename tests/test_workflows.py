"""The workflow catalogue: what counts as a readable model, and how a graph is
handed out.

The conversion of a UI-format graph needs the server's node registry, so these
tests use API-format graphs, which the catalogue passes through untouched — the
shape of the catalogue, not the vendored converter, is what is under test here.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from support import PLUGIN_DIR  # noqa: F401  (puts the plugin on sys.path)

from sillytalk_plugin import workflows


def api_graph(text: str = "hello") -> dict:
    return {"1": {"class_type": "CLIPTextEncode", "inputs": {"text": text}}}


class CatalogueTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self._original = workflows.WORKFLOW_DIR
        workflows.WORKFLOW_DIR = self.dir
        workflows.forget_cache()
        self.addCleanup(lambda: setattr(workflows, "WORKFLOW_DIR", self._original))
        self.addCleanup(workflows.forget_cache)
        self.addCleanup(self._tmp.cleanup)

    def write(self, code: str, data) -> Path:
        path = self.dir / f"{code}.json"
        path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
        return path

    def test_reads_variants_and_defaults(self):
        self.write(
            "my-model",
            {
                "label": "My Model",
                "defaults": {"width": 1328, "height": 1328},
                "variants": [
                    {"refs": 2, "workflow": api_graph()},
                    {"refs": 0, "workflow": api_graph()},
                ],
            },
        )
        spec = workflows.model("my-model")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.refs, (0, 2), "the variants come back ordered by reference count")
        self.assertEqual(spec.max_refs, 2)
        self.assertEqual(spec.width, 1328)
        self.assertEqual(spec.describe()["label"], "My Model")

    def test_model_codes_are_sorted(self):
        self.write("b-model", {"variants": [{"refs": 0, "workflow": api_graph()}]})
        self.write("a-model", {"variants": [{"refs": 0, "workflow": api_graph()}]})
        self.assertEqual([spec.code for spec in workflows.models()], ["a-model", "b-model"])

    def test_broken_file_is_reported_not_raised(self):
        self.write("broken", "{ not json")
        self.assertIsNone(workflows.model("broken"))
        entry = next(item for item in workflows.entries() if item.code == "broken")
        self.assertIn("not valid JSON", entry.error)

    def test_missing_variants_is_an_error(self):
        self.write("empty", {"defaults": {}, "variants": []})
        entry = next(item for item in workflows.entries() if item.code == "empty")
        self.assertIn("non-empty list", entry.error)

    def test_duplicate_reference_count_is_an_error(self):
        self.write(
            "twice",
            {"variants": [{"refs": 1, "workflow": api_graph()}, {"refs": 1, "workflow": api_graph()}]},
        )
        entry = next(item for item in workflows.entries() if item.code == "twice")
        self.assertIn("two variants", entry.error)

    def test_a_graph_that_is_not_a_graph_is_an_error(self):
        self.write("nograph", {"variants": [{"refs": 0, "workflow": {"note": "nothing here"}}]})
        entry = next(item for item in workflows.entries() if item.code == "nograph")
        self.assertIn("neither", entry.error)

    def test_unknown_and_pathish_codes_are_refused(self):
        self.assertIsNone(workflows.model("nope"))
        self.assertIsNone(workflows.model("../config"))
        self.assertIsNone(workflows.model("a/b"))

    def test_api_graph_hands_out_a_copy(self):
        self.write("copy", {"variants": [{"refs": 0, "workflow": api_graph("original")}]})
        spec = workflows.model("copy")
        first = workflows.api_graph(spec, 0)
        first["1"]["inputs"]["text"] = "mutated by the run"
        second = workflows.api_graph(spec, 0)
        self.assertEqual(second["1"]["inputs"]["text"], "original")

    def test_variant_without_its_reference_count_says_so(self):
        self.write("text-only", {"variants": [{"refs": 0, "workflow": api_graph()}]})
        spec = workflows.model("text-only")
        with self.assertRaises(workflows.WorkflowError) as caught:
            spec.graph_for(1)
        self.assertIn("no variant for 1 reference", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
