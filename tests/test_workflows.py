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

SIZES = {"square": [1328, 1328], "portrait": [1140, 1472], "landscape": [1472, 1140]}


def api_graph(text: str = "hello") -> dict:
    return {"1": {"class_type": "CLIPTextEncode", "inputs": {"text": text}}}


def sampler_graph(steps: int) -> dict:
    return {"2": {"class_type": "KSampler", "inputs": {"seed": 1, "steps": steps}}}


def loader_graph(slots: int) -> dict:
    """One encoder fed by `slots` numbered reference loaders."""
    graph = {"1": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["2", 0]}}}
    for number in range(1, slots + 1):
        node_id = str(100 + number)
        graph[node_id] = {"class_type": "LoadImage",
                          "inputs": {"image": f"PARAM_STRING_INPUT_IMAGE_FILENAME_{number}"}}
        graph["1"]["inputs"][f"images.image_{number}"] = [node_id, 0]
    return graph


def model_data(**extra) -> dict:
    """One readable catalogue file, with the two fields every model needs."""
    data = {"sizes": SIZES, "variants": [{"refs": 0, "workflow": api_graph()}]}
    data.update(extra)
    return data


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

    def error_of(self, code: str) -> str:
        entry = next(item for item in workflows.entries() if item.code == code)
        self.assertIsNotNone(entry.error, f"{code} was expected to be unusable")
        return entry.error or ""

    def test_reads_variants_sizes_and_the_default_file(self):
        self.write(
            "my-model",
            {
                "label": "My Model",
                "sizes": SIZES,
                "orientation": "portrait",
                "model_file": "my_derivative.safetensors",
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
        self.assertEqual(spec.size_for("portrait"), (1140, 1472))
        self.assertEqual(spec.size_for(None), (1140, 1472), "the file's orientation is the default")
        self.assertEqual(spec.model_file, "my_derivative.safetensors")
        described = spec.describe()
        self.assertEqual(described["label"], "My Model")
        self.assertEqual(described["sizes"], {"square": [1328, 1328], "portrait": [1140, 1472], "landscape": [1472, 1140]})
        self.assertEqual(described["orientation"], "portrait")
        self.assertEqual(described["model_file"], "my_derivative.safetensors")

    def test_square_is_the_default_orientation_when_the_file_names_none(self):
        self.write("plain", model_data())
        spec = workflows.model("plain")
        self.assertEqual(spec.size_for(None), (1328, 1328))
        self.assertIsNone(spec.model_file, "a file without model_file offers no choice of file")

    def test_the_first_declared_size_is_the_default_when_there_is_no_square(self):
        self.write("tall", model_data(sizes={"landscape": [1472, 1140]}))
        self.assertEqual(workflows.model("tall").size_for(None), (1472, 1140))

    def test_an_unknown_orientation_is_said_with_the_ones_that_exist(self):
        self.write("plain", model_data())
        spec = workflows.model("plain")
        with self.assertRaises(workflows.WorkflowError) as caught:
            spec.size_for("diagonal")
        message = str(caught.exception)
        self.assertIn("no diagonal size", message)
        self.assertIn("square 1328x1328", message)

    def test_sizes_are_required(self):
        self.write("sizeless", {"variants": [{"refs": 0, "workflow": api_graph()}]})
        self.assertIn("'sizes' must be an object", self.error_of("sizeless"))

    def test_an_orientation_nobody_draws_is_an_error(self):
        self.write("odd", model_data(sizes={"diagonal": [1000, 1000]}))
        self.assertIn("diagonal", self.error_of("odd"))

    def test_a_size_that_is_not_two_pixels_is_an_error(self):
        self.write("bad", model_data(sizes={"square": [1328]}))
        self.assertIn("sizes.square", self.error_of("bad"))
        self.write("worse", model_data(sizes={"square": [0, 0]}))
        self.assertIn("sizes.square", self.error_of("worse"))

    def test_an_orientation_the_sizes_do_not_define_is_an_error(self):
        self.write("lying", model_data(sizes={"square": [1328, 1328]}, orientation="portrait"))
        self.assertIn("does not define", self.error_of("lying"))

    def test_a_steps_range_is_read_and_published(self):
        self.write("stepped", model_data(steps={"min": 4, "max": 50},
                                         variants=[{"refs": 0, "workflow": sampler_graph(25)}]))
        spec = workflows.model("stepped")
        self.assertEqual(spec.steps_range, (4, 50))
        self.assertEqual(spec.describe()["steps"], {"min": 4, "max": 50})
        self.assertEqual(workflows.steps_default(spec), 25, "the graph's own number is the default")

    def test_a_fixed_step_count_is_min_equal_max(self):
        self.write("distilled", model_data(steps={"min": 4, "max": 4},
                                           variants=[{"refs": 0, "workflow": sampler_graph(4)}]))
        spec = workflows.model("distilled")
        self.assertEqual(spec.steps_range, (4, 4))
        self.assertEqual(workflows.steps_default(spec), 4)

    def test_a_model_without_steps_offers_no_choice(self):
        self.write("plain", model_data())
        self.assertIsNone(workflows.model("plain").steps_range)
        self.assertNotIn("steps", workflows.model("plain").describe())

    def test_a_malformed_steps_object_is_an_error(self):
        for broken in ({"min": 4}, {"min": 4, "max": 50, "default": 25},
                       {"min": 0, "max": 50}, {"min": 50, "max": 4}, {"min": "4", "max": "50"}, [4, 50]):
            with self.subTest(broken=broken):
                self.write("broken-steps", model_data(steps=broken))
                self.assertIn("'steps' must be", self.error_of("broken-steps"))

    def test_steps_default_is_none_when_the_variants_disagree(self):
        self.write("mixed", model_data(
            steps={"min": 4, "max": 50},
            variants=[{"refs": 0, "workflow": sampler_graph(25)},
                      {"refs": 1, "workflow": sampler_graph(8)}],
        ))
        self.assertIsNone(workflows.steps_default(workflows.model("mixed")))

    def test_model_codes_are_sorted(self):
        self.write("b-model", model_data())
        self.write("a-model", model_data())
        self.assertEqual([spec.code for spec in workflows.models()], ["a-model", "b-model"])

    def test_broken_file_is_reported_not_raised(self):
        self.write("broken", "{ not json")
        self.assertIsNone(workflows.model("broken"))
        self.assertIn("not valid JSON", self.error_of("broken"))

    def test_missing_variants_is_an_error(self):
        self.write("empty", model_data(variants=[]))
        self.assertIn("non-empty list", self.error_of("empty"))

    def test_duplicate_reference_count_is_an_error(self):
        self.write("twice", model_data(variants=[{"refs": 1, "workflow": api_graph()}, {"refs": 1, "workflow": api_graph()}]))
        self.assertIn("two variants", self.error_of("twice"))

    def test_a_graph_that_is_not_a_graph_is_an_error(self):
        self.write("nograph", model_data(variants=[{"refs": 0, "workflow": {"note": "nothing here"}}]))
        self.assertIn("neither", self.error_of("nograph"))

    def test_unknown_and_pathish_codes_are_refused(self):
        self.write("plain", model_data())
        self.assertIsNone(workflows.model("nope"))
        self.assertIsNone(workflows.model("../config"))
        self.assertIsNone(workflows.model("a/b"))

    def test_api_graph_hands_out_a_copy(self):
        self.write("copy", model_data(variants=[{"refs": 0, "workflow": api_graph("original")}]))
        spec = workflows.model("copy")
        first = workflows.api_graph(spec, 0)
        first["1"]["inputs"]["text"] = "mutated by the run"
        second = workflows.api_graph(spec, 0)
        self.assertEqual(second["1"]["inputs"]["text"], "original")

    def test_variant_without_its_reference_count_says_so(self):
        self.write("text-only", model_data())
        spec = workflows.model("text-only")
        with self.assertRaises(workflows.WorkflowError) as caught:
            spec.graph_for(1)
        self.assertIn("no variant for 1 reference", str(caught.exception))


class ReferenceRangeTest(unittest.TestCase):
    """A graph wired for the maximum serves every count below it: the run keeps
    the first slots and the rest is taken out, so one graph replaces ten."""

    setUp = CatalogueTest.setUp
    write = CatalogueTest.write
    error_of = CatalogueTest.error_of

    def write_wide(self, code: str = "wide", low: int = 1, high: int = 3, slots: int = 3) -> Path:
        return self.write(code, model_data(variants=[
            {"refs": 0, "workflow": api_graph()},
            {"refs": {"min": low, "max": high}, "workflow": loader_graph(slots)},
        ]))

    def test_a_range_serves_every_count_in_between(self):
        self.write_wide()
        spec = workflows.model("wide")
        self.assertEqual(spec.refs, (0, 1, 2, 3), "/sillytalk/models lists every count, not the range")
        self.assertEqual(spec.max_refs, 3)
        self.assertEqual(spec.describe()["refs"], [0, 1, 2, 3])

    def test_the_run_gets_only_the_slots_it_has_references_for(self):
        self.write_wide()
        spec = workflows.model("wide")
        one = workflows.api_graph(spec, 1)
        self.assertEqual(sorted(one["1"]["inputs"]), ["clip", "images.image_1"])
        three = workflows.api_graph(spec, 3)
        self.assertEqual(len([key for key in three["1"]["inputs"] if key.startswith("images.")]), 3)

    def test_a_range_that_promises_more_than_the_graph_wires_is_refused(self):
        self.write_wide("liar", low=1, high=5, slots=2)
        message = self.error_of("liar")
        self.assertIn("serves 1..5 reference images", message)
        self.assertIn("wires slots [1, 2]", message)

    def test_a_range_with_a_gap_in_the_numbering_is_refused(self):
        graph = loader_graph(2)
        graph["102"]["inputs"]["image"] = "PARAM_STRING_INPUT_IMAGE_FILENAME_3"
        self.write("gap", model_data(variants=[{"refs": {"min": 1, "max": 3}, "workflow": graph}]))
        self.assertIn("wires slots [1, 3]", self.error_of("gap"))

    def test_two_variants_that_claim_the_same_count_are_refused(self):
        self.write("overlap", model_data(variants=[
            {"refs": 1, "workflow": loader_graph(1)},
            {"refs": {"min": 1, "max": 2}, "workflow": loader_graph(2)},
        ]))
        self.assertIn("two variants for 1 reference image(s)", self.error_of("overlap"))

    def test_a_malformed_refs_is_an_error(self):
        for broken in ({"min": 1}, {"min": 1, "max": 3, "step": 1}, {"min": 3, "max": 1},
                       {"min": "1", "max": "3"}, -1, True, [1, 2]):
            with self.subTest(broken=broken):
                self.write("broken-refs", model_data(variants=[{"refs": broken, "workflow": api_graph()}]))
                self.assertIn("refs must be", self.error_of("broken-refs"))


if __name__ == "__main__":
    unittest.main()
