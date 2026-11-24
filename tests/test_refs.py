"""The reference slots a workflow was wired for, cut down to what a run sent."""

from __future__ import annotations

import unittest

from support import PLUGIN_DIR  # noqa: F401  (puts the plugin on sys.path)

from sillytalk_plugin import refs


def graph(loaders: int = 3, other: dict | None = None) -> dict:
    """One encoder fed by `loaders` reference loaders, numbered from one."""
    out = {"1": {"class_type": "TextEncodeQwenImage21", "inputs": {"clip": ["2", 0]}}}
    for number in range(1, loaders + 1):
        node_id = str(100 + number)
        out[node_id] = {"class_type": "LoadImage",
                        "inputs": {"image": f"PARAM_STRING_INPUT_IMAGE_FILENAME_{number}"}}
        out["1"]["inputs"][f"images.image_{number}"] = [node_id, 0]
    if other:
        out.update(other)
    return out


class SlotsTest(unittest.TestCase):
    def test_slots_come_back_in_number_order_with_their_numbers(self):
        found = refs.slots(graph(3))
        self.assertEqual([(number, node_id) for number, node_id, _node in found],
                         [(1, "101"), (2, "102"), (3, "103")])

    def test_a_graph_without_references_has_no_slots(self):
        self.assertEqual(refs.slots({"1": {"class_type": "KSampler", "inputs": {"steps": 4}}}), [])

    def test_a_loader_without_the_placeholder_is_not_a_slot(self):
        graph_ = graph(1)
        graph_["9"] = {"class_type": "LoadImage", "inputs": {"image": "someone-elses.png"}}
        self.assertEqual([number for number, _id, _n in refs.slots(graph_)], [1])


class TrimTest(unittest.TestCase):
    def test_the_slots_above_the_run_are_removed_with_their_inputs(self):
        graph_ = graph(3)
        removed = refs.trim(graph_, 1)
        self.assertEqual(removed, [2, 3])
        self.assertEqual(sorted(graph_), ["1", "101"], "the unused loaders are gone, the encoder stays")
        self.assertEqual(sorted(graph_["1"]["inputs"]), ["clip", "images.image_1"],
                         "the encoder no longer points at a node that is not there")

    def test_removal_goes_from_the_top_so_the_inputs_stay_contiguous(self):
        graph_ = graph(4)
        refs.trim(graph_, 2)
        self.assertEqual([key for key in graph_["1"]["inputs"] if key.startswith("images.")],
                         ["images.image_1", "images.image_2"],
                         "a gap in the numbering would make the encoder skip or refuse")

    def test_trimming_to_the_wired_maximum_changes_nothing(self):
        graph_ = graph(3)
        before = repr(graph_)
        self.assertEqual(refs.trim(graph_, 3), [])
        self.assertEqual(repr(graph_), before)

    def test_a_run_that_sends_nothing_leaves_no_loader(self):
        graph_ = graph(2)
        self.assertEqual(refs.trim(graph_, 0), [1, 2])
        self.assertEqual(sorted(graph_), ["1"])

    def test_a_link_from_somewhere_else_to_a_removed_slot_is_dropped_too(self):
        graph_ = graph(2)
        graph_["2"] = {"class_type": "PreviewImage", "inputs": {"images": ["102", 0]}}
        refs.trim(graph_, 1)
        self.assertEqual(sorted(graph_["2"]["inputs"]), [], "nothing may point at a removed node")


if __name__ == "__main__":
    unittest.main()
