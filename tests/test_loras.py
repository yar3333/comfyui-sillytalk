"""The LoRA slots: what counts as one, and what turning one off does to the graph.

A slot the run does not use has to leave the graph rather than sit in it at
strength 0 — `execution.validate_prompt` checks `lora_name` against the folder
listing, so a slot that stays would need a file that exists. These tests are
about the rewiring that replaces it.
"""

from __future__ import annotations

import unittest

from support import PLUGIN_DIR  # noqa: F401  (puts the plugin on sys.path)

from sillytalk_plugin import loras


def graph(slots: int = 3, head: str = "1") -> dict:
    """`head -> [slot 1] -> … -> [slot n] -> KSampler "2"`."""
    out = {
        head: {"class_type": "UNETLoaderMultiGPU", "inputs": {"unet_name": "PARAM_MODEL_FILE", "device": "cuda:0"}},
        "2": {"class_type": "KSampler", "inputs": {"model": [str(10 + slots - 1), 0]}},
    }
    for index in range(slots):
        node_id = str(10 + index)
        source = head if index == 0 else str(10 + index - 1)
        out[node_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": [source, 0], "lora_name": f"PARAM_LORA_{index + 1}", "strength_model": 1.0},
        }
    return out


class SlotsTest(unittest.TestCase):
    def test_slots_are_found_and_ordered_by_their_number(self):
        graph_ = graph()
        # Renumber the nodes so the node ids disagree with the slot numbers: the
        # client's list must follow the numbers, which survive a re-save by the UI.
        graph_["z"] = graph_.pop("12")
        graph_["z"]["inputs"]["model"] = ["11", 0]
        graph_["2"]["inputs"]["model"] = ["z", 0]
        found = loras.slots(graph_)
        self.assertEqual([index for index, _id, _node in found], [1, 2, 3])
        self.assertEqual([node_id for _index, node_id, _node in found], ["10", "11", "z"])

    def test_a_loader_without_a_placeholder_is_not_a_slot(self):
        graph_ = graph()
        graph_["10"]["inputs"]["lora_name"] = "Qwen-Image-Lightning-4steps.safetensors"
        self.assertEqual([index for index, _id, _n in loras.slots(graph_)], [2, 3])


class ApplyTest(unittest.TestCase):
    def test_no_loras_leave_the_loader_feeding_the_sampler(self):
        graph_ = graph()
        run = loras.apply(graph_, [])
        self.assertEqual(graph_["2"]["inputs"]["model"], ["1", 0])
        self.assertEqual(sorted(graph_), ["1", "2"])
        self.assertEqual(run.applied, [])
        self.assertEqual(len(run.bypassed), 3)

    def test_two_loras_fill_the_first_slots_and_drop_the_rest(self):
        graph_ = graph()
        run = loras.apply(graph_, [{"name": "a.safetensors"}, {"name": "b.safetensors", "strength": 0.4}])
        self.assertEqual(graph_["10"]["inputs"]["lora_name"], "a.safetensors")
        self.assertEqual(graph_["10"]["inputs"]["strength_model"], 1.0)
        self.assertEqual(graph_["11"]["inputs"]["lora_name"], "b.safetensors")
        self.assertEqual(graph_["11"]["inputs"]["strength_model"], 0.4)
        self.assertEqual(graph_["11"]["inputs"]["model"], ["10", 0], "the chain stays in order")
        self.assertEqual(graph_["2"]["inputs"]["model"], ["11", 0], "the sampler follows the last slot that stayed")
        self.assertNotIn("12", graph_)
        self.assertEqual(run.names, ["a.safetensors@1", "b.safetensors@0.4"])

    def test_the_bytes_of_a_chain_land_on_the_loader_that_carries_it(self):
        graph_ = graph()
        sizes = {"a.safetensors": 100, "b.safetensors": 50}
        run = loras.apply(graph_, [{"name": "a.safetensors"}, {"name": "b.safetensors"}], size_of=sizes.get)
        self.assertEqual(run.extra_weight(), {"1": 150}, "a LoRA costs memory on its model's card")

    def test_a_loader_the_chain_starts_at_a_lora_is_found_through_it(self):
        # The 2512 workflow has a fixed Lightning loader before its slots: the
        # slot's weight belongs to the model, not to the LoRA in front of it.
        graph_ = graph()
        graph_["10"]["inputs"]["lora_name"] = "lightning.safetensors"
        run = loras.apply(graph_, [{"name": "a.safetensors"}], size_of=lambda _n: 70)
        self.assertEqual(run.applied[0].loader, "1")
        self.assertEqual(run.extra_weight(), {"1": 70})

    def test_the_full_loader_does_not_merge_into_the_text_encoder(self):
        graph_ = graph()
        graph_["10"]["class_type"] = "LoraLoader"
        graph_["10"]["inputs"]["strength_clip"] = 1.0
        graph_["10"]["inputs"]["clip"] = ["3", 0]
        loras.apply(graph_, [{"name": "a.safetensors", "strength": 0.8}])
        self.assertEqual(graph_["10"]["inputs"]["strength_clip"], 0.0)
        self.assertEqual(graph_["10"]["inputs"]["strength_model"], 0.8)

    def test_more_loras_than_slots_is_said_with_both_numbers(self):
        graph_ = graph(slots=2)
        with self.assertRaises(loras.LoraError) as caught:
            loras.apply(graph_, [{"name": "a"}, {"name": "b"}, {"name": "c"}])
        self.assertIn("2 LoRA slot(s) and 3 LoRA(s)", str(caught.exception))

    def test_a_workflow_without_slots_says_how_to_give_it_them(self):
        graph_ = {"1": {"class_type": "UNETLoaderMultiGPU", "inputs": {"unet_name": "x.safetensors"}}}
        with self.assertRaises(loras.LoraError) as caught:
            loras.apply(graph_, [{"name": "a"}])
        self.assertIn("no LoRA slots", str(caught.exception))
        self.assertIn("LoraLoaderModelOnly", str(caught.exception))

    def test_a_workflow_without_slots_and_without_a_request_is_fine(self):
        graph_ = {"1": {"class_type": "UNETLoaderMultiGPU", "inputs": {"unet_name": "x.safetensors"}}}
        run = loras.apply(graph_, [])
        self.assertEqual(run.bypassed, [])

    def test_a_slot_nobody_fed_leaves_nothing_hanging(self):
        graph_ = graph(slots=1)
        graph_["2"]["inputs"]["model"] = ["10", 0]
        graph_["10"]["inputs"]["model"] = None
        loras.apply(graph_, [])
        self.assertIsNone(graph_["2"]["inputs"]["model"], "an unconnected input stays unconnected")


if __name__ == "__main__":
    unittest.main()
