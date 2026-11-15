"""The GPU bin packing: heaviest model to the emptiest card, and what happens
when no card can take it."""

from __future__ import annotations

import unittest
from unittest import mock

from support import TempDirTestCase, register_model_folder

from sillytalk_plugin import placement
from sillytalk_plugin.placement import Device, PlacementError

GIB = 1024 * 1024 * 1024


def unet_node(size_label: str, device: str = "cuda:0") -> dict:
    return {"class_type": "UNETLoaderMultiGPU", "inputs": {"unet_name": size_label, "device": device}}


def graph_of(**nodes: dict) -> dict:
    return {node_id: node for node_id, node in nodes.items()}


class PlacementTest(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.models = self.root / "models"
        self.models.mkdir()
        register_model_folder("diffusion_models", self.models)

    def model_file(self, name: str, size: int) -> str:
        path = self.models / name
        with path.open("wb") as handle:
            handle.truncate(size)
        return name

    def devices(self, *specs: tuple[int, int]) -> list[Device]:
        return [
            Device(id=f"cuda:{index}", name=f"Card {index}", total=total, free=free)
            for index, (free, total) in enumerate(specs)
        ]

    def place(self, graph: dict, devices: list[Device], headroom: int = 0) -> placement.Placement:
        with mock.patch.object(placement, "visible_devices", return_value=devices):
            return placement.place(graph, headroom=headroom)

    def test_heaviest_goes_to_the_emptiest_card(self):
        self.model_file("big.safetensors", 20 * GIB)
        self.model_file("small.safetensors", 1 * GIB)
        graph = graph_of(
            **{"1": unet_node("small.safetensors"), "2": unet_node("big.safetensors")}
        )
        result = self.place(graph, self.devices((6 * GIB, 24 * GIB), (22 * GIB, 24 * GIB)))
        self.assertEqual(graph["2"]["inputs"]["device"], "cuda:1", "the 20 GiB model takes the 22 GiB card")
        self.assertEqual(graph["1"]["inputs"]["device"], "cuda:0", "the small one takes what is left elsewhere")
        self.assertEqual([a.node for a in result.assignments], ["1", "2"], "the report is ordered by node id")

    def test_a_second_card_is_used_when_the_first_fills_up(self):
        self.model_file("a.safetensors", 10 * GIB)
        self.model_file("b.safetensors", 10 * GIB)
        graph = graph_of(**{"1": unet_node("a.safetensors"), "2": unet_node("b.safetensors")})
        self.place(graph, self.devices((15 * GIB, 24 * GIB), (20 * GIB, 24 * GIB)))
        chosen = {graph["1"]["inputs"]["device"], graph["2"]["inputs"]["device"]}
        self.assertEqual(chosen, {"cuda:1", "cuda:0"}, "10+10 GiB does not fit one 15 GiB card")

    def test_the_largest_model_carries_the_activation_headroom(self):
        self.model_file("only.safetensors", 10 * GIB)
        graph = graph_of(**{"1": unet_node("only.safetensors")})
        # 10 GiB of weights, 11 GiB free, 2 GiB of headroom asked for.
        with self.assertRaises(PlacementError) as caught:
            self.place(graph, self.devices((11 * GIB, 24 * GIB)), headroom=2 * GIB)
        self.assertIn("activation headroom", str(caught.exception))
        self.assertIn("12288 MiB", str(caught.exception))

    def test_a_file_already_placed_costs_nothing_more(self):
        self.model_file("shared.safetensors", 10 * GIB)
        graph = graph_of(**{"1": unet_node("shared.safetensors"), "2": unet_node("shared.safetensors")})
        result = self.place(graph, self.devices((12 * GIB, 24 * GIB), (23 * GIB, 24 * GIB)))
        self.assertEqual(graph["1"]["inputs"]["device"], graph["2"]["inputs"]["device"])
        second = next(a for a in result.assignments if a.node == "2")
        self.assertEqual(second.weight_bytes, 0)
        self.assertIn("already placed", second.note)

    def test_nothing_fits_says_the_numbers(self):
        self.model_file("huge.safetensors", 22 * GIB)
        graph = graph_of(**{"1": unet_node("huge.safetensors")})
        with self.assertRaises(PlacementError) as caught:
            self.place(graph, self.devices((4 * GIB, 24 * GIB), (6 * GIB, 24 * GIB)))
        message = str(caught.exception)
        self.assertIn("4096 MiB available", message)
        self.assertIn("6144 MiB available", message)
        self.assertIn("SILLYTALK_DEVICES", message)

    def test_the_servers_own_cache_counts_as_room(self):
        """A card full of ComfyUI's own weights is not a card full of somebody
        else's: the server can drop them, so the next run does fit."""
        self.model_file("big.safetensors", 18 * GIB)
        graph = graph_of(**{"1": unet_node("big.safetensors")})
        busy = Device(id="cuda:0", name="Card 0", total=24 * GIB, free=1 * GIB, reclaimable=20 * GIB)
        held_by_others = Device(id="cuda:1", name="Card 1", total=24 * GIB, free=1 * GIB)
        self.place(graph, [busy, held_by_others])
        self.assertEqual(graph["1"]["inputs"]["device"], "cuda:0")

    def test_a_resident_model_stays_on_the_card_that_holds_it(self):
        self.model_file("big.safetensors", 18 * GIB)
        graph = graph_of(**{"1": unet_node("big.safetensors")})
        resident = {str(self.models / "big.safetensors"): ("cuda:1", 18 * GIB)}
        with mock.patch.object(placement, "resident_models", return_value=({"cuda:1": 18 * GIB}, resident)):
            result = self.place(graph, self.devices((22 * GIB, 24 * GIB), (5 * GIB, 24 * GIB)))
        self.assertEqual(graph["1"]["inputs"]["device"], "cuda:1", "reloading 18 GiB to save an empty card is silly")
        self.assertEqual(result.assignments[0].weight_bytes, 0)
        self.assertIn("already resident", result.assignments[0].note)

    def test_models_on_mixed_cards_are_not_half_followed(self):
        register_model_folder("text_encoders", self.models)
        self.model_file("a.safetensors", 2 * GIB)
        self.model_file("b.safetensors", 3 * GIB)
        graph = graph_of(**{"1": {"class_type": "DualCLIPLoaderMultiGPU",
                                  "inputs": {"clip_name1": "a.safetensors", "clip_name2": "b.safetensors",
                                             "device": "cuda:0"}}})
        resident = {
            str(self.models / "a.safetensors"): ("cuda:0", 2 * GIB),
            str(self.models / "b.safetensors"): ("cuda:1", 3 * GIB),
        }
        with mock.patch.object(placement, "resident_models", return_value=({}, resident)):
            self.place(graph, self.devices((22 * GIB, 24 * GIB), (21 * GIB, 24 * GIB)))
        self.assertEqual(graph["1"]["inputs"]["device"], "cuda:0", "both files go to the emptiest card together")

    def test_a_server_without_a_gpu_leaves_the_workflow_alone(self):
        graph = graph_of(**{"1": unet_node("whatever.safetensors")})
        with mock.patch.object(placement, "visible_devices", return_value=[]):
            result = placement.place(graph)
        self.assertEqual(graph["1"]["inputs"]["device"], "cuda:0")
        self.assertIn("no CUDA device", result.note)

    def test_a_graph_without_device_inputs_is_not_a_problem(self):
        graph = graph_of(**{"1": {"class_type": "KSampler", "inputs": {"seed": 1}}})
        result = self.place(graph, self.devices((20 * GIB, 24 * GIB)))
        self.assertEqual(result.assignments, [])
        self.assertIn("no device inputs", result.note)

    def test_allowed_devices_filter(self):
        with mock.patch.dict(placement.os.environ, {"SILLYTALK_DEVICES": "cuda:1,cuda:2"}):
            self.assertEqual(placement.allowed_devices(), {"cuda:1", "cuda:2"})
        with mock.patch.dict(placement.os.environ, {}, clear=True):
            self.assertIsNone(placement.allowed_devices())

    def test_headroom_reads_its_environment_variable(self):
        with mock.patch.dict(placement.os.environ, {"SILLYTALK_VRAM_HEADROOM_GB": "0.5"}):
            self.assertEqual(placement.headroom_bytes(), int(0.5 * GIB))
        with mock.patch.dict(placement.os.environ, {"SILLYTALK_VRAM_HEADROOM_GB": "wide"}):
            self.assertEqual(placement.headroom_bytes(), int(placement.DEFAULT_HEADROOM_GB * GIB))


class ResidentIdentityTest(unittest.TestCase):
    """Which attribute of a loaded model names the file it came from."""

    class Patcher:
        def __init__(self, model):
            self.model = model

    def test_cached_patcher_init_on_the_patcher_is_read(self):
        """A current ComfyUI keeps it on the ModelPatcher, which is the object
        that rebuilds itself from it — the spelling that answers on a real server."""
        patcher = self.Patcher(object())
        patcher.cached_patcher_init = (object(), ("/models/unet.safetensors", {"foo": 1}))
        self.assertEqual(placement._patcher_file(patcher), "/models/unet.safetensors")

    def test_cached_patcher_init_is_read(self):
        model = mock.Mock(cached_patcher_init=(object(), ("/models/unet.safetensors", {})), spec=["cached_patcher_init"])
        self.assertEqual(placement._patcher_file(self.Patcher(model)), "/models/unet.safetensors")

    def test_a_list_of_files_yields_its_first(self):
        model = mock.Mock(
            cached_patcher_init=(object(), (["/t/clip_a.safetensors", "/t/clip_b.safetensors"], "/emb", 0, {})),
            spec=["cached_patcher_init"],
        )
        self.assertEqual(placement._patcher_file(self.Patcher(model)), "/t/clip_a.safetensors")

    def test_a_model_without_a_trace_is_none(self):
        class Bare:
            pass

        self.assertIsNone(placement._patcher_file(self.Patcher(Bare())))

    def test_the_warning_property_is_left_alone(self):
        """On a model built from separate files, `model_config.checkpoint_info`
        is a property that logs a complaint about whoever reads it. Answering
        from the patcher must not touch it."""

        class Config:
            @property
            def checkpoint_info(self):
                raise AssertionError("checkpoint_info was read")

        class Model:
            def __init__(self):
                self.model_config = Config()

        patcher = self.Patcher(Model())
        patcher.cached_patcher_init = (object(), ("/models/unet.safetensors", {}))
        self.assertEqual(placement._patcher_file(patcher), "/models/unet.safetensors")

    def test_a_checkpoint_that_holds_its_info_is_still_read(self):
        class Info:
            location = "/models/one.safetensors"

        class Model:
            def __init__(self):
                self.model_config = type("Config", (), {})()
                self.model_config.checkpoint_info = Info()

        self.assertEqual(placement._patcher_file(self.Patcher(Model())), "/models/one.safetensors")


class PlaceableNodeTest(unittest.TestCase):
    def test_device_input_recognised_without_the_registry(self):
        graph = graph_of(
            **{
                "1": {"class_type": "VAELoaderMultiGPU", "inputs": {"vae_name": "v.safetensors", "device": "cuda:2"}},
                "2": {"class_type": "KSampler", "inputs": {"seed": 3}},
                "3": {"class_type": "Something", "inputs": {"device": "not a device"}},
            }
        )
        found = [node_id for node_id, _ in placement.placeable_nodes(graph)]
        self.assertEqual(found, ["1"])

    def test_a_placeholder_device_counts_as_placeable(self):
        graph = graph_of(**{"1": {"class_type": "X", "inputs": {"device": "PARAM_DEVICE"}}})
        self.assertEqual([node_id for node_id, _ in placement.placeable_nodes(graph)], ["1"])


if __name__ == "__main__":
    unittest.main()
