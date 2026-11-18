"""The job record: the values a run writes into its graph, what the plugin does
with a placeholder it cannot answer, and when a file stops being kept."""

from __future__ import annotations

import os
import time
import unittest
from pathlib import Path
from unittest import mock

from support import TempDirTestCase, register_model_folder

from sillytalk_plugin import jobs, model_files, workflows
from sillytalk_plugin.workflows import ModelSpec

SIZES = (("square", (1328, 1328)), ("portrait", (1140, 1472)))


def spec_with(graph: dict, code: str = "test-model", root: Path | None = None, **extra) -> ModelSpec:
    path = (root or Path(".")) / f"{code}.json"
    return ModelSpec(
        code=code,
        file=path,
        label=code,
        sizes=SIZES,
        default_orientation="square",
        graphs=((0, graph),),
        **extra,
    )


def lora_chain(start: int = 20, slots: int = 3) -> dict:
    """A model loader feeding `slots` PARAM_LORA_n slots in a chain."""
    graph = {
        "1": {"class_type": "UNETLoaderMultiGPU", "inputs": {"unet_name": "PARAM_MODEL_FILE", "device": "cuda:0"}},
        "2": {"class_type": "KSampler", "inputs": {"model": [str(start + slots - 1), 0]}},
    }
    for index in range(slots):
        node_id = str(start + index)
        source = "1" if index == 0 else str(start + index - 1)
        graph[node_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {"model": [source, 0], "lora_name": f"PARAM_LORA_{index + 1}", "strength_model": 1.0},
        }
    return graph


class BuildGraphTest(unittest.TestCase):
    def build(self, graph: dict, refs: list[Path], prompt: str = "an apple", model_file: str | None = None,
              loras: list[dict] | None = None) -> dict:
        spec = spec_with(graph, model_file=model_file)
        job = jobs.Job(id="11111111-2222-3333-4444-555555555555", model=spec.code, width=1328, height=768, seed=42)
        job.refs = refs
        with mock.patch.object(workflows, "api_graph", return_value=graph):
            out, _run = jobs._build_graph(spec, job, prompt, model_file, loras or [])
        return out

    def test_the_run_values_land_in_the_graph(self):
        graph = {
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "PARAM_PROMPT"}},
            "9": {"class_type": "EmptySD3LatentImage", "inputs": {"width": "PARAM_INT_OUTPUT_IMAGE_WIDTH",
                                                                  "height": "PARAM_INT_OUTPUT_IMAGE_HEIGHT"}},
            "10": {"class_type": "KSampler", "inputs": {"seed": "PARAM_SEED"}},
            "13": {"class_type": "SaveImage", "inputs": {"filename_prefix": "PARAM_STRING_OUTPUT_IMAGE_FILENAME"}},
        }
        out = self.build(graph, [])
        self.assertEqual(out["5"]["inputs"]["text"], "an apple")
        self.assertEqual(out["9"]["inputs"]["width"], 1328)
        self.assertEqual(out["9"]["inputs"]["height"], 768)
        self.assertEqual(out["10"]["inputs"]["seed"], 42)
        self.assertEqual(out["13"]["inputs"]["filename_prefix"], "sillytalk/11111111")

    def test_one_reference_is_unnumbered(self):
        graph = {"7": {"class_type": "LoadImage", "inputs": {"image": "PARAM_STRING_INPUT_IMAGE_FILENAME"}}}
        one = self.build(graph, [Path("input/sillytalk_ab_apple.png")])
        self.assertEqual(one["7"]["inputs"]["image"], "sillytalk_ab_apple.png")

    def test_two_references_are_numbered_from_one(self):
        graph = {
            "8": {"class_type": "LoadImage", "inputs": {"image": "PARAM_STRING_INPUT_IMAGE_FILENAME_1"}},
            "9": {"class_type": "LoadImage", "inputs": {"image": "PARAM_STRING_INPUT_IMAGE_FILENAME_2"}},
        }
        two = self.build(graph, [Path("input/first.png"), Path("input/second.png")])
        self.assertEqual(two["8"]["inputs"]["image"], "first.png")
        self.assertEqual(two["9"]["inputs"]["image"], "second.png")

    def test_a_placeholder_nobody_answers_is_named(self):
        graph = {"1": {"class_type": "X", "inputs": {"text": "PARAM_MOONLIGHT"}}}
        with self.assertRaises(jobs.JobError) as caught:
            self.build(graph, [])
        self.assertIn("PARAM_MOONLIGHT", str(caught.exception))

    def test_a_leftover_device_placeholder_is_tolerated(self):
        graph = {"1": {"class_type": "UNETLoaderMultiGPU", "inputs": {"device": "PARAM_DEVICE"}}}
        out = self.build(graph, [])
        self.assertEqual(out["1"]["inputs"]["device"], "PARAM_DEVICE", "placement overwrites it next")

    def test_the_chosen_model_file_lands_in_the_loader(self):
        graph = {"1": {"class_type": "UNETLoaderMultiGPU", "inputs": {"unet_name": "PARAM_MODEL_FILE"}}}
        out = self.build(graph, [], model_file="my_derivative.safetensors")
        self.assertEqual(out["1"]["inputs"]["unet_name"], "my_derivative.safetensors")


class PickTest(TempDirTestCase):
    """A name the server does not have is answered with the names it does, so
    the client that offered a pick of its own still says what went wrong."""

    def setUp(self):
        super().setUp()
        self.models = self.root / "diffusion_models"
        self.loras = self.root / "loras"
        for folder in (self.models, self.loras):
            folder.mkdir()
        (self.models / "qwen_image_2.1_int8_convrot.safetensors").write_bytes(b"x" * 10)
        (self.loras / "style.safetensors").write_bytes(b"y" * 100)
        register_model_folder("diffusion_models", self.models)
        register_model_folder("loras", self.loras)
        model_files.forget_cache()
        self.addCleanup(model_files.forget_cache)

    def test_no_request_keeps_the_workflows_own_file(self):
        spec = spec_with({}, model_file="qwen_image_2.1_int8_convrot.safetensors")
        self.assertEqual(jobs._pick_model_file(spec, None), "qwen_image_2.1_int8_convrot.safetensors")
        self.assertEqual(jobs._pick_model_file(spec, "   "), "qwen_image_2.1_int8_convrot.safetensors")

    def test_a_file_the_server_has_is_accepted(self):
        spec = spec_with({}, model_file="qwen_image_2.1_int8_convrot.safetensors")
        self.assertEqual(jobs._pick_model_file(spec, "qwen_image_2.1_int8_convrot.safetensors"),
                         "qwen_image_2.1_int8_convrot.safetensors")

    def test_a_file_that_is_not_there_lists_the_ones_that_are(self):
        spec = spec_with({}, model_file="qwen_image_2.1_int8_convrot.safetensors")
        with self.assertRaises(jobs.JobError) as caught:
            jobs._pick_model_file(spec, "nope.safetensors")
        message = str(caught.exception)
        self.assertIn("no 'nope.safetensors'", message)
        self.assertIn("qwen_image_2.1_int8_convrot.safetensors", message)

    def test_a_workflow_without_the_placeholder_refuses_the_choice(self):
        spec = spec_with({}, model_file=None)
        with mock.patch.object(jobs, "_placeholder_names", return_value=set()):
            with self.assertRaises(jobs.JobError) as caught:
                jobs._pick_model_file(spec, "anything.safetensors")
        self.assertIn("no swappable model file", str(caught.exception))

    def test_loras_keep_their_order_and_strength(self):
        (self.loras / "second.safetensors").write_bytes(b"z" * 10)
        chosen = jobs._pick_loras([{"name": "second.safetensors", "strength": 0.6}, {"name": "style.safetensors"}])
        self.assertEqual([entry["name"] for entry in chosen], ["second.safetensors", "style.safetensors"])
        self.assertEqual(chosen[0]["strength"], 0.6)
        self.assertEqual(chosen[1]["strength"], 1.0, "an unstated strength is a full one")

    def test_an_unknown_lora_lists_the_ones_that_are(self):
        with self.assertRaises(jobs.JobError) as caught:
            jobs._pick_loras([{"name": "ghost.safetensors"}])
        message = str(caught.exception)
        self.assertIn("no 'ghost.safetensors'", message)
        self.assertIn("style.safetensors", message)

    def test_a_lora_without_a_name_is_refused(self):
        with self.assertRaises(jobs.JobError) as caught:
            jobs._pick_loras([{"strength": 0.5}])
        self.assertIn("needs a 'name'", str(caught.exception))


class ErrorTextTest(unittest.TestCase):
    def test_the_execution_error_message_is_what_comes_back(self):
        entry = {
            "status": {
                "status_str": "error",
                "completed": True,
                "messages": [["execution_error", {"exception_message": "CUDA out of memory",
                                                  "node_id": "1", "node_type": "KSampler"}]],
            }
        }
        self.assertEqual(jobs._entry_error(entry), "CUDA out of memory (node 1 KSampler)")

    def test_an_error_without_a_message_still_says_something(self):
        self.assertEqual(jobs._entry_error({"status": {"status_str": "error"}}),
                         "the server reported an error without saying what")

    def test_a_validate_prompt_error_dict(self):
        self.assertEqual(jobs._error_text({"type": "no_prompt", "message": "No prompt provided"}), "No prompt provided")


class RetentionTest(TempDirTestCase):
    def test_a_settled_job_loses_its_picture_after_the_window(self):
        image = self.out / "sillytalk" / "abcd1234_00001.png"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"png")
        job = jobs.Job(id="abcd1234", model="m", width=8, height=8, seed=1, status=jobs.SUCCEEDED, image=image)
        job.settled = time.time() - jobs.RETENTION_SECONDS - 1
        jobs._jobs[job.id] = job
        self.addCleanup(jobs._jobs.clear)

        jobs.sweep()
        self.assertFalse(image.exists(), "the server does not keep somebody else's picture")

    def test_a_fresh_picture_survives_a_sweep(self):
        image = self.out / "sillytalk" / "beef0001_00001.png"
        image.parent.mkdir(parents=True)
        image.write_bytes(b"png")
        job = jobs.Job(id="beef0001", model="m", width=8, height=8, seed=1, status=jobs.SUCCEEDED, image=image)
        job.settled = time.time()
        jobs._jobs[job.id] = job
        self.addCleanup(jobs._jobs.clear)

        jobs.sweep()
        self.assertTrue(image.exists())
        self.assertEqual(jobs.image_bytes(job), b"png")

    def test_a_picture_the_server_loses_explains_itself(self):
        job = jobs.Job(id="gone0001", model="m", width=8, height=8, seed=1, status=jobs.SUCCEEDED,
                       image=self.out / "sillytalk" / "missing.png")
        with self.assertRaises(jobs.JobError) as caught:
            jobs.image_bytes(job)
        self.assertIn("gone from the server", str(caught.exception))

    def test_the_references_go_when_the_job_settles(self):
        ref = self.inp / "sillytalk_ref.png"
        ref.write_bytes(b"ref")
        job = jobs.Job(id="ref00001", model="m", width=8, height=8, seed=1, refs=[ref])
        jobs._settle(job, jobs.SUCCEEDED, None)
        self.assertFalse(ref.exists())
        self.assertEqual(job.refs, [])


def stale(path: Path, seconds_ago: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"png")
    when = time.time() - seconds_ago
    os.utime(path, (when, when))
    return path


class OrphanSweepTest(TempDirTestCase):
    """`output/sillytalk/` is this plugin's folder, and a job record lives only
    in this process's memory — so a restart orphans whatever the dead process
    had not handed out, and nothing else would ever clean it."""

    def setUp(self):
        super().setUp()
        jobs._jobs.clear()
        self.addCleanup(jobs._jobs.clear)
        jobs._orphan_scan_at = 0.0
        self.addCleanup(setattr, jobs, "_orphan_scan_at", 0.0)

    def test_a_picture_an_earlier_process_left_is_removed(self):
        stale(self.out / "sillytalk" / "dead0001_00001_.png", jobs.RETENTION_SECONDS + 10)
        jobs.sweep()
        self.assertFalse((self.out / "sillytalk" / "dead0001_00001_.png").exists())

    def test_a_picture_a_job_still_remembers_survives(self):
        path = stale(self.out / "sillytalk" / "live0001_00001_.png", jobs.RETENTION_SECONDS + 10)
        jobs._jobs["live0001-full-id"] = jobs.Job(id="live0001", model="m", width=8, height=8, seed=1,
                                                  status=jobs.SUCCEEDED, image=path)
        jobs.sweep()
        self.assertTrue(path.exists(), "the job itself decides when its picture goes")

    def test_a_young_picture_is_left_alone(self):
        path = self.out / "sillytalk" / "new00001_00001_.png"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"png")
        jobs.sweep()
        self.assertTrue(path.exists())

    def test_somebody_elses_output_is_never_touched(self):
        outside = self.out / "somebody-else.png"
        stale(outside, jobs.RETENTION_SECONDS + 10)
        jobs.sweep()
        self.assertTrue(outside.exists())


class FakeQueue:
    def __init__(self, history: dict | None = None, running: tuple = (), queued: tuple = ()):
        self._history = history or {}
        self._running = [(0, pid, {}, {}, [], {}) for pid in running]
        self._queued = [(0, pid, {}, {}, [], {}) for pid in queued]

    def get_history(self, prompt_id: str) -> dict:
        return self._history

    def get_current_queue_volatile(self) -> tuple[list, list]:
        return self._running, self._queued


class RefreshTest(unittest.TestCase):
    """What the server's own bookkeeping says about a job, and what the job must
    therefore report. An interrupted run is the case that matters: its history
    entry is written with `completed` false, so only the queue can say that it
    is over — and a job that never settles is a client that polls forever."""

    INTERRUPTED = {"p1": {"status": {"status_str": "error", "completed": False}, "outputs": {}}}

    def refresh(self, history=None, running=(), queued=(), **kwargs) -> jobs.Job:
        job = jobs.Job(id="job1", model="m", width=8, height=8, seed=1, prompt_id="p1", **kwargs)
        queue = FakeQueue(history=history, running=running, queued=queued)
        with mock.patch.object(jobs, "_server", return_value=mock.Mock(prompt_queue=queue)):
            jobs.refresh(job)
        return job

    def test_an_interrupted_run_is_cancelled(self):
        job = self.refresh(history=self.INTERRUPTED, cancel_requested=True)
        self.assertEqual(job.status, jobs.CANCELLED)
        self.assertIsNone(job.error)

    def test_a_run_the_server_stopped_on_its_own_is_failed(self):
        job = self.refresh(history=self.INTERRUPTED)
        self.assertEqual(job.status, jobs.FAILED)
        self.assertTrue(job.error)

    def test_a_run_still_executing_stays_running(self):
        job = self.refresh(history=self.INTERRUPTED, running=("p1",), cancel_requested=True)
        self.assertEqual(job.status, jobs.RUNNING)

    def test_a_job_that_never_started_is_cancelled(self):
        job = self.refresh(history={}, cancel_requested=True)
        self.assertEqual(job.status, jobs.CANCELLED)
        self.assertIn("no longer knows this job", job.error)

    def test_a_job_still_in_the_queue_is_not_over(self):
        job = self.refresh(history={}, queued=("p1",))
        self.assertEqual(job.status, jobs.QUEUED)

    def test_a_finished_run_that_wrote_nothing_is_failed(self):
        job = self.refresh(history={"p1": {"status": {"status_str": "success", "completed": True},
                                           "outputs": {"9": {"images": []}}}})
        self.assertEqual(job.status, jobs.FAILED)
        self.assertIn("wrote no picture", job.error)


if __name__ == "__main__":
    unittest.main()
