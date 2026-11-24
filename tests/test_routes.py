"""The generate endpoint as a request: what it accepts, and what a refused one
leaves behind on the server.

The handlers are called directly with a stand-in request rather than through an
aiohttp test server — the parsing and the cleanup are what needs covering, and
neither needs a socket.
"""

from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest import mock

from support import TempDirTestCase, lora_header, model_header, register_model_folder, write_safetensors

from sillytalk_plugin import jobs, lora_fit, model_files, routes

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 64


class FakePart:
    def __init__(self, name: str, filename: str | None = None, data: bytes = b""):
        self.name = name
        self.filename = filename
        self._data = data
        self._sent = False

    async def text(self) -> str:
        return self._data.decode()

    async def read_chunk(self) -> bytes:
        if self._sent:
            return b""
        self._sent = True
        return self._data

    async def read(self) -> bytes:
        return self._data


class FakeRequest:
    content_type = "multipart/form-data; boundary=x"

    def __init__(self, *parts: FakePart):
        self._parts = list(parts)

    async def multipart(self):
        parts = list(self._parts)

        class Reader:
            async def next(self):
                return parts.pop(0) if parts else None

        return Reader()


def field(name: str, value) -> FakePart:
    return FakePart(name, data=str(value).encode())


def picture(filename: str, data: bytes = PNG) -> FakePart:
    return FakePart("image", filename=filename, data=data)


class FakeJob:
    id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"

    def describe(self) -> dict:
        return {"job_id": self.id, "status": "queued"}


class GenerateTest(TempDirTestCase):
    def left_behind(self) -> list[str]:
        return sorted(path.name for path in self.inp.iterdir())

    async def post(self, *parts, start=None):
        self.starter = start if start is not None else mock.AsyncMock(return_value=FakeJob())
        with mock.patch.object(jobs, "start", self.starter):
            response = await routes.generate(FakeRequest(*parts))
        return response

    def sent(self) -> dict:
        """What the request handed to `jobs.start`, by keyword."""
        _args, kwargs = self.starter.call_args
        return kwargs

    async def test_a_supported_picture_lands_in_input(self):
        response = await self.post(field("model", "m"), field("prompt", "an apple"), picture("cat.png"))
        self.assertEqual(response.status, 202)
        saved = self.left_behind()
        self.assertEqual(len(saved), 1)
        self.assertTrue(saved[0].startswith("sillytalk_") and saved[0].endswith("_1_cat.png"), saved)

    async def test_a_file_that_is_not_a_supported_picture_is_refused(self):
        response = await self.post(field("model", "m"), field("prompt", "an apple"), picture("notes.txt", b"hello"))
        self.assertEqual(response.status, 400)
        self.assertIn("not a supported picture", response.body.decode())
        self.assertEqual(self.left_behind(), [], "a refused upload must not stay on the server")

    async def test_a_picture_without_an_extension_is_refused(self):
        response = await self.post(field("model", "m"), field("prompt", "an apple"), picture("cat"))
        self.assertEqual(response.status, 400)
        self.assertEqual(self.left_behind(), [])

    async def test_a_request_refused_halfway_leaves_its_earlier_pictures_behind_nothing(self):
        """The cap is hit on the last picture; the ones already written have no
        job to own them, so the request itself must remove them."""
        parts = [field("model", "m"), field("prompt", "an apple")]
        parts += [picture(f"r{i}.png") for i in range(routes.MAX_REF_FILES + 1)]
        response = await self.post(*parts)
        self.assertEqual(response.status, 400)
        self.assertIn("reference images", response.body.decode())
        self.assertEqual(self.left_behind(), [])

    async def test_a_job_that_was_refused_by_the_catalogue_leaves_nothing_behind(self):
        async def refuse(*args, **kwargs):
            raise jobs.JobError("no workflow named 'm' in this server's catalogue")

        response = await self.post(field("model", "m"), field("prompt", "an apple"), picture("cat.png"), start=refuse)
        self.assertEqual(response.status, 400)
        self.assertEqual(self.left_behind(), [])

    async def test_a_started_job_keeps_its_references(self):
        """Until the run settles, the graph still has to read them."""
        response = await self.post(field("model", "m"), field("prompt", "an apple"), picture("cat.png"))
        self.assertEqual(response.status, 202)
        self.assertEqual(len(self.left_behind()), 1)

    async def test_the_prompt_length_cap_says_so(self):
        response = await self.post(field("model", "m"), field("prompt", "x" * (routes.MAX_PROMPT_CHARS + 1)))
        self.assertEqual(response.status, 400)
        self.assertIn("longer than", response.body.decode())

    async def test_an_unknown_field_does_not_break_the_request(self):
        response = await self.post(field("model", "m"), field("prompt", "an apple"), FakePart("junk", data=b"12345"))
        self.assertEqual(response.status, 202)

    async def test_a_plain_post_is_refused_before_anything_is_read(self):
        class Plain:
            content_type = "application/json"

            async def multipart(self):
                raise AssertionError("must not be reached")

        response = await routes.generate(Plain())
        self.assertEqual(response.status, 400)

    async def test_the_orientation_the_file_and_the_loras_are_passed_on(self):
        response = await self.post(
            field("model", "qwen-image-2.1"),
            field("prompt", "an apple"),
            field("orientation", "portrait"),
            field("model_file", "my_derivative.safetensors"),
            field("loras", '[{"name":"style.safetensors","strength":0.7}]'),
        )
        self.assertEqual(response.status, 202)
        sent = self.sent()
        self.assertEqual(sent["orientation"], "portrait")
        self.assertEqual(sent["model_file"], "my_derivative.safetensors")
        self.assertEqual(sent["loras"], [{"name": "style.safetensors", "strength": 0.7}])

    async def test_fields_that_were_not_sent_reach_the_job_as_none(self):
        await self.post(field("model", "m"), field("prompt", "an apple"))
        sent = self.sent()
        self.assertIsNone(sent["orientation"], "the model's own default is the job's business, not the route's")
        self.assertIsNone(sent["model_file"])
        self.assertEqual(sent["loras"], [])
        self.assertIsNone(sent["steps"], "the workflow's wired count is the default, so the route sends nothing")

    async def test_the_step_count_is_passed_on_as_a_number(self):
        response = await self.post(field("model", "m"), field("prompt", "an apple"), field("steps", "8"))
        self.assertEqual(response.status, 202)
        self.assertEqual(self.sent()["steps"], 8)

    async def test_steps_that_are_not_a_whole_number_are_refused(self):
        response = await self.post(field("model", "m"), field("prompt", "an apple"), field("steps", "many"))
        self.assertEqual(response.status, 400)
        self.assertIn("whole number", response.body.decode())

    async def test_loras_that_are_not_json_are_refused_with_what_the_field_wants(self):
        response = await self.post(field("model", "m"), field("prompt", "an apple"), field("loras", "style.safetensors"))
        self.assertEqual(response.status, 400)
        self.assertIn("not valid JSON", response.body.decode())

    async def test_a_lora_entry_without_a_name_is_refused(self):
        response = await self.post(field("model", "m"), field("prompt", "an apple"), field("loras", '[{"s":1}]'))
        self.assertEqual(response.status, 400)
        self.assertIn("needs a name", response.body.decode())

    async def test_too_many_loras_are_refused_before_the_job(self):
        many = [{"name": f"l{i}"} for i in range(routes.MAX_LORAS + 1)]
        response = await self.post(field("model", "m"), field("prompt", "an apple"), field("loras", json.dumps(many)))
        self.assertEqual(response.status, 400)
        self.assertIn("no more than", response.body.decode())


def swappable_spec(code: str = "m", model_file: str = "base.safetensors") -> SimpleNamespace:
    """A catalogue entry as `ModelSpec.describe()` returns it, for a workflow whose
    model file the client may change."""
    return SimpleNamespace(
        code=code, label=code.upper(), max_refs=2, refs=[0], model_file=model_file,
        default_orientation="square",
        sizes=(("square", (1328, 1328)), ("portrait", (1140, 1472))),
        describe=lambda: {"code": code, "label": code.upper(), "max_refs": 2, "refs": [0],
                          "orientation": "square", "sizes": {"square": [1328, 1328]},
                          "model_file": model_file},
    )


class CatalogueTest(TempDirTestCase):
    """`/sillytalk/models` is what the settings dialog fills its pick from, so
    the server's own folders have to be in the answer — a client that had to
    type a file name would be the design this plugin exists to remove."""

    def setUp(self):
        super().setUp()
        self.models = self.root / "diffusion_models"
        self.loras = self.root / "loras"
        for folder in (self.models, self.loras):
            folder.mkdir()
        (self.models / "base.safetensors").write_bytes(b"x" * 1024)
        (self.loras / "style.safetensors").write_bytes(b"y" * 1024)
        register_model_folder("diffusion_models", self.models)
        register_model_folder("loras", self.loras)
        model_files.forget_cache()
        lora_fit.forget_cache()
        self.addCleanup(model_files.forget_cache)
        self.addCleanup(lora_fit.forget_cache)

    async def get(self, specs):
        with mock.patch.object(routes, "catalogue_models", return_value=specs), \
                mock.patch.object(routes.loras_module, "available_slots", return_value=5):
            response = await routes.list_models(FakeRequest())
        return json.loads(response.body.decode())

    async def test_the_answer_offers_the_files_of_the_server(self):
        body = await self.get([swappable_spec()])
        model = body["models"][0]
        self.assertEqual([entry["name"] for entry in model["model_files"]["options"]], ["base.safetensors"])
        self.assertEqual([entry["name"] for entry in model["loras"]["options"]], ["style.safetensors"])
        self.assertEqual(model["loras"]["max"], 5)

    async def test_a_model_that_offers_steps_publishes_the_range_and_the_graph_default(self):
        spec = swappable_spec()
        described = spec.describe()
        described["steps"] = {"min": 4, "max": 50}
        spec.describe = lambda: described
        with mock.patch.object(routes, "steps_default", return_value=25):
            body = await self.get([spec])
        self.assertEqual(body["models"][0]["steps"], {"min": 4, "max": 50, "default": 25})

    async def test_a_workflow_without_a_swappable_file_says_there_is_none(self):
        spec = SimpleNamespace(code="m", model_file=None,
                               describe=lambda: {"code": "m", "sizes": {}, "orientation": "square",
                                                 "model_file": None})
        body = await self.get([spec])
        self.assertNotIn("model_files", body["models"][0])
        self.assertNotIn("fits", body["models"][0]["loras"]["options"][0],
                         "with no file to merge into there is nothing to judge")

    async def test_a_lora_says_per_file_whether_it_merges_and_why(self):
        write_safetensors(self.models / "base.safetensors",
                          model_header({"transformer_blocks.0.attn.to_q": (4096, 4096)}))
        write_safetensors(self.models / "other.safetensors",
                          model_header({"transformer_blocks.0.attn.to_q": (3072, 3072)}))
        write_safetensors(self.loras / "style.safetensors",
                          lora_header({"transformer_blocks.0.attn.to_q": (4096, 4096)}))
        model_files.forget_cache()
        lora_fit.forget_cache()
        body = await self.get([swappable_spec()])
        option = body["models"][0]["loras"]["options"][0]
        self.assertEqual(option["fits"]["base.safetensors"]["fits"], "yes")
        self.assertNotIn("reason", option["fits"]["base.safetensors"], "nothing to warn about says nothing")
        self.assertEqual(option["fits"]["other.safetensors"]["fits"], "no")
        self.assertIn("at a size this model does not have", option["fits"]["other.safetensors"]["reason"])

    async def test_the_verdicts_do_not_leak_into_the_cached_folder_listing(self):
        """The listing is shared by every model; writing a verdict into it would give
        them all the first one's answer and outlive the request that made it."""
        write_safetensors(self.models / "base.safetensors",
                          model_header({"transformer_blocks.0.attn.to_q": (4096, 4096)}))
        write_safetensors(self.loras / "style.safetensors",
                          lora_header({"transformer_blocks.0.attn.to_q": (4096, 4096)}))
        model_files.forget_cache()
        lora_fit.forget_cache()
        await self.get([swappable_spec()])
        self.assertNotIn("fits", model_files.listing(model_files.LORA_FOLDER)[0])

    async def test_a_workflow_with_no_slots_says_nothing_about_loras(self):
        write_safetensors(self.models / "base.safetensors",
                          model_header({"transformer_blocks.0.attn.to_q": (4096, 4096)}))
        write_safetensors(self.loras / "style.safetensors",
                          lora_header({"transformer_blocks.0.attn.to_q": (4096, 4096)}))
        model_files.forget_cache()
        lora_fit.forget_cache()
        with mock.patch.object(routes, "catalogue_models", return_value=[swappable_spec()]), \
                mock.patch.object(routes.loras_module, "available_slots", return_value=0):
            body = json.loads((await routes.list_models(FakeRequest())).body.decode())
        self.assertEqual(body["models"][0]["loras"]["max"], 0)
        self.assertNotIn("fits", body["models"][0]["loras"]["options"][0])


class LoraFieldTest(unittest.TestCase):
    def test_a_bare_name_is_a_lora_at_full_strength(self):
        self.assertEqual(routes._loras('["a.safetensors"]'), ([{"name": "a.safetensors"}], ""))

    def test_an_empty_field_is_no_loras(self):
        self.assertEqual(routes._loras(None), ([], ""))
        self.assertEqual(routes._loras("  "), ([], ""))

    def test_an_object_instead_of_an_array_is_refused(self):
        _out, problem = routes._loras('{"name":"a"}')
        self.assertIn("must be a JSON array", problem)

    def test_a_strength_that_is_not_a_number_is_named(self):
        _out, problem = routes._loras('[{"name":"a","strength":"lots"}]')
        self.assertIn("a", problem)
        self.assertIn("not a number", problem)


class NamingTest(unittest.TestCase):
    def test_a_path_in_a_filename_cannot_escape_input(self):
        from sillytalk_plugin.routes import _safe_name

        name = _safe_name("../../etc/passwd.png", 1, "deadbeef")
        self.assertEqual(name, "sillytalk_deadbeef_1_passwd.png")


if __name__ == "__main__":
    import unittest

    unittest.main()
