"""The generate endpoint as a request: what it accepts, and what a refused one
leaves behind on the server.

The handlers are called directly with a stand-in request rather than through an
aiohttp test server — the parsing and the cleanup are what needs covering, and
neither needs a socket.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

from support import TempDirTestCase

from sillytalk_plugin import jobs, routes

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

    async def post(self, *parts, start=None, spec=SimpleNamespace(width=1328, height=768)):
        starter = start if start is not None else mock.AsyncMock(return_value=FakeJob())
        with mock.patch.object(routes, "load_model", return_value=spec), mock.patch.object(jobs, "start", starter):
            response = await routes.generate(FakeRequest(*parts))
        return response

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
        """The cap is hit on the ninth picture; the eight already written have
        no job to own them, so the request itself must remove them."""
        parts = [field("model", "m"), field("prompt", "an apple")] + [picture(f"r{i}.png") for i in range(9)]
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


class NamingTest(unittest.TestCase):
    def test_a_path_in_a_filename_cannot_escape_input(self):
        from sillytalk_plugin.routes import _safe_name

        name = _safe_name("../../etc/passwd.png", 1, "deadbeef")
        self.assertEqual(name, "sillytalk_deadbeef_1_passwd.png")


if __name__ == "__main__":
    import unittest

    unittest.main()
