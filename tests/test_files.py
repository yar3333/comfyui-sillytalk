"""The moved /files/delete and /files/list endpoints.

Their safety rule is the point of them — a client may only ever reach inside
ComfyUI's own output/, input/ or temp/ — so that is what these tests push on.
"""

from __future__ import annotations

import json
import os
import time

from support import TempDirTestCase, unittest

from sillytalk_plugin import files


class FakeRequest:
    """Just enough of an aiohttp request for these two handlers."""

    def __init__(self, body=None, query=None):
        self._body = body
        self.query = query or {}
        self.remote = "test"

    async def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def payload(response) -> dict:
    return json.loads(response.body)


class DeleteFilesTest(TempDirTestCase):
    async def delete(self, body):
        return payload(await files.delete_files(FakeRequest(body)))

    async def test_removes_a_file_it_is_pointed_at(self):
        (self.out / "keep.png").write_bytes(b"x")
        (self.out / "gone.png").write_bytes(b"y")
        result = await self.delete({"files": [{"filename": "gone.png", "type": "output"}]})
        self.assertTrue(result["ok"])
        self.assertEqual(result["deleted"], ["output/gone.png"])
        self.assertTrue((self.out / "keep.png").exists())
        self.assertFalse((self.out / "gone.png").exists())

    async def test_a_file_that_is_already_gone_counts_as_deleted(self):
        result = await self.delete({"files": [{"filename": "never.png", "type": "output"}]})
        self.assertTrue(result["ok"])
        self.assertEqual(result["absent"], ["output/never.png"])

    async def test_traversal_is_refused(self):
        secret = self.root / "config.json"
        secret.write_text("secret")
        result = await self.delete({"files": [{"filename": "../config.json", "type": "output"}]})
        self.assertFalse(result["ok"])
        self.assertTrue(secret.exists())
        self.assertIn("not a relative path", result["skipped"][0]["reason"])

    async def test_an_absolute_path_is_refused(self):
        result = await self.delete({"files": [{"filename": "/etc/passwd", "type": "output"}]})
        self.assertFalse(result["ok"])
        self.assertIn("not a relative path", result["skipped"][0]["reason"])

    async def test_a_subfolder_traversal_is_refused(self):
        result = await self.delete({"files": [{"filename": "x.png", "subfolder": "../../", "type": "input"}]})
        self.assertFalse(result["ok"])

    async def test_an_unknown_type_is_refused(self):
        result = await self.delete({"files": [{"filename": "x.png", "type": "elsewhere"}]})
        self.assertFalse(result["ok"])
        self.assertIn("type must be one of", result["skipped"][0]["reason"])

    async def test_a_directory_is_left_alone(self):
        (self.out / "adir").mkdir()
        result = await self.delete({"files": [{"filename": "adir", "type": "output"}]})
        self.assertFalse(result["ok"])
        self.assertIn("not a regular file", result["skipped"][0]["reason"])
        self.assertTrue((self.out / "adir").is_dir())

    async def test_an_empty_list_is_refused(self):
        self.assertFalse((await self.delete({"files": []}))["ok"])

    async def test_a_body_that_is_not_an_object_is_refused(self):
        response = await files.delete_files(FakeRequest([1, 2]))
        self.assertEqual(response.status, 400)

    async def test_unreadable_json_is_400(self):
        response = await files.delete_files(FakeRequest(ValueError("nope")))
        self.assertEqual(response.status, 400)


class ListFilesTest(TempDirTestCase):
    async def list(self, query):
        return payload(await files.list_files(FakeRequest(query=query)))

    async def test_lists_relative_names_with_their_folder(self):
        (self.out / "a.png").write_bytes(b"1")
        (self.out / "sub").mkdir()
        (self.out / "sub" / "b.png").write_bytes(b"22")
        result = await self.list({"type": "output"})
        names = {entry["filename"]: entry for entry in result["files"]}
        self.assertEqual(set(names), {"a.png", "b.png"})
        self.assertEqual(names["b.png"]["subfolder"], "sub")
        self.assertEqual(names["b.png"]["bytes"], 2)
        self.assertEqual(result["total"], 2)

    async def test_prefix_narrows_the_listing(self):
        (self.out / "sillytalk").mkdir()
        (self.out / "sillytalk" / "x.png").write_bytes(b"1")
        (self.out / "other.png").write_bytes(b"1")
        result = await self.list({"type": "output", "prefix": "sillytalk/"})
        self.assertEqual([entry["filename"] for entry in result["files"]], ["x.png"])

    async def test_older_than_keeps_only_stale_files(self):
        old = self.out / "old.png"
        old.write_bytes(b"1")
        past = time.time() - 7200
        os.utime(old, (past, past))
        (self.out / "new.png").write_bytes(b"1")
        result = await self.list({"type": "output", "older_than_min": 60})
        self.assertEqual([entry["filename"] for entry in result["files"]], ["old.png"])

    async def test_the_limit_reports_truncation(self):
        for index in range(5):
            (self.out / f"f{index}.png").write_bytes(b"1")
        result = await self.list({"type": "output", "limit": 2})
        self.assertEqual(len(result["files"]), 2)
        self.assertTrue(result["truncated"])
        self.assertEqual(result["total"], 5)

    async def test_a_bad_number_is_400(self):
        response = await files.list_files(FakeRequest(query={"type": "output", "limit": "many"}))
        self.assertEqual(response.status, 400)

    async def test_an_unknown_type_is_400(self):
        response = await files.list_files(FakeRequest(query={"type": "nope"}))
        self.assertEqual(response.status, 400)


if __name__ == "__main__":
    unittest.main()
