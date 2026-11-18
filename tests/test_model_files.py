"""The file listings a client picks from.

The whole point of the feature is that a name is never typed by hand, so what
matters here is that the listing is the server's own folder — subfolders
included — and that a file dropped in becomes selectable without a restart.
"""

from __future__ import annotations

import unittest

from support import TempDirTestCase, register_model_folder

from sillytalk_plugin import model_files


class ListingTest(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.folder = self.root / "diffusion_models"
        (self.folder / "sub").mkdir(parents=True)
        (self.folder / "b_model.safetensors").write_bytes(b"x" * (2 * 1024 * 1024))
        (self.folder / "A_model.safetensors").write_bytes(b"y")
        (self.folder / "sub" / "derivative.safetensors").write_bytes(b"z" * 3)
        (self.folder / ".hidden").write_bytes(b"")
        register_model_folder("diffusion_models", self.folder)
        model_files.forget_cache()
        self.addCleanup(model_files.forget_cache)

    def test_the_folder_is_listed_by_name_with_its_size(self):
        listed = model_files.listing("diffusion_models")
        self.assertEqual(
            [entry["name"] for entry in listed],
            ["A_model.safetensors", "b_model.safetensors", "sub/derivative.safetensors"],
            "sorted by name, case-insensitively, subfolders included",
        )
        by_name = {entry["name"]: entry for entry in listed}
        self.assertEqual(by_name["b_model.safetensors"]["mb"], 2)
        self.assertEqual(by_name["A_model.safetensors"]["mb"], 0)

    def test_an_unknown_folder_is_an_empty_choice_not_an_error(self):
        self.assertEqual(model_files.listing("nowhere"), [])

    def test_known_answers_for_a_name_the_folder_has(self):
        self.assertTrue(model_files.known("diffusion_models", "b_model.safetensors"))
        self.assertTrue(model_files.known("diffusion_models", "sub/derivative.safetensors"))
        self.assertFalse(model_files.known("diffusion_models", "gone.safetensors"))
        self.assertFalse(model_files.known("diffusion_models", ""))

    def test_size_of_measures_the_file_and_zeroes_without_one(self):
        self.assertEqual(model_files.size_of("diffusion_models", "b_model.safetensors"), 2 * 1024 * 1024)
        self.assertEqual(model_files.size_of("diffusion_models", "gone.safetensors"), 0)

    def test_a_file_dropped_in_appears_once_the_cache_expires(self):
        self.assertNotIn("new.safetensors", [entry["name"] for entry in model_files.listing("diffusion_models")])
        (self.folder / "new.safetensors").write_bytes(b"n")
        self.assertNotIn("new.safetensors", [entry["name"] for entry in model_files.listing("diffusion_models")],
                         "the listing is cached, so a busy app does not rescan per probe")
        model_files.forget_cache()
        self.assertIn("new.safetensors", [entry["name"] for entry in model_files.listing("diffusion_models")])


if __name__ == "__main__":
    unittest.main()
