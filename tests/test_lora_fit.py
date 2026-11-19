"""Whether a LoRA merges into a model file, decided from their headers.

The trap this guards is the one ComfyUI walks into silently: it skips every LoRA
tensor whose shape disagrees with the model, reports the run `succeeded`, and hands
back the picture the model makes without the LoRA. So the interesting cases here
are the ones where a file is present, named plausibly and still has to be refused
— and the ones where it must not be, because refusing what this module merely
failed to understand would be worse than letting a person use it.
"""

from __future__ import annotations

from support import TempDirTestCase, lora_header, model_header, register_model_folder, write_safetensors

from sillytalk_plugin import lora_fit, model_files

LORA = "style.safetensors"
MODEL = "qwen_image_2.1_int8_convrot.safetensors"


class FitTest(TempDirTestCase):
    def setUp(self):
        super().setUp()
        self.lora_dir = self.root / "loras"
        self.model_dir = self.root / "diffusion_models"
        self.lora_dir.mkdir()
        self.model_dir.mkdir()
        register_model_folder("loras", self.lora_dir)
        register_model_folder("diffusion_models", self.model_dir)
        lora_fit.forget_cache()
        model_files.forget_cache()
        self.addCleanup(lora_fit.forget_cache)
        self.addCleanup(model_files.forget_cache)

    def judge(self, lora_layers, model_layers, *, lora_name=LORA, model_name=MODEL,
              lora_prefix="diffusion_model.", down="lora_down", up="lora_up", model_prefix="model.diffusion_model."):
        write_safetensors(self.lora_dir / lora_name,
                          lora_header(lora_layers, prefix=lora_prefix, down=down, up=up))
        write_safetensors(self.model_dir / model_name, model_header(model_layers, prefix=model_prefix))
        return lora_fit.fit(lora_name, model_name)

    def test_a_lora_whose_layers_the_model_has_at_the_same_size_fits(self):
        layers = {f"transformer_blocks.{i}.attn.to_q": (4096, 4096) for i in range(3)}
        verdict = self.judge(layers, layers)
        self.assertEqual(verdict["fits"], "yes")
        self.assertEqual((verdict["matched"], verdict["mismatched"]), (3, 0))
        self.assertEqual(verdict["reason"], "")

    def test_every_layer_at_a_different_size_is_the_silent_no_op_and_is_refused(self):
        verdict = self.judge({f"transformer_blocks.{i}.attn.to_q": (3072, 3072) for i in range(3)},
                             {f"transformer_blocks.{i}.attn.to_q": (4096, 4096) for i in range(3)})
        self.assertEqual(verdict["fits"], "no")
        self.assertEqual((verdict["matched"], verdict["mismatched"]), (0, 3))
        self.assertIn("3 of the layers it names are at a size this model does not have", verdict["reason"])

    def test_one_stale_tensor_among_many_that_fit_is_not_a_refusal(self):
        """ComfyUI skips the one and merges the rest, which is what the user asked for."""
        good = {f"transformer_blocks.{i}.attn.to_q": (4096, 4096) for i in range(6)}
        own = dict(good, **{"transformer_blocks.9.attn.to_v": (4096, 4096)})
        asked = dict(good, **{"transformer_blocks.9.attn.to_v": (3072, 3072)})
        verdict = self.judge(asked, own)
        self.assertEqual(verdict["fits"], "yes")
        self.assertEqual((verdict["matched"], verdict["mismatched"]), (6, 1))
        self.assertIn("1 of the layers it names is at a different size and ComfyUI will skip it",
                      verdict["reason"])

    def test_more_wrong_than_right_is_a_refusal_and_says_both_counts(self):
        asked = {"a.to_q": (3072, 3072), "b.to_q": (3072, 3072), "c.to_q": (4096, 4096)}
        own = {"a.to_q": (4096, 4096), "b.to_q": (4096, 4096), "c.to_q": (4096, 4096)}
        verdict = self.judge(asked, own)
        self.assertEqual(verdict["fits"], "no")
        self.assertIn("2 of the layers it names are at a size this model does not have while 1 fits",
                      verdict["reason"])

    def test_a_layer_this_model_does_not_have_at_all_is_said_and_not_refused(self):
        """The rest merges, so it is no reason to refuse — but the style is weaker than asked
        and ComfyUI mentions it on the console only."""
        asked = {f"transformer_blocks.{i}.{layer}": (4096, 4096)
                 for i in range(2) for layer in ("attn.to_q", "attn.to_k", "img_mlp.gate_up")}
        verdict = self.judge(asked, {k: v for k, v in asked.items() if "gate_up" not in k})
        self.assertEqual(verdict["fits"], "yes")
        self.assertEqual((verdict["matched"], verdict["mismatched"]), (4, 0))
        self.assertEqual(verdict["reason"],
                         "2 of the layers it names are not in this model at all and ComfyUI will skip them")

    def test_a_lora_carrying_both_spellings_of_the_mlp_is_whole_on_either_model(self):
        """Qwen-Image 2.1 has checkpoints with a fused `gate_up` and ones with `gate_layer` +
        `proj`. A LoRA carrying both is complete on either, so the spelling the loaded file
        does not use must not be reported as skipped."""
        attn = {"transformer_blocks.0.attn.to_q": (4096, 4096)}
        both = dict(attn, **{f"transformer_blocks.0.img_mlp.{n}": (4096, 4096)
                             for n in ("gate_up", "gate_layer", "proj")})
        for layout, model_layers in (("fused", dict(attn, **{"transformer_blocks.0.img_mlp.gate_up": (4096, 4096)})),
                                     ("split", dict(attn, **{f"transformer_blocks.0.img_mlp.{n}": (4096, 4096)
                                                             for n in ("gate_layer", "proj")}))):
            with self.subTest(layout=layout):
                lora_fit.forget_cache()
                verdict = self.judge(both, model_layers)
                self.assertEqual(verdict["fits"], "yes")
                self.assertEqual(verdict["reason"], "", f"the {layout} model covers it through one spelling")

    def test_half_of_the_split_mlp_is_not_the_other_spelling_and_is_still_a_gap(self):
        asked = {f"transformer_blocks.0.img_mlp.{n}": (4096, 4096) for n in ("gate_up", "gate_layer", "proj")}
        verdict = self.judge(asked, {"transformer_blocks.0.img_mlp.gate_layer": (4096, 4096)})
        self.assertEqual(verdict["fits"], "yes")
        self.assertIn("2 of the layers it names are not in this model at all", verdict["reason"])

    def test_a_stale_size_and_a_missing_layer_are_both_said(self):
        verdict = self.judge({"a.to_q": (4096, 4096), "b.to_q": (3072, 3072), "c.to_k": (4096, 4096)},
                             {"a.to_q": (4096, 4096), "b.to_q": (4096, 4096)})
        self.assertEqual(verdict["fits"], "yes")
        self.assertEqual(verdict["reason"],
                         "1 of the layers it names is at a different size and ComfyUI will skip it; "
                         "1 of the layers it names is not in this model at all and ComfyUI will skip it")

    def test_a_text_encoder_lora_changes_nothing_here_and_is_refused(self):
        """These graphs merge into the model only; strength_clip is fixed at 0."""
        layers = {f"text_encoders.clip_l.layers.{i}.attn.q": (1024, 1024) for i in range(2)}
        verdict = self.judge(layers, layers)
        self.assertEqual(verdict["fits"], "no")
        self.assertIn("text encoder", verdict["reason"])

    def test_a_lora_carrying_the_text_encoder_too_is_judged_on_its_model_layers(self):
        layers = {"text_encoders.clip_l.layers.0.attn.q": (1024, 1024), "transformer_blocks.0.attn.to_q": (4096, 4096)}
        verdict = self.judge(layers, {"transformer_blocks.0.attn.to_q": (4096, 4096)})
        self.assertEqual(verdict["fits"], "yes")

    def test_the_naming_of_the_wild_reaches_the_same_verdict(self):
        layers = {"transformer_blocks.0.attn.to_q": (4096, 4096)}
        for prefix in ("model.diffusion_model.", "diffusion_model.", "lora_unet_", "base_model.model.", ""):
            for down, up in (("lora_down", "lora_up"), ("lora_A", "lora_B")):
                with self.subTest(prefix=prefix, halves=f"{down}/{up}"):
                    lora_fit.forget_cache()
                    self.assertEqual(self.judge(layers, layers, lora_prefix=prefix, down=down, up=up)["fits"], "yes")

    def test_a_quantised_scale_does_not_claim_the_layer_it_sits_next_to(self):
        """An int8 checkpoint carries weight_scale beside weight, at a shape of its own."""
        layers = {"transformer_blocks.0.attn.to_q": (4096, 4096)}
        header = model_header(layers)
        header["model.diffusion_model.transformer_blocks.0.attn.to_q.weight_scale"] = ("F8_E4M3", [4096, 1])
        header["model.diffusion_model.transformer_blocks.0.attn.to_q.weight_zero_point"] = ("F8_E4M3", [4096, 1])
        write_safetensors(self.lora_dir / LORA, lora_header(layers))
        write_safetensors(self.model_dir / MODEL, header)
        self.assertEqual(lora_fit.fit(LORA, MODEL)["fits"], "yes")

    def test_nothing_matched_by_name_is_a_flag_not_a_refusal(self):
        verdict = self.judge({"blocks.0.self_attn.qkv": (4096, 4096)},
                             {"transformer_blocks.0.attn.to_q": (4096, 4096)})
        self.assertEqual(verdict["fits"], "unknown")
        self.assertIn("may be for another one", verdict["reason"])

    def test_an_alpha_entry_is_not_read_as_a_layer_of_its_own(self):
        layers = {"transformer_blocks.0.attn.to_q": (4096, 4096)}
        header = lora_header(layers)
        header["diffusion_model.transformer_blocks.0.attn.to_q.alpha"] = ("F32", [])
        write_safetensors(self.lora_dir / LORA, header)
        write_safetensors(self.model_dir / MODEL, model_header(layers))
        verdict = lora_fit.fit(LORA, MODEL)
        self.assertEqual((verdict["fits"], verdict["matched"]), ("yes", 1))

    def test_a_file_that_is_not_safetensors_gets_no_verdict(self):
        (self.lora_dir / LORA).write_bytes(b"not a safetensors file")
        write_safetensors(self.model_dir / MODEL, model_header({"a.to_q": (4096, 4096)}))
        self.assertIsNone(lora_fit.fit(LORA, MODEL))

    def test_a_file_that_is_not_there_gets_no_verdict(self):
        write_safetensors(self.model_dir / MODEL, model_header({"a.to_q": (4096, 4096)}))
        self.assertIsNone(lora_fit.fit(LORA, MODEL))
        self.assertIsNone(lora_fit.fit(LORA, "gone.safetensors"))

    def test_a_header_that_is_not_json_gets_no_verdict(self):
        (self.lora_dir / LORA).write_bytes((256).to_bytes(8, "little") + b"x" * 256)
        write_safetensors(self.model_dir / MODEL, model_header({"a.to_q": (4096, 4096)}))
        self.assertIsNone(lora_fit.fit(LORA, MODEL))

    def test_a_verdict_follows_the_file_it_came_from(self):
        """A LoRA replaced under the same name is judged again without emptying the cache."""
        self.assertEqual(self.judge({"a.to_q": (4096, 4096)}, {"a.to_q": (4096, 4096)})["fits"], "yes")
        write_safetensors(self.lora_dir / LORA, lora_header({"a.to_q": (3072, 3072)}))
        self.assertEqual(lora_fit.fit(LORA, MODEL)["fits"], "no")

    def test_the_same_lora_is_judged_per_file(self):
        layers = {"a.to_q": (4096, 4096)}
        self.judge(layers, layers, model_name="fits.safetensors")
        write_safetensors(self.model_dir / "other.safetensors", model_header({"a.to_q": (3072, 3072)}))
        self.assertEqual(lora_fit.fit(LORA, "fits.safetensors")["fits"], "yes")
        self.assertEqual(lora_fit.fit(LORA, "other.safetensors")["fits"], "no")


if __name__ == "__main__":
    from support import unittest

    unittest.main()
