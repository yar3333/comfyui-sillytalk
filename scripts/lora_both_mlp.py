#!/usr/bin/env python
"""Give a Qwen-Image 2.1 LoRA both MLP layouts, so it merges into either checkpoint.

ComfyUI builds the model to match the checkpoint it loads — `model_detection.py` sets
`fused_mlp = gate_up is not None` — so the names inside a loaded model are the names in
its file. A LoRA that names `img_mlp.gate_up` therefore lands whole on a fused
checkpoint and silently skips one layer per block on a split one, and the other way
round for a LoRA naming `gate_layer`/`proj`. Nothing refuses it; the style is just a
little weaker than asked, which is the quiet failure ComfyUI is prone to.

The two layouts are the same SwiGLU:

    fused:  gate, up = self.gate_up(x).chunk(2, dim=-1); silu(gate) * up   # comfy/ops.py
    split:  self.out(F.silu(self.gate_layer(x)) * self.proj(x))   # comfy/ldm/qwen_image21

so a fused LoRA's B splits in half along its output rows and the same A serves both —
ΔW = B @ A, and B's first half is the gate. Exact, not an approximation, same rank.

Writing both layouts into one file costs a few tens of megabytes and makes the LoRA
complete on either checkpoint. The spelling the loaded model does not use is skipped,
and ComfyUI says so: one `lora key not loaded` line per tensor of it — measured on a
32-block LoRA at 64 lines on a split checkpoint and 128 on a fused one, on the first
load of a run and none after, since the merged model is cached. Noise on the console,
not a lost part of the style: the same runs skipped exactly those and nothing else.

    <comfyui>/.venv/bin/python scripts/lora_both_mlp.py <lora.safetensors> [more…]

(needs torch, so the ComfyUI venv is the one that has it). Refuses a file that has
nothing to convert, and never touches the input: the result is written next to it
as `<name> [both-mlp].safetensors`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

FUSED = ".img_mlp.gate_up."


def convert(tensors: dict[str, torch.Tensor]) -> tuple[dict[str, torch.Tensor], int]:
    """The file's tensors plus the other layout's halves, and how many layers were added."""
    out = dict(tensors)
    added = 0
    for key, b in tensors.items():
        if FUSED not in key or not key.endswith(".lora_B.weight"):
            continue
        a_key = key[: -len("lora_B.weight")] + "lora_A.weight"
        a = tensors.get(a_key)
        if a is None:
            continue
        if b.shape[0] % 2:
            raise SystemExit(f"{key}: an odd {b.shape[0]} rows cannot be a [gate; up] pair")
        half = b.shape[0] // 2
        # safetensors forbids two tensors sharing one storage, so every copy is its own.
        for which, rows in (("gate_layer", slice(0, half)), ("proj", slice(half, None))):
            out[key.replace(FUSED, f".img_mlp.{which}.")] = b[rows].contiguous().clone()
            out[a_key.replace(FUSED, f".img_mlp.{which}.")] = a.clone()
            added += 1
        # The scale factor belongs to the pair, and an absent alpha would silently mean
        # a default one for the halves.
        alpha_key = key[: -len("lora_B.weight")] + "alpha"
        alpha = tensors.get(alpha_key)
        if alpha is not None:
            for which in ("gate_layer", "proj"):
                out[alpha_key.replace(FUSED, f".img_mlp.{which}.")] = alpha.clone()
    return out, added


def main(paths: list[str]) -> int:
    for name in paths:
        src = Path(name)
        with safe_open(src, framework="pt") as handle:
            metadata = handle.metadata()
        tensors = load_file(src)
        out, added = convert(tensors)
        if not added:
            print(f"{src.name}: nothing to convert — no fused img_mlp.gate_up pair in here")
            continue
        dst = src.with_name(f"{src.stem} [both-mlp].safetensors")
        save_file(out, dst, metadata=metadata)
        grew = dst.stat().st_size - src.stat().st_size
        print(f"{src.name}: +{added // 2} layers of the other layout → {dst.name} "
              f"({src.stat().st_size / 1e6:.0f} → {dst.stat().st_size / 1e6:.0f} MB, "
              f"+{grew / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv[1:]))
