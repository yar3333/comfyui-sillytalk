"""Whether a LoRA can merge into a model file, decided from their headers.

ComfyUI matches a LoRA's tensors against the model by shape and *skips* the ones
that do not fit. A LoRA trained for another model therefore draws a picture that
looks exactly like the one without it, the job reports `succeeded`, and only this
server's console says `lora key not loaded` twenty times. This module answers the
same question before the job is queued.

It reads the safetensors header and nothing else: the first 8 bytes of the file
are the length of a JSON table naming every tensor with its dtype and shape, so a
20 GB checkpoint costs a few hundred kilobytes and no torch is involved. Measured
on a real folder: five LoRAs against six checkpoints, 169 ms for all thirty pairs
cold and a few microseconds each once the files have been read.

What it can and cannot say:

* a tensor whose layer the model has but at a different size is proof the LoRA was
  not made for it; when those outweigh the layers that do fit, that is the
  silent-skip case and it is refused — a lone stale tensor among hundreds is not,
  because the rest merges and ComfyUI skips only that one;
* a LoRA whose every target is a text encoder cannot do anything here either,
  because these graphs merge into the model only (`strength_clip` is fixed at 0),
  so that is refused too;
* a LoRA that matches nothing by name at all is *unknown*, not wrong: the naming
  of LoRA files varies (peft, DCNV3, diffusers) and this module deliberately
  understands the common few. Unknown LoRAs are allowed and flagged, because
  refusing a file this code simply did not recognise would be worse than letting
  a person use it.
"""

from __future__ import annotations

import json
import logging
import re
import struct
from pathlib import Path
from typing import Any

from . import model_files

logger = logging.getLogger(__name__)

# The prefixes a LoRA and a checkpoint may each wear for the same layer. ComfyUI
# keeps its own longer table in comfy/lora.py; these are the ones that appear in
# files people actually drop into models/loras/.
_PREFIXES = ("model.diffusion_model.", "diffusion_model.", "lora_unet_", "base_model.model.")

# What marks the two halves of a low-rank pair, across the formats in the wild.
_DOWN_RE = re.compile(r"\.(lora_down|lora_A)\.weight$")
_UP_RE = re.compile(r"\.(lora_up|lora_B)\.weight$")
_ALPHA_RE = re.compile(r"\.alpha$")
_SUFFIX_RE = re.compile(r"(\.weight|\.bias|_scale)$")

# A LoRA for the text encoder rather than the diffusion model.
_TEXT_PREFIXES = ("text_encoders.", "lora_te", "text_encoder.")

# A header claiming to be bigger than this is not a header.
_MAX_HEADER = 64 * 1024 * 1024

# A header is a few hundred kilobytes, so the cache is bounded by the folder rather
# than by anything a person would notice — unless someone keeps a thousand files
# in it, in which case starting over is cheaper than growing without end.
_MAX_CACHED_HEADERS = 128

# path -> (mtime, size, parsed header)
_header_cache: dict[str, tuple[int, int, dict[str, Any]]] = {}
# (lora name, model name) -> ((lora signature, model signature), verdict)
_verdict_cache: dict[tuple[str, str], tuple[tuple[Any, Any], dict[str, Any]]] = {}


def fit(lora_name: str, model_name: str) -> dict[str, Any] | None:
    """How `lora_name` (models/loras/) stands against `model_name` (diffusion_models/).

    `None` when either file is not on disk or one of them is not safetensors —
    which is not a verdict, and the caller treats it as "unknown, let it through".
    """
    lora = model_files.full_path(model_files.LORA_FOLDER, lora_name)
    model = model_files.full_path(model_files.DIFFUSION_FOLDER, model_name)
    if lora is None or model is None:
        return None

    # A file's header only changes when the file does, and mtime plus size say
    # whether it did, so a verdict is trusted exactly as long as the bytes under
    # it are. Two stats per call is what that costs.
    signature = (_signature(lora), _signature(model))
    key = (lora_name, model_name)
    cached = _verdict_cache.get(key)
    if cached is not None and cached[0] == signature:
        return cached[1]

    targets = _lora_targets(_headers(lora))
    if not targets:
        return None
    shapes = _model_shapes(_headers(model))
    if not shapes:
        return None

    verdict = _judge(targets, shapes)
    _verdict_cache[key] = (signature, verdict)
    return verdict


def _judge(targets: dict[str, tuple[int, int]], shapes: dict[str, tuple[int, int]]) -> dict[str, Any]:
    matched = 0
    mismatched = 0
    for base, (lora_in, lora_out) in targets.items():
        own = shapes.get(base)
        if own is None:
            continue
        if own == (lora_out, lora_in):
            matched += 1
        else:
            mismatched += 1

    text_only = all(base.startswith(_TEXT_PREFIXES) for base in targets)
    if text_only:
        return {
            "fits": "no",
            "matched": 0,
            "mismatched": 0,
            "reason": (
                "it is a LoRA for the text encoder — these graphs merge LoRAs into the model only "
                "(strength_clip is fixed at 0), so it would change nothing"
            ),
        }
    # Wrong sizes only decide the matter when they outweigh the right ones. A LoRA
    # that fits 700 layers and carries 1 stale tensor merges fine — ComfyUI skips
    # the one — while 0 fitting against 128 wrong is the silent no-op this module
    # exists to catch.
    if mismatched and mismatched > matched:
        verb, pronoun = _agreement(mismatched)
        return {
            "fits": "no",
            "matched": matched,
            "mismatched": mismatched,
            "reason": (
                f"{mismatched} of the layers it names {verb} at a size this model does not have"
                + ("" if not matched else f" while {matched} {('fits' if matched == 1 else 'fit')}")
                + f", so ComfyUI would load it and skip {pronoun} — the picture would be close to "
                "the one you get without it"
            ),
        }
    if matched:
        verb, pronoun = _agreement(mismatched)
        return {
            "fits": "yes",
            "matched": matched,
            "mismatched": mismatched,
            "reason": ""
            if not mismatched
            else f"{mismatched} of the layers it names {verb} at a different size and ComfyUI will "
            f"skip {pronoun}",
        }
    return {
        "fits": "unknown",
        "matched": 0,
        "mismatched": 0,
        "reason": "none of its layer names occur in this model — it may be for another one",
    }


def _agreement(count: int) -> tuple[str, str]:
    """The verb and the pronoun that read correctly after `count` layers."""
    return ("is", "it") if count == 1 else ("are", "them")


def _signature(path: Path) -> tuple[int, int] | None:
    """What identifies a file's bytes for the caches; None when it cannot be stat'ed."""
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size


def _headers(path: Path) -> dict[str, Any]:
    """The header table of a file, parsed once per version of it."""
    signature = _signature(path)
    if signature is None:
        return {}
    hit = _header_cache.get(str(path))
    if hit is not None and hit[:2] == signature:
        return hit[2]
    parsed = _read_header(path)
    if len(_header_cache) >= _MAX_CACHED_HEADERS:
        _header_cache.clear()
    _header_cache[str(path)] = (signature[0], signature[1], parsed)
    return parsed


def _read_header(path: Path) -> dict[str, Any]:
    """The JSON table at the front of a safetensors file, {} for anything else."""
    if path.suffix.lower() != ".safetensors":
        return {}
    try:
        with path.open("rb") as handle:
            raw = handle.read(8)
            if len(raw) < 8:
                return {}
            (length,) = struct.unpack("<Q", raw)
            if length < 2 or length > _MAX_HEADER:
                return {}
            body = handle.read(length)
    except OSError as err:
        logger.warning("sillytalk: could not read the header of %s (%s)", path, err)
        return {}
    try:
        meta = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return {}
    return meta if isinstance(meta, dict) else {}


def _base_name(key: str) -> str:
    """The layer a tensor belongs to, with the LoRA and container naming removed."""
    name = key
    for prefix in _PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]
            break
    name = _DOWN_RE.sub("", name)
    name = _UP_RE.sub("", name)
    name = _ALPHA_RE.sub("", name)
    return _SUFFIX_RE.sub("", name)


def _lora_targets(meta: dict[str, Any]) -> dict[str, tuple[int, int]]:
    """`{layer: (in, out)}` a LoRA claims to patch, from the shapes of its two halves."""
    found: dict[str, dict[str, int]] = {}
    for key, entry in meta.items():
        if key == "__metadata__" or not isinstance(entry, dict):
            continue
        shape = entry.get("shape")
        if not isinstance(shape, list) or not shape:
            continue
        base = _base_name(key)
        if not base:
            continue
        half = found.setdefault(base, {})
        if _DOWN_RE.search(key) or key.endswith("lora_A.weight"):
            half["in"] = int(shape[-1])
        elif _UP_RE.search(key) or key.endswith("lora_B.weight"):
            half["out"] = int(shape[0])
    return {b: (d.get("in", 0), d.get("out", 0)) for b, d in found.items() if d}


def _model_shapes(meta: dict[str, Any]) -> dict[str, tuple[int, int]]:
    """`{layer: (out, in)}` of a checkpoint's own 2-D weights."""
    out: dict[str, tuple[int, int]] = {}
    for key, entry in meta.items():
        if key == "__metadata__" or not isinstance(entry, dict):
            continue
        # An int8/fp8 checkpoint carries a `weight_scale` (sometimes a
        # `weight_zero_point`) beside every quantised weight, at a shape of its
        # own. That is not the weight, and letting it claim the layer would
        # compare a rank-1 scale against a LoRA.
        if key.endswith(("_scale", "_zero_point", "_absmax")):
            continue
        shape = entry.get("shape")
        if not isinstance(shape, list) or len(shape) < 2:
            continue
        base = _base_name(key)
        if not base:
            continue
        out.setdefault(base, (int(shape[0]), int(shape[1])))
    return out


def forget_cache() -> None:
    """Drops every cached header and verdict (tests, and a folder changed on purpose)."""
    _header_cache.clear()
    _verdict_cache.clear()
