"""Which GPU each model loader of a run lands on.

The reason this lives on the server rather than in the client: only the server
can ask torch how much VRAM is free right now. On a box where the same cards
also serve LLM inference, a device chosen once in a client config is wrong the
moment another model is loaded — and a loader pinned to a full card does not
fail politely, it takes the server down with an out-of-memory.

The rule is a first-fit-decreasing bin packing over the free VRAM: the heaviest
model of the graph goes to the emptiest card, then the next, each with what the
previous ones of this run already took off that card. The largest single model
also carries a headroom for activations, which is where the sampling happens.

Every `device` input of the graph is written by this module. A device value in a
workflow file is therefore ignored — that is the point of the plugin, and it is
said in the README rather than hidden in the code.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Activation headroom on top of the largest model, in GiB. A 1328x1328 Qwen job
# needs well over a gigabyte of it, and free VRAM reported by torch is the only
# honest number available before the run.
DEFAULT_HEADROOM_GB = 2.0

# Input names that carry a model file, and where ComfyUI keeps it. Anything
# unmapped is looked for in FALLBACK_FOLDERS, which costs a stat() per folder
# and only happens for a loader this table does not know.
FOLDER_BY_INPUT = {
    "unet_name": "diffusion_models",
    "model_name": "diffusion_models",
    "diffusion_model": "diffusion_models",
    "ckpt_name": "checkpoints",
    "clip_name": "text_encoders",
    "clip_name1": "text_encoders",
    "clip_name2": "text_encoders",
    "text_encoder_name": "text_encoders",
    "vae_name": "vae",
    "lora_name": "loras",
    "control_net_name": "controlnet",
}
FALLBACK_FOLDERS = ["diffusion_models", "text_encoders", "vae", "loras", "checkpoints", "clip"]

# A device string a workflow may already hold. Used to recognise a device input
# on a class whose INPUT_TYPES() cannot be called without arguments.
_DEVICE_PREFIXES = ("cuda:", "cpu", "npu:", "xpu:", "mps", "hip:", "privateuseone")


class PlacementError(Exception):
    """No card can take what the graph needs — said with the numbers."""


@dataclass
class Device:
    id: str
    name: str
    total: int
    free: int
    # What ComfyUI itself holds here and can drop to make room. It is the whole
    # difference between a card that is full and a card that is busy: a second
    # run of the same workflow needs nothing new, because its weights are
    # already sitting on this card. VRAM held by ANOTHER process is not in here
    # and never can be reclaimed.
    reclaimable: int = 0
    taken: int = 0  # what this run has already put on it

    @property
    def left(self) -> int:
        return self.free + self.reclaimable - self.taken

    def describe(self) -> dict[str, Any]:
        return {
            "device": self.id,
            "name": self.name,
            "vram_total_mb": round(self.total / 1024 / 1024),
            "vram_free_mb": round(self.free / 1024 / 1024),
            "vram_reclaimable_mb": round(self.reclaimable / 1024 / 1024),
            "vram_used_by_run_mb": round(self.taken / 1024 / 1024),
        }


@dataclass
class Assignment:
    node: str
    class_type: str
    device: str
    weight_bytes: int
    note: str = ""

    def describe(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "node": self.node,
            "class": self.class_type,
            "device": self.device,
            "mb": round(self.weight_bytes / 1024 / 1024),
        }
        if self.note:
            out["note"] = self.note
        return out


@dataclass
class Placement:
    assignments: list[Assignment] = field(default_factory=list)
    devices: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""

    def describe(self) -> dict[str, Any]:
        out: dict[str, Any] = {"placements": [a.describe() for a in self.assignments], "devices": self.devices}
        if self.note:
            out["note"] = self.note
        return out

    def summary(self) -> str:
        if not self.assignments:
            return self.note or "nothing to place"
        return ", ".join(f"{a.class_type}@{a.device}" for a in self.assignments)


def headroom_bytes() -> int:
    raw = os.environ.get("SILLYTALK_VRAM_HEADROOM_GB", "").strip()
    gb = DEFAULT_HEADROOM_GB
    if raw:
        try:
            gb = float(raw)
        except ValueError:
            logger.warning("sillytalk: SILLYTALK_VRAM_HEADROOM_GB=%r is not a number, using %s", raw, DEFAULT_HEADROOM_GB)
    return max(0, int(gb * 1024 * 1024 * 1024))


def allowed_devices() -> set[str] | None:
    """The device ids this plugin may use, or None for "any the server has".

    `SILLYTALK_DEVICES=cuda:0,cuda:1` keeps a ComfyUI off cards the owner
    reserves for other inference.
    """
    raw = os.environ.get("SILLYTALK_DEVICES", "").strip()
    if not raw:
        return None
    return {part.strip() for part in raw.split(",") if part.strip()}


def resident_models() -> tuple[dict[str, int], dict[str, tuple[str, int]]]:
    """What this server already holds in VRAM, and which file each byte came from.

    Returns `(bytes per device, model file -> (device, bytes))`. Both halves are
    best effort: a ComfyUI that does not keep the list, or keeps models whose
    file this version does not name, simply reports less — and the placement
    falls back to plain bin packing instead of guessing.
    """
    per_device: dict[str, int] = {}
    per_file: dict[str, tuple[str, int]] = {}
    try:
        import comfy.model_management as model_management
    except Exception:
        return per_device, per_file
    for loaded in list(getattr(model_management, "current_loaded_models", [])):
        try:
            patcher = loaded.model
            device = str(patcher.current_loaded_device())
            size = int(patcher.loaded_size())
        except Exception:
            continue
        if size <= 0:
            continue
        per_device[device] = per_device.get(device, 0) + size
        path = _patcher_file(patcher)
        if path:
            per_file[path] = (device, size)
    return per_device, per_file


def _patcher_file(patcher: Any) -> str | None:
    """The file a loaded model was read from, from whichever attribute exists.

    `cached_patcher_init` is the one that answers on a current ComfyUI: it is
    whatever the server would reload this model from, which is precisely the
    identity wanted here. The rest are older or checkpoint-shaped spellings, and
    None means "resident, but I cannot say with what" — the VRAM still counts as
    reclaimable on that card, only the preference of staying put is lost.
    """
    model = getattr(patcher, "model", None)
    for candidate in (
        # The patcher carries it in a current ComfyUI (it is what the patcher
        # rebuilds itself from); the model is checked too, in case the loader
        # left it one level down.
        _first_path(getattr(patcher, "cached_patcher_init", None)),
        _first_path(getattr(model, "cached_patcher_init", None)),
        _checkpoint_location(model),
        getattr(model, "model_checkpoint_path", None),
    ):
        if isinstance(candidate, str) and candidate:
            return candidate
    return None


def _checkpoint_location(model: Any) -> str | None:
    """Where a checkpoint-style model says it came from, if it says.

    `model_config.checkpoint_info` is read only when the object actually holds
    it: on a model built from separate files that name is a property which logs
    a complaint about whoever reads it, and a helper that filled the server log
    with warnings about itself would be rude.
    """
    config = getattr(model, "model_config", None)
    if config is None or "checkpoint_info" not in getattr(config, "__dict__", {}):
        return None
    location = getattr(config.checkpoint_info, "location", None)
    return location if isinstance(location, str) and location else None


def _first_path(cached_init: Any) -> str | None:
    """The path inside a `cached_patcher_init` of `(factory, (path, ...))`.

    A dual text encoder carries a list of files there, so the first string at
    either depth is taken; a loader of several files is only followed when every
    one of them is accounted for, so a partial answer costs nothing.
    """
    if not isinstance(cached_init, (tuple, list)) or len(cached_init) < 2:
        return None
    args = cached_init[1]
    if isinstance(args, str):
        return args
    if not isinstance(args, (tuple, list)):
        return None
    for item in args:
        if isinstance(item, str) and item:
            return item
        if isinstance(item, (tuple, list)):
            for nested in item:
                if isinstance(nested, str) and nested:
                    return nested
            return None
    return None


def visible_devices() -> list[Device]:
    """The compute devices torch sees, with the VRAM a run could actually use.

    Read per run on purpose: both numbers are the whole input to the decision
    and both change whenever any process loads or unloads a model.
    """
    try:
        import torch
    except ImportError:
        return []
    if not hasattr(torch, "cuda") or not torch.cuda.is_available():
        return []
    reclaimable, _ = resident_models()
    devices: list[Device] = []
    allowed = allowed_devices()
    for index in range(torch.cuda.device_count()):
        device = f"cuda:{index}"
        if allowed is not None and device not in allowed:
            continue
        try:
            free, total = torch.cuda.mem_get_info(index)
        except Exception as err:  # a card that cannot answer is not a candidate
            logger.warning("sillytalk: cuda:%d did not answer mem_get_info (%s)", index, err)
            continue
        try:
            name = torch.cuda.get_device_name(index)
        except Exception:
            name = device
        devices.append(
            Device(id=device, name=name, total=int(total), free=int(free), reclaimable=int(reclaimable.get(device, 0)))
        )
    return devices


def _class_offers_device(class_type: str) -> bool | None:
    """True/False when the node class says, None when asking is not possible."""
    try:
        import nodes

        obj_class = nodes.NODE_CLASS_MAPPINGS.get(class_type)
        if obj_class is None:
            return None
        spec = obj_class.INPUT_TYPES()
        if not isinstance(spec, dict):
            return None
        for group in ("required", "optional"):
            for name, schema in (spec.get(group) or {}).items():
                if name == "device":
                    return True
        return False
    except Exception:
        return None


def _looks_like_device(value: Any) -> bool:
    return isinstance(value, str) and value.strip().startswith(_DEVICE_PREFIXES)


def placeable_nodes(graph: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """The nodes of a graph that carry a device to choose."""
    found: list[tuple[str, dict[str, Any]]] = []
    for node_id, node in graph.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict) or "device" not in inputs:
            continue
        class_type = str(node.get("class_type") or "")
        offers = _class_offers_device(class_type)
        if offers is False:
            continue
        if offers is None and not (_looks_like_device(inputs["device"]) or inputs["device"] == "PARAM_DEVICE"):
            continue
        found.append((str(node_id), node))
    return found


def _model_files(node: dict[str, Any]) -> list[Path]:
    """The model files a loader node names, resolved on the server."""
    import folder_paths

    inputs = node.get("inputs") or {}
    paths: list[Path] = []
    for name, value in inputs.items():
        if name == "device" or not isinstance(value, str) or not value or value.startswith("PARAM_"):
            continue
        folders = [FOLDER_BY_INPUT[name]] if name in FOLDER_BY_INPUT else FALLBACK_FOLDERS
        for folder in folders:
            try:
                full = folder_paths.get_full_path(folder, value)
            except Exception:
                full = None
            if full:
                paths.append(Path(full))
                break
    return paths


def _size_of(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _one_device_for(files: list[Path], lookup) -> str | None:
    """The single device every one of `files` sits on, or None.

    `lookup(path)` answers a device id or None. Files on mixed devices count as
    no answer at all — a loader puts its weights on one card, so half an answer
    would move weights the server has already loaded.
    """
    if not files:
        return None
    found: set[str] = set()
    for path in files:
        device = lookup(path)
        if device is None:
            return None
        found.add(device)
    return found.pop() if len(found) == 1 else None


def place(graph: dict[str, Any], headroom: int | None = None) -> Placement:
    """Writes a `device` into every loader of `graph` (in place) and reports how.

    A graph with no device inputs and a server with no visible GPU both come
    back without an error: there is nothing to decide, and refusing to draw
    because of that would be worse than letting the workflow's own values stand.
    """
    result = Placement()
    nodes_to_place = placeable_nodes(graph)
    if not nodes_to_place:
        result.note = "the workflow has no device inputs to place"
        return result

    devices = visible_devices()
    if not devices:
        result.note = "torch reports no CUDA device — the workflow keeps its own device values"
        logger.info("sillytalk: %s", result.note)
        return result
    result.devices = [device.describe() for device in devices]

    # Each loader node is measured by the model files it names. Heaviest first:
    # a big model left for last would find no card with a span wide enough,
    # while the small ones fit almost anywhere.
    prepared: list[tuple[int, str, dict[str, Any], str, list[Path]]] = []
    for node_id, node in nodes_to_place:
        files = _model_files(node)
        prepared.append((sum(_size_of(path) for path in files), node_id, node, str(node.get("class_type") or ""), files))
    prepared.sort(key=lambda item: (-item[0], item[1]))

    # A file costs VRAM once however many nodes load it — the server caches
    # weights by file — so the second loader joins the card that already holds
    # it instead of asking for room again.
    _, resident_files = resident_models()
    # A loaded model is often named by its bare file name rather than the full
    # path, so both spellings find it.
    resident_by_name: dict[str, tuple[str, int]] = {}
    for key, where in resident_files.items():
        resident_by_name.setdefault(key, where)
        resident_by_name.setdefault(Path(key).name, where)

    device_of_file: dict[Path, str] = {}
    reserve = headroom_bytes() if headroom is None else headroom
    largest = prepared[0][0] if prepared else 0

    def resident_device_of(path: Path) -> str | None:
        where = resident_by_name.get(str(path)) or resident_by_name.get(path.name)
        return where[0] if where else None

    for size, node_id, node, class_type, files in prepared:
        # The cheapest load is the one already done: keep a model on the card
        # that holds it and the run costs nothing for it.
        resident = _one_device_for(files, resident_device_of)
        if resident and any(device.id == resident for device in devices):
            node["inputs"]["device"] = resident
            for path in files:
                device_of_file.setdefault(path, resident)
            result.assignments.append(
                Assignment(
                    node=node_id,
                    class_type=class_type,
                    device=resident,
                    weight_bytes=0,
                    note="weights already resident on this device",
                )
            )
            continue

        already = _one_device_for(files, device_of_file.get)
        if already:
            node["inputs"]["device"] = already
            result.assignments.append(
                Assignment(
                    node=node_id,
                    class_type=class_type,
                    device=already,
                    weight_bytes=0,
                    note="weights already placed by another node of this run",
                )
            )
            continue

        wanted = sum(_size_of(path) for path in files if path not in device_of_file)
        if size == largest:
            wanted += reserve
        candidate = max(devices, key=lambda device: (device.left, device.id))
        if candidate.left < wanted:
            table = "; ".join(
                f"{device.id} ({device.name}) has {round(device.left / 1024 / 1024)} MiB available "
                f"({round(device.free / 1024 / 1024)} free + "
                f"{round(device.reclaimable / 1024 / 1024)} of this server's own cache)"
                for device in devices
            )
            raise PlacementError(
                f"the {class_type} loader of node {node_id} needs about "
                f"{round(wanted / 1024 / 1024)} MiB (model weights plus activation headroom) and no card has that "
                f"much: {table}. Free a card, lower the size, or widen SILLYTALK_DEVICES."
            )
        candidate.taken += wanted
        node["inputs"]["device"] = candidate.id
        for path in files:
            device_of_file.setdefault(path, candidate.id)
        result.assignments.append(
            Assignment(node=node_id, class_type=class_type, device=candidate.id, weight_bytes=wanted)
        )

    result.assignments.sort(key=lambda a: int(a.node) if a.node.isdigit() else 0)
    logger.info("sillytalk placement: %s", result.summary())
    return result
