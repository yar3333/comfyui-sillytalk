"""The catalogue of models this plugin can draw with.

A model is one JSON file in `<plugin>/workflows/<code>.json`; the file stem IS
the model code a client sends. The format:

    {
      "label": "Qwen-Image 2.1",               // optional, for the UI
      "sizes": {                               // what the client picks an
        "square":    [880, 880],               // ORIENTATION of — the server
        "portrait":  [752, 976],               // owns the pixel numbers
        "landscape": [976, 752]
      },
      "orientation": "square",                 // used when the client says none
      "model_file": "qwen_image_2.1_int8_convrot.safetensors",  // PARAM_MODEL_FILE's default
      "steps": {"min": 4, "max": 50},           // optional: the range a client may ask for
      "variants": [
        { "refs": 0, "workflow": { ...ComfyUI UI-format graph... } },
        { "refs": {"min": 1, "max": 10}, "workflow": { ...10 reference slots... } }
      ]
    }

`sizes` is what a picture of this model is drawn at; the client sends
`orientation` and gets the pair back in the job. `steps` declares whether the
client may choose a step count at all: the number wired into the graph is the
default, and a model distilled for a fixed few steps says so with `min == max`
(for example `{"min": 4, "max": 4}`). A file without it takes no `steps` field.

`refs` is either the one reference count a graph was drawn for, or
`{"min": …, "max": …}` for a graph wired with the maximum number of reference
slots — the plugin fills the first *k* and takes the rest out (`refs.py`). A
range is checked against the placeholders the graph actually carries, so a file
cannot promise ten references and wire two. Only a reference that is a single
loader can be cut that way: a graph whose reference is a chain of its own gets
one variant per count.

`model_file` is the file the
workflow's `PARAM_MODEL_FILE` loader uses when the client does not name one —
the client may name any file of that folder, which is how a derivative of the
model gets used without touching a workflow file.

LoRA slots are not declared here: they are the `LoraLoaderModelOnly` nodes of
the graph whose `lora_name` holds `PARAM_LORA_1`…`PARAM_LORA_5` (see
`loras.py`). A workflow without them simply cannot take LoRAs, and says so.

The graph is kept in the format ComfyUI saves, so it stays editable in the
ComfyUI UI: open the file, drag the nodes around, save, paste the graph back
under `workflow`. The conversion to the API format happens in-process with the
server's own node registry (the vendored converter) and is cached against the
file's mtime, so an edited file is converted again on the next job.

Nothing here is importable without ComfyUI's `nodes` module being loaded, which
is why the catalogue is read lazily: custom nodes are imported while the
registry is still being filled, and a graph converted then would lose every
class a sibling custom node registers after this one.
"""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import refs as refs_module
from .params import find_placeholders

logger = logging.getLogger(__name__)

PLUGIN_DIR = Path(__file__).resolve().parent.parent
WORKFLOW_DIR = PLUGIN_DIR / "workflows"

# The orientations a client may ask for. Anything else is a client bug, and the
# error names the ones this model does define.
ORIENTATIONS = ("square", "portrait", "landscape")


class WorkflowError(Exception):
    """A model is unknown, or its file cannot be turned into a runnable graph."""


@dataclass(frozen=True)
class Variant:
    """One graph and the reference counts it serves.

    Exactly one count for a graph drawn per count, which is what a reference that
    is a chain of its own needs, or a whole range for a graph wired with the
    maximum number of single-loader slots — `refs.trim` shortens that one per run.
    """

    refs: tuple[int, ...]
    graph: dict[str, Any]


@dataclass(frozen=True)
class ModelSpec:
    """One readable workflow file."""

    code: str
    file: Path
    label: str
    # orientation -> (width, height), in the order the file declares them.
    sizes: tuple[tuple[str, tuple[int, int]], ...] = ()
    default_orientation: str = "square"
    # The file PARAM_MODEL_FILE falls back to, or None when the graph has no
    # such placeholder and therefore offers no choice of model file.
    model_file: str | None = None
    # The step counts a request may ask for, or None when the graph owns the
    # sampler settings outright. The number wired into the graph is the default,
    # so a file distilled for four steps declares min == max == 4.
    steps_range: tuple[int, int] | None = None
    # The graphs this file offers, ordered by their first reference count.
    variants: tuple[Variant, ...] = ()

    @property
    def refs(self) -> tuple[int, ...]:
        return tuple(sorted({count for variant in self.variants for count in variant.refs}))

    @property
    def max_refs(self) -> int:
        return max(self.refs) if self.variants else -1

    def size_for(self, orientation: str | None) -> tuple[int, int]:
        """The pixel pair of one orientation, or the model's own default."""
        wanted = orientation or self.default_orientation
        for name, pair in self.sizes:
            if name == wanted:
                return pair
        raise WorkflowError(
            f"the model '{self.code}' has no {wanted} size; it draws "
            f"{', '.join(f'{name} {width}x{height}' for name, (width, height) in self.sizes)}"
        )

    def variant_for(self, refs: int) -> Variant:
        for variant in self.variants:
            if refs in variant.refs:
                return variant
        raise WorkflowError(
            f"the model '{self.code}' has no variant for {refs} reference image(s); "
            f"its file defines {list(self.refs)}"
        )

    def graph_for(self, refs: int) -> dict[str, Any]:
        return self.variant_for(refs).graph

    def describe(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "code": self.code,
            "label": self.label,
            "max_refs": self.max_refs,
            "refs": list(self.refs),
            "orientation": self.default_orientation,
            "sizes": {name: [width, height] for name, (width, height) in self.sizes},
            "model_file": self.model_file,
        }
        if self.steps_range:
            low, high = self.steps_range
            out["steps"] = {"min": low, "max": high}
        return out


@dataclass
class CatalogueEntry:
    """One file of the catalogue as it was read — usable or not.

    The self-test prints this table, so the reason a file is unusable travels
    with it instead of only reaching the server log.
    """

    code: str
    file: str
    label: str = ""
    refs: list[int] = field(default_factory=list)
    sizes: dict[str, tuple[int, int]] = field(default_factory=dict)
    orientation: str = "square"
    model_file: str | None = None
    steps_range: tuple[int, int] | None = None
    # (the reference counts it serves, its graph) as the file declares them.
    variants: list[tuple[tuple[int, ...], dict[str, Any]]] = field(default_factory=list)
    error: str | None = None


# code -> (mtime of the file, refs -> API-format graph). Keyed by mtime so an
# edited workflow file converts again instead of being shadowed by the old copy.
_api_cache: dict[str, tuple[float, dict[int, dict[str, Any]]]] = {}


def _read_file(path: Path) -> CatalogueEntry:
    entry = CatalogueEntry(code=path.stem, file=str(path))
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as err:
        entry.error = f"not readable — {err}"
        return entry
    except json.JSONDecodeError as err:
        entry.error = f"not valid JSON — {err}"
        return entry
    if not isinstance(raw, dict):
        entry.error = "expected an object with 'variants'"
        return entry

    entry.label = str(raw.get("label") or path.stem)
    sizes = raw.get("sizes")
    if not isinstance(sizes, dict) or not sizes:
        entry.error = "'sizes' must be an object of orientation -> [width, height]"
        return entry
    for name, pair in sizes.items():
        if name not in ORIENTATIONS:
            entry.error = f"sizes.{name} is not one of {', '.join(ORIENTATIONS)}"
            return entry
        pixels = _pixel_pair(pair)
        if pixels is None:
            entry.error = f"sizes.{name} must be [width, height] of positive pixels"
            return entry
        entry.sizes[name] = pixels
    declared = str(raw.get("orientation") or "").strip()
    if declared and declared not in entry.sizes:
        entry.error = f"'orientation' is {declared}, which 'sizes' does not define"
        return entry
    entry.orientation = declared or ("square" if "square" in entry.sizes else next(iter(entry.sizes)))
    model_file = raw.get("model_file")
    if model_file is not None and not (isinstance(model_file, str) and model_file.strip()):
        entry.error = "'model_file' must be a file name or absent"
        return entry
    entry.model_file = str(model_file).strip() if model_file else None
    steps = raw.get("steps")
    if steps is not None:
        parsed_steps = _steps_range(steps)
        if parsed_steps is None:
            entry.error = "'steps' must be an object of exactly 'min' and 'max' whole numbers, min <= max"
            return entry
        entry.steps_range = parsed_steps

    variants = raw.get("variants")
    if not isinstance(variants, list) or not variants:
        entry.error = "'variants' must be a non-empty list of {refs, workflow}"
        return entry

    seen: set[int] = set()
    for index, variant in enumerate(variants):
        if not isinstance(variant, dict):
            entry.error = f"variants[{index}] is not an object"
            return entry
        counts = _refs_counts(variant.get("refs"))
        if counts is None:
            entry.error = (
                f"variants[{index}].refs must be an integer >= 0, or an object of exactly "
                f"'min' and 'max' whole numbers with min <= max"
            )
            return entry
        clash = [count for count in counts if count in seen]
        if clash:
            entry.error = f"variants[{index}]: two variants for {clash[0]} reference image(s)"
            return entry
        graph = variant.get("workflow")
        if not isinstance(graph, dict) or not graph:
            entry.error = f"variants[{index}].workflow must be a non-empty ComfyUI graph"
            return entry
        if "nodes" not in graph and not _is_api_format(graph):
            entry.error = (
                f"variants[{index}].workflow holds neither 'nodes' (the UI format) nor "
                f"'class_type' (the API format)"
            )
            return entry
        if len(counts) > 1:
            wired = _wired_slots(graph)
            if wired != set(counts):
                entry.error = (
                    f"variants[{index}] serves {counts[0]}..{counts[-1]} reference images but wires "
                    f"{_slot_text(wired)}; a range needs one numbered "
                    f"PARAM_STRING_INPUT_IMAGE_FILENAME_n slot per reference"
                )
                return entry
        seen.update(counts)
        entry.variants.append((counts, graph))

    entry.variants.sort(key=lambda item: item[0][0])
    entry.refs = sorted(seen)
    return entry


def _refs_counts(value: Any) -> tuple[int, ...] | None:
    """`2` or `{"min": 1, "max": 10}` as every reference count that variant serves.

    The range form is what a graph wired for the maximum declares: one number
    means "this graph, drawn for exactly this many references", and the ten
    almost identical graphs that spelling would need are what the range avoids.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return (value,) if value >= 0 else None
    if not isinstance(value, dict) or set(value) != {"min", "max"}:
        return None
    limits: list[int] = []
    for name in ("min", "max"):
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            return None
        limits.append(item)
    low, high = limits
    if low > high:
        return None
    return tuple(range(low, high + 1))


def _wired_slots(graph: dict[str, Any]) -> set[int]:
    """The reference numbers a graph actually carries, in either file format.

    Read from the placeholders rather than from the converted graph, because this
    runs while the catalogue is being read — before anything may touch the node
    registry.
    """
    found: set[int] = set()
    for name in find_placeholders(graph):
        match = refs_module.SLOT_RE.match(name)
        if match:
            found.add(int(match.group(1)))
    return found


def _slot_text(numbers: set[int]) -> str:
    return "no PARAM_STRING_INPUT_IMAGE_FILENAME slot" if not numbers else f"slots {sorted(numbers)}"


def _steps_range(value: Any) -> tuple[int, int] | None:
    """`{"min": 4, "max": 50}` as a pair of positive step counts.

    Exactly these two keys: a `default` would be a second source of the number
    the graph already wires, and a `fixed` flag says nothing that `min == max`
    does not.
    """
    if not isinstance(value, dict) or set(value) != {"min", "max"}:
        return None
    out: list[int] = []
    for name in ("min", "max"):
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, int) or item < 1:
            return None
        out.append(item)
    return (out[0], out[1]) if out[0] <= out[1] else None


def _pixel_pair(value: Any) -> tuple[int, int] | None:
    """`[1328, 1328]` as a pair of positive pixels, or None."""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    out: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            return None
        out.append(item)
    return out[0], out[1]


def entries() -> list[CatalogueEntry]:
    """Every workflow file of the catalogue, readable or not."""
    found: list[CatalogueEntry] = []
    try:
        files = sorted(WORKFLOW_DIR.glob("*.json"))
    except OSError:
        return found
    for path in files:
        found.append(_read_file(path))
    return found


def models() -> list[ModelSpec]:
    """The models a client can ask for, by code."""
    specs: list[ModelSpec] = []
    for entry in entries():
        if entry.error:
            logger.error("sillytalk workflow %s: %s", entry.file, entry.error)
            continue
        specs.append(_spec(entry))
    return sorted(specs, key=lambda spec: spec.code)


def model(code: str) -> ModelSpec | None:
    """One model by code, or None when no readable file defines it."""
    if not code or "/" in code or "\\" in code or "\0" in code:
        return None
    path = WORKFLOW_DIR / f"{code}.json"
    if not path.is_file():
        return None
    entry = _read_file(path)
    if entry.error:
        logger.error("sillytalk workflow %s: %s", entry.file, entry.error)
        return None
    return _spec(entry)


def _spec(entry: CatalogueEntry) -> ModelSpec:
    return ModelSpec(
        code=entry.code,
        file=Path(entry.file),
        label=entry.label,
        sizes=tuple(entry.sizes.items()),
        default_orientation=entry.orientation,
        model_file=entry.model_file,
        steps_range=entry.steps_range,
        variants=tuple(Variant(refs=counts, graph=graph) for counts, graph in entry.variants),
    )


def api_graph(spec: ModelSpec, refs: int) -> dict[str, Any]:
    """The API-format graph of one variant, ready to submit to the queue.

    Converted once per file version and handed out as a copy, because every run
    writes its own prompt, seed and file names into it — and, for a variant wired
    with the maximum number of references, with the slots this run does not use
    already taken out.
    """
    # Raises with the counts this file does define when the run is not one of them.
    spec.graph_for(refs)
    try:
        mtime = spec.file.stat().st_mtime
    except OSError:
        mtime = 0.0
    cached = _api_cache.get(spec.code)
    if cached is None or cached[0] != mtime:
        _api_cache[spec.code] = (mtime, _convert_all(spec))
        cached = _api_cache[spec.code]
    converted = cached[1].get(refs)
    if converted is None:
        raise WorkflowError(f"the '{spec.code}' variant for {refs} references converted to nothing")
    out = copy.deepcopy(converted)
    refs_module.trim(out, refs)
    return out


def _is_api_format(graph: dict[str, Any]) -> bool:
    """Whether a graph is already what `/prompt` wants (a workflow file may be).

    Decided here rather than by the converter so that an API-format catalogue
    file never pulls in a 1400-line module to be told what it already is.
    """
    if "nodes" in graph:
        return False
    values = [node for node in graph.values() if isinstance(node, dict)]
    return bool(values) and all("class_type" in node for node in values)


def _convert_all(spec: ModelSpec) -> dict[int, dict[str, Any]]:
    """refs -> API graph, converting each distinct graph once.

    A range variant is one graph under several keys, so the conversion walks the
    variants rather than the counts: ten references out of one template must not
    mean ten conversions of the same file.
    """
    out: dict[int, dict[str, Any]] = {}
    converted: dict[int, dict[str, Any]] = {}
    for variant in spec.variants:
        key = id(variant.graph)
        api = converted.get(key)
        if api is None:
            if _is_api_format(variant.graph):
                api = variant.graph
            else:
                # Only a UI-format graph needs the registry, so only then is the
                # converter imported: at custom-node import time the registry is
                # still filling, and this runs on the first job instead.
                from .vendor.workflow_converter import WorkflowConverter

                api = WorkflowConverter.convert_to_api(copy.deepcopy(variant.graph))
            if not isinstance(api, dict) or not api:
                raise WorkflowError(
                    f"the '{spec.code}' variant for {variant.refs[0]} reference image(s) did not convert "
                    f"(is its graph a ComfyUI workflow this server can load?)"
                )
            converted[key] = api
        for count in variant.refs:
            out[count] = api
    logger.info("sillytalk: converted the '%s' workflow (%s)", spec.code, ", ".join(str(r) for r in sorted(out)))
    return out


def steps_default(spec: ModelSpec) -> int | None:
    """The step count the graph wires, when every variant agrees on one.

    Read out of the converted graph rather than declared in the catalogue: the
    number a UI user sees on the KSampler is the number a client gets when it
    sends no `steps`, so a graph edited in ComfyUI keeps that answer true.
    """
    found: set[int] = set()
    for variant in spec.variants:
        try:
            graph = api_graph(spec, variant.refs[-1])
        except WorkflowError:
            continue
        for node in graph.values():
            if not isinstance(node, dict) or not isinstance(node.get("inputs"), dict):
                continue
            value = node["inputs"].get("steps")
            if isinstance(value, int) and not isinstance(value, bool):
                found.add(value)
    return found.pop() if len(found) == 1 else None


def forget_cache() -> None:
    """Drops the converted graphs (tests, and a catalogue swapped at runtime)."""
    _api_cache.clear()
