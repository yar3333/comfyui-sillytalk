"""The catalogue of models this plugin can draw with.

A model is one JSON file in `<plugin>/workflows/<code>.json`; the file stem IS
the model code a client sends. The format:

    {
      "label": "Qwen-Image 2512",              // optional, for the UI
      "sizes": {                               // what the client picks an
        "square":    [1328, 1328],             // ORIENTATION of — the server
        "portrait":  [1136, 1472],             // owns the pixel numbers
        "landscape": [1472, 1136]
      },
      "orientation": "square",                 // used when the client says none
      "model_file": "qwen_image_2512_fp8.safetensors",  // PARAM_MODEL_FILE's default
      "variants": [
        { "refs": 0, "workflow": { ...ComfyUI UI-format graph... } },
        { "refs": 1, "workflow": { ... } },
        { "refs": 2, "workflow": { ... } }
      ]
    }

`sizes` is what a picture of this model is drawn at; the client sends
`orientation` and gets the pair back in the job. `model_file` is the file the
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

logger = logging.getLogger(__name__)

PLUGIN_DIR = Path(__file__).resolve().parent.parent
WORKFLOW_DIR = PLUGIN_DIR / "workflows"

# The orientations a client may ask for. Anything else is a client bug, and the
# error names the ones this model does define.
ORIENTATIONS = ("square", "portrait", "landscape")


class WorkflowError(Exception):
    """A model is unknown, or its file cannot be turned into a runnable graph."""


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
    # refs -> the UI-format graph of that reference count, ascending.
    graphs: tuple[tuple[int, dict[str, Any]], ...] = ()

    @property
    def refs(self) -> tuple[int, ...]:
        return tuple(refs for refs, _ in self.graphs)

    @property
    def max_refs(self) -> int:
        return self.graphs[-1][0] if self.graphs else -1

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

    def graph_for(self, refs: int) -> dict[str, Any]:
        for count, graph in self.graphs:
            if count == refs:
                return graph
        raise WorkflowError(
            f"the model '{self.code}' has no variant for {refs} reference image(s); "
            f"its file defines {list(self.refs)}"
        )

    def describe(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "label": self.label,
            "max_refs": self.max_refs,
            "refs": list(self.refs),
            "orientation": self.default_orientation,
            "sizes": {name: [width, height] for name, (width, height) in self.sizes},
            "model_file": self.model_file,
        }


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

    variants = raw.get("variants")
    if not isinstance(variants, list) or not variants:
        entry.error = "'variants' must be a non-empty list of {refs, workflow}"
        return entry

    seen: dict[int, dict[str, Any]] = {}
    for index, variant in enumerate(variants):
        if not isinstance(variant, dict):
            entry.error = f"variants[{index}] is not an object"
            return entry
        refs = variant.get("refs")
        if not isinstance(refs, int) or isinstance(refs, bool) or refs < 0:
            entry.error = f"variants[{index}].refs must be an integer >= 0"
            return entry
        if refs in seen:
            entry.error = f"variants[{index}]: two variants for {refs} reference image(s)"
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
        seen[refs] = graph

    entry.refs = sorted(seen)
    return entry


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
    path = Path(entry.file)
    raw = json.loads(path.read_text(encoding="utf-8"))
    graphs = {int(v["refs"]): v["workflow"] for v in raw["variants"]}
    return ModelSpec(
        code=entry.code,
        file=path,
        label=entry.label,
        sizes=tuple(entry.sizes.items()),
        default_orientation=entry.orientation,
        model_file=entry.model_file,
        graphs=tuple(sorted(graphs.items())),
    )


def api_graph(spec: ModelSpec, refs: int) -> dict[str, Any]:
    """The API-format graph of one variant, ready to submit to the queue.

    Converted once per (file version, refs) and handed out as a copy, because
    every run writes its own prompt, seed and file names into it.
    """
    ui_graph = spec.graph_for(refs)
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
    return copy.deepcopy(converted)


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
    out: dict[int, dict[str, Any]] = {}
    for refs, graph in spec.graphs:
        if _is_api_format(graph):
            api = graph
        else:
            # Only a UI-format graph needs the registry, so only then is the
            # converter imported: at custom-node import time the registry is
            # still filling, and this runs on the first job instead.
            from .vendor.workflow_converter import WorkflowConverter

            api = WorkflowConverter.convert_to_api(copy.deepcopy(graph))
        if not isinstance(api, dict) or not api:
            raise WorkflowError(
                f"the '{spec.code}' variant for {refs} references did not convert "
                f"(is its graph a ComfyUI workflow this server can load?)"
            )
        out[refs] = api
    logger.info("sillytalk: converted the '%s' workflow (%s)", spec.code, ", ".join(str(r) for r in sorted(out)))
    return out


def forget_cache() -> None:
    """Drops the converted graphs (tests, and a catalogue swapped at runtime)."""
    _api_cache.clear()
