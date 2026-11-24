"""The LoRA slots of a workflow, switched on and off for one run.

A workflow offers LoRAs by carrying a chain of loader nodes whose `lora_name`
holds a placeholder:

    UNETLoader -> [LoraLoaderModelOnly PARAM_LORA_1] -> … -> [PARAM_LORA_5] -> KSampler

Wiring it once in the file (rather than building nodes here) keeps the graph
editable in the ComfyUI UI — the slots are visible, reorderable, and a run that
uses none of them is the graph as drawn.

A slot the run does not use is taken out of the graph and its consumers are
pointed at whatever fed it. That is what ComfyUI's own "Bypass" does in the
browser before it sends a prompt; there is no off-switch in the prompt format
itself, and it cannot be a strength of 0 either:

  * `LoraLoader.load_lora` does return early at strength 0 without reading the
    file, but
  * `execution.validate_prompt` still checks `lora_name` against the folder
    listing (`value_not_in_list`), so a disabled slot would have to name a file
    that exists — which there is none of in a `loras/` folder nobody filled.

An unused slot that is not in the graph is neither validated nor executed, so
removing it is both honest and free.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass, field
from typing import Any, Callable

# The classes this plugin treats as a LoRA slot. Model-only is what a qwen-image
# derivative ships and what survives a quantised text encoder; the full loader is
# accepted too so a workflow drawn in the UI keeps working.
LORA_CLASSES = ("LoraLoaderModelOnly", "LoraLoader")
SLOT_PREFIX = "PARAM_LORA_"
SLOT_RE = re.compile(r"^PARAM_LORA_(\d+)$")
# How many slots a workflow file is expected to carry. Not a limit this module
# enforces — it is the number the README tells whoever edits a workflow to wire.
SLOT_COUNT = 5


class LoraError(Exception):
    """The run asked for LoRAs the workflow cannot take."""


@dataclass
class Applied:
    """One slot that stayed in the graph."""

    node: str
    name: str
    strength: float
    # The loader whose model this LoRA is merged into — where its bytes land.
    loader: str
    bytes: int = 0


@dataclass
class LoraRun:
    applied: list[Applied] = field(default_factory=list)
    bypassed: list[str] = field(default_factory=list)

    @property
    def names(self) -> list[str]:
        return [f"{a.name}@{a.strength:g}" for a in self.applied]

    def extra_weight(self) -> dict[str, int]:
        """Bytes to add to each loader's placement, keyed by loader node."""
        out: dict[str, int] = {}
        for applied in self.applied:
            out[applied.loader] = out.get(applied.loader, 0) + applied.bytes
        return out

    def describe(self) -> list[dict[str, Any]]:
        return [{"name": a.name, "strength": a.strength, "node": a.node} for a in self.applied]


def available_slots(spec: Any) -> int:
    """How many LoRA slots one model offers (its widest variant).

    Reads the converted graph, which is cached against the file's mtime, so this
    is a dict scan by the time the catalogue is asked for it.
    """
    from .workflows import api_graph

    best = 0
    for variant in spec.variants:
        try:
            best = max(best, len(slots(api_graph(spec, variant.refs[-1]))))
        except Exception:
            continue
    return best


def slots(graph: dict[str, Any]) -> list[tuple[int, str, dict[str, Any]]]:
    """`(index, node_id, node)` of every slot, in the order they run.

    Ordered by the number in the placeholder rather than by node id: the number
    is what the client's list is matched against, and a graph re-saved by the UI
    renumbers its nodes freely.
    """
    found: list[tuple[int, str, dict[str, Any]]] = []
    for node_id, node in graph.items():
        if not isinstance(node, dict):
            continue
        if str(node.get("class_type") or "") not in LORA_CLASSES:
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        match = SLOT_RE.match(str(inputs.get("lora_name") or ""))
        if not match:
            continue
        found.append((int(match.group(1)), str(node_id), node))
    found.sort(key=lambda item: (item[0], item[1]))
    return found


def apply(
    graph: dict[str, Any],
    chosen: list[dict[str, Any]],
    size_of: Callable[[str], int] | None = None,
) -> LoraRun:
    """Fills the first `len(chosen)` slots and removes the rest (in place).

    `chosen` is `[{"name": …, "strength": …}]` in the order the caller wants them
    applied; `size_of(name)` reports the file size so the placement can count the
    LoRA onto the card its model sits on.
    """
    run = LoraRun()
    found = slots(graph)
    if not found:
        if chosen:
            raise LoraError(
                "this workflow has no LoRA slots — wire up to "
                f"{SLOT_COUNT} LoraLoaderModelOnly nodes whose lora_name is {SLOT_PREFIX}1…{SLOT_COUNT} "
                "into it to make it take LoRAs"
            )
        return run
    if len(chosen) > len(found):
        raise LoraError(
            f"this workflow has {len(found)} LoRA slot(s) and {len(chosen)} LoRA(s) were asked for"
        )

    # Ascending, because a bypass hands its consumers whatever fed it — which may
    # itself be a later slot that is bypassed in turn, and that bypass finds the
    # consumers already pointing at it.
    filled = 0
    for _number, node_id, node in found:
        if filled < len(chosen):
            _fill(node, chosen[filled], node_id, graph, run, size_of)
            filled += 1
            continue
        _bypass(graph, node_id, run)
    return run


def _fill(
    node: dict[str, Any],
    choice: dict[str, Any],
    node_id: str,
    graph: dict[str, Any],
    run: LoraRun,
    size_of: Callable[[str], int] | None,
) -> None:
    inputs = node["inputs"]
    name = str(choice.get("name") or "")
    strength = choice.get("strength")
    inputs["lora_name"] = name
    inputs["strength_model"] = 1.0 if strength is None else float(strength)
    if "strength_clip" in inputs:
        # A full LoraLoader would merge into the text encoder too; the client
        # chooses LoRAs for the picture, not for the encoder, and a quantised
        # one cannot take the merge at all.
        inputs["strength_clip"] = 0.0
    run.applied.append(
        Applied(
            node=node_id,
            name=name,
            strength=float(inputs["strength_model"]),
            loader=_loader_of(graph, node_id),
            bytes=size_of(name) if size_of else 0,
        )
    )


def _loader_of(graph: dict[str, Any], node_id: str) -> str:
    """The node whose model reaches this slot — the first one that is not a slot.

    The chain is linear, so walking `model` upstream ends at the model loader,
    and that is the card the merged weights will occupy.
    """
    seen: set[str] = {node_id}
    current = node_id
    while True:
        node = graph.get(current)
        if not isinstance(node, dict):
            return current
        source = (node.get("inputs") or {}).get("model")
        if not isinstance(source, (list, tuple)) or len(source) != 2:
            return current
        upstream = str(source[0])
        if upstream in seen or upstream not in graph:
            return upstream
        seen.add(upstream)
        current = upstream


def _bypass(graph: dict[str, Any], node_id: str, run: LoraRun) -> None:
    """Points every consumer of `node_id` at what fed it, and drops the node."""
    node = graph.get(node_id) or {}
    source = (node.get("inputs") or {}).get("model")
    for other_id, other in list(graph.items()):
        if other_id == node_id or not isinstance(other, dict):
            continue
        inputs = other.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for key, value in inputs.items():
            if isinstance(value, (list, tuple)) and len(value) == 2 and str(value[0]) == node_id:
                inputs[key] = copy.deepcopy(source) if isinstance(source, (list, tuple)) else None
    graph.pop(node_id, None)
    run.bypassed.append(node_id)
