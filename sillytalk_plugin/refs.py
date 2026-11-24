"""The reference-image slots of a workflow, filled down to what a run sent.

A workflow takes reference pictures by carrying one loader per picture whose
image widget holds a numbered placeholder:

    LoadImage(PARAM_STRING_INPUT_IMAGE_FILENAME_1) ─┐
    LoadImage(PARAM_STRING_INPUT_IMAGE_FILENAME_2) ─┴→ TextEncodeQwenImage21

Wiring the maximum in the file and shortening the graph per run — rather than one
graph per reference count — keeps the catalogue readable and the graph editable
in the ComfyUI UI: the slots are visible and reorderable, and a run that sends
`k` pictures is the graph as drawn with the last slots taken out. Ten graphs that
differ in two nodes are ten copies to re-save whenever the sampler, the size or a
LoRA slot changes, and nine of them are dead weight.

A slot the run does not use is removed rather than left pointing at nothing, for
the same reasons an unused LoRA slot is (see `loras.py`): `execution.validate_prompt`
checks every `LoadImage` name against the flat contents of `input/`, so an empty
slot is a refused prompt, and this plugin's own leftover-placeholder check refuses
a `PARAM_*` that reached the queue. A node outside the graph is neither validated
nor executed.

Trimming goes from the top, so the surviving inputs stay numbered `1..k` without a
gap — the order the encoder node walks them in.

Only a reference that is one loader feeding an optional input of another node can
be trimmed this way. A reference that is a chain of its own — `LoadImage` →
`ImageScaleToTotalPixels` → `VAEEncode` → `ReferenceLatent`, as Qwen-Image-Edit
style graphs draw it — must keep its chain: removing the loader would leave the
next node with a missing input. Such a workflow declares one variant per count
instead of a range, and `workflows.py` accepts that too.
"""

from __future__ import annotations

import re
from typing import Any

SLOT_RE = re.compile(r"^PARAM_STRING_INPUT_IMAGE_FILENAME_(\d+)$")


def slots(graph: dict[str, Any]) -> list[tuple[int, str, dict[str, Any]]]:
    """`(number, node_id, node)` of every reference slot, ordered by number.

    Ordered by the number in the placeholder rather than by node id, because that
    number is the order the pictures were sent in and a graph re-saved by the UI
    renumbers its nodes freely.
    """
    found: list[tuple[int, str, dict[str, Any]]] = []
    for node_id, node in graph.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for value in inputs.values():
            match = SLOT_RE.match(value) if isinstance(value, str) else None
            if match:
                found.append((int(match.group(1)), str(node_id), node))
                break
    found.sort(key=lambda item: (item[0], item[1]))
    return found


def trim(graph: dict[str, Any], keep: int) -> list[int]:
    """Removes every slot above `keep` from `graph`, in place.

    Returns the numbers taken out, so a caller can log which references the run
    did not send.
    """
    dropped = [(number, node_id) for number, node_id, _node in slots(graph) if number > keep]
    if not dropped:
        return []
    gone = {node_id for _number, node_id in dropped}
    for _number, node_id in dropped:
        graph.pop(node_id, None)
    # Whatever consumed a removed slot loses the input too: a node left pointing
    # at an id that is no longer in the graph is not a graph any more.
    for node in graph.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for key, value in list(inputs.items()):
            if isinstance(value, (list, tuple)) and len(value) == 2 and str(value[0]) in gone:
                inputs.pop(key)
    return [number for number, _node_id in dropped]
