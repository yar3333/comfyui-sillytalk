"""The `PARAM_*` placeholders a workflow file may hold.

A widget value that IS one of these names (whole string, no surrounding text)
gets the run's value. A value that merely contains one stays untouched, so a
prompt template in a workflow cannot be half-replaced by accident.

The device of a model loader is deliberately NOT a placeholder: this plugin
decides it per run (see `placement.py`), and a `PARAM_DEVICE` left in an older
workflow file is simply overwritten along with every other device value.
"""

from __future__ import annotations

from typing import Any

PLACEHOLDER_PREFIX = "PARAM_"


def substitute(node: Any, values: dict[str, Any]) -> Any:
    """Returns a copy of `node` with the placeholders replaced by `values`."""
    if isinstance(node, str):
        return values[node] if node in values else node
    if isinstance(node, list):
        return [substitute(item, values) for item in node]
    if isinstance(node, dict):
        return {key: substitute(value, values) for key, value in node.items()}
    return node


def find_placeholders(node: Any, into: set[str] | None = None) -> set[str]:
    """Every `PARAM_*` value still present — a run with leftovers is a bug in
    the workflow file or in this plugin, and naming them is the whole error."""
    found = into if into is not None else set()
    if isinstance(node, str):
        if node.startswith(PLACEHOLDER_PREFIX):
            found.add(node)
    elif isinstance(node, list):
        for item in node:
            find_placeholders(item, found)
    elif isinstance(node, dict):
        for value in node.values():
            find_placeholders(value, found)
    return found
