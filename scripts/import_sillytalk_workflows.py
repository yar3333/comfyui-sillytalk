#!/usr/bin/env python3
"""Imports a SillyTalk workflow file into this plugin's catalogue format.

    python3 scripts/import_sillytalk_workflows.py <source-dir-or-file> [...] [--out workflows]

The source is the format sillytalk used to ship: one JSON file per model holding
the UI-format graph of each reference count under the keys `text`, `oneImage` and
`twoImages`. The output is this plugin's format — a `variants` list where each
entry says how many references it takes — so a fourth reference is a new entry
rather than a new key name somebody has to guess.

A `device` widget of a model loader is rewritten to the first card: this plugin
overwrites every device value at run time, and leaving `PARAM_DEVICE` in the file
would only suggest that the client still has a say in it.

Run it again after editing a graph in the ComfyUI UI and saving it back over the
sillytalk file, or edit the catalogue file directly and keep the same shape.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

VARIANT_KEYS = (("text", 0), ("oneImage", 1), ("twoImages", 2))
LABELS = {
    "qwen-image-2.1": "Qwen-Image 2.1 (references through the same model)",
    "qwen-image-2512": "Qwen-Image 2512 (references through Qwen-Image-Edit 2511)",
}
DEFAULT_SIZE = {"qwen-image-2.1": 1328, "qwen-image-2512": 1328}
FALLBACK_DEVICE = "cuda:0"


def strip_devices(node: Any) -> int:
    """Pins every device widget to a concrete card; returns how many.

    In the UI format a widget is a position in `widgets_values`, not a key, so
    the placeholder is matched by value — and in the API format it would be
    `inputs.device`, which the same rule catches.
    """
    changed = 0
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, str) and value == "PARAM_DEVICE":
                node[key] = FALLBACK_DEVICE
                changed += 1
            else:
                changed += strip_devices(value)
    elif isinstance(node, list):
        for index, item in enumerate(node):
            if isinstance(item, str) and item == "PARAM_DEVICE":
                node[index] = FALLBACK_DEVICE
                changed += 1
            else:
                changed += strip_devices(item)
    return changed


def convert(source: Path) -> dict[str, Any]:
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{source}: expected an object of workflow variants")
    if "variants" in raw:  # already this format: normalise the devices and keep it
        strip_devices(raw)
        return raw

    variants = []
    for key, refs in VARIANT_KEYS:
        graph = raw.get(key)
        if not isinstance(graph, dict) or not graph:
            continue
        strip_devices(graph)
        variants.append({"refs": refs, "workflow": graph})
    if not variants:
        raise ValueError(f"{source}: holds none of {', '.join(key for key, _ in VARIANT_KEYS)}")

    code = source.stem
    size = DEFAULT_SIZE.get(code, 1024)
    return {
        "label": LABELS.get(code, code),
        "defaults": {"width": size, "height": size},
        "variants": variants,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sources", nargs="+", help="sillytalk workflow file(s) or folder(s)")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "workflows"))
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files: list[Path] = []
    for source in args.sources:
        path = Path(source)
        if path.is_dir():
            files.extend(sorted(path.glob("*.json")))
        elif path.is_file():
            files.append(path)
        else:
            print(f"not found: {path}", file=sys.stderr)
            return 2

    for path in files:
        try:
            data = convert(path)
        except (ValueError, json.JSONDecodeError) as err:
            print(f"skipped {path}: {err}", file=sys.stderr)
            continue
        target = out_dir / f"{path.stem}.json"
        target.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        refs = ", ".join(str(variant["refs"]) for variant in data["variants"])
        print(f"{path} -> {target} (variants for {refs} reference images)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
