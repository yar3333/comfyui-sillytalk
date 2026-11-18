"""The files of this server that a client is allowed to pick.

The point of the whole feature: whoever wants a derivative of a model drops the
file into `models/diffusion_models/` (or a LoRA into `models/loras/`) and it
appears in the app's settings dialog — they never type a file name, because a
name typed by hand is a name that can be wrong in a way only the server knows
about.

The listing is ComfyUI's own (`folder_paths.get_filename_list`), which is the
same list the node's widget offers, subfolders included. It is cached for a few
seconds because `/sillytalk/models` is asked on every probe of every client and
a folder of a few hundred files is a real scan.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# How long a listing is trusted. Long enough to collapse the probes of a busy
# app into one scan, short enough that a file dropped in is selectable without
# restarting the server.
CACHE_SECONDS = 15.0

# The two folders this plugin offers a choice from. The text encoder and the VAE
# are deliberately not here: a qwen-image derivative changes the diffusion model,
# and picking a mismatched encoder is a way to make a picture worse.
DIFFUSION_FOLDER = "diffusion_models"
LORA_FOLDER = "loras"

_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}


def listing(folder: str) -> list[dict[str, Any]]:
    """`[{"name": …, "mb": …}]` for one model folder, by name."""
    now = time.time()
    cached = _cache.get(folder)
    if cached is not None and now - cached[0] < CACHE_SECONDS:
        return cached[1]
    found = _scan(folder)
    _cache[folder] = (now, found)
    return found


def _scan(folder: str) -> list[dict[str, Any]]:
    import folder_paths

    try:
        names = list(folder_paths.get_filename_list(folder) or [])
    except Exception as err:  # a folder this ComfyUI does not know is an empty choice
        logger.warning("sillytalk: could not list the %s folder (%s)", folder, err)
        return []
    out: list[dict[str, Any]] = []
    for name in names:
        entry: dict[str, Any] = {"name": str(name)}
        full = full_path(folder, str(name))
        if full is not None:
            try:
                entry["mb"] = round(full.stat().st_size / 1024 / 1024)
            except OSError:
                pass
        out.append(entry)
    out.sort(key=lambda item: item["name"].casefold())
    return out


def full_path(folder: str, name: str) -> Path | None:
    """Where a listed file actually is, or None when it is not on disk."""
    if not name:
        return None
    import folder_paths

    try:
        found = folder_paths.get_full_path(folder, name)
    except Exception:
        return None
    return Path(found) if found else None


def size_of(folder: str, name: str) -> int:
    """Bytes of one file, 0 when it cannot be measured (it still may run)."""
    full = full_path(folder, name)
    if full is None:
        return 0
    try:
        return full.stat().st_size
    except OSError:
        return 0


def known(folder: str, name: str) -> bool:
    """Whether the server really has that file under that name."""
    return any(entry["name"] == name for entry in listing(folder))


def forget_cache() -> None:
    """Drops the cached listings (tests, and a folder changed on purpose)."""
    _cache.clear()
