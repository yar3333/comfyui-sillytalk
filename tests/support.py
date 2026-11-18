"""Shared setup for the plugin's tests.

The plugin modules reach for ComfyUI's `folder_paths` / `nodes` lazily, except
`files.py`, which binds `folder_paths`'s directory getters at import. So the
stub installed here keeps its state in a mutable dict that each test repoints:
replacing the module in `sys.modules` would leave the already-bound getters
pointing at the first stub.

Run the suite with the interpreter that runs ComfyUI (it has aiohttp):

    <comfyui>/.venv/bin/python -m unittest discover -s tests -t tests
"""

from __future__ import annotations

import atexit
import json
import shutil
import struct
import sys
import tempfile
import types
import unittest
from pathlib import Path

PLUGIN_DIR = Path(__file__).resolve().parent.parent
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

# Repointed per test; the stub's getters read it when they are called.
_DIRS: dict[str, Path] = {}
# folder_name -> base folders a model file may live in.
_MODEL_PATHS: dict[str, list[Path]] = {}


def _install_stub() -> types.ModuleType:
    module = types.ModuleType("folder_paths")
    module.get_output_directory = lambda: str(_DIRS["output"])
    module.get_input_directory = lambda: str(_DIRS["input"])
    module.get_temp_directory = lambda: str(_DIRS["temp"])

    def register(folder_name: str, path: Path) -> None:
        _MODEL_PATHS.setdefault(folder_name, []).append(path)

    def get_full_path(folder_name: str, filename: str):
        for base in _MODEL_PATHS.get(folder_name, []):
            candidate = base / filename
            if candidate.is_file():
                return str(candidate)
        return None

    def get_filename_list(folder_name: str):
        """The folder's files by relative name, the way ComfyUI lists them."""
        names: list[str] = []
        for base in _MODEL_PATHS.get(folder_name, []):
            if not base.is_dir():
                continue
            for candidate in base.rglob("*"):
                if candidate.is_file() and not candidate.name.startswith("."):
                    names.append(str(candidate.relative_to(base)))
        return sorted(names)

    module.register = register
    module.get_full_path = get_full_path
    module.get_filename_list = get_filename_list
    sys.modules["folder_paths"] = module
    return module


_install_stub()
_ROOT = Path(tempfile.mkdtemp(prefix="comfyui-sillytalk-tests-"))
for _name in ("output", "input", "temp"):
    (_ROOT / _name).mkdir(parents=True, exist_ok=True)
_DIRS.update({name: _ROOT / name for name in ("output", "input", "temp")})
atexit.register(shutil.rmtree, _ROOT, True)


def register_model_folder(folder_name: str, path: Path) -> None:
    """Tells the stub where a model folder is, for the placement tests."""
    _MODEL_PATHS.setdefault(folder_name, []).append(path)


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int]]]) -> Path:
    """A file whose safetensors header lists `tensors` as `name: (dtype, shape)`.

    Only the header is written, because nothing in the plugin reads past it: a test
    can hold a "20 GB" checkpoint in a few hundred bytes.
    """
    meta = {
        name: {"dtype": dtype, "shape": list(shape), "data_offsets": [0, 0]}
        for name, (dtype, shape) in tensors.items()
    }
    body = json.dumps(meta).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(body)) + body)
    return path


def model_header(layers: dict[str, tuple[int, int]], prefix: str = "model.diffusion_model.") -> dict:
    """A checkpoint header from `{layer: (out, in)}`, in ComfyUI's own naming."""
    return {f"{prefix}{base}.weight": ("BF16", [out, width]) for base, (out, width) in layers.items()}


def lora_header(
    layers: dict[str, tuple[int, int]],
    prefix: str = "diffusion_model.",
    down: str = "lora_down",
    up: str = "lora_up",
    rank: int = 32,
) -> dict:
    """A LoRA header from `{layer: (in, out)}`, as one of the formats in the wild."""
    out: dict[str, tuple[str, list[int]]] = {}
    for base, (width, height) in layers.items():
        out[f"{prefix}{base}.{down}.weight"] = ("BF16", [rank, width])
        out[f"{prefix}{base}.{up}.weight"] = ("BF16", [height, rank])
    return out


class TempDirTestCase(unittest.IsolatedAsyncioTestCase):
    """Own output/input/temp folders.

    IsolatedAsyncioTestCase rather than TestCase: the endpoint handlers are
    coroutines, and a coroutine test method on a plain TestCase is never
    awaited — it reports as passing without running anything.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.out = self.root / "output"
        self.inp = self.root / "input"
        self.temp = self.root / "temp"
        for folder in (self.out, self.inp, self.temp):
            folder.mkdir(parents=True)
        _DIRS.update({"output": self.out, "input": self.inp, "temp": self.temp})
        _MODEL_PATHS.clear()
        self.addCleanup(self._tmp.cleanup)


__all__ = [
    "PLUGIN_DIR",
    "TempDirTestCase",
    "lora_header",
    "model_header",
    "register_model_folder",
    "unittest",
    "write_safetensors",
]
