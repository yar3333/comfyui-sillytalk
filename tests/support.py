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
import shutil
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

    module.register = register
    module.get_full_path = get_full_path
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


__all__ = ["PLUGIN_DIR", "TempDirTestCase", "register_model_folder", "unittest"]
