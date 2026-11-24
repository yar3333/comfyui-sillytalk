# AGENTS.md

Guidance for working in this repository. Read `README.md` first: it is the
specification — the API, the workflow file format, the placement rule and the
LoRA-fit verdicts are all described there in detail. This file covers what a
change has to respect, and the traps that are not obvious from the code.

## What this is

A ComfyUI custom node (no nodes of its own, only HTTP routes) that turns a
ComfyUI server into an image-generation service. A client sends a prompt,
reference pictures and a model code; the server owns the workflow graph, the GPU
placement, the file lifecycle and the LoRA checks.

Python 3.10+, standard library only. `aiohttp` comes from ComfyUI. Do not add a
dependency — `dependencies = []` in `pyproject.toml` is deliberate, and the
plugin has to load inside whatever environment ComfyUI already runs in.

## Layout

```
__init__.py                     ComfyUI entry point: registers routes, prints the banner
sillytalk_plugin/
    routes.py                   HTTP surface, request parsing, __version__
    jobs.py                     one run: build graph, submit, poll, hand back PNG, sweep files
    workflows.py                the catalogue (<plugin>/workflows/*.json) and UI -> API conversion
    placement.py                which cuda:N every loader of a run gets
    loras.py                    LoRA slots: fill, bypass, bytes per loader
    refs.py                     reference slots: how many of them a run keeps
    lora_fit.py                 safetensors-header check: does this LoRA merge into this file
    model_files.py              the server's folder listings (diffusion_models, loras)
    params.py                   PARAM_* placeholder substitution
    files.py                    POST /files/delete, GET /files/list
    vendor/workflow_converter.py  unmodified third-party copy — never edit it
workflows/<code>.json           the model catalogue; file stem is the model code
scripts/                        maintenance tools, not imported by the plugin
tests/                          unittest suite, no ComfyUI, no GPU, no network
```

## Rules a change must respect

**The server owns the decisions.** Pixels behind an orientation, the card each
loader lands on, the retention of a picture — none of that may move into the
request. If a client needs a new knob, add it as a request field and validate it
here, next to the folders and devices it refers to.

**Import ComfyUI lazily.** `folder_paths`, `nodes`, `torch`, `execution`,
`server`, `comfy.model_management` are imported *inside functions*, never at
module scope. That is what lets the whole unit suite run with a stubbed
`folder_paths` and no ComfyUI installed. The one exception is the root
`__init__.py`, which must import `server.PromptServer` to register routes, and
`files.py`, which binds the directory getters at import (the test stub keeps
their state in a mutable dict for exactly that reason).

**One source of the version.** `__version__` lives in `sillytalk_plugin/routes.py`
and is re-exported by `__init__.py`. Bump it, `pyproject.toml`, and add a
version-history entry to `README.md` in the same change — all three have said
1.2.1 since that release, and the banner printed at import comes from
`__version__`, so none of them may drift.

**Placeholders are whole strings.** `params.py` replaces a widget value only when
the entire string is `PARAM_*`; `"a prefix PARAM_PROMPT"` stays untouched, on
purpose. A leftover `PARAM_*` at submit time is a `400` that names it. There is no
device placeholder: placement writes every `device` input, and a device value in a
workflow file is ignored. Do not reintroduce client-side device selection.

**Unused LoRA slots are pruned, not disabled.** `validate_prompt` rejects any
`lora_name` that is not in the folder listing, so a slot cannot stay in the graph
with an empty name or strength 0. `loras.py` bypasses it the way ComfyUI's Ctrl+B
does: re-point consumers at the upstream `model` link, then remove the node.
`strength_clip` is forced to `0` — these graphs use quantised text encoders.

**Placement is measured per run.** Free VRAM is read from torch at submit time.
`reclaimable` is only what this ComfyUI's own model cache holds; VRAM held by
another process is never counted. A file already resident stays on its card. A
graph with no device inputs, or a server with no visible GPU, is not an error —
the workflow keeps its own values and the job says so.

**Files belong to the job.** Reference pictures are written to `input/` under a
`sillytalk_<token>_…` name and removed when the job settles — including a request
that failed halfway. Results go to `output/sillytalk/<job8>/` (the workflow's own
`SaveImage` with `PARAM_STRING_OUTPUT_IMAGE_FILENAME`) and are deleted when fetched
or after `RETENTION_SECONDS`. Job records live in process memory: a restart
orphans the old pictures, which `_sweep_orphans` cleans up. Do not add a
background task at import time; sweeping happens on request, by design.

**`lora_fit` is header-only and its three verdicts are a design decision.**
`yes` merges (with a note when layers will be skipped), `no` is refused before
the queue, `unknown` is allowed on purpose — refusing a naming convention this
code does not recognise would reject working files. The refusal threshold is
"more mismatched layers than matched", not "any mismatch", and the two spellings
of the Qwen-Image 2.1 MLP are not counted as a gap. Read the module docstring
before touching that arithmetic; the comments in `tests/test_lora_fit.py` are the
cases written down.

**The catalogue file stem is the API.** `workflows/<code>.json` defines the model
code, `sizes` is required and holds the pixels the model actually draws (ComfyUI
rounds to a multiple of 16 — verify against a real run), `steps` is optional and
declares the `min`/`max` a request may ask for (the number wired into the graph
stays the default; a distilled model says `min == max`), and `variants` holds one
graph per reference count — or a range, `{"refs": {"min": 1, "max": 10}}`, for a
graph wired with that many single-loader slots, which the run cuts down to what it
sent (`refs.py`). A range is checked against the placeholders the graph carries, and
only a reference that is one loader can be cut: a reference that is a chain of nodes
of its own needs a variant per count. A file that fails to parse only removes itself from
`/sillytalk/models` and shows up in `/sillytalk/health` under `broken_workflows`.
Graphs may be UI format (converted in-process, cached by file mtime) or API format.
`vendor/workflow_converter.py` is an unmodified public-domain copy (v2.3.0) —
never edit it; replace it wholesale if it must be synced.

**No legacy-format support.** When the catalogue or request format changes, write
only the new format. No migration layers, no fallback reads. `scripts/` is where
one-off converters for old inputs belong (`import_sillytalk_workflows.py` is the
existing example).

## Testing

The suite needs `aiohttp` for the route, job and file modules; the rest runs on
plain Python. Use the interpreter that runs ComfyUI, or any venv with `aiohttp`:

```bash
python -m unittest discover -s tests -t tests
```

`tests/support.py` installs the `folder_paths` stub and provides
`write_safetensors` / `model_header` / `lora_header` so a "20 GB checkpoint" can
be a few hundred bytes of header. Tests that touch endpoints or folders subclass
`TempDirTestCase` and call `register_model_folder`. Async test methods require
`IsolatedAsyncioTestCase` — on a plain `TestCase` they silently do not run.

There is no GPU, no ComfyUI and no network in the suite. A change that needs a
live server to be believed (placement OOM messages, conversion of a new graph,
`lora key not loaded` counts) belongs in `README.md` as a measured result, not
only in the commit message.

## Conventions

- Comments and docstrings explain **why**: the constraint, the ComfyUI behaviour
  being worked around, the measurement behind a threshold. This codebase is
  heavy on them and that is intentional; match it. Do not add comments that say
  what the next line does.
- Errors that reach a client are one sentence of English that says what to do
  next, and list the valid options (`no workflow named 'x': a, b`, `this server
  has no 'f': it has …`).
- Public text (README, docstrings, error messages, commit messages, issues) is
  English.
- Line length is not enforced by a linter; the longest non-vendored line is 129
  columns, so aim for ~120 and do not worry about an occasional overflow.
  `from __future__ import annotations` and type hints are used throughout the
  plugin.
- Commit messages are prose about the user-visible effect, often with measured
  numbers, and a release commit starts with `comfyui-sillytalk <version>: …`.
  Match that style rather than `feat:`/`fix:` prefixes.
- Do not commit generated pictures, `__pycache__`, `user/` or `.qwen/tmp/`
  (already ignored).

## Environment knobs

`SILLYTALK_DEVICES` (default: all visible) and `SILLYTALK_VRAM_HEADROOM_GB`
(default `2.0`) are read per call in `placement.py`. Keep them read lazily from
the environment so tests and a long-running server behave.
