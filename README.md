# comfyui-sillytalk

A ComfyUI custom node that turns a ComfyUI server into an image-generation
service for [SillyTalk](https://github.com/yar3333/sillytalk) — and for anything
else that wants a picture rather than a graph.

The client sends **a prompt, its reference pictures and a model code**. It gets
back a job it polls, and then the finished PNG. Everything else stays on the
server, where it belongs:

- which workflow graph a model code means,
- **which GPU each model loader lands on**, chosen from the VRAM that is
  actually free at the moment the job is submitted,
- where the picture is written and when it is removed again.

```bash
curl -s -F model=qwen-image-2.1 \
        -F "prompt=a red cube on a white table, studio light" \
        http://localhost:8188/sillytalk/generate
```

```json
{"job_id": "6edaa0fb-0653-4e6b-92e9-851b8666fd62", "status": "queued", "orientation": "square",
 "width": 1328, "height": 1328, "steps": 25, "model_file": "qwen_image_2.1_int8_convrot.safetensors",
 "loras": [{"name": "my-style.safetensors", "strength": 0.6, "node": "9"}],
 "placements": [{"node": "1", "class": "UNETLoaderMultiGPU", "device": "cuda:2", "mb": 6921},
                {"node": "2", "class": "CLIPLoaderMultiGPU", "device": "cuda:3", "mb": 10966}]}
```

The client asks for a **shape** — `square`, `portrait` or `landscape` — and the
answer says what that became in pixels. Every model was trained at its own
numbers, and only this server knows them, so they live in its workflow file and
nowhere else.

## Why the split is this way

The alternative — a client that holds workflow files and picks devices — goes
wrong in two places.

A workflow file in the client means every client has to know the shape of the
server: node names, links, the API format, which model file is installed. The
server is the only place that knows those.

A device chosen in a client config is a guess about a moment that has not
happened yet. On a machine where the same cards also serve LLM inference, or
where two ComfyUI instances share them, `cuda:0` chosen once in a config file is
wrong the first time anything else loads — and a loader pinned to a full card
does not fail politely, it takes the run down with an out-of-memory. Free VRAM
is only knowable at submit time, by asking torch, on the server.

So the plugin owns the decision and reports it back. Every job response says
which card each loader got and why, so a surprising placement is visible instead
of mysterious.

## Install

Drop the folder into `ComfyUI/custom_nodes/` and restart:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yar3333/comfyui-sillytalk
```

Python 3.10 or newer. No dependencies beyond what ComfyUI itself uses.

A working install says this in the log:

```
[comfyui-sillytalk] v1.3.0 — POST /sillytalk/generate, GET /sillytalk/models
```

and answers `GET /sillytalk/health`.

The workflow files in [`workflows/`](workflows/) are the catalogue; see
[Adding a model](#adding-a-model) for what to put there. The nodes a workflow
uses must exist on the server — the shipped ones use the multi-GPU loaders
(`UNETLoaderMultiGPU`, `CLIPLoaderMultiGPU`, `VAELoaderMultiGPU`), which is the
point of letting the server place them.

## API

| Method | Path | What |
| --- | --- | --- |
| GET | `/sillytalk` | what this is, and the endpoint list |
| GET | `/sillytalk/health` | version, catalogue, devices, free VRAM, resident models |
| GET | `/sillytalk/models` | the model codes a client may ask for |
| POST | `/sillytalk/generate` | multipart: prompt + pictures → `202` with a `job_id` |
| GET | `/sillytalk/jobs/{job_id}` | where the run is, and how it was placed |
| GET | `/sillytalk/jobs/{job_id}/image` | the finished picture, as bytes |
| DELETE | `/sillytalk/jobs/{job_id}` | drop it from the queue, or interrupt it |
| POST | `/files/delete` | remove files from `output/`, `input/`, `temp/` |
| GET | `/files/list` | list them |

Nothing here is authenticated, like the rest of a default ComfyUI install. Keep
the server on a network you trust, as you already would for `/prompt`.

### GET /sillytalk/models

One entry per usable workflow, and next to each the files of this server that it
could be pointed at:

```json
{"models": [{
  "code": "qwen-image-2.1",
  "label": "Qwen-Image 2.1 (references through the same model)",
  "max_refs": 2,
  "orientation": "square",
  "sizes": {"square": [1328, 1328], "portrait": [1136, 1472], "landscape": [1472, 1136]},
  "steps": {"min": 4, "max": 50, "default": 25},
  "model_file": "qwen_image_2.1_int8_convrot.safetensors",
  "model_files": {"folder": "diffusion_models",
                  "options": [{"name": "qwen_image_2.1_int8_convrot.safetensors", "mb": 6919},
                              {"name": "my-qwen-derivative.safetensors", "mb": 20000}]},
  "loras": {"max": 5,
            "options": [{"name": "my-style.safetensors", "mb": 224,
                         "fits": {"qwen_image_2.1_int8_convrot.safetensors": {"fits": "no"},
                                  "my-qwen-derivative.safetensors": {"fits": "yes"},
                                  "qwen_image_2512_fp8_e4m3fn.safetensors": {"fits": "yes"}}}]}
}]}
```

`model_files` is only there when the workflow has a swappable file, and
`steps` only when the model lets a request choose the step count: `default` is the
number its graph wires, `min`/`max` what a request may ask for instead.
`loras.max` is the number of slots its graph wired — `0` for a workflow that takes
no LoRAs. The two option lists are read from the server's folders (subfolders
included, under their relative names), so a file the user drops in becomes a
choice in the client without either side being reconfigured. The listing is cached
for 15 seconds, which is long enough for a settings dialog to open quickly and
short enough that a new file appears without a restart.

Each LoRA also answers, per model file, whether it would merge into that one —
see [Which LoRA fits which file](#which-lora-fits-which-file). `reason` is the
sentence the client shows and is present whenever there is one to show: for a `no`,
for an `unknown`, and for a `yes` that would land only partly. `fits` is absent for
a pair whose headers could not be read, which is not a verdict; a LoRA with no
`fits` at all is offered as before.

### POST /sillytalk/generate

`multipart/form-data`:

| Field | Required | Notes |
| --- | --- | --- |
| `model` | yes | a code from `/sillytalk/models` |
| `prompt` | yes | at most 8000 characters |
| `image` | no | repeatable, in order. PNG, JPEG, WebP or BMP |
| `orientation` | no | `square`, `portrait` or `landscape`; defaults to the model's own. The pixels behind it are the model's, so a shape it cannot draw is a `400` that lists the ones it can |
| `model_file` | no | a file of `models/diffusion_models/` to load instead of the one the workflow names — how a derivative someone dropped there gets used |
| `loras` | no | a JSON array of `{"name": …, "strength": …}` (strength optional, default `1.0`), named in the order they merge. Bare strings are accepted too. At most 8 |
| `seed` | no | a new one per job when omitted |
| `steps` | no | a whole number inside the range the model declares (`/sillytalk/models` → `steps.min`/`steps.max`). Omitted, the number the workflow wires is used. A model distilled for a fixed count accepts only that count; a model whose catalogue has no `steps` refuses the field |

`model_file` and `loras` are checked against the server's own folders before the
job is queued, so a name that is not there is answered with the list that is:

```json
{"error": "this server has no 'my-qwen.safetensors' in models/diffusion_models/ — it has: qwen_image_2.1_int8_convrot.safetensors, qwen_image_2512_fp8_e4m3fn.safetensors. Drop the file there and ask again."}
```

A LoRA the file being loaded cannot take is refused the same way, because the
alternative is a run that reports itself a success and draws the picture the model
makes without it:

```json
{"error": "a LoRA of this run does not fit 'qwen_image_2.1_int8_convrot.safetensors': 'Qwen-Image-Lightning-4steps-V1.0.safetensors' — 128 of the layers it names are at a size this model does not have, so ComfyUI would load it and skip them — the picture would be close to the one you get without it. What models/loras/ has that it either fits or cannot judge: nothing."}
```

Limits: 8 reference pictures, 32 MB each, 96 MB per request. A file whose name
carries no supported extension is refused — a reference has to be a picture, and
saving something else under `.png` only moves the failure into the run.

```bash
curl -s -F model=qwen-image-2512 \
        -F "prompt=turn these two objects into one calm still life" \
        -F image=@ref1.png -F image=@ref2.png \
        -F orientation=portrait \
        -F model_file=my-qwen-derivative.safetensors \
        -F 'loras=[{"name":"my-style.safetensors","strength":0.6}]' \
        -F seed=42 \
        http://localhost:8188/sillytalk/generate
```

`202` returns the job, including the placement decided at submit time. `400`
returns `{"error": "…"}` for anything the client got wrong, including a model
code that is not in the catalogue and a graph that no card can hold.

### Getting the picture

```bash
id=6edaa0fb-0653-4e6b-92e9-851b8666fd62
curl -s http://localhost:8188/sillytalk/jobs/$id          # status
curl -s http://localhost:8188/sillytalk/jobs/$id/image -o out.png
```

Statuses are `queued`, `running`, `succeeded`, `failed`, `cancelled`.
`/image` answers `409` while the job is still going, `410` when it failed or the
picture has already been discarded, `404` for a job this process never started.

`DELETE` cancels: a queued job is taken out of the queue, a running one is
interrupted.

### What the server keeps, and for how long

- A reference picture is written into `input/` under a `sillytalk_…` name and
  deleted when the job settles. A request that was refused part-way leaves
  nothing behind.
- The result is written into `output/sillytalk/<job>/` by the workflow's own
  `SaveImage`. `/image` returns those bytes and removes the file. If nobody
  fetches it, the copy is discarded 15 minutes after the job finished.
- Job records live an hour, or the 500 most recent, whichever comes first.

This is why the plugin exists next to a `SaveImage` rather than instead of one:
the picture is a real file on the server while the run is alive, and it is gone
once the client has it.

### POST /files/delete, GET /files/list

ComfyUI core has no way to remove a file over HTTP — its only DELETE route is
`/userdata/{file}`, and `DELETE /api/assets/{id}` removes the database row while
the file on disk stays. These two endpoints fill that gap, and they live here
because this plugin is what writes files.

```json
{"files": [{"filename": "a.png", "subfolder": "", "type": "output"}]}
```

`filename`/`subfolder`/`type` are exactly what ComfyUI reports in a job history
(or in the `/upload/image` response). Up to 64 files per request.

```json
{"ok": true, "deleted": ["output/a.png"], "absent": ["output/already-gone.png"], "skipped": []}
```

A file that is already gone is `absent`, so retrying after a lost response is
harmless. `ok` is false only when `skipped` is not empty, and each entry there
carries the reason.

Only regular files inside ComfyUI's own `output/`, `input/` or `temp/` can be
removed: absolute paths, drive letters, UNC paths, `~` and `..` are refused, and
so is anything whose resolved path ends up outside that directory — a junction
or symlink inside `output/` cannot be used to reach elsewhere. Nothing is
globbed, recursive or shell-based: one unlink per exact filename.

`GET /files/list?type=output|input|temp` takes optional `prefix=`,
`older_than_min=`, `limit=` (default 200, max 2000) and `all=1`, newest first:

```json
{"type": "output", "directory": "/srv/comfyui/output", "total": 1, "truncated": false,
 "files": [{"filename": "a.png", "subfolder": "", "type": "output", "bytes": 1048576, "modified": 1762430000}]}
```

## GPU placement

Every `device` input in a graph is written by the plugin at submit time. A
device value inside a workflow file is ignored — that is the point.

The rule is first-fit-decreasing bin packing over the VRAM that is actually
available:

1. Each loader node is measured by the model files it names, on disk.
2. Heaviest first. The largest model also carries an activation headroom
   (2 GiB by default) — that is where sampling happens.
3. A model goes to the card with the most room, where room is *free VRAM, plus
   what this server's own cache holds there and could drop, minus what this run
   already put on it*.
4. A file that is already resident stays where it is: reloading 20 GB to free an
   empty card would be silly, and the second loader of a file another node of
   the same run already placed costs nothing at all.

The distinction in step 3 is the whole difference between a card that is *full*
and a card that is *busy*. VRAM held by another process is not reclaimable and
is never counted. VRAM held by this ComfyUI's own model cache is, because the
server drops it when asked.

When nothing fits, the job is refused before it is queued, with the numbers:

```
the UNETLoaderMultiGPU loader of node 1 needs about 21532 MiB (model weights plus
activation headroom) and no card has that much: cuda:0 (AMD Radeon Graphics) has
206 MiB available (206 free + 0 of this server's own cache); cuda:2 has 14444 MiB
available (14444 free + 0 …). Free a card, lower the size, or widen SILLYTALK_DEVICES.
```

### Environment

| Variable | Default | Meaning |
| --- | --- | --- |
| `SILLYTALK_DEVICES` | all visible | comma-separated device ids this plugin may use, e.g. `cuda:2,cuda:3` |
| `SILLYTALK_VRAM_HEADROOM_GB` | `2.0` | activation headroom kept for the largest model |

`SILLYTALK_DEVICES` matters on a machine where some cards are reserved for other
work. Note that a runtime filtering these by its own ordinal — ROCm's
`HIP_VISIBLE_DEVICES` is the usual one — numbers cards differently from the
vendor tool, so check which `cuda:N` a card really is before reserving it.

If the server reports no CUDA device at all, or a workflow has no device inputs,
nothing is refused: the workflow keeps its own values and the job says so.

## The workflow catalogue

One JSON file per model in [`workflows/`](workflows/). **The file name without
its extension is the model code** a client sends.

```json
{
  "label": "Qwen-Image 2.1 (references through the same model)",
  "sizes": {
    "square": [1328, 1328],
    "portrait": [1136, 1472],
    "landscape": [1472, 1136]
  },
  "orientation": "square",
  "model_file": "qwen_image_2.1_int8_convrot.safetensors",
  "steps": { "min": 4, "max": 50 },
  "variants": [
    { "refs": 0, "workflow": { … } },
    { "refs": 1, "workflow": { … } },
    { "refs": 2, "workflow": { … } }
  ]
}
```

`sizes` is what the model was trained at, per shape, and it is required: a client
sends a shape and this is the table the pixels come from. Write the numbers the
picture actually comes back with — ComfyUI rounds a size to a multiple of 16, so
a portrait declared `1140` is drawn `1136` wide, and a catalogue that promises
what never arrives is worse than one that says nothing. `orientation` is the
shape used when a request names none (default `square`); a model that lists only
`square` answers every request with it.

`model_file` is the file the graph loads by default. Naming it is what makes the
file swappable: `/sillytalk/models` then publishes the whole folder, and a client
can ask for another one of it. A workflow without it — one whose LoRA is welded
in, say — keeps its own file and says so by not offering a choice.

`steps` is optional and is the one sampler setting a client may move. It declares
the range, not the number: the step count wired into the graph is what a request
that names none runs with, and `/sillytalk/models` reports it as `steps.default`.
A model distilled down to four steps declares `{ "min": 4, "max": 4 }`, so asking
for twenty-five is a `400` instead of the smear that a step LoRA produces away
from its count. A workflow without `steps` accepts no such field at all — its
graph owns the sampler, as it owns the shape and the devices.

A `workflow` is either the API format (what *Save (API)* produces) or the UI
format (what *Save* produces). UI graphs are converted once, on first use, with
the vendored converter, and the result is cached until the file's modification
time changes.

A file that does not parse, or whose variants are broken, does not take the rest
of the catalogue with it: it disappears from `/sillytalk/models` and appears in
`/sillytalk/health` under `broken_workflows` with the reason.

### Placeholders

A graph asks for the run's values by holding the whole placeholder as a string
value — `"PARAM_PROMPT"`, not `"a prefix PARAM_PROMPT"`.

| Placeholder | Type | Value |
| --- | --- | --- |
| `PARAM_PROMPT` | string | the prompt |
| `PARAM_SEED` | int | the seed |
| `PARAM_MODEL_FILE` | string | a file of `models/diffusion_models/` — the workflow's own unless the request named another |
| `PARAM_LORA_1` … `_5` | string | the LoRA of that slot, from `models/loras/` |
| `PARAM_LORA_STRENGTH_1` … `_5` | float | its strength |
| `PARAM_INT_OUTPUT_IMAGE_WIDTH` | int | the width the requested shape means for this model |
| `PARAM_INT_OUTPUT_IMAGE_HEIGHT` | int | the height the requested shape means for this model |
| `PARAM_STRING_OUTPUT_IMAGE_FILENAME` | string | `sillytalk/<job>` — for `SaveImage`'s `filename_prefix` |
| `PARAM_STRING_INPUT_IMAGE_FILENAME` | string | the reference, for a 1-reference variant |
| `PARAM_STRING_INPUT_IMAGE_FILENAME_1` … `_8` | string | the references, for a variant with several |

Reference values are flat file names in `input/`, because that is what a
`LoadImage` node lists in a current ComfyUI. A placeholder the plugin does not
know is a `400` that lists what it does — a typo fails loudly at submit time
rather than drawing a picture of the string `PARAM_PROMPT`.

### LoRA slots

A graph that can take LoRAs carries a chain of up to five
`LoraLoaderModelOnly` nodes, wired one into the next between the model loader and
everything that uses the model:

```
UNETLoaderMultiGPU → LoRA 1 → LoRA 2 → … → LoRA 5 → KSampler
```

Each holds `PARAM_LORA_N` and `PARAM_LORA_STRENGTH_N`. At submit time the plugin
fills the first *N* slots with the LoRAs the request named and **removes the rest
from the graph**, re-pointing whatever consumed a removed slot at the loader that
fed it. That is what ComfyUI's own Ctrl+B does — its bypass is a client-side
prune and relink, not a flag — and it is the only way here: `validate_prompt`
rejects a `lora_name` that is not in the folder list, so an unused slot cannot be
left in the prompt holding an empty string, and on a server whose `models/loras/`
is empty there would be no valid name to leave at all.

Two consequences worth knowing:

- The number of slots is a property of the workflow file, and `/sillytalk/models`
  reports it (`loras.max`), together with the files the folder holds. Ask for
  more LoRAs than there are slots and the answer says so before anything runs.
- A LoRA is a few hundred megabytes that has to live somewhere, so its size is
  added to the loader node that carries it when the cards are chosen. A run with
  three LoRAs on a big UNET can land them on different cards than the same run
  without them, and the job's `placements` say where each one went.

`strength_clip` is forced to `0` on a slot that has one. These graphs run a
quantised text encoder that a CLIP-side merge would not survive, so a LoRA here
only ever touches the model.

A LoRA trained for a different model used to be the one thing this server could
not see: ComfyUI matches a LoRA's tensors by shape and **skips** the ones that do
not fit, so a file whose width is 3072 meeting a model that is 4096 wide produced
a console full of `lora key not loaded: …` and a picture that looked exactly like
the one without it, from a job that reported itself `succeeded`. It is seen now —
see the next section — and refused before the queue.

### Which LoRA fits which file

`/sillytalk/models` answers per LoRA and per model file whether the two merge, and
`POST /sillytalk/generate` refuses what it can prove will not. The check reads the
safetensors header and nothing else: the first 8 bytes of the file are the length
of a JSON table naming every tensor with its dtype and shape, so a 20 GB
checkpoint costs a few hundred kilobytes and no torch is involved. Five LoRAs
against six checkpoints — thirty pairs, on a folder of 90 GB — is 153 ms cold and
under a millisecond once the headers have been read. A verdict is trusted for as
long as the mtime and size of both files say they are unchanged.

The LoRA's two halves give the `in` and `out` of every layer it wants to patch;
the checkpoint's own weights give the same pair under its own naming. The names of
the wild are normalised first: the prefixes `model.diffusion_model.`,
`diffusion_model.`, `lora_unet_` and `base_model.model.`, the halves
`lora_down`/`lora_A` and `lora_up`/`lora_B`, the `.alpha` entries, and the
`weight_scale` an int8 or fp8 checkpoint carries beside every quantised weight —
which is not the weight, and would compare a rank-1 scale against a LoRA.

Three answers, and the difference between the last two is the whole design:

| | meaning | what happens |
| --- | --- | --- |
| `yes` | of the layers it names that this model has, all are at that size | merged; a note naming what ComfyUI will skip — a layer at a stale size, or one this model does not have at all |
| `no` | proof it is for another file: more named layers at the wrong size than at the right one, or every target a text encoder | refused with a `400`, greyed in the client with the reason |
| `unknown` | none of its layer names occur in this model | allowed, and flagged |

`unknown` is allowed on purpose. LoRA naming varies — peft, DCNV3, diffusers, and
whatever a training script wrote out — and this reads the common few. A file this
code did not recognise is not a file that will not work, and refusing it would
turn a working feature into one that quietly stopped accepting other people's
models. What `no` claims is the opposite: not "I did not understand this" but "I
understood it, and ComfyUI would load it and drop it".

A text-encoder LoRA is a `no` for a reason of its own: these graphs force
`strength_clip` to `0`, so such a file would change nothing even where its layers
exist.

The rule that weighs wrong against right is deliberately not "any mismatch
refuses". A LoRA that fits 700 layers and carries 1 stale tensor merges fine —
ComfyUI skips the one — and refusing it would be this check's own false alarm.
Zero fitting against 128 wrong is the silent no-op, and that is what gets refused.

A `yes` can still be a partial LoRA, and says so. A file naming 192 layers of a
model that has 160 of them merges and draws, just with less of the style than was
asked for, and ComfyUI mentions it only on the console — so the reason counts the
layers that will be skipped. One thing is deliberately not counted: the two
spellings of the Qwen-Image 2.1 MLP. A checkpoint carries either a fused
`img_mlp.gate_up` or a split `gate_layer` + `proj`, and ComfyUI builds the model to
match the file it loaded (`model_detection.py` sets `fused_mlp = gate_up is not
None`), so a LoRA naming both spellings is complete on either checkpoint and its
unused keys are not a gap.

`scripts/lora_both_mlp.py` makes such a file out of a single-spelling one. The two
spellings are the same SwiGLU — `comfy/ops.py` reads a fused tensor as
`gate, up = x.chunk(2, dim=-1)`, the split path computes
`out(F.silu(gate_layer(x)) * proj(x))` — so a fused `lora_B` splits along its
output rows, first half the gate, and the same `lora_A` serves both halves. That is
exact rather than approximate: ΔW = B @ A, and the two halves of B are exactly the
two halves of ΔW. Measured on three real LoRAs: 160 of 224 layers on a split
checkpoint before, 224 of 224 after, and 192 of 192 on a fused one, for 34 MB on an
80 MB file.

Drawn through on a live server both ways — 52 s on the split `REDQW21`, 58 s on the
fused `qwen_image_2.1_int8_convrot` — and each skipped exactly the tensors of the
spelling that model does not use and nothing else: 64 `lora key not loaded` lines on
the split one, 128 on the fused one, once per run rather than per step since the
merged model is cached. Those lines name keys ComfyUI has no place for; the change
itself lands whole, which is the other side of the single-spelling file, whose 32
skipped layers are part of the style.

### Adding a model

1. Build the workflow in the UI and save it. Make one variant per number of
   references the model can take, each with `LoadImage` nodes holding
   `PARAM_STRING_INPUT_IMAGE_FILENAME…`, a text encoder holding `PARAM_PROMPT`,
   a `KSampler` holding `PARAM_SEED`, and a `SaveImage` whose `filename_prefix`
   is `PARAM_STRING_OUTPUT_IMAGE_FILENAME`.
2. Put `PARAM_MODEL_FILE` in the UNET loader and add the LoRA chain described
   under [LoRA slots](#lora-slots) if the model should take them. Both are
   optional — leave them out and the workflow keeps its own file and no LoRAs.
   Add `steps` if a client should be able to choose the step count; the number
   left on the `KSampler` is the default, and a distilled graph should declare
   `min` equal to `max`.
3. Save it as `workflows/<your-model-code>.json` in the format above, with its
   `sizes` filled in from what the model was trained at.
4. Restart ComfyUI, or touch the file to drop the conversion cache, and check
   `GET /sillytalk/models`.

`scripts/import_sillytalk_workflows.py` does this for the older
`{text, oneImage, twoImages}` triple SillyTalk used to ship: it merges the three
files into one catalogue file and strips the device values the client used to
write.

## Tests

They run against the interpreter that runs ComfyUI, since that is the one with
aiohttp:

```bash
<comfyui>/.venv/bin/python -m unittest discover -s tests -t tests
```

No GPU, no server and no network: `folder_paths` is stubbed, and the placement
tests hand the code a list of cards with made-up VRAM numbers. What is covered
is the arithmetic that decides a placement, the catalogue's parsing rules, the
placeholder substitution, the job record's lifetime, and the file endpoints'
path handling.

## Vendored code

`sillytalk_plugin/vendor/workflow_converter.py` is a copy of the converter from
[SethRobinson/comfyui-workflow-to-api-converter-endpoint](https://github.com/SethRobinson/comfyui-workflow-to-api-converter-endpoint)
(v2.3.0), which is public domain. [`vendor/NOTICE.md`](vendor/NOTICE.md) carries
the source, author and licence text. It is vendored rather than called over HTTP
so that a plugin does not depend on another plugin being installed, and it is
kept unmodified so a future sync stays a copy and paste.

Everything else in this repository is MIT — see [`LICENSE`](LICENSE).

## Version history

**1.3.0** — A client may choose the step count: `steps` in the request, a
`min`/`max` range per model in the catalogue, and the number the graph wires stays
the default (and is reported as `steps.default`). A distilled model declares
`min == max`, so the four-step LoRA cannot be asked to run twenty-five; a model
whose catalogue has no range refuses the field outright, and a graph whose own
step count sits outside its declared range is refused before the queue.

**1.2.1** — A `yes` that is only part of a LoRA says which layers it loses, and
`scripts/lora_both_mlp.py` writes a LoRA carrying both spellings of the
Qwen-Image 2.1 MLP — the fused `img_mlp.gate_up` and the split `gate_layer` +
`proj` — so one file merges completely into either kind of checkpoint. The two
spellings are not counted as a gap when a LoRA carries both.

**1.2.0** — The plugin decides which LoRA fits which file. It reads the
safetensors headers of both — a few hundred kilobytes, no torch — and says `yes`,
`no` or `unknown` per pair in `/sillytalk/models`, so the client can grey out what
would merge into nothing and say why; `POST /sillytalk/generate` refuses a `no`
instead of drawing the picture the model makes without the LoRA. This is the
silent-skip case: ComfyUI drops tensors whose shapes disagree and still reports
the run a success.

**1.1.0** — The server decides the pixels, the client the shape: `width`/`height`
are gone from the request and a workflow file carries `sizes` per orientation.
`/sillytalk/models` publishes the contents of `models/diffusion_models/` and
`models/loras/`, so a file the user drops in becomes selectable in SillyTalk's
settings instead of something to type, and a run can name a `model_file` of its
own and up to eight LoRAs with their strengths. Workflows that want LoRAs wire a
chain of `LoraLoaderModelOnly` slots; the unused ones are taken out of the graph
at submit time, and a LoRA's bytes count towards the card its loader lands on.

**1.0.0** — First release. `/sillytalk/*` generation endpoints with server-side
GPU placement, the workflow catalogue, the `/files/delete` and `/files/list`
cleanup endpoints moved here from
`comfyui-workflow-to-api-converter-endpoint`, and the vendored workflow
converter.
