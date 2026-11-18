"""The HTTP surface of the plugin.

    GET    /sillytalk                 what this plugin is, and its endpoints
    GET    /sillytalk/health          version, catalogue, devices and free VRAM
    GET    /sillytalk/models          the models a client may ask for, with the
                                      files and LoRAs this server can offer
    POST   /sillytalk/generate        multipart: prompt + model + images -> 202 job_id
    GET    /sillytalk/jobs/{id}       where the run is, and how it was placed
    GET    /sillytalk/jobs/{id}/image the finished picture (bytes)
    DELETE /sillytalk/jobs/{id}       drop it from the queue, or interrupt it

The client sends a prompt, its reference pictures, a model code and an
orientation — plus, when it wants them, one model file of this server's folder
and the LoRAs to merge — and gets a job it polls. Nothing about GPUs, folder
paths, or the shape of a workflow graph crosses this boundary: the sizes behind
an orientation and the cards behind the weights are decided here.
"""

from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path
from typing import Any

from aiohttp import web

from . import jobs, model_files, placement
from . import loras as loras_module
from .jobs import CANCELLED, FAILED, RUNNING
from .workflows import WORKFLOW_DIR, entries
from .workflows import models as catalogue_models

logger = logging.getLogger(__name__)

__version__ = "1.1.0"

MAX_PROMPT_CHARS = 8000
MAX_REF_FILES = 8
MAX_REF_BYTES = 32 * 1024 * 1024
MAX_TOTAL_BYTES = 96 * 1024 * 1024
# A run may stack this many LoRAs; a workflow with fewer slots says so itself.
MAX_LORAS = 8
ALLOWED_REF_SUFFIXES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
                        ".bmp": "image/bmp"}
TEXT_FIELDS = ("model", "prompt", "orientation", "model_file", "loras", "seed")


def _json(data: Any, status: int = 200) -> web.Response:
    return web.json_response(data, status=status, dumps=lambda payload: json.dumps(payload, ensure_ascii=False))


def _input_dir() -> Path:
    import folder_paths

    return Path(folder_paths.get_input_directory()).resolve()


def _safe_name(filename: str, index: int, token: str) -> str:
    """A name that cannot collide and cannot carry a path.

    The reference is loaded back by the workflow's LoadImage node, whose list in
    this ComfyUI version is the flat contents of input/ — so the name stays flat
    and gets a prefix that says who put it here.
    """
    suffix = Path(str(filename or "")).suffix.lower()
    if suffix not in ALLOWED_REF_SUFFIXES:
        suffix = ".png"
    stem = "".join(char if char.isalnum() or char in "-_." else "_" for char in Path(str(filename or "image")).stem)[:48]
    return f"sillytalk_{token}_{index}_{stem or 'ref'}{suffix}"


async def index(request: web.Request) -> web.Response:
    return _json(
        {
            "name": "comfyui-sillytalk",
            "version": __version__,
            "description": "Image generation for SillyTalk: a prompt, reference pictures and a model code in, "
            "a finished PNG out. The server picks the GPUs.",
            "endpoints": [
                "GET /sillytalk/health",
                "GET /sillytalk/models",
                "POST /sillytalk/generate (multipart: model, prompt, orientation, model_file, loras, seed, image)",
                "GET /sillytalk/jobs/{job_id}",
                "GET /sillytalk/jobs/{job_id}/image",
                "DELETE /sillytalk/jobs/{job_id}",
                "POST /files/delete",
                "GET /files/list",
            ],
        }
    )


async def health(request: web.Request) -> web.Response:
    try:
        from comfyui_version import __version__ as comfyui_version
    except ImportError:
        comfyui_version = ""
    readable = [entry for entry in entries() if not entry.error]
    broken = [{"code": entry.code, "file": entry.file, "error": entry.error} for entry in entries() if entry.error]
    _, resident_files = placement.resident_models()
    return _json(
        {
            "name": "comfyui-sillytalk",
            "version": __version__,
            "comfyui_version": comfyui_version,
            "workflow_dir": str(WORKFLOW_DIR),
            "models": [entry.code for entry in readable],
            "broken_workflows": broken,
            "devices": [device.describe() for device in placement.visible_devices()],
            # What the server already holds, so a caller can see why the next
            # job will not have to load anything.
            "resident_models": [
                {"model": model, "device": where[0], "mb": round(where[1] / 1024 / 1024)}
                for model, where in sorted(resident_files.items())
            ],
        }
    )


async def list_models(request: web.Request) -> web.Response:
    """The catalogue, with what this server can actually be pointed at.

    `model_files.options` and `loras.options` are the contents of the server's
    own folders, so the client offers a pick instead of a text field: a file the
    user dropped in becomes selectable without either side being reconfigured.
    `loras.max` is how many slots that model's workflow wired — 0 means it takes
    no LoRAs at all.
    """
    diffusion = model_files.listing(model_files.DIFFUSION_FOLDER)
    loras = model_files.listing(model_files.LORA_FOLDER)
    described = []
    for spec in catalogue_models():
        entry = spec.describe()
        if spec.model_file:
            entry["model_files"] = {"folder": model_files.DIFFUSION_FOLDER, "options": diffusion}
        entry["loras"] = {"max": loras_module.available_slots(spec), "options": loras}
        described.append(entry)
    return _json({"models": described})


async def generate(request: web.Request) -> web.Response:
    jobs.sweep()
    if not request.content_type.startswith("multipart/"):
        return _json({"error": "send multipart/form-data with model, prompt and optional image parts"}, 400)

    token = uuid.uuid4().hex[:8]
    fields: dict[str, str] = {}
    refs: list[Path] = []
    total = 0
    started = False
    try:
        reader = await request.multipart()
        while True:
            part = await reader.next()
            if part is None:
                break
            name = part.name or ""
            if name in TEXT_FIELDS:
                raw = await part.text()
                if name == "prompt" and len(raw) > MAX_PROMPT_CHARS:
                    return _json({"error": f"the prompt is longer than {MAX_PROMPT_CHARS} characters"}, 400)
                fields[name] = raw.strip()
            elif name == "image":
                if len(refs) >= MAX_REF_FILES:
                    return _json({"error": f"no more than {MAX_REF_FILES} reference images"}, 400)
                saved = await _save_ref(part, len(refs) + 1, token)
                if isinstance(saved, web.Response):
                    return saved
                total += saved.stat().st_size
                if total > MAX_TOTAL_BYTES:
                    return _json({"error": f"the reference images together exceed {MAX_TOTAL_BYTES // (1024 * 1024)} MB"}, 400)
                refs.append(saved)
            else:
                await part.read()  # unknown field: drain it, the request stays readable

        model_code = fields.get("model", "")
        prompt = fields.get("prompt", "")
        if not model_code:
            return _json({"error": "the 'model' field is required"}, 400)
        if not prompt:
            return _json({"error": "the 'prompt' field is required"}, 400)
        chosen, problem = _loras(fields.get("loras"))
        if problem:
            return _json({"error": problem}, 400)

        job = await jobs.start(
            model_code,
            prompt,
            refs,
            orientation=(fields.get("orientation") or "").strip() or None,
            model_file=(fields.get("model_file") or "").strip() or None,
            loras=chosen,
            seed=_seed(fields.get("seed")),
        )
        started = True
        return _json(job.describe(), 202)
    except jobs.JobError as err:
        return _json({"error": str(err)}, 400)
    except web.HTTPError:
        raise
    except Exception as err:
        logger.exception("sillytalk: the generate request failed")
        return _json({"error": f"internal error: {err}"}, 500)
    finally:
        # A reference picture belongs to the job once there is one, which then
        # removes it when the run settles. A request refused halfway through has
        # no job, and nothing else on this server would ever remove it.
        if not started:
            _drop_paths(refs)


async def _save_ref(part, index_of_ref: int, token: str):
    """Streams one reference picture into input/, with a size cap."""
    suffix = Path(str(part.filename or "")).suffix.lower()
    if suffix not in ALLOWED_REF_SUFFIXES:
        return _json({"error": f"'{part.filename}' is not a supported picture ({', '.join(sorted(ALLOWED_REF_SUFFIXES))})"}, 400)
    name = _safe_name(str(part.filename or ""), index_of_ref, token)
    target = _input_dir() / name
    written = 0
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        while True:
            chunk = await part.read_chunk()
            if not chunk:
                break
            written += len(chunk)
            if written > MAX_REF_BYTES:
                handle.close()
                target.unlink(missing_ok=True)
                return _json({"error": f"'{part.filename}' is larger than {MAX_REF_BYTES // (1024 * 1024)} MB"}, 400)
            handle.write(chunk)
    if written == 0:
        target.unlink(missing_ok=True)
        return _json({"error": f"'{part.filename}' arrived empty"}, 400)
    return target


def _loras(raw: str | None) -> tuple[list[dict[str, Any]] | None, str]:
    """The `loras` field: a JSON array of `{"name": …, "strength": …}`.

    A JSON string rather than repeated parts because a name and its strength
    belong together — two parallel lists would let a client send three names and
    two strengths, and the pairing would be the server's guess.
    """
    text = (raw or "").strip()
    if not text:
        return [], ""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as err:
        return None, f"the 'loras' field is not valid JSON: {err}"
    if not isinstance(parsed, list):
        return None, "the 'loras' field must be a JSON array of {name, strength}"
    if len(parsed) > MAX_LORAS:
        return None, f"no more than {MAX_LORAS} LoRAs in one run"
    out: list[dict[str, Any]] = []
    for item in parsed:
        if isinstance(item, str):
            out.append({"name": item})
            continue
        if isinstance(item, dict) and item.get("name"):
            entry: dict[str, Any] = {"name": str(item["name"])}
            if item.get("strength") is not None:
                try:
                    entry["strength"] = float(item["strength"])
                except (TypeError, ValueError):
                    return None, f"the strength of {item['name']} is not a number"
            out.append(entry)
            continue
        return None, "every entry of 'loras' needs a name"
    return out, ""


def _seed(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    try:
        parsed = int(str(value))
    except ValueError:
        return None
    return parsed % 2147483647


async def job_status(request: web.Request) -> web.Response:
    jobs.sweep()
    job = jobs.get(request.match_info["job_id"])
    if job is None:
        return _json({"error": "no such job on this server (it answers only for jobs this process started)"}, 404)
    jobs.refresh(job)
    return _json(job.describe())


async def job_image(request: web.Request) -> web.Response:
    jobs.sweep()
    job = jobs.get(request.match_info["job_id"])
    if job is None:
        return _json({"error": "no such job on this server (it answers only for jobs this process started)"}, 404)
    jobs.refresh(job)
    if job.status in (RUNNING, jobs.QUEUED):
        return _json({"error": f"the job is still {job.status}"}, 409)
    if job.status in (FAILED, CANCELLED):
        return _json({"error": job.error or job.status}, 410)
    try:
        body = jobs.image_bytes(job)
    except jobs.JobError as err:
        return _json({"error": str(err)}, 410)
    suffix = job.image.suffix.lower() if job.image else ".png"
    return web.Response(
        body=body,
        content_type=ALLOWED_REF_SUFFIXES.get(suffix, "image/png"),
        headers={
            "Content-Disposition": f'inline; filename="{job.id[:8]}{suffix}"',
            "Cache-Control": "no-store",
        },
    )


async def job_cancel(request: web.Request) -> web.Response:
    job = jobs.get(request.match_info["job_id"])
    if job is None:
        return _json({"error": "no such job on this server"}, 404)
    jobs.refresh(job)
    jobs.cancel(job)
    return _json(job.describe())


def _drop_paths(paths: list[Path]) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def register(routes) -> None:
    """Mounts every route of the plugin on the server's aiohttp app."""
    routes.get("/sillytalk")(index)
    routes.get("/sillytalk/")(index)
    routes.get("/sillytalk/health")(health)
    routes.get("/sillytalk/models")(list_models)
    routes.post("/sillytalk/generate")(generate)
    routes.get("/sillytalk/jobs/{job_id}")(job_status)
    routes.get("/sillytalk/jobs/{job_id}/image")(job_image)
    routes.delete("/sillytalk/jobs/{job_id}")(job_cancel)
