"""A generation job, from the request to the picture it hands back.

The job is this plugin's own record; the ComfyUI queue stays the one thing that
actually runs anything. A job therefore never re-implements the queue — it
submits a validated prompt the way the server's own `/prompt` handler does, and
then reads the server's own bookkeeping (running set, history) to say where the
run is. That is also why a job that outlives a server restart is simply gone:
the record lives in this process, and a client that asks for it after a restart
gets a 404 and treats it as a failure.

Files follow the job, not the other way round. The references a run uploaded
are removed as soon as the run reaches a terminal state, and the picture is
kept just long enough for the client to fetch it (`RETENTION_SECONDS`) and then
removed too — the server is not a picture archive, and the client keeps its own
copy.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import placement as placement_module
from .params import find_placeholders, substitute
from .workflows import ModelSpec, WorkflowError, model as load_model

logger = logging.getLogger(__name__)

QUEUED = "queued"
RUNNING = "running"
SUCCEEDED = "succeeded"
FAILED = "failed"
CANCELLED = "cancelled"
TERMINAL = (SUCCEEDED, FAILED, CANCELLED)

# How long a finished picture waits for its client after the job settled. Long
# enough for a retry after a lost response, short enough that a client which
# never comes back does not fill the output folder.
RETENTION_SECONDS = 900
# The records themselves are dropped after this, settled jobs only.
RECORD_RETENTION_SECONDS = 3600
MAX_RECORDS = 500

# SaveImage writes under this prefix, so a leftover from a client that vanished
# is recognisable (and inside our own corner of output/).
OUTPUT_PREFIX = "sillytalk"


class JobError(Exception):
    """Something the client can act on: a bad model, a missing placeholder."""


@dataclass
class Job:
    id: str
    model: str
    width: int
    height: int
    seed: int
    status: str = QUEUED
    error: str | None = None
    prompt_id: str | None = None
    placement: placement_module.Placement | None = None
    refs: list[Path] = field(default_factory=list)
    image: Path | None = None
    created: float = field(default_factory=time.time)
    settled: float | None = None
    cancel_requested: bool = False

    @property
    def output_prefix(self) -> str:
        return f"{OUTPUT_PREFIX}/{self.id[:8]}"

    def describe(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "job_id": self.id,
            "model": self.model,
            "status": self.status,
            "width": self.width,
            "height": self.height,
            "seed": self.seed,
            "seconds": round((self.settled or time.time()) - self.created, 1),
        }
        if self.error:
            out["error"] = self.error
        if self.placement is not None:
            out.update(self.placement.describe())
        return out


_jobs: dict[str, Job] = {}


def get(job_id: str) -> Job | None:
    return _jobs.get(job_id)


def sweep() -> None:
    """Drops what a vanished client left behind. Called from the request
    handlers rather than a timer: a custom node that starts a task at import
    time is a task nobody owns, and a picture waiting for the next request is
    nobody's problem."""
    now = time.time()
    for job in list(_jobs.values()):
        if job.status not in TERMINAL:
            continue
        if job.settled and now - job.settled > RETENTION_SECONDS:
            _remove_image(job)
        if job.settled and now - job.settled > RECORD_RETENTION_SECONDS:
            _jobs.pop(job.id, None)
    # A client that generates in a loop must not grow this forever.
    while len(_jobs) > MAX_RECORDS:
        oldest = min(_jobs.values(), key=lambda job: job.created)
        if oldest.status not in TERMINAL:
            break
        _remove_image(oldest)
        _jobs.pop(oldest.id, None)
    _sweep_orphans(now)


_orphan_scan_at = 0.0
ORPHAN_SCAN_INTERVAL_SECONDS = 60.0


def _sweep_orphans(now: float) -> None:
    """Removes the pictures a previous process left in `output/sillytalk/`.

    A job record lives in this process's memory, so a restart orphans every
    picture the dead process had not handed out yet, and nothing else on this
    server knows that folder is ours to clean. Nothing younger than the
    retention is touched, and neither is anything a job this process still
    remembers owns — which is what keeps a client fetching while it fetches.
    """
    global _orphan_scan_at
    if now - _orphan_scan_at < ORPHAN_SCAN_INTERVAL_SECONDS:
        return
    _orphan_scan_at = now
    folder = _output_dir() / OUTPUT_PREFIX
    if not folder.is_dir():
        return
    owned = {job.id[:8] for job in _jobs.values()}
    for path in folder.iterdir():
        try:
            if not path.is_file() or now - path.stat().st_mtime < RETENTION_SECONDS:
                continue
        except OSError:
            continue
        if path.name.split("_", 1)[0] in owned:
            continue
        logger.info("sillytalk: removing %s, left behind by an earlier process", path)
        _unlink(path)


async def start(
    model_code: str,
    prompt: str,
    ref_paths: list[Path],
    width: int,
    height: int,
    seed: int | None,
) -> Job:
    """Builds the run's graph and puts it on the server's queue."""
    spec: ModelSpec | None = load_model(model_code)
    if spec is None:
        known = ", ".join(entry.code for entry in _readable_entries()) or "(none)"
        raise JobError(f"no workflow named '{model_code}' in this server's catalogue: {known}")
    if len(ref_paths) > spec.max_refs:
        raise JobError(
            f"the '{model_code}' workflow takes at most {spec.max_refs} reference image(s), "
            f"{len(ref_paths)} were sent"
        )

    job = Job(
        id=str(uuid.uuid4()),
        model=spec.code,
        width=width,
        height=height,
        seed=seed if seed is not None else uuid.uuid4().int % 2147483647,
        refs=list(ref_paths),
    )
    _jobs[job.id] = job

    try:
        graph = _build_graph(spec, job, prompt)
        job.placement = placement_module.place(graph)
        await _submit(graph, job)
    except Exception as err:
        job.status = FAILED
        job.error = str(err)
        job.settled = time.time()
        _drop_refs(job)
        logger.warning("sillytalk job %s failed before the queue: %s", job.id, err)
        raise JobError(str(err)) from err

    logger.info("sillytalk job %s queued (model=%s, %dx%d, %s)", job.id, spec.code, width, height, job.placement.summary())
    return job


def _build_graph(spec: ModelSpec, job: Job, prompt: str) -> dict[str, Any]:
    from .workflows import api_graph

    graph = api_graph(spec, len(job.refs))
    values: dict[str, Any] = {
        "PARAM_PROMPT": prompt,
        "PARAM_SEED": job.seed,
        "PARAM_INT_OUTPUT_IMAGE_WIDTH": job.width,
        "PARAM_INT_OUTPUT_IMAGE_HEIGHT": job.height,
        "PARAM_STRING_OUTPUT_IMAGE_FILENAME": job.output_prefix,
    }
    # The reference names follow the AiPic convention the workflow files were
    # written with: one reference is unnumbered, two are numbered from one.
    if len(job.refs) == 1:
        values["PARAM_STRING_INPUT_IMAGE_FILENAME"] = job.refs[0].name
    elif len(job.refs) > 1:
        for index, path in enumerate(job.refs, start=1):
            values[f"PARAM_STRING_INPUT_IMAGE_FILENAME_{index}"] = path.name

    graph = substitute(graph, values)
    leftovers = find_placeholders(graph)
    # PARAM_DEVICE is the one placeholder this plugin answers by itself, and an
    # older workflow file still holds it in every loader.
    leftovers.discard("PARAM_DEVICE")
    if leftovers:
        raise JobError(
            f"the '{spec.code}' workflow asks for {', '.join(sorted(leftovers))}, which this plugin does not "
            f"provide — a workflow file may only use PARAM_PROMPT, PARAM_SEED, "
            f"PARAM_INT_OUTPUT_IMAGE_WIDTH/_HEIGHT, PARAM_STRING_OUTPUT_IMAGE_FILENAME and "
            f"PARAM_STRING_INPUT_IMAGE_FILENAME(_1/_2)"
        )
    return graph


def _readable_entries():
    from .workflows import entries

    return [entry for entry in entries() if not entry.error]


def _server():
    from server import PromptServer

    return PromptServer.instance


async def _submit(graph: dict[str, Any], job: Job) -> None:
    """Queues the graph the way the server's own /prompt handler does."""
    import execution

    server = _server()
    prompt_id = str(uuid.uuid4())
    valid = await execution.validate_prompt(prompt_id, graph, None)
    ok, error, outputs_to_execute, node_errors = valid[0], valid[1], valid[2], valid[3]
    if not ok:
        detail = _error_text(error) if error else "the server refused the graph"
        raise JobError(
            f"the server refused the '{job.model}' workflow: {detail} "
            f"(usually a custom node or a model file of the workflow is missing on this server)"
        )
    number = server.number
    server.number += 1
    extra_data = {"create_time": int(time.time() * 1000), "comfy_usage_source": "comfyui-sillytalk"}
    server.prompt_queue.put((number, prompt_id, graph, extra_data, list(outputs_to_execute), {}))
    job.prompt_id = prompt_id


def refresh(job: Job) -> Job:
    """Reads the server's own state into the job, and cleans up on the way out."""
    if job.status in TERMINAL or not job.prompt_id:
        return job
    server = _server()
    prompt_id = job.prompt_id

    history = server.prompt_queue.get_history(prompt_id)
    entry = history.get(prompt_id) if history else None
    running, queued = server.prompt_queue.get_current_queue_volatile()
    in_queue = any(item[1] == prompt_id for item in queued)
    is_running = any(item[1] == prompt_id for item in running)

    if entry is None:
        if is_running:
            job.status = RUNNING
        elif not in_queue:
            # Neither queued, running, nor in the history: the server forgot it
            # (a restart, or someone wiped the history) — the job is over.
            _settle(job, CANCELLED if job.cancel_requested else FAILED,
                    "the server no longer knows this job (it restarted or its history was cleared)")
        return job

    status = entry.get("status") or {}
    if not status.get("completed"):
        # An interrupted run lands here: the server writes its history entry
        # with `completed` false, so the entry alone cannot say whether the run
        # is over. The queue can.
        if is_running or in_queue:
            job.status = RUNNING
            return job
        if job.cancel_requested:
            _settle(job, CANCELLED, None)
        else:
            _settle(job, FAILED, _entry_error(entry) or "the server stopped the run without saying why")
        return job

    if status.get("status_str") == "error":
        if job.cancel_requested:
            _settle(job, CANCELLED, None)
        else:
            _settle(job, FAILED, _entry_error(entry))
        return job

    image = _output_file(entry)
    if image is None:
        _settle(job, FAILED, "the job finished but its SaveImage node wrote no picture")
        return job
    job.image = image
    _settle(job, SUCCEEDED, None)
    return job


def _settle(job: Job, status: str, error: str | None) -> None:
    job.status = status
    job.error = error
    job.settled = time.time()
    _drop_refs(job)
    if status != SUCCEEDED:
        _remove_image(job)
    logger.info("sillytalk job %s %s%s", job.id, status, f" — {error}" if error else "")


def cancel(job: Job) -> Job:
    """Takes the job out of the queue, or interrupts it if it is running."""
    job.cancel_requested = True
    if job.status in TERMINAL:
        return job
    server = _server()
    removed = False
    if job.prompt_id:
        removed = server.prompt_queue.delete_queue_item(lambda item: item[1] == job.prompt_id)
        if not removed:
            server.prompt_queue.interrupt_if_running(job.prompt_id)
    if removed:
        # A job that never started will never reach the history.
        _settle(job, CANCELLED, None)
    return job


def image_bytes(job: Job) -> bytes:
    """The finished picture, read from the server's output folder."""
    if job.status != SUCCEEDED or job.image is None:
        raise JobError(f"job {job.id} is {job.status}, it has no picture to send")
    path = job.image
    if not path.is_file():
        raise JobError(
            f"the picture of job {job.id} is gone from the server ({path}) — "
            f"it is kept for {RETENTION_SECONDS // 60} minutes after the job finishes"
        )
    return path.read_bytes()


def _drop_refs(job: Job) -> None:
    """Removes the reference pictures the run uploaded."""
    for path in job.refs:
        _unlink(path)
    job.refs = []


def _remove_image(job: Job) -> None:
    """Removes the copy this plugin wrote. The `output/sillytalk/` folder itself
    stays: it is this plugin's namespace, and an empty one costs nothing that a
    directory scan by somebody else would not cost."""
    if job.image is not None:
        _unlink(job.image)
        job.image = None


def _unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as err:
        logger.warning("sillytalk: could not remove %s (%s)", path, err)


def _output_dir() -> Path:
    import folder_paths

    return Path(folder_paths.get_output_directory()).resolve()


def _output_file(entry: dict[str, Any]) -> Path | None:
    """The picture the run's SaveImage node reported in the history."""
    base = _output_dir()
    for output in (entry.get("outputs") or {}).values():
        for image in (output or {}).get("images") or []:
            name = image.get("filename")
            if not name:
                continue
            subfolder = str(image.get("subfolder") or "")
            if str(image.get("type") or "output") != "output":
                continue
            path = (base / subfolder / name).resolve()
            try:
                path.relative_to(base)
            except ValueError:
                continue
            if path.is_file():
                return path
    return None


def _error_text(value: Any) -> str:
    """The sentence a failed run actually says, out of the server's structures."""
    if isinstance(value, str):
        return value[:2000]
    if isinstance(value, dict):
        for key in ("exception_message", "message", "details", "type"):
            found = value.get(key)
            if isinstance(found, str) and found:
                node = value.get("node_id") or value.get("node")
                class_type = value.get("node_type") or value.get("class_type")
                where = f" (node {node} {class_type})" if node or class_type else ""
                return f"{found}{where}"[:2000]
        return str(value)[:2000]
    if isinstance(value, list):
        parts = [_error_text(item) for item in value]
        return "; ".join(part for part in parts if part)[:2000]
    return str(value)[:2000] if value is not None else ""


def _entry_error(entry: dict[str, Any]) -> str:
    """What a failed run said, out of the history entry the server left behind."""
    status = entry.get("status") or {}
    for message in status.get("messages") or []:
        if isinstance(message, (list, tuple)) and len(message) > 1 and isinstance(message[1], dict):
            text = _error_text(message[1])
            if text:
                return text
        elif isinstance(message, str) and message:
            return message
    for key in ("exception_message", "message", "details"):
        found = status.get(key)
        if isinstance(found, str) and found:
            return found[:2000]
    return "the server reported an error without saying what"
