"""comfyui-sillytalk — image generation for the SillyTalk chat app.

SillyTalk sends a prompt, its reference pictures and a model code; this plugin
owns everything else about the run: which workflow graph the model code means,
which GPU each model loader lands on, where the picture goes and when it is
removed again. The client therefore needs no workflow file, no device name and
no knowledge of ComfyUI's API format.

Endpoints (see `sillytalk_plugin/routes.py`):

    GET    /sillytalk                 what this is
    GET    /sillytalk/health          version, catalogue, devices, free VRAM
    GET    /sillytalk/models          the model codes and what each takes
    POST   /sillytalk/generate        multipart -> 202 {job_id}
    GET    /sillytalk/jobs/{id}       status + placement
    GET    /sillytalk/jobs/{id}/image the finished PNG
    DELETE /sillytalk/jobs/{id}       cancel

Plus the file cleanup endpoints ComfyUI does not have (`POST /files/delete`,
`GET /files/list`), which live here because this plugin is what writes files on
the server.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from server import PromptServer
except ImportError as err:  # pragma: no cover - only when installed outside ComfyUI
    logger.error("comfyui-sillytalk: PromptServer is not importable — install this folder into ComfyUI's custom_nodes/")
    raise err

from .sillytalk_plugin import files, routes  # noqa: E402  (needs the server module above)

routes.register(PromptServer.instance.routes)

# Losing the cleanup endpoints must never take the generation endpoints down.
try:
    files.register_file_endpoints(PromptServer.instance.routes)
except Exception as err:  # pragma: no cover
    logger.error("comfyui-sillytalk: /files/delete and /files/list not registered: %s", err)

# This plugin adds endpoints, not nodes; the empty mappings keep ComfyUI Manager
# and every importer that reads them from guessing.
NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}

__version__ = routes.__version__

logger.info("comfyui-sillytalk v%s registered (GET /sillytalk for the endpoints)", __version__)
print(f"[comfyui-sillytalk] v{__version__} — POST /sillytalk/generate, GET /sillytalk/models")
