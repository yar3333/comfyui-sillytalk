"""Third-party code kept inside this plugin.

`workflow_converter.py` is a copy of the converter from
https://github.com/SethRobinson/comfyui-workflow-to-api-converter-endpoint
(Unlicense / public domain, see NOTICE.md). It is vendored rather than depended
on so that a SillyTalk server needs exactly one custom node: the workflow files
of this plugin are converted in-process, over the same node registry the server
is about to execute them with.
"""
