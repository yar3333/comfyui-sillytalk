"""The implementation of the comfyui-sillytalk plugin.

    routes.py     the HTTP surface
    jobs.py       one run: submit, follow, hand back the picture, clean up
    workflows.py  the model catalogue and the UI -> API conversion of its graphs
    placement.py  which GPU each model loader of a run gets
    params.py     the PARAM_* placeholders a workflow file may use
    files.py      /files/delete and /files/list
    vendor/       the workflow converter this plugin runs in-process
"""
