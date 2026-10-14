"""Prompt registry dispatching by TASK_TYPE.

Each task (VLN / OBJNAV / EQA) has its own prompt module. Access prompts via
`get_prompts(task_type)` which returns the appropriate module; read constants
off it (e.g. `P.LA_PROMPT_BACKTRACK`).

The VLN constants are also re-exported at module level for backward
compatibility with existing `from .utils.prompts import *` usage.
"""
from . import prompts_vln, prompts_objnav, prompts_eqa

PROMPT_REGISTRY = {
    "VLN": prompts_vln,
    "OBJNAV": prompts_objnav,
    "EQA": prompts_eqa,
}


def get_prompts(task_type: str):
    if task_type not in PROMPT_REGISTRY:
        raise KeyError(
            f"Unknown TASK_TYPE {task_type!r}; expected one of {sorted(PROMPT_REGISTRY)}"
        )
    return PROMPT_REGISTRY[task_type]


_vln_exports = [n for n in dir(prompts_vln) if not n.startswith("_")]
for _n in _vln_exports:
    globals()[_n] = getattr(prompts_vln, _n)
__all__ = _vln_exports + ["PROMPT_REGISTRY", "get_prompts"]
