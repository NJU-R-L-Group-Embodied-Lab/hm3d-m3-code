"""Capture a golden snapshot of the LaViRA prompt / mode surface (no simulator, no GPU).

The snapshot pins everything that decides what the LaViRA LA/VA/verifier
see for the paper mode:

* ``k2-async-keep-goals-pa`` (LaViRA full);
* the policy-visible communication flags of the mode;
* every runner attribute that ``benchmark.ma_search.main`` sets for the mode
  (the real ``main`` runs; only the simulator-backed runner is replaced by a
  recorder), plus its start-up log line;
* the LA instruction text on fixed fake inputs (goals, teammate
  progress_analysis, found state, 4- and 8-view panoramas) and the LA JSON
  field-contract rewrite;
* the claim-verifier prompts;
* all module-level prompt constants of the LaViRA modules;
* an AST fingerprint (comments and docstrings ignored) of every function and
  method in those modules, so edits to the simulator-only episode loop are
  caught even though it cannot run here.

Needs the project Python environment (``vlnce_baselines`` must import), but no
Habitat simulator instance, dataset, API key or GPU. Run from ``sim-code/``::

    python tests/golden/capture_lavira_prompts.py --out tests/golden/lavira_prompts.json

``tests/golden/check_lavira_prompts.py`` re-captures and compares.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import importlib
import inspect
import io
import json
import os
import re
import sys
import tempfile
from types import SimpleNamespace
from typing import Any, Dict, List

SIM_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if SIM_ROOT not in sys.path:
    sys.path.insert(0, SIM_ROOT)

PAPER_MODES = (
    "k2-async-keep-goals-pa",   # LaViRA (full)
)

# Flags the paper launchers pass for every LaViRA row (THINKING_BUDGET=1024).
PAPER_FLAGS = (
    "--use-vlm", "--clean-found-state", "--disable-path-budget",
    "--no-team-topdown-map", "--la-thinking-budget", "1024",
)

LAVIRA_MODULES = (
    "benchmark.ma_search",
    "benchmark.async_board_pa",
    "benchmark.pano_views",
    "benchmark.coord_shell",
)

# Release identity scrub: string literals renamed on purpose, keyed by the
# sha256 prefix of the old literal (old -> new). Applied when fingerprinting so
# the rename alone is not reported as a change.
STRING_RENAMES = {
    "9803cfe46998a142": "lavira_arrival_double_check",
}
# Names that equal a former literal under the default environment
# (LLM_MODEL unset -> api_auth default "qwen3.5-27b").
NAME_AS_CONSTANT = {
    "_LLM_MODEL": "qwen3.5-27b",
}


# ---------------------------------------------------------------- helpers
def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(v) for v in value)
    if isinstance(value, re.Pattern):
        return "re:" + value.pattern
    return "<%s>" % type(value).__name__


def _goals() -> List[dict]:
    return [
        {"goal_index": 0, "object_category": "chair", "modality": "ON",
         "position": [1.0, 2.0, 3.0], "rep_point": [4.0, 5.0, 6.0],
         "reach_points": [[7.0, 8.0, 9.0]]},
        {"goal_index": 1, "object_category": "stool", "modality": "TN",
         "tn_text": "a white stool under a table"},
        {"goal_index": 2, "object_category": "sofa", "modality": "IIN"},
        {"goal_index": 5, "object_category": "bed", "modality": "IIN"},
    ]


# goal_pil is keyed by goal_index; goal 5 has no rendered reference image.
GOAL_PIL = {2: "IMG_goal_2"}


def _states(n: int, step: int) -> List[SimpleNamespace]:
    texts = [
        "I checked the hallway and a bedroom; no chair yet.",
        "",
        ("Kitchen explored: counters, fridge, a white stool near the island. " * 3).strip(),
    ]
    out = []
    for i in range(n):
        text = texts[i % len(texts)]
        out.append(SimpleNamespace(
            agent_id=i,
            last_progress_analysis=text,
            last_progress_step=(step - 3 - i) if text else -1,
        ))
    return out


SCENARIOS = (
    {"name": "start_k2_agent0", "n": 2, "agent": 0, "step": 0,
     "found": [False, False, False, False], "pano_n": 4, "stride": 90.0},
    {"name": "mid_k2_agent1_found1", "n": 2, "agent": 1, "step": 17,
     "found": [False, True, False, False], "pano_n": 4, "stride": 90.0},
    {"name": "k3_agent2_8view", "n": 3, "agent": 2, "step": 40,
     "found": [True, False, False, True], "pano_n": 8, "stride": 45.0},
    {"name": "k2_agent1_step3", "n": 2, "agent": 1, "step": 3,
     "found": [False, False, True, False], "pano_n": 4, "stride": 90.0},
)

# Placeholders the LA prompt in vlnce_baselines/ZS_Evaluator_mp.py contains.
LA_TEMPLATE = (
    "2. THEN, write `reasoning_action` and choose the next action based on the UPDATED TODO\n"
    "   plus what you observe.\n"
    "{\n"
    '    "progress_analysis": "<brief assessment of what you\'ve observed and done so far>",\n'
    '    "reasoning_action": "<reasoning for the chosen action, given the updated TODO + observations>",\n'
    "}\n"
    "Look at the current 4-directional views and the 4-dir views again.\n"
    "This is a single-agent LaViRA decision. "
    "Do not include map_understanding. Use only first-person views, "
    "navigation history, and the target list.\n"
)


def _la_prompt_after_contract(instruction: str, num_frames: int) -> str:
    """Mirror of the marker dispatch in ZS_Evaluator_mp (LA prompt assembly)."""
    from benchmark.async_board_pa import apply_la_json_field_contract

    prompt = LA_TEMPLATE
    ins = instruction or ""
    if "ASYNC_BOARD_PA_FIELD_CONTRACT" in ins or "ASYNC_IND_PA_FIELD_CONTRACT" in ins:
        prompt = apply_la_json_field_contract(
            prompt,
            independent_plus_pa="ASYNC_IND_PA_FIELD_CONTRACT" in ins,
            pano_n=num_frames,
        )
    return prompt


# ---------------------------------------------------------------- main() capture
class _RunnerRecorder:
    """Stands in for MASearchRunner: records ctor kwargs and every attribute."""

    def __init__(self, **kwargs):
        self.__dict__["_ctor"] = dict(kwargs)
        self.__dict__["_evaluate"] = None

    def evaluate(self, adapters, seed=0):
        self.__dict__["_evaluate"] = {"n_adapters": len(adapters), "seed": seed}

    def evaluate_dynamic(self, *a, **k):  # pragma: no cover - not used here
        raise AssertionError("dynamic queue not exercised")


def _run_main(argv: List[str]) -> Dict[str, Any]:
    import benchmark.ma_search as MS

    saved = (MS.MASearchRunner, MS._load_cfg, MS._set_num_agents,
             argparse.ArgumentParser.parse_args)
    box: Dict[str, Any] = {}

    def _parse_args(self, args=None, namespace=None):
        ns = saved[3](self, args, namespace)
        box["args"] = vars(ns).copy()
        box["actions"] = {
            a.dest: {"options": list(a.option_strings), "default": _jsonable(a.default),
                     "required": bool(a.required)}
            for a in self._actions if a.dest != "help"
        }
        box["choices"] = {a.dest: list(a.choices) for a in self._actions if a.choices}
        return ns

    def _recorder(**kwargs):
        rec = _RunnerRecorder(**kwargs)
        box["runner"] = rec
        return rec

    MS.MASearchRunner = _recorder
    MS._load_cfg = lambda path: SimpleNamespace(config_path=path)
    MS._set_num_agents = lambda cfg, n: setattr(cfg, "num_agents", n)
    argparse.ArgumentParser.parse_args = _parse_args
    out = io.StringIO()
    try:
        with tempfile.TemporaryDirectory() as td:
            ep_path = os.path.join(td, "episodes.json")
            with open(ep_path, "w") as f:
                json.dump({"episodes": []}, f)
            full = ["--episodes", ep_path, "--iin-images-dir", os.path.join(td, "no_iin")] + argv
            with contextlib.redirect_stdout(out):
                MS.main(full)
    finally:
        (MS.MASearchRunner, MS._load_cfg, MS._set_num_agents,
         argparse.ArgumentParser.parse_args) = saved
    rec = box["runner"]
    # Effective runner configuration: what main() set, else the class default.
    effective = {
        k: v for k, v in vars(saved[0]).items()
        if not k.startswith("__") and not callable(v)
        and not isinstance(v, (staticmethod, classmethod, property))
    }
    effective.update(vars(rec))
    attrs = {k: _jsonable(v) for k, v in sorted(effective.items())
             if k not in ("iin_images_dir",)}
    log = [line for line in out.getvalue().splitlines()
           if not line.startswith("[ma-search] verify=")]
    return {"attrs": attrs, "log": log, "args": box["args"],
            "actions": box["actions"], "choices": box["choices"]}


# ---------------------------------------------------------------- instructions
def _mode_instruction(attrs: Dict[str, Any], sc: dict) -> Dict[str, Any]:
    """Mirror of the per-agent LA instruction branch in MASearchRunner._run_episode
    for the configuration the runner attributes select (paper modes only)."""
    import benchmark.async_board_pa as ABP
    import benchmark.ma_search as MS
    from benchmark.pano_views import views_phrase

    goals = _goals()
    i, n, step = sc["agent"], sc["n"], sc["step"]
    pano_n, stride = sc["pano_n"], sc["stride"]
    visible_found = list(sc["found"])
    states = _states(n, step)
    goal_images = GOAL_PIL if attrs.get("multimodal_search") else {}

    fhint = ""
    active_stop_goal_id = None
    if attrs.get("use_qwen_claim_scoring"):
        if attrs.get("use_policy_selected_claim_goal") or (
            not attrs.get("bind_claim_to_focus") and attrs.get("use_qwen_goal_focus")
        ):
            fhint += MS._policy_selected_claim_prompt(
                goals, visible_found, pano_n=pano_n, pano_stride_deg=stride)
    refs: List[Any] = []
    if attrs.get("use_async_board_pa"):
        instruction, r = ABP.build_async_board_instruction(
            goals, [-1] * len(goals), states, agent_id=i, current_step=step,
            goal_pil=goal_images, fhint=fhint)
        refs = [x["goal_id"] for x in r]
        branch = "board_pa"
    elif attrs.get("use_async_ind_pa"):
        body, r = MS._build_clean_target_instruction(goals, visible_found, goal_images, fhint)
        instruction = ABP.build_async_ind_pa_instruction(
            body, states, agent_id=i, current_step=step,
            views_phrase=views_phrase(pano_n, stride))
        refs = [x["goal_id"] for x in r]
        branch = "ind_pa"
    elif attrs.get("clean_found_state"):
        instruction, r = MS._build_clean_target_instruction(goals, visible_found, goal_images, fhint)
        refs = [x["goal_id"] for x in r]
        branch = "clean"
    else:
        branch = "legacy"
        instruction = "<legacy instruction path>"
    return {
        "branch": branch,
        "instruction": instruction,
        "reference_images": refs,
        "active_stop_goal_id": active_stop_goal_id,
        "has_field_contract": ABP.has_la_field_contract(instruction),
        "la_prompt": _la_prompt_after_contract(instruction, pano_n),
    }


def _pano_phrases() -> Dict[str, Any]:
    import benchmark.async_board_pa as ABP
    import benchmark.ma_search as MS
    import benchmark.pano_views as PV

    out: Dict[str, Any] = {}
    for n, stride in ((4, 90.0), (8, 45.0), (6, 60.0)):
        key = "%d_%g" % (n, stride)
        out[key] = {
            "views_phrase": PV.views_phrase(n, stride),
            "doublecheck": PV.doublecheck_phrase(n, stride),
            "spin_plan": list(PV.spin_plan(n, stride)),
            "view_index_legend": PV.view_index_legend(n),
            "current_views_label": ABP.current_views_section_label(n),
            "claim_doublecheck": MS._claim_doublecheck_sentence(n, stride),
            "view_specs": _jsonable(PV.view_specs(n)),
        }
    out["history_label"] = ABP.HISTORY_SECTION_LABEL
    return out


# ---------------------------------------------------------------- constants + AST
def _module_constants(mod) -> Dict[str, Any]:
    out = {}
    for name, val in vars(mod).items():
        if name.startswith("__") or "KEY" in name.upper():
            continue
        if inspect.ismodule(val) or inspect.isclass(val) or callable(val):
            continue
        if isinstance(val, (str, int, float, bool, tuple, list, dict, set, frozenset, re.Pattern)):
            out[name] = _jsonable(val)
    return out


class _Normalize(ast.NodeTransformer):
    def _strip_doc(self, node):
        body = getattr(node, "body", None)
        if body and isinstance(body[0], ast.Expr) and isinstance(
                getattr(body[0], "value", None), ast.Constant) and isinstance(body[0].value.value, str):
            node.body = body[1:] or [ast.Pass()]
        return node

    def visit_FunctionDef(self, node):
        self._strip_doc(node)
        self.generic_visit(node)
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self._strip_doc(node)
        self.generic_visit(node)
        return node

    def visit_Constant(self, node):
        if isinstance(node.value, str):
            key = hashlib.sha256(node.value.encode("utf-8")).hexdigest()[:16]
            if key in STRING_RENAMES:
                return ast.copy_location(ast.Constant(value=STRING_RENAMES[key], kind=None), node)
        return node

    def visit_Name(self, node):
        if node.id in NAME_AS_CONSTANT and isinstance(node.ctx, ast.Load):
            return ast.copy_location(ast.Constant(value=NAME_AS_CONSTANT[node.id], kind=None), node)
        return node


def _fingerprints(mod) -> Dict[str, str]:
    path = inspect.getsourcefile(mod)
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    out: Dict[str, str] = {}

    def add(qual, node):
        node = _Normalize().visit(ast.fix_missing_locations(node))
        dump = ast.dump(node, include_attributes=False)
        out[qual] = hashlib.sha256(dump.encode("utf-8")).hexdigest()[:20]

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add(node.name, node)
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    add("%s.%s" % (node.name, sub.name), sub)
            fields = [s for s in node.body if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef))]
            holder = ast.ClassDef(name=node.name, bases=[], keywords=[], body=fields or [ast.Pass()],
                                  decorator_list=[])
            add("%s.<fields>" % node.name, holder)
    return out


# ---------------------------------------------------------------- capture
def capture() -> Dict[str, Any]:
    import benchmark.ma_search as MS

    snap: Dict[str, Any] = {"paper_modes": list(PAPER_MODES), "modes": {}}
    first = None
    for mode in PAPER_MODES:
        run = _run_main(["--control-mode", mode] + list(PAPER_FLAGS))
        first = first or run
        attrs = run["attrs"]
        snap["modes"][mode] = {
            "flags_clean": MS._control_mode_flags(mode, True),
            "flags_unclean": MS._control_mode_flags(mode, False),
            "flags_no_frontier": MS._control_mode_flags(mode, True, True),
            "runner_attrs": attrs,
            "startup_log": run["log"],
            "parsed_args": _jsonable({k: v for k, v in run["args"].items()
                                      if k not in ("episodes", "iin_images_dir")}),
            "instructions": {sc["name"]: _mode_instruction(attrs, sc) for sc in SCENARIOS},
        }
    defaults = _run_main(["--control-mode", PAPER_MODES[0]])
    snap["argparse"] = {
        "defaults": _jsonable({k: v for k, v in defaults["args"].items()
                               if k not in ("episodes", "iin_images_dir", "control_mode")}),
        "actions": {k: v for k, v in defaults["actions"].items() if k != "control_mode"},
        "control_mode_choices": defaults["choices"].get("control_mode", []),
    }
    snap["pano"] = _pano_phrases()
    snap["constants"] = {}
    snap["ast"] = {}
    for name in LAVIRA_MODULES:
        mod = importlib.import_module(name)
        short = name.split(".")[-1]
        snap["constants"][short] = _module_constants(mod)
        snap["ast"][short] = _fingerprints(mod)
    return snap


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "lavira_prompts.json"))
    args = p.parse_args(argv)
    snap = capture()
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(snap, f, indent=1, sort_keys=True, ensure_ascii=False)
        f.write("\n")
    print("wrote %s (%d modes)" % (args.out, len(snap["modes"])))


if __name__ == "__main__":
    main()
