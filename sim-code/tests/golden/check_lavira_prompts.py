"""Self-test: the LaViRA prompt / mode surface still matches the paper code.

``lavira_prompts.json`` was captured with ``capture_lavira_prompts.py`` on the
code that produced the paper numbers (before the release cleanup) and then
restricted to the released paper mode (LaViRA full): entries that belonged only
to other modes were dropped, together with runner switches and start-up log
fields that are off in this mode. This script re-captures on the current tree
and compares. It exits non-zero on any difference that is not one of the
intentional release changes listed below.

Run from ``sim-code/`` in the project environment (no simulator, GPU or API key)::

    python tests/golden/check_lavira_prompts.py
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import capture_lavira_prompts as CAP  # noqa: E402

GOLDEN = os.path.join(HERE, "lavira_prompts.json")

# Paper alias accepted by --control-mode in the release.
ALIASES = {
    "full": "k2-async-keep-goals-pa",
}

# ---- intentional release changes -------------------------------------------
# CLI flags that only tuned non-paper modes (qwen observer / goal focus,
# semantic contracts, frontier assignment, opting into the team top-down map). Their runner attributes keep the
# same class defaults, which the per-mode runner_attrs comparison enforces.
REMOVED_ARGS = {
    "force_team_topdown_map",
    "no_frontier",
    "ovn_confirm",
    "qwen_observer_cooldown_steps",
    "qwen_observer_min_confidence",
    "qwen_observer_object_only",
    "qwen_focus_cooldown_steps",
    "qwen_focus_min_confidence",
    "semantic_contract_lease_steps",
}
# Functions whose source changed on purpose. Their behaviour for the paper
# mode is pinned by the flag / runner-attribute / log / prompt sections.
# ma_search's episode loop and the async_board_pa contract helpers lost the
# branches of the modes that are not released; those branches were off in
# the paper mode. _verify_goal lost the --ovn-confirm gate (off in the paper
# mode). The adapted baselines' optional Verifier V2 path and its prompts are
# not shipped: those baselines use the detector FOUND gate. The runner's class fields and the run-config block of
# _build_metrics_result lost the switches of those modes (all False here).
CHANGED_FUNCTIONS = {
    "ma_search": {"_control_mode_flags", "main",
                  "MASearchRunner._run_episode", "MASearchRunner.<fields>",
                  "MASearchRunner._build_metrics_result", "MASearchRunner._verify_goal"},
    "async_board_pa": {"apply_la_json_field_contract", "has_la_field_contract"},
}
NEW_FUNCTIONS = {
    "ma_search": {"resolve_control_mode"},
}
# Development switches that are off in the paper mode were removed together with
# their modules (target contracts, goal leases, qwen target observer / goal
# focus, team top-down map). Their flags, runner attributes and golden entries
# were dropped after asserting they were False / None in the paper mode; the
# helpers below served only those switches.
REMOVED_FUNCTIONS: Dict[str, set] = {
    "ma_search": {"_announcement_target_text", "_bbox_confidence",
                  "_compose_soft_contract_focus_prompt", "_depth_at_bbox_metres",
                  "_select_decoupled_claim_goal", "_select_opportunistic_visible_goal",
                  "_select_soft_contract_claim_goal", "_shares_direct_teammate_focus",
                  "MASearchRunner._get_qwen_goal_focus_observer",
                  "MASearchRunner._get_qwen_target_observer"},
}
# Module-level names added (all values still compared when present in both).
NEW_CONSTANTS = {
    "ma_search": {"PAPER_CONTROL_MODES", "CONTROL_MODE_ALIASES", "_LLM_MODEL"},
}
# FAIL_R_M was the near/far radius of the per-episode failure-analysis dump,
# which is not shipped; it never affected the policy.
# MAX_LAST_METER_STEPS was the target-contract last-metre limit (removed above).
REMOVED_CONSTANTS: Dict[str, set] = {
    "ma_search": {"FAIL_R_M", "MAX_LAST_METER_STEPS"},
}


def _diff(path: str, old: Any, new: Any, out: List[str]) -> None:
    if old == new:
        return
    if isinstance(old, dict) and isinstance(new, dict):
        for k in sorted(set(old) | set(new)):
            if k not in new:
                out.append("%s.%s: removed" % (path, k))
            elif k not in old:
                out.append("%s.%s: added" % (path, k))
            else:
                _diff("%s.%s" % (path, k), old[k], new[k], out)
        return
    o, n = json.dumps(old, ensure_ascii=False), json.dumps(new, ensure_ascii=False)
    if isinstance(old, str) and isinstance(new, str):
        k = next((i for i, (a, b) in enumerate(zip(old, new)) if a != b), min(len(old), len(new)))
        lo = max(0, k - 40)
        o = "...%s..." % json.dumps(old[lo:k + 80], ensure_ascii=False)
        n = "...%s..." % json.dumps(new[lo:k + 80], ensure_ascii=False)
    out.append("%s: %s -> %s" % (path, o[:300], n[:300]))


def _drop(d: Dict[str, Any], keys) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if k not in keys}


def compare(golden: Dict[str, Any], cur: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    for mode in CAP.PAPER_MODES:
        g, c = dict(golden["modes"][mode]), dict(cur["modes"][mode])
        g["parsed_args"] = _drop(g["parsed_args"], REMOVED_ARGS)
        _diff("modes.%s" % mode, g, c, errs)

    g_args, c_args = golden["argparse"], cur["argparse"]
    _diff("argparse.defaults", _drop(g_args["defaults"], REMOVED_ARGS), c_args["defaults"], errs)
    _diff("argparse.actions", _drop(g_args["actions"], REMOVED_ARGS), c_args["actions"], errs)
    want_choices = set(CAP.PAPER_MODES) | set(ALIASES)
    if set(c_args["control_mode_choices"]) != want_choices:
        errs.append("argparse.control_mode_choices: %r" % c_args["control_mode_choices"])
    if not set(CAP.PAPER_MODES) <= set(g_args["control_mode_choices"]):
        errs.append("golden is missing a paper mode")

    for key in ("pano",):
        _diff(key, golden[key], cur[key], errs)

    for mod in sorted(set(golden["constants"]) | set(cur["constants"])):
        g = _drop(golden["constants"].get(mod, {}), REMOVED_CONSTANTS.get(mod, set()))
        c = _drop(cur["constants"].get(mod, {}), NEW_CONSTANTS.get(mod, set()))
        _diff("constants.%s" % mod, g, c, errs)

    for mod in sorted(set(golden["ast"]) | set(cur["ast"])):
        skip_g = CHANGED_FUNCTIONS.get(mod, set()) | REMOVED_FUNCTIONS.get(mod, set())
        skip_c = CHANGED_FUNCTIONS.get(mod, set()) | NEW_FUNCTIONS.get(mod, set())
        g = _drop(golden["ast"].get(mod, {}), skip_g)
        c = _drop(cur["ast"].get(mod, {}), skip_c)
        for name in sorted(set(g) | set(c)):
            if g.get(name) != c.get(name):
                errs.append("ast.%s.%s: %s -> %s" % (mod, name, g.get(name), c.get(name)))
        for name in CHANGED_FUNCTIONS.get(mod, set()) | NEW_FUNCTIONS.get(mod, set()):
            if name not in cur["ast"].get(mod, {}):
                errs.append("ast.%s.%s: expected in current code" % (mod, name))
    return errs


def check_aliases(cur: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    for alias, mode in ALIASES.items():
        run = CAP._run_main(["--control-mode", alias] + list(CAP.PAPER_FLAGS))
        want = cur["modes"][mode]
        _diff("alias.%s.runner_attrs" % alias, want["runner_attrs"], run["attrs"], errs)
        _diff("alias.%s.startup_log" % alias, want["startup_log"], run["log"], errs)
    return errs


def main() -> int:
    with open(GOLDEN, encoding="utf-8") as f:
        golden = json.load(f)
    cur = json.loads(json.dumps(CAP.capture(), sort_keys=True))
    errs = compare(golden, cur) + check_aliases(cur)
    if errs:
        print("LaViRA golden check FAILED (%d differences):" % len(errs))
        for e in errs[:200]:
            print("  " + e)
        return 1
    print("LaViRA golden check passed (%d paper mode(s), %d alias(es))"
          % (len(CAP.PAPER_MODES), len(ALIASES)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
