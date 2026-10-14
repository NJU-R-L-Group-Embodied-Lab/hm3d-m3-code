"""Run the offline self-tests (no simulator episode, GPU or API key needed).

This is the single entry point for the self-tests (``scripts/run_selftests.sh``
wraps it). Run it from ``sim-code/``::

    python tests/run_all.py                  # every test whose deps are installed
    python tests/run_all.py --tier cv        # only tiers minimal + cv (no torch / Habitat)
    python tests/run_all.py metrics zeroshot # only tests whose name contains a filter
    python tests/run_all.py --list           # print the manifest

Each test runs in its own process (``python -m tests.selftests.<name>``) and is
reported as PASS / FAIL / SKIP. Tiers, cumulative from left to right:

    minimal     numpy, pillow, defusedxml
    cv          + opencv-python-headless, matplotlib
    torch       + torch
    habitat     + habitat-sim / habitat-lab (the project environment)
    home_robot  + the DM3 home_robot fork (third_party, see benchmark/dm3_official)

A test is SKIPped when it is above ``--tier``, when a package of its tier is not
importable (FAIL with ``--strict``), or when a data file it needs is missing
(always SKIP; no data is shipped with the code). An unexpected
``ModuleNotFoundError`` of a third-party package is also a SKIP (FAIL with
``--strict``); a missing module of this repository is always a FAIL.
Exit status: 0 if nothing FAILed.
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import os
import re
import subprocess
import sys
import time
from typing import Dict, List, NamedTuple, Optional, Tuple

SIM_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TIERS = ["minimal", "cv", "torch", "habitat", "home_robot"]
TIER_MODULES: Dict[str, List[str]] = {
    "minimal": ["numpy", "PIL"],
    "cv": ["cv2", "matplotlib"],
    "torch": ["torch"],
    "habitat": ["habitat_sim", "habitat"],
    "home_robot": ["home_robot"],
}
REPO_PACKAGES = ("benchmark", "vlnce_baselines", "habitat_extensions", "services", "tests")


class Test(NamedTuple):
    name: str  # module under tests/selftests, or a script path relative to sim-code/
    tier: str
    data: Tuple[str, ...] = ()  # files (relative to sim-code/) the test needs


# The manifest: every self-test and the lowest tier it runs in.
MANIFEST: List[Test] = [
    # minimal: numpy + pillow only
    Test("benchmark_selftest", "minimal"),
    Test("clean_baseline_selftest", "minimal"),
    Test("coord_shell_selftest", "minimal"),
    Test("gen_tn_selftest", "minimal"),
    Test("goat_goal_sample_selftest", "minimal"),
    Test("metrics_selftest", "minimal"),
    Test("render_iin_selftest", "minimal"),
    Test("semantic_goals_selftest", "minimal"),
    Test("work_queue_selftest", "minimal"),
    Test("zeroshot_selftest", "minimal"),
    # cv: + opencv-python-headless, matplotlib
    Test("frontier_selftest", "cv"),
    Test("iin_verify_selftest", "cv"),
    Test("od_found_selftest", "cv"),
    Test("ovon_od_selftest", "cv"),
    Test("ma_visualizer_selftest", "cv"),
    Test("mp3d_obs_selftest", "cv"),
    # torch
    Test("ma_decision_protocol_selftest", "torch"),
    Test("ma_vlm_parallel_selftest", "torch"),
    Test("pano_views_selftest", "torch"),
    # habitat (project environment)
    Test("async_board_pa_selftest", "habitat"),
    Test("ma_nav_spin_selftest", "habitat"),
    Test("ma_search_clean_prompt_selftest", "habitat"),
    Test("tests/golden/check_lavira_prompts.py", "habitat"),
    # home_robot (DM3 fork)
    Test("dm3_runner_selftest", "home_robot"),
]


def _is_script(t: Test) -> bool:
    return t.name.endswith(".py")


def _missing_modules(tier: str) -> List[str]:
    need: List[str] = []
    for t in TIERS[: TIERS.index(tier) + 1]:
        need += TIER_MODULES[t]
    out = []
    for mod in need:
        try:
            found = importlib.util.find_spec(mod) is not None
        except (ImportError, ValueError):
            found = False
        if not found:
            out.append(mod)
    return out


def _manifest_errors() -> List[str]:
    listed = {t.name for t in MANIFEST if not _is_script(t)}
    on_disk = {
        os.path.splitext(os.path.basename(p))[0]
        for p in glob.glob(os.path.join(SIM_ROOT, "tests", "selftests", "*_selftest.py"))
    }
    errs = [f"tests/selftests/{n}.py is not in tests/run_all.py MANIFEST" for n in sorted(on_disk - listed)]
    errs += [f"MANIFEST entry {n} has no tests/selftests/{n}.py" for n in sorted(listed - on_disk)]
    errs += [f"MANIFEST entry {t.name} does not exist" for t in MANIFEST
             if _is_script(t) and not os.path.isfile(os.path.join(SIM_ROOT, t.name))]
    errs += [f"MANIFEST entry {t.name} has unknown tier {t.tier}" for t in MANIFEST if t.tier not in TIERS]
    return errs


def _classify_failure(log_text: str, strict: bool) -> Tuple[str, str]:
    m = re.search(r"ModuleNotFoundError: No module named '([^']+)'", log_text)
    if m:
        reason = m.group(0)
        if m.group(1).split(".")[0] in REPO_PACKAGES:
            return "FAIL", reason + " (repository module)"
        return ("FAIL" if strict else "SKIP"), reason
    last = [ln for ln in log_text.strip().splitlines() if ln.strip()]
    return "FAIL", (last[-1][:120] if last else "")


def run_one(t: Test, log_dir: str, timeout_s: int) -> Tuple[str, str]:
    if _is_script(t):
        cmd = [sys.executable, t.name]
        log = os.path.join(log_dir, os.path.splitext(os.path.basename(t.name))[0] + ".log")
    else:
        cmd = [sys.executable, "-m", "tests.selftests." + t.name]
        log = os.path.join(log_dir, t.name + ".log")
    with open(log, "w") as fh:
        try:
            rc = subprocess.run(cmd, cwd=SIM_ROOT, stdout=fh, stderr=subprocess.STDOUT,
                                timeout=timeout_s).returncode
        except subprocess.TimeoutExpired:
            return "FAIL", f"timeout after {timeout_s}s"
    if rc == 0:
        return "PASS", ""
    with open(log, errors="replace") as fh:
        return _classify_failure(fh.read(), strict=False)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("filters", nargs="*", help="run only tests whose name contains one of these")
    ap.add_argument("--tier", default=TIERS[-1], choices=TIERS,
                    help="highest tier to run; tests above it are skipped (default: all)")
    ap.add_argument("--strict", action="store_true",
                    help="a missing package of a selected tier is a FAIL instead of a SKIP")
    ap.add_argument("--log-dir", default=os.environ.get("LOG_DIR", "logs/selftests"))
    ap.add_argument("--timeout", type=int, default=int(os.environ.get("TIMEOUT_S", "600")))
    ap.add_argument("--list", action="store_true", help="print the manifest and exit")
    args = ap.parse_args(argv)
    if os.environ.get("STRICT") == "1":
        args.strict = True

    if args.list:
        for t in MANIFEST:
            extra = f"  data: {', '.join(t.data)}" if t.data else ""
            print(f"{t.tier:<11} {t.name}{extra}")
        return 0

    log_dir = args.log_dir if os.path.isabs(args.log_dir) else os.path.join(SIM_ROOT, args.log_dir)
    os.makedirs(log_dir, exist_ok=True)
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

    results: List[str] = []
    counts = {"PASS": 0, "FAIL": 0, "SKIP": 0}

    def record(status: str, name: str, reason: str) -> None:
        counts[status] += 1
        line = f"{status:<4}  {name:<55} {reason}".rstrip()
        results.append(line)
        print(line, flush=True)

    for err in _manifest_errors():
        record("FAIL", "manifest", err)

    max_rank = TIERS.index(args.tier)
    for t in MANIFEST:
        if args.filters and not any(f in t.name for f in args.filters):
            continue
        if TIERS.index(t.tier) > max_rank:
            record("SKIP", t.name, f"tier {t.tier} > --tier {args.tier}")
            continue
        missing = _missing_modules(t.tier)
        if missing:
            record("FAIL" if args.strict else "SKIP", t.name,
                   f"tier {t.tier} needs: {', '.join(missing)}")
            continue
        absent = [d for d in t.data if not os.path.exists(os.path.join(SIM_ROOT, d))]
        if absent:
            record("SKIP", t.name, f"missing data file: {absent[0]}")
            continue
        status, reason = run_one(t, log_dir, args.timeout)
        if status == "SKIP" and args.strict:
            status = "FAIL"
        record(status, t.name, reason)

    summary = f"pass={counts['PASS']} fail={counts['FAIL']} skip={counts['SKIP']}"
    with open(os.path.join(log_dir, "summary.txt"), "w") as fh:
        fh.write(f"# self-tests {time.strftime('%F %T %Z')} (tier <= {args.tier})\n")
        fh.write("\n".join(results) + "\n" + summary + "\n")
    print(f"{summary} (logs: {log_dir}, summary: {os.path.join(log_dir, 'summary.txt')})")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
