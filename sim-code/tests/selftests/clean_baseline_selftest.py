"""Static guard for clean HM3D-M3 baseline protocol.

This test intentionally avoids importing Habitat-backed runners. It checks the
source-level contracts that prevent evaluator GT from re-entering policy prompts
or hidden navigation controllers.
"""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2] / "benchmark"


def _src(name: str) -> str:
    return (ROOT / name).read_text()


def test_runner_clean_defaults() -> None:
    for name in ["ego_runner.py", "conav_runner.py"]:
        src = _src(name)
        assert "clean_found_state: bool = True" in src, name
        assert 'if not getattr(self, "clean_found_state", True):' in src, name
        assert "--legacy-gt-approach-lock" in src, name


def test_conav_prompt_uses_full_target_list() -> None:
    src = _src("conav_runner.py")
    assert 'remaining = sorted({goals[j]["object_category"] for j in range(M) if not found[j]})' not in src
    assert "target_list = [_goal_label(goals[j], j) for j in range(M)]" in src
    assert "last_obs, target_list)" in src


def _run_method_cases() -> dict:
    """method -> MODULE_ARGS tokens parsed from scripts/run_method.sh."""
    import re
    import shlex

    sh = (ROOT.parent / "scripts" / "run_method.sh").read_text()
    body = sh.split("method_module_and_args() {", 1)[1].split("\n}\n", 1)[0]
    cases = {}
    for name, args in re.findall(r"^\s+([a-z0-9_]+)\)\s*\n\s+MODULE_ARGS=\((.*?)\)\s*\n\s+;;",
                                 body, flags=re.M | re.S):
        cases[name] = shlex.split(args.replace("\\\n", " "))
    return cases


EXPECTED_METHODS = {
    "lavira_full": ["benchmark.ma_search", "--use-vlm", "--clean-found-state",
                    "--disable-path-budget", "--control-mode", "k2-async-keep-goals-pa",
                    "--no-team-topdown-map"],
    "vlfm": ["benchmark.zeroshot_runner", "--method", "vlfm"],
    "ego_swarm": ["benchmark.ego_runner"],
    "conav": ["benchmark.conav_runner", "--vlm", "qwen"],
    "dm3": ["benchmark.dm3_official.runner"],
}


def test_launchers_use_clean_lavira() -> None:
    METHODS = EXPECTED_METHODS
    cases = _run_method_cases()
    assert sorted(cases) == sorted(METHODS), (sorted(cases), sorted(METHODS))
    for name, tokens in cases.items():
        if "--la-thinking-budget" in tokens:
            i = tokens.index("--la-thinking-budget")
            assert tokens[i + 1] == "$THINKING_BUDGET", name
            tokens = tokens[:i] + tokens[i + 2:]
        assert tokens == METHODS[name], (name, tokens, METHODS[name])
        assert "--ovn-confirm" not in tokens and "--verifier-v2" not in tokens, name
        if name.startswith("lavira_"):
            for flag in ("--use-vlm", "--clean-found-state", "--disable-path-budget",
                         "--no-team-topdown-map", "--control-mode"):
                assert flag in tokens, (name, flag)
    assert METHODS["lavira_full"][METHODS["lavira_full"].index("--control-mode") + 1] \
        == "k2-async-keep-goals-pa"


def test_native_found_uses_od_gate() -> None:
    """Non-LaViRA found = open-vocabulary detector STOP, no GT geodesic."""
    for name in ["ego_runner.py", "conav_runner.py"]:
        src = _src(name)
        assert "claim_from_obs(" in src, name
        assert "ODFoundGate.from_runner" in src, name
        assert "if found[j] or self._geo_to_reach(sim, cur, reach[j]) > self.success_distance:" not in src, name
    official = _src("dm3_official/runner.py")
    assert "ODFoundGate.from_runner" in official
    assert "PROTOCOL_NAME" in official
    assert 'scored_found_by = self._score_official_gt_viewpoint(' not in official
    od = _src("od_found.py")
    assert 'PROTOCOL_NAME = "ovon_od_detector_stop"' in od
    assert "sim.geodesic_distance" not in od
    assert "geodesic_to_set" not in od


def test_zeroshot_found_matches_ego_perception() -> None:
    zs = _src("zeroshot_runner.py")
    assert "def _confirm_goal" not in zs
    assert "OpenVocabDetector" not in zs
    assert "present(" not in zs


def test_zeroshot_follows_original_papers() -> None:
    zs = _src("zeroshot_runner.py")
    assert "BLIP2ValueMap" in zs
    assert 'vlfm_backend: str = "blip2"' in zs
    assert "CLIP-ITM" not in zs
    assert "blip2-itm" in zs
    blip = _src("vlfm_value.py")
    assert "Seems like there is a {category} ahead." in blip
    assert "use_image_text_matching_head=use_itm" in blip
    assert 'match_head: str = "itc"' in blip


def main() -> None:
    test_runner_clean_defaults()
    test_conav_prompt_uses_full_target_list()
    test_launchers_use_clean_lavira()
    test_native_found_uses_od_gate()
    test_zeroshot_found_matches_ego_perception()
    test_zeroshot_follows_original_papers()
    print("clean baseline self-test passed")


if __name__ == "__main__":
    main()
