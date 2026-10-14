"""Self-test for clean ma_search target prompt construction."""
from __future__ import annotations

from benchmark.ma_search import (
    CONTROL_MODE_ALIASES,
    PAPER_CONTROL_MODES,
    resolve_control_mode,
    _build_clean_target_instruction,
    _control_mode_flags,
    _policy_selected_claim_prompt,
)


def test_clean_prompt_keeps_object_level_targets():
    goals = [
        {
            "goal_index": 0,
            "object_category": "chair",
            "modality": "ON",
            "position": [1, 2, 3],
            "rep_point": [4, 5, 6],
            "reach_points": [[7, 8, 9]],
        },
        {
            "goal_index": 1,
            "object_category": "chair",
            "modality": "TN",
            "tn_text": "a white stool under a table",
        },
        {
            "goal_index": 2,
            "object_category": "sofa",
            "modality": "IIN",
        },
    ]
    instruction, refs = _build_clean_target_instruction(
        goals,
        visible_verified_found=[False, True, False],
        goal_pil={2: object()},
    )

    assert "- goal_0 [ON]: chair" in instruction
    assert "- goal_1 [TN]: chair; description: a white stool under a table" in instruction
    assert "- goal_2 [IIN]: sofa; reference image #1" in instruction
    assert "Visible verified found targets: goal_1 [TN]: chair." in instruction
    assert instruction.count("chair") == 3
    assert "rep_point" not in instruction
    assert "reach_points" not in instruction
    assert "position" not in instruction
    assert len(refs) == 1


PAPER_FLAGS = {
    "share_found_state": False,
    "share_found_messages": False,
    "share_announcements": False,
    "share_assignment_hint": False,
    "use_frontier": False,
    "use_qwen_claim_scoring": True,
    "use_mce_primary_frontier": False,
}


def test_paper_control_modes_and_aliases():
    assert PAPER_CONTROL_MODES == ("k2-async-keep-goals-pa",)
    assert {resolve_control_mode(a): a for a in CONTROL_MODE_ALIASES} == {
        "k2-async-keep-goals-pa": "full",
    }
    for mode in PAPER_CONTROL_MODES:
        assert resolve_control_mode(mode) == mode
        for clean in (True, False):
            for no_frontier in (True, False):
                assert _control_mode_flags(mode, clean, no_frontier) == PAPER_FLAGS
    for dropped in ("k2-coop", "k2-async-board-pa", "k1-start0", "k2-independent-verifier-v2"):
        try:
            resolve_control_mode(dropped)
        except ValueError:
            pass
        else:
            raise AssertionError("non-paper mode accepted: %s" % dropped)


def test_policy_selected_claim_prompt_uses_only_public_unresolved_ids():
    goals = [
        {"goal_index": 0, "object_category": "chair", "modality": "ON"},
        {"goal_index": 2, "object_category": "plant", "modality": "TN"},
    ]
    prompt = _policy_selected_claim_prompt(goals, [True, False])
    assert "claim_goal_id" in prompt
    assert "goal_2" in prompt
    assert "goal_0" not in prompt
    all_ids = _policy_selected_claim_prompt(goals, [True, False], list_all=True)
    assert "goal_0" in all_ids and "goal_2" in all_ids
    assert "unresolved target" not in all_ids
    assert "position" not in prompt
    assert "rep_point" not in prompt and "reach_points" not in prompt


def main() -> None:
    test_clean_prompt_keeps_object_level_targets()
    test_paper_control_modes_and_aliases()
    test_policy_selected_claim_prompt_uses_only_public_unresolved_ids()
    print("ma_search clean prompt self-test passed")


if __name__ == "__main__":
    main()
