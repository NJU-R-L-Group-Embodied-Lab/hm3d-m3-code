"""Habitat-free tests for async object-board + progress_analysis broadcast."""
from __future__ import annotations

from types import SimpleNamespace

from benchmark.async_board_pa import (
    CURRENT_VIEWS_SECTION_LABEL,
    FIELD_CONTRACT_MARKER,
    HISTORY_SECTION_LABEL,
    IND_PA_FIELD_CONTRACT_MARKER,
    apply_la_json_field_contract,
    build_async_board_instruction,
    build_async_ind_pa_instruction,
    build_object_board,
    build_own_progress_note,
    build_progress_broadcast,
    build_teammate_progress_claims,
    collect_progress_records,
    has_la_field_contract,
)
from benchmark.ma_search import (
    _build_clean_target_instruction,
    _control_mode_flags,
)


def _goals():
    return [
        {
            "goal_index": 0,
            "object_category": "chair",
            "modality": "ON",
            "position": [1.0, 2.0, 3.0],
            "rep_point": [4.0, 5.0, 6.0],
            "reach_points": [[7.0, 8.0, 9.0]],
        },
        {
            "goal_index": 1,
            "object_category": "stool",
            "modality": "TN",
            "tn_text": "a white stool under a table",
        },
        {
            "goal_index": 2,
            "object_category": "sofa",
            "modality": "IIN",
        },
    ]


def test_object_board_found_unresolved_no_coords():
    board, refs = build_object_board(
        _goals(),
        found_by=[-1, 1, -1],
        goal_pil={2: object()},
    )
    assert "goal_0 [ON]: chair — unresolved" in board
    assert "goal_1 [TN]: stool; description: a white stool under a table — found by agent_1" in board
    assert "goal_2 [IIN]: sofa; reference image #1 — unresolved" in board
    assert "position" not in board
    assert "rep_point" not in board
    assert "reach_points" not in board
    assert "geodesic" not in board
    assert "1.0" not in board
    assert len(refs) == 1
    assert refs[0]["goal_id"] == "goal_2"


def test_progress_broadcast_full_text_age_and_none():
    long_pa = ("I am searching a kitchen with a long uncut paragraph. " * 8).strip()
    text = build_progress_broadcast(
        [
            {"agent_id": 0, "text": long_pa, "step": 12},
            {"agent_id": 1, "text": "", "step": -1},
            {"agent_id": 2, "text": "   ", "step": 20},
        ],
        current_step=20,
    )
    assert f"- agent_0 (8 steps ago): {long_pa}" in text
    assert "- agent_1: none" in text
    assert "- agent_2: none" in text
    assert long_pa in text
    assert "..." not in text


def test_collect_progress_records():
    states = [
        SimpleNamespace(agent_id=0, last_progress_analysis="alpha", last_progress_step=4),
        SimpleNamespace(agent_id=1, last_progress_analysis="", last_progress_step=-1),
    ]
    recs = collect_progress_records(states)
    assert recs == [
        {"agent_id": 0, "text": "alpha", "step": 4},
        {"agent_id": 1, "text": "", "step": -1},
    ]


def test_async_board_instruction_protocol():
    states = [
        SimpleNamespace(agent_id=0, last_progress_analysis="looking left", last_progress_step=6),
        SimpleNamespace(agent_id=1, last_progress_analysis="", last_progress_step=-1),
    ]
    instruction, refs = build_async_board_instruction(
        _goals(),
        [-1, 0, -1],
        states,
        agent_id=1,
        current_step=10,
        goal_pil={2: object()},
        fhint=" STOP protocol: claim_goal_id as exactly one of: goal_0, goal_2.",
    )
    assert "You are agent_1." in instruction
    assert FIELD_CONTRACT_MARKER in instruction
    assert "Field sources:" in instruction
    assert "progress_analysis: look ONLY at YOUR Navigation History image sequence" in instruction
    assert "Do not describe, caption, or quote the current 4-directional views" in instruction
    assert "reasoning, action, and STOP: use ALL inputs" in instruction
    assert "teammate progress_analysis" in instruction
    assert "stop=true proposes FOUND verification" in instruction
    assert "does not end the episode" in instruction
    assert "Do not output a stair field." in instruction
    assert "not STOP evidence" in instruction
    assert "Confirm FOUND with your own current 4-directional views only." in instruction
    assert "found by agent_0" in instruction
    assert "Your last progress_analysis" not in instruction
    assert "Do not use a previous own progress_analysis" not in instruction
    assert "Teammate claimed progress_analysis (for reasoning_action only" in instruction
    assert "do not copy into progress_analysis" in instruction
    assert "agent_0 (4 steps ago): looking left" in instruction
    assert "- agent_1: none" not in instruction
    assert "looking left" in instruction
    claims = instruction.split("Teammate claimed progress_analysis", 1)[1]
    assert "looking left" in claims
    assert "agent_1" not in claims
    assert "claim_goal_id" in instruction
    assert "stair" in instruction
    assert "position" not in instruction
    assert len(refs) == 1


def test_own_and_teammate_pa_blocks():
    records = [
        {"agent_id": 0, "text": "alpha", "step": 4},
        {"agent_id": 1, "text": "bravo", "step": 6},
    ]
    own = build_own_progress_note(records, 8, 1)
    claims = build_teammate_progress_claims(records, 8, 1)
    assert "Your last progress_analysis" in own
    assert "- agent_1 (2 steps ago): bravo" in own
    assert "alpha" not in own
    assert "Teammate claimed progress_analysis" in claims
    assert "- agent_0 (4 steps ago): alpha" in claims
    assert "bravo" not in claims
    assert "agent_1" not in claims


def test_apply_la_json_field_contract_rewrites_placeholders():
    prompt = (
        '2. THEN, write `reasoning_action` and choose the next action based on the UPDATED TODO\n'
        '   plus what you observe.\n'
        '{\n'
        '    "progress_analysis": "<brief assessment of what you\'ve observed and done so far>",\n'
        '    "reasoning_action": "<reasoning for the chosen action, given the updated TODO + observations>",\n'
        '}\n'
        "This is a single-agent LaViRA decision. "
        "Do not include map_understanding. Use only first-person views, "
        "navigation history, and the target list.\n"
    )
    out = apply_la_json_field_contract(prompt)
    assert "from Navigation History frames ONLY" in out
    assert "do not caption current 4-dir views" in out
    assert "use ALL inputs to choose action and STOP" in out
    assert "brief assessment of what you've observed" not in out
    assert "given the updated TODO + observations" not in out
    assert "single-agent LaViRA decision" not in out
    assert "progress_analysis: Navigation History frames only" in out
    assert "reasoning / action / STOP: use ALL inputs" in out
    assert "JSON field contract (mandatory for this decision):" in out
    assert "progress_analysis MUST use Navigation History frames only" in out
    assert "reasoning / action / STOP MUST use ALL inputs" in out
    assert "board, teammate progress_analysis" in out
    assert "Do not use a previous own progress_analysis" not in out
    assert "only visual source for progress_analysis" in HISTORY_SECTION_LABEL
    assert "together with every other input" in HISTORY_SECTION_LABEL
    assert "together with every other input" in CURRENT_VIEWS_SECTION_LABEL
    assert "Do not copy these frames into progress_analysis" in CURRENT_VIEWS_SECTION_LABEL


def test_paper_modes_share_one_flag_set():
    """The paper mode uses the async V2 / no-map / no-broadcast switches."""
    flags = _control_mode_flags("k2-async-keep-goals-pa", clean_found_state=True)
    assert flags == {
        "share_found_state": False,
        "share_found_messages": False,
        "share_announcements": False,
        "share_assignment_hint": False,
        "use_frontier": False,
        "use_qwen_claim_scoring": True,
        "use_mce_primary_frontier": False,
    }
    for mode in ("k2-async-keep-goals-pa", "full"):
        assert _control_mode_flags(mode, clean_found_state=True) == flags


def test_keep_goals_is_independent_plus_shared_pa():
    states = [
        SimpleNamespace(agent_id=0, last_progress_analysis="looking left", last_progress_step=6),
        SimpleNamespace(agent_id=1, last_progress_analysis="", last_progress_step=-1),
    ]
    body, refs = _build_clean_target_instruction(
        _goals(),
        [False, True, False],
        {2: object()},
        " STOP protocol: claim_goal_id as exactly one of: goal_0, goal_2.",
    )
    instruction = build_async_ind_pa_instruction(
        body,
        states,
        agent_id=1,
        current_step=10,
    )
    assert "You are agent_1." in instruction
    assert IND_PA_FIELD_CONTRACT_MARKER in instruction
    assert has_la_field_contract(instruction)
    assert FIELD_CONTRACT_MARKER not in instruction
    assert "Targets:" in instruction
    assert "Visible verified found targets:" in instruction
    assert "found by agent_" not in instruction
    assert "There is no object board. Teammate FOUND is not listed." in instruction
    assert "Teammate claimed progress_analysis" in instruction
    assert "looking left" in instruction
    assert "Your last progress_analysis" not in instruction
    assert "does not remove it from your list" not in instruction
    assert "ASYNC_KEEP_ALL_GOALS" not in instruction
    assert "claim_goal_id" in instruction
    assert len(refs) == 1
    out = apply_la_json_field_contract("plain prompt", independent_plus_pa=True)
    assert "target list, teammate progress_analysis" in out
    assert "board, teammate progress_analysis" not in out
    assert "full target list" not in out


def main() -> None:
    test_object_board_found_unresolved_no_coords()
    test_progress_broadcast_full_text_age_and_none()
    test_collect_progress_records()
    test_async_board_instruction_protocol()
    test_own_and_teammate_pa_blocks()
    test_apply_la_json_field_contract_rewrites_placeholders()
    test_paper_modes_share_one_flag_set()
    test_keep_goals_is_independent_plus_shared_pa()
    print("async_board_pa_selftest_ok")


if __name__ == "__main__":
    main()
