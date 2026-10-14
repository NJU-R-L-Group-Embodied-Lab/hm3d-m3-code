"""Habitat-free tests for 8-view 45° panorama layout and Independent+PA prompts."""
from __future__ import annotations

from types import SimpleNamespace

from benchmark.async_board_pa import (
    CURRENT_VIEWS_SECTION_LABEL,
    apply_la_json_field_contract,
    build_async_ind_pa_instruction,
    current_views_section_label,
)
from benchmark.pano_views import (
    direction_to_frame_idx,
    doublecheck_phrase,
    frame_idx_for_angle,
    navigate_action_choices,
    nearest_cardinal_name,
    parse_focus_view,
    parse_navigate_action,
    spin_plan,
    view_index_legend,
    view_specs,
    views_phrase,
)
from vlnce_baselines.multi.ma_decision import DIRECTION_TO_FRAME_IDX


def test_spin_plan_default_and_8x45():
    assert spin_plan(4, 90.0) == (12, 30.0, 3)
    assert spin_plan(8, 45.0) == (8, 45.0, 1)


def test_eight_view_cardinals():
    specs = view_specs(8)
    assert [s["angle"] for s in specs] == [0, 45, 90, 135, 180, 225, 270, 315]
    assert direction_to_frame_idx("forward", 8) == 0
    assert direction_to_frame_idx("left", 8) == 2
    assert direction_to_frame_idx("behind", 8) == 4
    assert direction_to_frame_idx("right", 8) == 6
    assert direction_to_frame_idx("forward", 4) == DIRECTION_TO_FRAME_IDX["forward"]
    assert direction_to_frame_idx("left", 4) == DIRECTION_TO_FRAME_IDX["left"]
    assert frame_idx_for_angle(45, 8) == 1
    assert parse_focus_view(3, 8) == 3
    assert parse_focus_view(9, 8) is None
    assert "2=90° LEFT" in view_index_legend(8)


def test_ind_pa_8view_prompt():
    states = [
        SimpleNamespace(agent_id=0, last_progress_analysis="looking left", last_progress_step=6),
        SimpleNamespace(agent_id=1, last_progress_analysis="", last_progress_step=-1),
    ]
    instruction = build_async_ind_pa_instruction(
        "Targets: goal_0",
        states,
        agent_id=1,
        current_step=10,
        views_phrase=views_phrase(8, 45.0),
    )
    assert "current 8-view 45° panorama" in instruction
    assert "4-directional views" not in instruction
    assert "8-VIEW 45°" in current_views_section_label(8)
    assert "4-DIRECTIONAL" in CURRENT_VIEWS_SECTION_LABEL
    coda = apply_la_json_field_contract("plain prompt", independent_plus_pa=True, pano_n=8)
    assert "navigate to view" in coda
    assert "Do not output focus_view" in coda
    assert "Cardinal actions stay" not in coda
    assert "8-view 45° panorama" in coda
    assert "four-direction" not in doublecheck_phrase(8, 45.0)
    assert "8-view 45°" in doublecheck_phrase(8, 45.0)
    section = current_views_section_label(8)
    assert "navigate to view k" in section
    assert "focus_view" in section  # "Do not output focus_view"


def test_eight_view_actions():
    items, desc = navigate_action_choices(8, blocked_indices=[4])
    assert len(items) == 7
    assert "navigate to view 4" not in desc
    assert "navigate to view 1" in desc
    assert "navigate to view 7" in desc
    assert parse_navigate_action("navigate to view 1", 8) == 1
    assert parse_navigate_action("navigate to view 7", 8) == 7
    assert parse_navigate_action("navigate to 45°", 8) == 1
    assert parse_navigate_action("navigate to left", 8) == 2
    assert parse_navigate_action("navigate to forward", 8) == 0
    assert parse_navigate_action("NAVIGATE", 8) is None
    assert nearest_cardinal_name(1, 8) == "forward"
    assert nearest_cardinal_name(2, 8) == "left"
    assert nearest_cardinal_name(7, 8) == "forward"


if __name__ == "__main__":
    test_spin_plan_default_and_8x45()
    test_eight_view_cardinals()
    test_ind_pa_8view_prompt()
    test_eight_view_actions()
    print("pano_views_selftest ok")
