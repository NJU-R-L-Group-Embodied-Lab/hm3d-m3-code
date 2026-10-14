"""Regression tests for the LaViRA two-phase STOP (approach, then arrival DCS)."""
from __future__ import annotations

import numpy as np

from vlnce_baselines.multi.ma_decision import (
    confirm_arrival_stop,
    decide,
    normalized_bbox_depth_meters,
)
from vlnce_baselines.multi.ma_mapping_planner import (
    WAYPOINT_REACH_M,
    map_cells_euclidean_m,
    world_xz_reached,
)


class _FakeVLM:
    def __init__(
        self,
        *,
        stop_signal: bool,
        confirm: bool = False,
        claim_goal_id: str = "",
    ) -> None:
        self.stop_signal = stop_signal
        self.confirm = confirm
        self.claim_goal_id = claim_goal_id
        self.calls = []
        self.stop_instruction = ""
        self.stop_claim_goal_id = None

    def navigate_or_backtrack(self, **_kwargs):
        self.calls.append("navigate")
        return {
            "action": "NAVIGATE",
            "direction": "forward",
            "stop_signal": self.stop_signal,
            "claim_goal_id": self.claim_goal_id,
        }

    def double_check_stop(self, *args, **_kwargs):
        self.calls.append("double_check")
        self.stop_instruction = str(args[0])
        self.stop_claim_goal_id = _kwargs.get("claim_goal_id")
        return self.confirm, {
            "decision": "STOP" if self.confirm else "CONTINUE",
            "view_index": 0,
            "bbox_2d": [100, 100, 900, 900],
        }

    def query_llm(self, **_kwargs):
        self.calls.append("query_llm")
        raise RuntimeError("intentional test stop before projection")


def _run(vlm, stop_events=None, *, active_goal_id="goal_0"):
    frame = {
        "rgb": np.zeros((8, 8, 3), dtype=np.uint8),
        "depth": np.full((8, 8, 1), 0.1, dtype=np.float32),
    }
    return decide(
        sim=object(),
        agent_id=0,
        panorama_frames=[frame] * 4,
        visited_targets=[{"panorama_frames": [frame] * 4}],
        vlm_agent=vlm,
        instruction="Find goal_0 [ON]: chair.",
        width=8,
        height=8,
        hfov_deg=90.0,
        depth_min=0.1,
        depth_max=5.0,
        current_step=12,
        active_goal_id=active_goal_id,
        stop_event_sink=(stop_events.append if stop_events is not None else None),
    )


def test_navigate_skips_double_check():
    vlm = _FakeVLM(stop_signal=False)
    assert _run(vlm) is None
    assert vlm.calls == ["navigate", "query_llm"]


def test_stop_signal_same_turn_does_not_double_check():
    vlm = _FakeVLM(stop_signal=True, confirm=True)
    events = []
    assert _run(vlm, events) is None
    assert vlm.calls == ["navigate", "query_llm"]
    assert [event["event_type"] for event in events] == ["la_stop_proposed"]


def test_policy_selected_goal_is_recorded_on_stop_proposal():
    vlm = _FakeVLM(stop_signal=True, confirm=True, claim_goal_id="goal_2")
    events = []
    _run(vlm, events, active_goal_id=None)
    assert events[0]["goal_id"] == "goal_2"


def test_arrival_double_check_confirms_without_depth_gate():
    vlm = _FakeVLM(stop_signal=True, confirm=True)
    events = []
    frame = {
        "rgb": np.zeros((8, 8, 3), dtype=np.uint8),
        "depth": np.full((8, 8, 1), 0.1, dtype=np.float32),
    }
    ok, verification = confirm_arrival_stop(
        vlm_agent=vlm,
        instruction="Find goal_2 [ON]: chair.",
        panorama_frames=[frame] * 4,
        visited_targets=[{"panorama_frames": [frame] * 4}],
        claim_goal_id="goal_2",
        stop_event_sink=events.append,
        agent_id=0,
        step=40,
    )
    assert ok is True
    assert verification["decision"] == "STOP"
    assert "depth_gate_passed" not in verification
    assert vlm.calls == ["double_check"]
    assert "Claim under verification: goal_2" in vlm.stop_instruction
    assert [event["event_type"] for event in events] == ["double_check_confirmed"]


def test_arrival_double_check_can_reject():
    vlm = _FakeVLM(stop_signal=True, confirm=False)
    events = []
    frame = {
        "rgb": np.zeros((8, 8, 3), dtype=np.uint8),
        "depth": np.full((8, 8, 1), 0.1, dtype=np.float32),
    }
    ok, _ = confirm_arrival_stop(
        vlm_agent=vlm,
        instruction="Find goal_0 [ON]: chair.",
        panorama_frames=[frame] * 4,
        visited_targets=[{"panorama_frames": [frame] * 4}],
        claim_goal_id="goal_0",
        stop_event_sink=events.append,
    )
    assert ok is False
    assert [event["event_type"] for event in events] == ["double_check_rejected"]


def test_map_euclidean_matches_lavira_075m():
    assert WAYPOINT_REACH_M == 0.75
    # 15 cells at 5 cm = 0.75 m; official uses strict <.
    assert abs(map_cells_euclidean_m((10, 10), (10, 25), 5.0) - 0.75) < 1e-9
    assert map_cells_euclidean_m((10, 10), (10, 24), 5.0) < 0.75
    assert map_cells_euclidean_m((10, 10), (10, 26), 5.0) > 0.75


def test_world_xz_matches_drawn_x():
    agent = (1.0, 0.0, 2.0)
    assert world_xz_reached(agent, (1.2, 0.0, 2.1))
    assert not world_xz_reached(agent, (2.0, 0.0, 2.0))
    assert not world_xz_reached(None, (1.0, 0.0, 2.0))


def test_normalized_bbox_depth_rejects_invalid_coordinates():
    frame = {"depth": np.full((480, 640, 1), 0.1, dtype=np.float32)}
    assert normalized_bbox_depth_meters(
        [frame], {"view_index": 0, "bbox_2d": [100, 100, 900, 900]}, 0.1, 5.0,
    ) == np.float32(0.1) * (5.0 - 0.1) + 0.1
    assert normalized_bbox_depth_meters(
        [frame], {"view_index": 0, "bbox_2d": [0, 0, 1000, 999]}, 0.1, 5.0,
    ) is None


def main() -> None:
    test_navigate_skips_double_check()
    test_stop_signal_same_turn_does_not_double_check()
    test_policy_selected_goal_is_recorded_on_stop_proposal()
    test_arrival_double_check_confirms_without_depth_gate()
    test_arrival_double_check_can_reject()
    test_map_euclidean_matches_lavira_075m()
    test_world_xz_matches_drawn_x()
    test_normalized_bbox_depth_rejects_invalid_coordinates()
    print("ma_decision STOP protocol self-test passed")


if __name__ == "__main__":
    main()
