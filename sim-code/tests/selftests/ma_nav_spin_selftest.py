"""Heading + legend checks for the in-place-spin fix."""
from __future__ import annotations

from habitat.sims.habitat_simulator.actions import HabitatSimActions

from vlnce_baselines.multi.ma_mapping_planner import SharedMappingPlanner
from vlnce_baselines.multi.ma_visualizer import scale_bbox_xyxy
from benchmark.ma_search import _hold_obs_list, _is_habitat_stop, _sim_step_obs_list
from habitat_extensions.multi_agent.multi_sim import _is_stop_action


def test_stg_heading_forward_when_aligned():
    # STG further along +col; heading 0 after Policy's -heading + 90° rot
    # should be treated as a small angle → FORWARD for a 30° turn budget.
    act, ang = SharedMappingPlanner._stg_heading_action(
        (10, 10), (10, 16), heading_deg=0.0, turn_angle=30.0
    )
    assert act == HabitatSimActions.MOVE_FORWARD, (act, ang)


def test_scale_bbox_xyxy_halves_on_half_tile():
    box = scale_bbox_xyxy(
        {"x1": 100, "y1": 40, "x2": 300, "y2": 200},
        (640, 480),
        (320, 240),
    )
    assert box == (50, 20, 150, 100)


def test_va_bbox_overlay_is_one_frame_only():
    from vlnce_baselines.multi.ma_runner import PerAgentState, consume_va_bbox_overlay

    s0 = PerAgentState(agent_id=0)
    s1 = PerAgentState(agent_id=1)
    s0.va_bbox_flash = {"x1": 1, "y1": 2, "x2": 3, "y2": 4}
    s0.va_bbox_flash_rgb = "va-rgb"
    live = ["live0", "live1"]
    rgbs, boxes = consume_va_bbox_overlay([s0, s1], live)
    assert rgbs == ["va-rgb", "live1"]
    assert boxes[0]["x1"] == 1 and boxes[1] is None
    rgbs2, boxes2 = consume_va_bbox_overlay([s0, s1], live)
    assert rgbs2 == ["live0", "live1"]
    assert boxes2 == [None, None]


def test_k1_stop_is_hold_not_sim_step():
    assert _is_habitat_stop(HabitatSimActions.STOP)
    assert _is_habitat_stop("stop")
    assert _is_habitat_stop(int(HabitatSimActions.STOP))
    assert _is_habitat_stop(HabitatSimActions.MOVE_FORWARD) is False
    assert _is_stop_action(HabitatSimActions.STOP)
    assert _is_stop_action("STOP")
    assert _is_stop_action(HabitatSimActions.TURN_LEFT) is False

    class DummySim:
        def __init__(self):
            self.stepped = []
            self._prev_sim_obs = {"rgb": "held"}

        def step(self, action):
            if _is_habitat_stop(action):
                raise AssertionError("No move_fn for action 'stop'")
            self.stepped.append(action)
            return {"rgb": "moved"}

    dummy = DummySim()
    held = _sim_step_obs_list(dummy, [HabitatSimActions.STOP], 1)
    assert held[0]["rgb"] == "held"
    assert dummy.stepped == []
    moved = _sim_step_obs_list(dummy, [HabitatSimActions.MOVE_FORWARD], 1)
    assert moved[0]["rgb"] == "moved"
    assert dummy.stepped == [HabitatSimActions.MOVE_FORWARD]
    assert _hold_obs_list(dummy, 1)[0]["rgb"] == "held"


def main():
    test_stg_heading_forward_when_aligned()
    test_scale_bbox_xyxy_halves_on_half_tile()
    test_va_bbox_overlay_is_one_frame_only()
    test_k1_stop_is_hold_not_sim_step()
    print("ma_nav_spin self-test passed")


if __name__ == "__main__":
    main()
