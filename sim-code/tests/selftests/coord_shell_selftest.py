"""Self-test for :mod:`benchmark.coord_shell` — offline, no Habitat/GPU."""
from __future__ import annotations

from benchmark import coord_shell, metrics


def _close(a, b, eps=1e-6):
    return abs(a - b) <= eps


def test_channel():
    ch = coord_shell.CoordChannel(congestion_radius=1.0)
    # 5 steps; agents far apart except step 3 (within 1m on the ground plane).
    far = [[0.0, 0.0, 0.0], [10.0, 0.0, 10.0]]
    near = [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]]
    diff_floor = [[0.0, 0.0, 0.0], [0.2, 3.0, 0.0]]  # close in XZ but a floor up
    for step_pts in (far, far, near, far, diff_floor):
        ch.observe_positions(step_pts)
    ch.message(0, payload_bytes=128)
    ch.message(0, payload_bytes=128)
    ch.message(1, payload_bytes=64)
    ch.false_positive()
    s = ch.summary()
    assert s["comm_volume"] == 3 and s["comm_bytes"] == 320, s
    assert _close(s["congestion"], 1 / 5), s            # only the 'near' step
    assert s["false_positives"] == 1
    assert 0.0 <= s["comm_gini"] <= 1.0
    print(f"  [ok] channel: comm={s['comm_volume']} congestion={s['congestion']} "
          f"FP={s['false_positives']} comm_gini={s['comm_gini']}")
    return s


def test_feeds_metrics():
    """channel.summary() merges into result and surfaces in team_metrics."""
    ch = coord_shell.CoordChannel()
    ch.message(0); ch.message(0); ch.false_positive()
    ep = {
        "episode_id": "x", "num_agents": 2,
        "goals_set": [{"goal_index": 0}, {"goal_index": 1}],
        "difficulty": {"allocation_critical": False, "cross_floor": False},
        "routing": {"oracle_makespan": 1.0, "oracle_assignment": [0, 1],
                    "start_goal_geo": [[1.0, 21.0], [21.0, 1.0]],
                    "goal_goal_geo": [[0.0, 20.0], [20.0, 0.0]]},
    }
    result = {"per_agent": [{"path_length": 1.0, "goals_found": [0]},
                            {"path_length": 1.0, "goals_found": [1]}]}
    result.update(ch.summary())
    m = metrics.team_metrics(ep, result)
    assert m["comm_volume"] == 2 and m["false_positives"] == 1, m
    print(f"  [ok] feeds metrics: comm_volume={m['comm_volume']} FP={m['false_positives']}")


def test_responsibilities():
    assert len(coord_shell.SHELL_RESPONSIBILITIES) == 5
    print(f"  [ok] {len(coord_shell.SHELL_RESPONSIBILITIES)} shell responsibilities defined")


def main():
    print("== coord_shell self-test (offline) ==")
    test_channel()
    test_feeds_metrics()
    test_responsibilities()
    print("ALL COORD-SHELL SELF-TESTS PASSED")


if __name__ == "__main__":
    main()
