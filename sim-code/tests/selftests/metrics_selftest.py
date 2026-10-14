"""Self-test for :mod:`benchmark.metrics` — offline, no Habitat/GPU.

Builds a synthetic 2-agent / 2-goal episode (goals at opposite ends, each near
one agent's start) and checks the metric behaviours we rely on:

* a balanced perfect split → MSPL high, fairness perfect, CG > 1;
* one agent doing both goals → worse MSPL, unfair, CG ≈ 1;
* a partial run (one goal missed) → SR 0, MSPL 0, Progress 0.5;
* all metrics stay within their valid ranges.
"""
from __future__ import annotations

from benchmark import metrics


def _episode():
    # start_goal[k][j], goal_goal[i][j] — agent0 near goal0, agent1 near goal1,
    # goals 20 apart. Oracle split: each agent ~1.0 makespan.
    sg = [[1.0, 21.0], [21.0, 1.0]]
    gg = [[0.0, 20.0], [20.0, 0.0]]
    return {
        "episode_id": "synth-0",
        "num_agents": 2,
        "goals_set": [{"goal_index": 0}, {"goal_index": 1}],
        "difficulty": {"allocation_critical": True},
        "routing": {
            "oracle_makespan": 1.0,              # balanced split
            "oracle_assignment": [0, 1],
            "start_goal_geo": sg,
            "goal_goal_geo": gg,
        },
    }


def _close(a, b, eps=1e-6):
    return abs(a - b) <= eps


def test_balanced_split():
    ep = _episode()
    res = {"per_agent": [
        {"path_length": 1.0, "goals_found": [0]},
        {"path_length": 1.0, "goals_found": [1]},
    ]}
    m = metrics.team_metrics(ep, res)
    assert m["success"] == 1 and _close(m["progress"], 1.0)
    assert m["mspl"] > 0.99, m["mspl"]                       # makespan == oracle
    assert _close(m["fairness_gini"], 0.0)                   # 1 goal each = equal
    assert _close(m["fairness_entropy"], 1.0)
    assert m["cooperation_gain_oracle"] > 1.5                # 1 robot ~21 vs team ~1
    print(f"  [ok] balanced: MSPL={m['mspl']:.2f} gini={m['fairness_gini']:.2f} "
          f"CG_oracle={m['cooperation_gain_oracle']:.1f}")
    return m


def test_one_agent_does_all():
    ep = _episode()
    # agent0 visits both goals (1 -> goal0 -> goal1 = 1 + 20 = 21), agent1 idle
    res = {"per_agent": [
        {"path_length": 21.0, "goals_found": [0, 1]},
        {"path_length": 0.0, "goals_found": []},
    ]}
    m = metrics.team_metrics(ep, res)
    assert m["success"] == 1
    assert m["mspl"] < 0.1, m["mspl"]                        # makespan 21 vs oracle 1
    assert m["fairness_gini"] > 0.4                          # 2 vs 0 = very unequal
    assert m["cooperation_gain_realized"] is not None and m["cooperation_gain_realized"] < 1.1
    print(f"  [ok] one-does-all: MSPL={m['mspl']:.2f} gini={m['fairness_gini']:.2f} "
          f"CG_real={m['cooperation_gain_realized']:.2f}")


def test_partial_failure():
    ep = _episode()
    res = {"per_agent": [
        {"path_length": 1.0, "goals_found": [0]},
        {"path_length": 5.0, "goals_found": []},
    ]}
    m = metrics.team_metrics(ep, res)
    assert m["success"] == 0 and _close(m["progress"], 0.5)
    assert _close(m["mspl"], 0.0)                            # SR=0 -> MSPL 0
    print(f"  [ok] partial: SR={m['success']} progress={m['progress']} MSPL={m['mspl']}")


def test_ranges_and_redundancy():
    ep = _episode()
    res = {"per_agent": [
        {"path_length": 2.0, "goals_found": [0], "explored_cells": None},
        {"path_length": 2.0, "goals_found": [1]},
    ],
        "explored_cells": [{(0, 0), (0, 1), (1, 1)}, {(0, 1), (1, 1), (2, 2)}]}
    m = metrics.team_metrics(ep, res)
    for k in ("progress", "mspl", "spl_sum", "fairness_gini", "fairness_entropy"):
        assert 0.0 <= m[k] <= 1.0, (k, m[k])
    # explored overlap: sizes=6, union=4 -> redundancy = (6-4)/4 = 0.5
    assert _close(m["redundancy"], 0.5), m["redundancy"]
    print(f"  [ok] ranges valid; redundancy={m['redundancy']}")


def test_aggregate():
    ep = _episode()
    runs = [
        {"per_agent": [{"path_length": 1.0, "goals_found": [0]}, {"path_length": 1.0, "goals_found": [1]}]},
        {"per_agent": [{"path_length": 21.0, "goals_found": [0, 1]}, {"path_length": 0.0, "goals_found": []}]},
    ]
    ms = [metrics.team_metrics(ep, r) for r in runs]
    agg = metrics.aggregate(ms)
    assert agg["n"] == 2 and agg["SR"] == 1.0
    assert agg["alloc_critical"]["n"] == 2
    assert 2 in agg["by_M"] or len(agg["by_M"]) >= 1
    print(f"  [ok] aggregate: SR={agg['SR']} MSPL={agg['MSPL']:.2f} "
          f"alloc_critical.n={agg['alloc_critical']['n']}")


def test_coordination_channels():
    """overlap_ratio / workload_std / SRAT(micro) / comm-congestion-FP passthrough."""
    ep = _episode()
    res = {
        "per_agent": [
            {"path_length": 2.0, "goals_found": [0]},
            {"path_length": 6.0, "goals_found": [1]},
        ],
        "explored_cells": [{(0, 0), (0, 1), (1, 1)}, {(0, 1), (1, 1), (2, 2)}],
        "comm_volume": 12, "congestion": 0.3, "false_positives": 1,
    }
    m = metrics.team_metrics(ep, res)
    # sizes=6, union=4 -> overlap_ratio = (6-4)/6 = 0.333..
    assert _close(m["overlap_ratio"], (6 - 4) / 6, eps=1e-3), m["overlap_ratio"]
    assert _close(m["workload_std"], 0.0)                 # 1 goal each -> equal counts
    assert _close(m["path_std"], 2.0)                     # std([2,6]) = 2
    assert m["comm_volume"] == 12 and m["congestion"] == 0.3 and m["false_positives"] == 1
    assert m["targets_found"] == 2
    print(f"  [ok] coord channels: overlap={m['overlap_ratio']:.3f} "
          f"path_std={m['path_std']:.1f} comm={m['comm_volume']} FP={m['false_positives']}")


def test_aggregate_new_channels():
    """SRAT micro, Overlap/Comm/Congestion/FP means, cross-floor slice."""
    flat = _episode(); flat["difficulty"] = {"allocation_critical": True, "cross_floor": False}
    xf = _episode(); xf["episode_id"] = "synth-xf"
    xf["difficulty"] = {"allocation_critical": False, "cross_floor": True}
    full = {"per_agent": [{"path_length": 1.0, "goals_found": [0]},
                          {"path_length": 1.0, "goals_found": [1]}],
            "comm_volume": 10, "congestion": 0.2, "false_positives": 0}
    partial = {"per_agent": [{"path_length": 1.0, "goals_found": [0]},
                             {"path_length": 5.0, "goals_found": []}],
               "comm_volume": 4, "congestion": 0.5, "false_positives": 2}
    ms = [metrics.team_metrics(flat, full), metrics.team_metrics(xf, partial)]
    agg = metrics.aggregate(ms)
    # micro SRAT: (2 + 1) targets found / (2 + 2) total = 0.75; macro Progress = (1+0.5)/2 = 0.75
    assert _close(agg["SRAT"], 0.75), agg["SRAT"]
    assert _close(agg["Comm_Volume"], 7.0) and _close(agg["Congestion"], 0.35)
    assert _close(agg["False_Positives"], 1.0)
    assert agg["cross_floor"]["n"] == 1 and agg["cross_floor"]["SR"] == 0.0
    print(f"  [ok] aggregate channels: SRAT={agg['SRAT']:.2f} comm={agg['Comm_Volume']:.1f} "
          f"xfloor.n={agg['cross_floor']['n']}")


def main():
    print("== metrics self-test (offline) ==")
    test_balanced_split()
    test_one_agent_does_all()
    test_partial_failure()
    test_ranges_and_redundancy()
    test_aggregate()
    test_coordination_channels()
    test_aggregate_new_channels()
    print("ALL METRICS SELF-TESTS PASSED")


if __name__ == "__main__":
    main()
