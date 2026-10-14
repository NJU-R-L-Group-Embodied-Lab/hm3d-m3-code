"""Self-test for :mod:`benchmark.frontier` — offline, synthetic occupancy masks."""
from __future__ import annotations

import numpy as np

from benchmark.frontier import assign_frontiers, assign_frontiers_mce, detect_frontiers


def test_detect():
    H = W = 200
    explored = np.zeros((H, W), np.uint8)
    obstacle = np.zeros((H, W), np.uint8)
    # two explored blobs separated by unknown -> two frontier rings
    explored[20:60, 20:60] = 1
    explored[140:180, 140:180] = 1
    fr = detect_frontiers(explored, obstacle, min_cluster=8)
    assert len(fr) >= 2, f"expected >=2 frontier clusters, got {len(fr)}"
    # centroids should sit near the two blobs
    cents = [(c[0], c[1]) for c in fr]
    near_a = any(abs(c[0] - 40) < 30 and abs(c[1] - 40) < 30 for c in cents)
    near_b = any(abs(c[0] - 160) < 30 and abs(c[1] - 160) < 30 for c in cents)
    assert near_a and near_b, f"frontiers not near both blobs: {cents}"
    print(f"  [ok] detect: {len(fr)} clusters at {[(round(c[0]),round(c[1])) for c in fr]}")
    return fr


def test_assign_distinct():
    # 2 agents, 2 frontiers at opposite corners; each agent near one
    frontiers = [(40.0, 40.0, 50), (160.0, 160.0, 50)]
    agents = [(45.0, 45.0), (155.0, 155.0)]
    a = assign_frontiers(agents, frontiers)
    assert sorted(a) == [0, 1], f"assignment must be distinct, got {a}"
    assert a[0] == 0 and a[1] == 1, f"each agent should take its nearer frontier: {a}"
    print(f"  [ok] assign distinct: {a}")


def test_assign_avoids_clumping():
    # both agents near frontier 0; distinct assignment must still split them
    frontiers = [(40.0, 40.0, 80), (160.0, 160.0, 30)]
    agents = [(42.0, 42.0), (50.0, 50.0)]
    a = assign_frontiers(agents, frontiers)
    assert sorted(a) == [0, 1], f"both near f0 but must split: {a}"
    print(f"  [ok] anti-clump: {a} (one sent to the far frontier)")


def test_fewer_frontiers_than_agents():
    frontiers = [(100.0, 100.0, 40)]
    agents = [(10.0, 10.0), (190.0, 190.0)]
    a = assign_frontiers(agents, frontiers)
    assert a == [0, 0], f"single frontier -> both assigned to it: {a}"
    print(f"  [ok] surplus agents share: {a}")


def test_mce_repels_and_distinct():
    # Co-located agents, two frontiers of equal size far apart: MCE must still split.
    frontiers = [(40.0, 40.0, 50), (160.0, 160.0, 50)]
    agents = [(100.0, 100.0), (101.0, 101.0)]
    agent_xz = [(0.0, 0.0), (0.1, 0.1)]
    frontier_xz = [(0.0, 0.0), (10.0, 10.0)]
    a = assign_frontiers_mce(
        agents, frontiers,
        agent_world_xz=agent_xz,
        frontier_world_xz=frontier_xz,
        repulse_lambda=0.6,
    )
    assert sorted(a) == [0, 1], f"MCE must assign distinct frontiers, got {a}"
    print(f"  [ok] mce distinct: {a}")


def main():
    print("== frontier self-test (offline) ==")
    test_detect()
    test_assign_distinct()
    test_assign_avoids_clumping()
    test_fewer_frontiers_than_agents()
    test_mce_repels_and_distinct()
    print("ALL FRONTIER SELF-TESTS PASSED")


if __name__ == "__main__":
    main()
