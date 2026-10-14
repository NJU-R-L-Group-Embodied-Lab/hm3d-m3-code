"""End-to-end self-test with a pure-Python MockNav — no Habitat, no data.

Exercises routing math, episode assembly (distributed starts + joint
reachability + difficulty labels) and the official sampler, asserting the
invariants the real pipeline must also satisfy. Run anywhere::

    python -m tests.selftests.benchmark_selftest

The MockNav approximates a scene as an open box with Euclidean geodesics — fine
for validating control flow and the routing logic. The Habitat path is
validated separately by running ``benchmark.generate`` inside the container.
"""
from __future__ import annotations

import numpy as np

from benchmark import config, routing, sample_official
from benchmark.generate import SPACING_WINDOWS, BaseNav, assign_modalities, try_make_episode
from benchmark.goal_sources import GoalInstance, SceneGoals


class MockNav(BaseNav):
    """Open-box scene; geodesic == Euclidean (x,z); everything navigable."""

    def __init__(self, extent=12.0, seed=0):
        self.extent = extent
        self.navigable_area = float(extent * extent)
        self._rng = np.random.default_rng(seed)

    def geodesic(self, a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        return float(np.hypot(a[0] - b[0], a[2] - b[2]))

    def snap(self, p):
        p = np.asarray(p, dtype=np.float32).copy()
        p[0] = float(np.clip(p[0], 0.0, self.extent))
        p[2] = float(np.clip(p[2], 0.0, self.extent))
        return p

    def is_navigable(self, p):
        p = np.asarray(p)
        return 0.0 <= p[0] <= self.extent and 0.0 <= p[2] <= self.extent

    def island_radius(self, p):
        return 10.0

    def random_point(self):
        return np.array([self._rng.uniform(0, self.extent), 0.0,
                         self._rng.uniform(0, self.extent)], dtype=np.float32)


def _mock_scene(n_cat=5, inst_per_cat=3, extent=12.0, seed=0) -> SceneGoals:
    rng = np.random.default_rng(seed)
    cats = config.HM3D_CATEGORIES[:n_cat]
    by_cat = {}
    oid = 0
    for c in cats:
        insts = []
        for _ in range(inst_per_cat):
            pos = [float(rng.uniform(0, extent)), 0.0, float(rng.uniform(0, extent))]
            vps = [{"position": pos, "rotation": [0, 0, 0, 1], "iou": 0.5}]
            insts.append(GoalInstance(c, oid, f"{c}_{oid}", pos, vps))
            oid += 1
        by_cat[c] = insts
    return SceneGoals("mock-scene", "mock/scene.basis.glb", "mock.json", "mock.navmesh", by_cat)


def test_routing_math():
    # 2 agents, 2 goals at opposite corners; starts each near one goal.
    nav = MockNav()
    starts = [np.array([0, 0, 0.0]), np.array([10, 0, 10.0])]
    goals = [np.array([0, 0, 1.0]), np.array([10, 0, 9.0])]
    r = routing.route_summary(nav, starts, goals)
    assert r["all_reachable"]
    # oracle is the minimum -> never worse than geometric or balanced
    assert r["oracle_makespan"] <= r["geometric_makespan"] + 1e-6
    assert r["oracle_makespan"] <= r["balanced_makespan"] + 1e-6
    # split assignment beats both-to-one: oracle makespan ~1, not ~ width
    assert r["oracle_makespan"] < 3.0
    print("  [ok] routing math: oracle<=geometric, split detected")


def test_allocation_critical():
    # Both goals clustered near agent 0; geometric sends both to agent 0,
    # oracle still must (only one agent is near) -> low gap. Now craft a case
    # where geometric overloads one agent though a split is cheaper.
    nav = MockNav()
    starts = [np.array([0, 0, 0.0]), np.array([1, 0, 0.0])]
    goals = [np.array([0, 0, 10.0]), np.array([1, 0, 10.0]), np.array([0.5, 0, 0.2])]
    r = routing.route_summary(nav, starts, goals)
    assert r["oracle_makespan"] <= r["geometric_makespan"] + 1e-6
    print(f"  [ok] allocation gap computed: gap={r['allocation_gap']:.3f} "
          f"critical={r['allocation_critical']}")


def test_episode_assembly():
    nav = MockNav(seed=1)
    scene = _mock_scene(seed=1)
    rng = np.random.default_rng(7)
    eps = []
    for _ in range(40):
        ep = try_make_episode(nav, scene, K=2, M=3, modality_mode="mixed",
                              spacing_mode="mixed", success_distance=1.0,
                              rng=rng, max_tries=80)
        if ep:
            ep["episode_id"] = f"mock_{len(eps):04d}"
            eps.append(ep)
    assert len(eps) >= 20, f"only made {len(eps)} episodes"
    for ep in eps:
        assert ep["num_agents"] == 2 and len(ep["agents"]) == 2
        assert len(ep["goals_set"]) == 3
        cats = [g["object_category"] for g in ep["goals_set"]]
        assert len(set(cats)) == 3, "goals must be distinct categories"
        assert ep["routing"]["oracle_makespan"] is not None
        assert ep["routing"]["oracle_makespan"] <= ep["routing"]["geometric_makespan"] + 1e-6
        # joint reachability: every goal reachable from >=1 start (finite col min)
        sg = np.asarray(ep["routing"]["start_goal_geo"])
        assert np.all(np.isfinite(sg).any(axis=0))
        assert ep["budget_geo"] is None or ep["budget_geo"] > 0
        assert set(g["modality"] for g in ep["goals_set"]) <= set(config.MODALITIES)
    print(f"  [ok] episode assembly: {len(eps)} episodes, all invariants hold")
    return eps


def test_k4_episode():
    nav = MockNav(seed=2)
    scene = _mock_scene(n_cat=5, seed=2)
    rng = np.random.default_rng(3)
    ep = None
    for _ in range(60):
        ep = try_make_episode(
            nav, scene, K=4, M=3, modality_mode="mixed",
            spacing_mode="near", success_distance=1.0, rng=rng, max_tries=80,
        )
        if ep:
            break
    assert ep is not None
    assert ep["num_agents"] == 4 and len(ep["agents"]) == 4
    assert len(ep["goals_set"]) == 3
    sg = np.asarray(ep["routing"]["start_goal_geo"])
    assert sg.shape == (4, 3)
    assert np.all(np.isfinite(sg).any(axis=0))
    print("  [ok] K=4 M=3 assembly")


def test_same_floor_episode():
    nav = MockNav(seed=4)
    scene = _mock_scene(n_cat=5, seed=4)
    rng = np.random.default_rng(5)
    ep = try_make_episode(
        nav, scene, K=2, M=3, modality_mode="mixed",
        spacing_mode="mixed", success_distance=1.0, rng=rng, max_tries=80,
        same_floor=True,
    )
    assert ep is not None
    ys = [a["start_position"][1] for a in ep["agents"]]
    ys += [g["rep_point"][1] for g in ep["goals_set"]]
    assert max(ys) - min(ys) < config.FLOOR_HEIGHT_M
    assert ep["difficulty"].get("same_floor") is True
    print("  [ok] same-floor assembly")


def test_mini_km_quota():
    quota = {(2, 3): 4, (2, 4): 4, (2, 5): 4, (3, 3): 4, (3, 4): 4, (3, 5): 4, (4, 3): 4, (4, 4): 4, (4, 5): 4}
    pool = []
    scenes = ["s%02d" % i for i in range(12)]
    n = 0
    for s in scenes:
        for k, m in sample_official.KM_CELLS:
            for _ in range(3):
                pool.append({
                    "episode_id": "%s_%05d" % (s, n),
                    "scene_key": s,
                    "num_agents": k,
                    "goals_set": [{"object_category": "c%d" % j} for j in range(m)],
                })
                n += 1
    rng = np.random.default_rng(7)
    mini = sample_official._take_km_quota(pool, quota, rng, min_per_scene=2, max_per_scene=4)
    assert len(mini) == 36
    hist = sample_official.histogram(mini)
    for km, q in quota.items():
        assert hist["by_KM"]["K%d-M%d" % km] == q
    by_s = {}
    for ep in mini:
        by_s[ep["scene_key"]] = by_s.get(ep["scene_key"], 0) + 1
    assert len(by_s) == 12
    assert min(by_s.values()) >= 2
    mini2 = sample_official._take_km_quota(pool, quota, np.random.default_rng(7), min_per_scene=2, max_per_scene=4)
    assert [e["episode_id"] for e in mini2] == [e["episode_id"] for e in mini]
    print("  [ok] mini KM quota: 9 cells exact, 12 scenes covered, deterministic")


def test_mixed3_modalities():
    rng = np.random.default_rng(0)
    for M in (3, 4, 5):
        for _ in range(80):
            tags = assign_modalities(M, "mixed3", rng)
            assert len(tags) == M
            assert {"ON", "IIN", "TN"} <= set(tags), tags
    tags3 = assign_modalities(3, "mixed3", np.random.default_rng(1))
    assert sorted(tags3) == ["IIN", "ON", "TN"]
    nav = MockNav(seed=2)
    scene = _mock_scene(seed=2)
    rng = np.random.default_rng(3)
    got = 0
    for _ in range(30):
        ep = try_make_episode(
            nav, scene, K=2, M=3, modality_mode="mixed3",
            spacing_mode="mixed", success_distance=1.0, rng=rng, max_tries=80,
        )
        if not ep:
            continue
        got += 1
        assert {"ON", "IIN", "TN"} <= set(g["modality"] for g in ep["goals_set"])
    assert got >= 5, got
    print("  [ok] mixed3: every episode has ON+IIN+TN")


def main():
    print("== HM3D-M3 benchmark self-test (MockNav, no Habitat) ==")
    test_routing_math()
    test_allocation_critical()
    test_episode_assembly()
    test_k4_episode()
    test_same_floor_episode()
    test_mixed3_modalities()
    test_mini_km_quota()
    print("ALL SELF-TESTS PASSED")


if __name__ == "__main__":
    main()
