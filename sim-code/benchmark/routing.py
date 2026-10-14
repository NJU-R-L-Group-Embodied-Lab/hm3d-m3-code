"""Multi-agent routing geometry: geodesic matrices and team makespan.

Given K agent starts and M goal reach-points we compute:

* ``start_goal[k][j]`` — geodesic from agent k's start to goal j,
* ``goal_goal[i][j]``   — geodesic between goal reach-points.

From these we derive the **team makespan** under different allocations:

* ``oracle``    — min over all goal->agent assignments of the max per-agent route
                  (each agent's route = nearest-neighbour open tour over its goals).
                  Exact brute force while ``K**M`` is small, else balanced-greedy.
* ``geometric`` — each goal to its *nearest agent by start distance*, load-blind
                  (what a distance-weighted frontier heuristic, e.g. DM3, tends to).
* ``balanced``  — load-aware greedy (earliest-finishing agent grabs its nearest
                  unclaimed goal); a strong, fair geometric baseline.

The relative gap ``(geometric - oracle) / oracle`` flags *allocation-critical*
episodes — the ones where semantic allocation should beat geometry.

All routes use geodesic NN-TSP (open path). M is small (<=10) so this is cheap.
Makespan is the right team objective for cooperative search (DM3 reports MSPL,
defined via the team makespan); SPL-style normalisation is applied by the runner.
"""
from __future__ import annotations

import itertools
from typing import List, Sequence, Tuple

import numpy as np

from .config import ALLOCATION_CRITICAL_GAP, ORACLE_BRUTE_FORCE_CAP

INF = float("inf")


def build_geo_matrices(nav, starts: Sequence, goal_pts: Sequence) -> Tuple[np.ndarray, np.ndarray]:
    """Return (start_goal[K,M], goal_goal[M,M]); unreachable legs are +inf."""
    K, M = len(starts), len(goal_pts)
    sg = np.full((K, M), INF, dtype=np.float64)
    for k in range(K):
        for j in range(M):
            sg[k, j] = nav.geodesic(starts[k], goal_pts[j])
    gg = np.zeros((M, M), dtype=np.float64)
    for i in range(M):
        for j in range(i + 1, M):
            d = nav.geodesic(goal_pts[i], goal_pts[j])
            gg[i, j] = gg[j, i] = d
    return sg, gg


def nn_route_len(start_to_g: np.ndarray, gg: np.ndarray, idxs: Sequence[int]) -> float:
    """Open nearest-neighbour tour length: start -> all goals in ``idxs``.

    +inf if any required leg is unreachable.
    """
    remaining = list(idxs)
    if not remaining:
        return 0.0
    # first hop: from the agent start to its nearest assigned goal
    cur = min(remaining, key=lambda x: start_to_g[x])
    total = float(start_to_g[cur])
    remaining.remove(cur)
    while remaining:
        nxt = min(remaining, key=lambda x: gg[cur, x])
        total += float(gg[cur, nxt])
        cur = nxt
        remaining.remove(nxt)
    return total if np.isfinite(total) else INF


def makespan_for_assignment(assign: Sequence[int], sg: np.ndarray, gg: np.ndarray, K: int) -> Tuple[float, List[float], List[List[int]]]:
    """assign[j] = agent index for goal j. Returns (makespan, per_agent_cost, groups)."""
    groups: List[List[int]] = [[] for _ in range(K)]
    for j, a in enumerate(assign):
        groups[a].append(j)
    costs = [nn_route_len(sg[a], gg, groups[a]) for a in range(K)]
    return max(costs), costs, groups


def oracle_makespan(sg: np.ndarray, gg: np.ndarray) -> Tuple[float, List[int], bool]:
    """Min-max team makespan. Returns (makespan, assignment, exact?).

    Brute force over K**M assignments while tractable; otherwise fall back to the
    balanced-greedy assignment and mark the result approximate.
    """
    K, M = sg.shape
    if K ** M <= ORACLE_BRUTE_FORCE_CAP:
        best_ms, best_assign = INF, None
        for assign in itertools.product(range(K), repeat=M):
            ms, _, _ = makespan_for_assignment(assign, sg, gg, K)
            if ms < best_ms:
                best_ms, best_assign = ms, list(assign)
        return best_ms, best_assign, True
    ms, assign, _ = balanced_greedy(sg, gg)
    return ms, assign, False


def geometric_assignment(sg: np.ndarray) -> List[int]:
    """Load-blind: each goal -> nearest agent by start geodesic distance."""
    return [int(np.argmin(sg[:, j])) for j in range(sg.shape[1])]


def balanced_greedy(sg: np.ndarray, gg: np.ndarray) -> Tuple[float, List[int], List[float]]:
    """Load-aware greedy: the earliest-finishing agent claims its nearest unclaimed goal."""
    K, M = sg.shape
    assign = [-1] * M
    cur_pos = [None] * K           # current goal index per agent (None = at start)
    cur_cost = [0.0] * K
    unclaimed = set(range(M))
    while unclaimed:
        a = int(np.argmin(cur_cost))
        if cur_pos[a] is None:
            j = min(unclaimed, key=lambda x: sg[a, x])
            leg = float(sg[a, j])
        else:
            j = min(unclaimed, key=lambda x: gg[cur_pos[a], x])
            leg = float(gg[cur_pos[a], j])
        assign[j] = a
        cur_cost[a] += leg
        cur_pos[a] = j
        unclaimed.discard(j)
    return max(cur_cost), assign, cur_cost


def route_summary(nav, starts: Sequence, goal_pts: Sequence) -> dict:
    """Full routing analysis for one episode (matrices + the three makespans)."""
    sg, gg = build_geo_matrices(nav, starts, goal_pts)
    oracle_ms, oracle_assign, exact = oracle_makespan(sg, gg)
    geo_assign = geometric_assignment(sg)
    geo_ms, _, _ = makespan_for_assignment(geo_assign, sg, gg, len(starts))
    bal_ms, bal_assign, _ = balanced_greedy(sg, gg)

    if np.isfinite(oracle_ms) and oracle_ms > 1e-6:
        gap = (geo_ms - oracle_ms) / oracle_ms
    else:
        gap = 0.0
    return {
        "start_goal": sg.tolist(),
        "goal_goal": gg.tolist(),
        "oracle_makespan": _f(oracle_ms),
        "oracle_assignment": oracle_assign,
        "oracle_exact": bool(exact),
        "geometric_makespan": _f(geo_ms),
        "geometric_assignment": geo_assign,
        "balanced_makespan": _f(bal_ms),
        "balanced_assignment": bal_assign,
        "allocation_gap": round(float(gap), 4),
        "allocation_critical": bool(gap > ALLOCATION_CRITICAL_GAP and np.isfinite(geo_ms)),
        "all_reachable": bool(np.isfinite(oracle_ms)),
    }


def _f(x: float):
    return round(float(x), 4) if np.isfinite(x) else None
