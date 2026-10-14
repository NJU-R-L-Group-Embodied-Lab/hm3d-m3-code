"""Frontier detection + distinct geometric assignment for multi-agent coverage.

The decentralized LA agents tend to drift into the same area, so goals nobody
visits never enter the approach-lock radius. Frontier coordination fixes the
*coverage* ceiling: extract exploration frontiers (the boundary between explored
and unknown) on the shared map and hand each agent a DIFFERENT frontier so the
team spreads out.

This module is pure ``numpy``/``opencv`` (no Habitat), so it is unit-testable on
synthetic occupancy masks. Assignment here is purely **geometric** (nearest,
load-balanced, distinct) — the Co-NavGPT2 nearest/co_ut baseline; a semantic
relevance score is layered on later for the method version.
"""
from __future__ import annotations

import itertools
import math
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


def detect_frontiers(explored: np.ndarray, obstacle: np.ndarray,
                     min_cluster: int = 12) -> List[Tuple[float, float, int]]:
    """Cluster frontier cells and return centroids.

    Args:
        explored: HxW bool/0-1 mask of explored cells (agent_0 frame).
        obstacle: HxW bool/0-1 mask of known obstacles.
        min_cluster: drop frontier clusters smaller than this (cells).

    Returns list of ``(col, row, size)`` centroids, largest first.
    """
    explored = (np.asarray(explored) > 0).astype(np.uint8)
    obstacle = (np.asarray(obstacle) > 0).astype(np.uint8)
    free = (explored & (1 - obstacle)).astype(np.uint8)
    unknown = (1 - explored).astype(np.uint8)

    # frontier = free cell touching unknown (and not an obstacle)
    k = np.ones((3, 3), np.uint8)
    unknown_dil = cv2.dilate(unknown, k, iterations=1)
    frontier = (free & unknown_dil & (1 - obstacle)).astype(np.uint8)
    # NOTE: frontier rings are intrinsically 1-2 cells thick, so do NOT apply a
    # morphological open here (it would erode them away); rely on the
    # connected-component min_cluster filter to drop noise instead.

    num, _labels, stats, centroids = cv2.connectedComponentsWithStats(frontier, 8)
    out = []
    for i in range(1, num):
        size = int(stats[i, cv2.CC_STAT_AREA])
        if size >= min_cluster:
            cx, cy = centroids[i]  # (col, row)
            out.append((float(cx), float(cy), size))
    out.sort(key=lambda t: t[2], reverse=True)
    return out


def _dist(a, b) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def assign_frontiers(agent_cells: Sequence[Tuple[float, float]],
                     frontiers: Sequence[Tuple[float, float, int]],
                     dist_fn=None) -> List[Optional[int]]:
    """Assign each agent a DISTINCT frontier minimizing total travel.

    ``dist_fn(agent_idx, frontier_idx) -> float`` overrides the default cell
    Euclidean distance (e.g. to pass geodesic distance). When there are fewer
    frontiers than agents, the surplus agents share the globally-nearest ones.
    Returns a frontier index (or None) per agent.
    """
    K = len(agent_cells)
    F = len(frontiers)
    if F == 0:
        return [None] * K
    if dist_fn is None:
        def dist_fn(ai, fi):
            return _dist(agent_cells[ai], (frontiers[fi][0], frontiers[fi][1]))

    # distinct assignment via brute force over injective maps (K small, F small);
    # if F < K, fall back to per-agent nearest (sharing allowed).
    if F >= K and K <= 4 and F <= 8:
        best, best_cost = None, float("inf")
        for combo in itertools.permutations(range(F), K):
            cost = sum(dist_fn(ai, combo[ai]) for ai in range(K))
            if cost < best_cost:
                best, best_cost = combo, cost
        return list(best)
    # surplus agents or large F: greedy distinct then nearest
    assign: List[Optional[int]] = [None] * K
    used = set()
    order = sorted(range(K), key=lambda ai: min(dist_fn(ai, fi) for fi in range(F)))
    for ai in order:
        cand = [fi for fi in range(F) if fi not in used] or list(range(F))
        fi = min(cand, key=lambda fi: dist_fn(ai, fi))
        assign[ai] = fi
        used.add(fi)
    return assign


def assign_frontiers_mce(
    agent_cells: Sequence[Tuple[float, float]],
    frontiers: Sequence[Tuple[float, float, int]],
    *,
    agent_world_xz: Optional[Sequence[Tuple[float, float]]] = None,
    frontier_world_xz: Optional[Sequence[Tuple[float, float]]] = None,
    repulse_lambda: float = 0.6,
    dist_scale_m: float = 5.0,
    dist_fn=None,
) -> List[Optional[int]]:
    """Map-Complementary Exploration assignment on a shared map.

    Each agent gets a **distinct** frontier maximising expected coverage per
    travel cost while repelling from teammates' *current positions* (and from
    frontiers assigned earlier in the greedy pass). This is the policy core of
    MCE-Nav: fix double-coverage on the union occupancy map without goal leases
    or free-form chat.

    Score (higher better)::

        score(ai, fi) = (size/max_size) * exp(-d/D0)
                        + λ * tanh(min_dist_to_teammates / 4)

    Then a greedy distinct assignment (agents ordered by best available score)
    is applied. Falls back to :func:`assign_frontiers` when F or K is 0.
    """
    K = len(agent_cells)
    F = len(frontiers)
    if K == 0:
        return []
    if F == 0:
        return [None] * K

    sizes = [max(1, int(fr[2])) for fr in frontiers]
    max_sz = float(max(sizes) or 1)

    if dist_fn is None:
        def dist_fn(ai, fi):  # noqa: F811
            return _dist(agent_cells[ai], (frontiers[fi][0], frontiers[fi][1]))

    # Precompute pairwise geometric costs.
    dmat = np.zeros((K, F), dtype=np.float64)
    for ai in range(K):
        for fi in range(F):
            dmat[ai, fi] = float(dist_fn(ai, fi))

    def teammate_repulsion(ai: int, fi: int, taken_frontiers: set) -> float:
        """Distance of frontier fi to other agents (and to already-taken frontiers)."""
        reps = []
        if agent_world_xz is not None and frontier_world_xz is not None:
            fx, fz = frontier_world_xz[fi]
            for aj in range(K):
                if aj == ai:
                    continue
                ax, az = agent_world_xz[aj]
                reps.append(math.hypot(fx - ax, fz - az))
        # Also repel from frontiers already claimed in this assignment round
        # (spreads the team even when agents are co-located).
        if frontier_world_xz is not None and taken_frontiers:
            fx, fz = frontier_world_xz[fi]
            for ofi in taken_frontiers:
                ox, oz = frontier_world_xz[ofi]
                reps.append(math.hypot(fx - ox, fz - oz))
        if not reps:
            return 0.0
        return float(np.tanh(min(reps) / 4.0))

    def score(ai: int, fi: int, taken: set) -> float:
        d = dmat[ai, fi]
        if not np.isfinite(d):
            return -1e9
        info = sizes[fi] / max_sz
        reach = math.exp(-d / max(dist_scale_m, 1e-3))
        rep = teammate_repulsion(ai, fi, taken)
        # Prefer large, reachable, far-from-teammate frontiers.
        return info * reach + float(repulse_lambda) * rep

    assign: List[Optional[int]] = [None] * K
    used: set = set()
    # Agents with the best current option first (stable spread).
    remaining_agents = list(range(K))
    while remaining_agents:
        best_ai, best_fi, best_sc = None, None, -1e18
        for ai in remaining_agents:
            cand = [fi for fi in range(F) if fi not in used] or list(range(F))
            for fi in cand:
                sc = score(ai, fi, used)
                if sc > best_sc:
                    best_ai, best_fi, best_sc = ai, fi, sc
        if best_ai is None:
            break
        assign[best_ai] = best_fi
        used.add(best_fi)
        remaining_agents.remove(best_ai)
    return assign
