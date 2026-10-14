"""P3: team metrics for multi-agent unordered multi-object navigation.

Computes, from an episode (its stored oracle makespan ``d*`` and geodesic
matrices) plus a run result (per-agent travelled path length + which goals each
agent found):

* **Progress / SR** — fraction of the goal-set found / whether ALL were found.
* **MSPL** — ``SR · d* / max{d*, max_j d_j}`` (DM3's metric; ``d*`` = oracle
  min-max makespan from P1, ``d_j`` = agent j's travelled length). The team
  objective for cooperative search: dominated by the *slowest* agent (makespan),
  not the sum.
* **SPL_sum** — sum-based SPL using the oracle per-agent route lengths.
* **Fairness** — load balance over per-agent found-goal counts: Gini (0 = equal)
  and normalised entropy (1 = equal). DM3 reports neither.
* **Cooperation Gain (CG)** — single-agent-does-all makespan / team makespan
  (>1 ⇒ the team beats one robot). Reported both oracle and realised. DM3 has
  only a scaling number, no formal CG.
* **Redundancy** — ``(Σ|explored_k| − |∪explored|)/|∪explored|`` when per-agent
  explored cell-sets are supplied (overlap of exploration); else ``None``.

All inputs are plain dicts/lists so this is unit-testable with no Habitat/GPU.
The geometry reuses :mod:`benchmark.routing` on the matrices already stored in
each episode by the P1 generator.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .routing import nn_route_len

INF = float("inf")


# ---------------------------------------------------------------- fairness math
def gini(values: Sequence[float]) -> float:
    """Gini coefficient in [0,1]; 0 = perfectly equal. Empty/all-zero ⇒ 0."""
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0 or x.sum() == 0:
        return 0.0
    x = np.sort(np.clip(x, 0, None))
    n = x.size
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum() - (n + 1) * x.sum()) / (n * x.sum()))


def normalized_entropy(counts: Sequence[float]) -> float:
    """Normalised Shannon entropy of the load distribution in [0,1]; 1 = equal."""
    x = np.asarray(counts, dtype=np.float64)
    n = x.size
    if n <= 1 or x.sum() == 0:
        return 1.0 if n <= 1 else 0.0
    p = x / x.sum()
    p = p[p > 0]
    h = -(p * np.log(p)).sum()
    return float(h / np.log(n))


# ---------------------------------------------------------------- routing helpers
def single_agent_makespan(sg: np.ndarray, gg: np.ndarray) -> float:
    """Best single agent visits ALL goals: min over agents of its NN-TSP route."""
    K, M = sg.shape
    allg = list(range(M))
    return min(nn_route_len(sg[a], gg, allg) for a in range(K))


def assignment_route_lengths(sg: np.ndarray, gg: np.ndarray, assignment: Sequence[int]) -> List[float]:
    """Per-agent NN-TSP route length under a goal->agent assignment."""
    K = sg.shape[0]
    groups: List[List[int]] = [[] for _ in range(K)]
    for j, a in enumerate(assignment):
        groups[a].append(j)
    return [nn_route_len(sg[a], gg, groups[a]) for a in range(K)]


# ---------------------------------------------------------------- main metric
def team_metrics(episode: Dict, result: Dict) -> Dict:
    """Compute the full metric dict for one episode run.

    ``result`` = {
        "per_agent": [{"path_length": float, "goals_found": [goal_idx, ...]},
                      ...],                # one entry per agent
        "explored_cells": [set(...), ...]  # OPTIONAL, per-agent visited cells
    }
    """
    M = len(episode["goals_set"])
    K = episode["num_agents"]
    routing = episode["routing"]
    d_star = routing.get("oracle_makespan")
    sg = np.asarray(routing["start_goal_geo"], dtype=np.float64)
    gg = np.asarray(routing["goal_goal_geo"], dtype=np.float64)

    per_agent = result["per_agent"]
    path_lengths = [float(a.get("path_length", 0.0)) for a in per_agent]
    found_sets = [set(a.get("goals_found", [])) for a in per_agent]
    found_counts = [len(s) for s in found_sets]
    union_found = set().union(*found_sets) if found_sets else set()

    progress = len(union_found) / M if M else 0.0
    sr = 1 if len(union_found) == M and M > 0 else 0

    team_makespan = max(path_lengths) if path_lengths else INF
    total_path = sum(path_lengths)

    # MSPL (makespan-normalised SPL)
    if sr and d_star and np.isfinite(team_makespan) and team_makespan > 0:
        mspl = float(d_star / max(d_star, team_makespan))
    else:
        mspl = 0.0

    # SPL_sum using oracle per-agent route lengths as the sum-optimal reference
    oracle_assign = routing.get("oracle_assignment")
    if oracle_assign is not None:
        lstar_sum = sum(assignment_route_lengths(sg, gg, oracle_assign))
    else:
        lstar_sum = d_star or 0.0
    if sr and lstar_sum > 0 and total_path > 0:
        spl_sum = float(lstar_sum / max(lstar_sum, total_path))
    else:
        spl_sum = 0.0

    # Fairness over found-goal counts (only meaningful with >1 agent)
    fair_gini = gini(found_counts)
    fair_entropy = normalized_entropy(found_counts)

    # Cooperation gain
    single_ms = single_agent_makespan(sg, gg)
    cg_oracle = float(single_ms / d_star) if (d_star and d_star > 0 and np.isfinite(single_ms)) else None
    cg_realized = (float(single_ms / team_makespan)
                   if (np.isfinite(single_ms) and np.isfinite(team_makespan) and team_makespan > 0)
                   else None)

    # Redundancy + overlap-ratio (optional, from per-agent explored cell-sets).
    #   redundancy   = (Σ|E_k| − |∪E_k|) / |∪E_k|   (DM3-style, unbounded)
    #   overlap_ratio= (Σ|E_k| − |∪E_k|) / Σ|E_k|    (Explore-Bench, in [0,1])
    redundancy = None
    overlap_ratio = None
    explored = result.get("explored_cells")
    if explored:
        sizes = sum(len(s) for s in explored)
        union = len(set().union(*explored)) if explored else 0
        redundancy = float((sizes - union) / union) if union else 0.0
        overlap_ratio = float((sizes - union) / sizes) if sizes else 0.0

    # Workload imbalance: std of per-agent found-goal counts and travelled length.
    workload_std = float(np.std(found_counts)) if found_counts else 0.0
    path_std = float(np.std(path_lengths)) if path_lengths else 0.0

    # --- Coordination-cost channels (logged by the runner's coordination shell;
    #   None when a method does not report them). Communication overhead,
    #   spatial congestion, and wrong-instance stops (open-vocab / instance FP).
    comm_volume = result.get("comm_volume")          # messages or bytes exchanged
    congestion = result.get("congestion")            # agent-agent close-proximity time
    false_positives = result.get("false_positives")  # count of wrong-object stops

    return {
        "episode_id": episode.get("episode_id"),
        "M": M, "num_agents": K,
        "progress": round(progress, 4),
        "success": sr,
        "targets_found": len(union_found),       # for micro-averaged SRAT
        "mspl": round(mspl, 4),
        "spl_sum": round(spl_sum, 4),
        "team_makespan": round(team_makespan, 3) if np.isfinite(team_makespan) else None,
        "d_star": d_star,
        "fairness_gini": round(fair_gini, 4),
        "fairness_entropy": round(fair_entropy, 4),
        "workload_std": round(workload_std, 4),
        "path_std": round(path_std, 4),
        "found_counts": found_counts,
        "cooperation_gain_oracle": round(cg_oracle, 4) if cg_oracle is not None else None,
        "cooperation_gain_realized": round(cg_realized, 4) if cg_realized is not None else None,
        "redundancy": round(redundancy, 4) if redundancy is not None else None,
        "overlap_ratio": round(overlap_ratio, 4) if overlap_ratio is not None else None,
        "comm_volume": comm_volume,
        "congestion": congestion,
        "false_positives": false_positives,
        "cross_floor": episode["difficulty"].get("cross_floor"),
        "allocation_critical": episode["difficulty"].get("allocation_critical"),
    }


def aggregate(metrics: List[Dict]) -> Dict:
    """Dataset-level means + slices that surface what aggregate SR/MSPL hide."""
    if not metrics:
        return {}

    def mean(key, subset=None):
        rows = subset if subset is not None else metrics
        vals = [m[key] for m in rows if m.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    alloc = [m for m in metrics if m.get("allocation_critical")]
    xfloor = [m for m in metrics if m.get("cross_floor")]

    def micro_srat(rows=None):
        rows = rows if rows is not None else metrics
        tf = sum(m.get("targets_found", 0) for m in rows)
        tt = sum(m.get("M", 0) for m in rows)
        return float(tf / tt) if tt else None

    out = {
        "n": len(metrics),
        "SR": mean("success"),                 # joint success (all M found)
        "Progress": mean("progress"),          # macro: per-episode fraction
        "SRAT": micro_srat(),                  # micro: targets found / targets total
        "MSPL": mean("mspl"),
        "SPL_sum": mean("spl_sum"),
        "Fairness_Gini": mean("fairness_gini"),
        "Fairness_Entropy": mean("fairness_entropy"),
        "Workload_Std": mean("workload_std"),
        "Path_Std": mean("path_std"),
        "CG_oracle": mean("cooperation_gain_oracle"),
        "CG_realized": mean("cooperation_gain_realized"),
        "Redundancy": mean("redundancy"),
        "Overlap_Ratio": mean("overlap_ratio"),
        "Comm_Volume": mean("comm_volume"),
        "Congestion": mean("congestion"),
        "False_Positives": mean("false_positives"),
        # the allocation-critical slice — where semantic allocation should pay off
        "alloc_critical": {
            "n": len(alloc),
            "SR": mean("success", alloc),
            "MSPL": mean("mspl", alloc),
            "Fairness_Gini": mean("fairness_gini", alloc),
        },
        # cross-floor slice — multi-floor as an explicit difficulty axis
        "cross_floor": {
            "n": len(xfloor),
            "SR": mean("success", xfloor),
            "Progress": mean("progress", xfloor),
            "MSPL": mean("mspl", xfloor),
        },
        "by_M": {},
    }
    for M in sorted({m["M"] for m in metrics}):
        rows = [m for m in metrics if m["M"] == M]
        out["by_M"][M] = {"n": len(rows), "SR": mean("success", rows), "MSPL": mean("mspl", rows)}
    return out
