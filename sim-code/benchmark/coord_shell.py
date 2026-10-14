"""Coordination-shell interface + cost logger (HM3D-M3 P0).

Every HM3D-M3 baseline is a *single-agent policy wrapped in a coordination
shell*. The shell is the only multi-agent-specific surface; swapping the inner
policy (VLFM / ESC / ZSON / CoW / DD-PPO / LaViRA) must not require touching it.
This module *codifies* that contract (so the released runners — ego/dm3_official/
conav/zeroshot/ma_search — and future ones describe the same five responsibilities) and
provides :class:`CoordChannel`, the lightweight logger that turns a runner's
coordination traffic into the P2 metrics (``comm_volume`` / ``congestion`` /
``false_positives``) consumed by :func:`benchmark.metrics.team_metrics`.

The five shell responsibilities (the de-facto interface the runners share):

1. **Task allocation** — map the M unordered goals / frontiers onto the K agents
   (centralized VLM, CVRP tour, geometric+intent, decentralized LA ...).
2. **Communication** — what each agent shares (intent, detections, map deltas)
   and under what range/bandwidth (``r_comm``); every message is logged so the
   coordination *cost* is measured, not assumed free.
3. **Shared mapping** — co-registered occupancy/semantic map (see
   ``vlnce_baselines.multi.ma_mapping_planner.SharedMappingPlanner`` /
   ``comm_gated_layers``).
4. **Multi-goal termination & bookkeeping** — per-goal found-set, approach-lock,
   stop when all M verified (modality-aware via ``benchmark.iin_verify`` /
   ``benchmark.dm3_perception``).
5. **Inter-agent collision / congestion** — keep agents from occupying the same
   space; the congestion proxy quantifies how often they do.

:class:`CoordChannel` is pure-Python/numpy and Habitat-free so it is
unit-testable. A runner instantiates one per episode, calls ``message`` whenever
the shell exchanges information, ``observe_positions`` each sim step, and
``false_positive`` when an agent stops at a wrong/distractor instance; then folds
``channel.summary()`` into the per-episode ``result`` dict.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import config


@dataclass
class CoordChannel:
    """Per-episode coordination-cost logger.

    Parameters
    ----------
    congestion_radius : float
        Two agents closer than this (metres, world frame) count as "congested"
        for that sim step.
    """

    congestion_radius: float = 1.0
    _messages: int = 0
    _bytes: int = 0
    _steps: int = 0
    _congested_steps: int = 0
    _false_positives: int = 0
    _per_agent_msgs: Dict[int, int] = field(default_factory=dict)

    # -- communication -------------------------------------------------------
    def message(self, sender: int, payload_bytes: int = 0, n: int = 1) -> None:
        """Record ``n`` messages (optionally sized) emitted by ``sender``."""
        self._messages += int(n)
        self._bytes += int(payload_bytes)
        self._per_agent_msgs[sender] = self._per_agent_msgs.get(sender, 0) + int(n)

    # -- congestion ----------------------------------------------------------
    def observe_positions(self, positions: Sequence[Sequence[float]]) -> bool:
        """Feed all agents' world positions for one sim step.

        Returns True if this step was congested (any pair within
        ``congestion_radius``). Uses XZ (ground plane); y/height ignored so two
        agents on different floors are never "congested".
        """
        self._steps += 1
        pts = np.asarray(positions, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[0] < 2:
            return False
        has_y = pts.shape[1] >= 3
        xz = pts[:, [0, 2]] if has_y else pts[:, :2]
        y = pts[:, 1] if has_y else np.zeros(pts.shape[0])
        congested = False
        for i in range(xz.shape[0]):
            for j in range(i + 1, xz.shape[0]):
                # same floor only: a teammate one floor up is not "in the way".
                if abs(float(y[i] - y[j])) >= config.FLOOR_HEIGHT_M:
                    continue
                if float(np.linalg.norm(xz[i] - xz[j])) < self.congestion_radius:
                    congested = True
                    break
            if congested:
                break
        if congested:
            self._congested_steps += 1
        return congested

    # -- false positives -----------------------------------------------------
    def false_positive(self, n: int = 1) -> None:
        """An agent stopped at / verified a wrong (e.g. distractor) instance."""
        self._false_positives += int(n)

    # -- readout -------------------------------------------------------------
    def summary(self) -> Dict[str, Optional[float]]:
        """Fold into the per-episode ``result`` dict for :mod:`benchmark.metrics`."""
        congestion = (self._congested_steps / self._steps) if self._steps else 0.0
        msg_counts = list(self._per_agent_msgs.values())
        return {
            "comm_volume": self._messages,
            "comm_bytes": self._bytes,
            "congestion": round(float(congestion), 4),
            "false_positives": self._false_positives,
            "comm_gini": round(_gini(msg_counts), 4) if msg_counts else 0.0,
        }


def _gini(values: Sequence[float]) -> float:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0 or x.sum() == 0:
        return 0.0
    x = np.sort(np.clip(x, 0, None))
    n = x.size
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum() - (n + 1) * x.sum()) / (n * x.sum()))


# The responsibilities, exposed as data so a runner / test can assert coverage.
SHELL_RESPONSIBILITIES: List[str] = [
    "task_allocation",
    "communication",
    "shared_mapping",
    "multi_goal_termination",
    "collision_congestion",
]
