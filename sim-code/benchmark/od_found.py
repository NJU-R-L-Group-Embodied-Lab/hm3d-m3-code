"""Detector FOUND gate for the adapted (non-LaViRA) baselines.

Same STOP rule as `benchmark.ovon_od.OvonODSkill`:
OWL-ViT ≥0.4, TN CLIP on the box crop ≥0.24, IIN SuperGlue ≥24.5,
then stop when the agent is within the sensor-estimated standoff of that
box. No Habitat geodesic, no reach_points, no GT viewpoint.

Navigation stays with the calling method. This module only decides found.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from benchmark.ovon_od import (
    OvonODSkill,
    OwlClient,
    goal_spec,
    rho_theta_vlfm,
    sticky_stop_radius,
)

PROTOCOL_NAME = "ovon_od_detector_stop"


def _pos_xz(pos) -> Tuple[float, float]:
    p = np.asarray(pos, dtype=np.float32).reshape(-1)
    return float(p[0]), float(p[2])


def harvest_od_hits(
    skill_row: Sequence[OvonODSkill],
    resolved: Sequence[bool],
    pos_xz: Tuple[float, float],
    yaw: float,
) -> List[int]:
    """Goal indices that already satisfy OD+standoff. Does not run OWL."""
    hits: List[int] = []
    for j, skill in enumerate(skill_row):
        if j < len(resolved) and resolved[j]:
            continue
        if stop_ok(skill, pos_xz, yaw):
            hits.append(int(j))
    return hits


def stop_ok(skill: OvonODSkill, pos_xz: Tuple[float, float], yaw: float) -> bool:
    """Polar stop: the agent is within the sensor-estimated standoff of the box."""
    if not skill.detected or skill.goal_xz is None:
        return False
    rho, _ = rho_theta_vlfm(pos_xz, yaw, skill.goal_xz)
    stop_rho = float(rho)
    if skill.box_xz is not None:
        box_rho, _ = rho_theta_vlfm(pos_xz, yaw, skill.box_xz)
        stop_rho = min(stop_rho, float(box_rho))
    return stop_rho <= float(sticky_stop_radius(skill.best_match))


class ODFoundGate:
    """Per-episode, per-agent, per-goal OD skills sharing one OWL client."""

    def __init__(
        self,
        n_agents: int,
        goals: Sequence[dict],
        iin_lookup: Optional[Dict] = None,
        episode_id: str = "",
        owl_url: Optional[str] = None,
        owl_every: int = 1,
        hfov_deg: float = 79.0,
        dmin: float = 0.1,
        dmax: float = 5.0,
    ):
        url = (owl_url if owl_url is not None else os.environ.get("OWL_URL", "")).strip()
        if not url:
            raise RuntimeError("OWL_URL is required for ovon_od_detector_stop found")
        shm = (os.environ.get("OWL_SHM_DIR") or "").strip()
        self.client = OwlClient(url, shm_dir=shm, shm_name="od_%d" % os.getpid())
        self.goals = list(goals)
        self.iin_lookup = iin_lookup or {}
        self.episode_id = str(episode_id)
        self.owl_every = max(1, int(owl_every or os.environ.get("OWL_EVERY", "1")))
        self.hfov_deg = float(hfov_deg)
        self.dmin = float(dmin)
        self.dmax = float(dmax)
        self.skills: List[List[OvonODSkill]] = []
        for i in range(int(n_agents)):
            row: List[OvonODSkill] = []
            for _j in self.goals:
                skill = OvonODSkill(owl_url=None)
                skill.client = self.client
                row.append(skill)
            self.skills.append(row)

    @classmethod
    def from_runner(cls, runner: Any, n_agents: int, goals: Sequence[dict], episode_id: str):
        cfg = getattr(runner, "cfg", None)
        hfov, dmin, dmax = 79.0, 0.1, 5.0
        if cfg is not None:
            try:
                hfov = float(cfg.SIMULATOR.RGB_SENSOR.HFOV)
                dmin = float(cfg.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH)
                dmax = float(cfg.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH)
            except Exception:
                pass
        return cls(
            n_agents=n_agents,
            goals=goals,
            iin_lookup=getattr(runner, "iin_lookup", {}) or {},
            episode_id=episode_id,
            owl_url=os.environ.get("OWL_URL", ""),
            owl_every=int(os.environ.get("OWL_EVERY", "2")),
            hfov_deg=hfov,
            dmin=dmin,
            dmax=dmax,
        )

    def _spec(self, goal: dict) -> dict:
        path = self.iin_lookup.get((self.episode_id, int(goal.get("goal_index", -1))))
        return goal_spec(goal, path)

    def poll(
        self,
        agent_id: int,
        rgb,
        depth,
        pos,
        yaw: float,
        found: Sequence[bool],
        step: int,
    ) -> List[int]:
        if rgb is None or depth is None:
            return []
        i = int(agent_id)
        if i < 0 or i >= len(self.skills):
            return []
        pos_xz = _pos_xz(pos)
        hits: List[int] = []
        for j, skill in enumerate(self.skills[i]):
            if found[j]:
                continue
            if skill.detected and stop_ok(skill, pos_xz, yaw):
                hits.append(j)
        if int(step) % self.owl_every != 0:
            return hits
        for j, skill in enumerate(self.skills[i]):
            if found[j] or j in hits:
                continue
            skill.update(
                rgb, depth, self._spec(self.goals[j]), pos_xz, float(yaw),
                hfov_deg=self.hfov_deg, dmin=self.dmin, dmax=self.dmax,
            )
            if skill.detected and stop_ok(skill, pos_xz, float(yaw)):
                hits.append(j)
        return hits

    def approach_xz(self, agent_id: int, found: Sequence[bool]):
        """Sensor-estimated box to walk toward. Not a GT reach point."""
        i = int(agent_id)
        if i < 0 or i >= len(self.skills):
            return None
        for j, skill in enumerate(self.skills[i]):
            if found[j] or skill.goal_xz is None:
                continue
            if skill.detected or skill.approaching:
                return skill.goal_xz
        return None

    def summary(self) -> dict:
        c = self.client
        n_det = 0
        n_app = 0
        for row in self.skills:
            for skill in row:
                n_det += int(bool(skill.detected))
                n_app += int(bool(skill.approaching))
        return {
            "owl_url": c.url,
            "shm_dir": c.shm_dir,
            "n_detect": int(getattr(c, "n_detect", 0)),
            "n_box": int(getattr(c, "n_box", 0)),
            "n_err": int(getattr(c, "n_err", 0)),
            "n_shm": int(getattr(c, "n_shm", 0)),
            "n_jpeg": int(getattr(c, "n_jpeg", 0)),
            "skills_detected": n_det,
            "skills_approaching": n_app,
        }


def claim_from_obs(
    gate: ODFoundGate,
    sim,
    agent_id: int,
    obs: dict,
    step: int,
    found: List[bool],
    found_by: List[int],
    locked: Optional[List] = None,
) -> List[int]:
    from vlnce_baselines.multi.ma_decision import yaw_from_quat

    st = sim.get_agent_state(agent_id)
    return apply_od_found(
        gate, agent_id,
        None if obs is None else obs.get("rgb"),
        None if obs is None else obs.get("depth"),
        st.position, float(yaw_from_quat(st.rotation)), step,
        found, found_by, locked,
    )


def attach_od_metrics(tag: str, gate: Optional[ODFoundGate], metrics_dict: dict, result: dict):
    if gate is None:
        return
    summary = gate.summary()
    print(f"[{tag}][od-found] {summary}")
    metrics_dict["protocol"] = PROTOCOL_NAME
    metrics_dict["od_found"] = summary
    result["protocol"] = PROTOCOL_NAME
    result["od_found"] = summary


def sensor_approach_target(gate: ODFoundGate, agent_id: int, found: Sequence[bool], y: float):
    xz = gate.approach_xz(agent_id, found)
    if xz is None:
        return None
    return np.array([float(xz[0]), float(y), float(xz[1])], np.float32)


def apply_od_found(
    gate: ODFoundGate,
    agent_id: int,
    rgb,
    depth,
    pos,
    yaw: float,
    step: int,
    found: List[bool],
    found_by: List[int],
    locked: Optional[List] = None,
) -> List[int]:
    newly: List[int] = []
    for j in gate.poll(agent_id, rgb, depth, pos, yaw, found, step):
        if found[j]:
            continue
        found[j] = True
        found_by[j] = int(agent_id)
        if locked is not None and agent_id < len(locked) and locked[agent_id] == j:
            locked[agent_id] = None
        newly.append(j)
    return newly
