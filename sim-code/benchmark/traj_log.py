"""Shared per-step trajectory logging for the multi-agent runners.

Uniform schema so one visualizer renders every method on the same episode:
  {episode_id, scene_id, n_agents, starts_xz,
   goals[{idx,category,modality,xz,object_xz,rep_xz,reach_xz,found_by}],
   found_events[[step,agent,goal,category,modality,wx,wz]], steps,
   traj_xz[[ [x,z], ... ]]}
"""
from __future__ import annotations

import json


def _xz(point):
    return [float(point[0]), float(point[2])]


def _goal_object_xz(goal):
    point = goal.get("rep_point") or goal.get("position")
    if point is None:
        point = goal["reach_points"][0]
    return _xz(point)


def goal_xz(goals):
    return [_goal_object_xz(g) for g in goals]


def init_traj(starts, n):
    """Return (traj, found_events) seeded with each agent's start (x,z)."""
    traj = [[[float(starts[i][0]), float(starts[i][2])]] for i in range(n)]
    return traj, []


def dump_traj(save_dir, ep, n, starts, goals, found_by, found_events, steps, traj, gxz):
    td = {"episode_id": str(ep.episode_id), "scene_id": ep.scene_id, "n_agents": n,
          "starts_xz": [[float(s[0]), float(s[2])] for s in starts],
          "goals": [{"idx": j, "category": goals[j]["object_category"],
                     "modality": goals[j]["modality"], "xz": gxz[j],
                     "object_xz": _goal_object_xz(goals[j]),
                     "rep_xz": _xz(goals[j].get("rep_point") or goals[j].get("position") or goals[j]["reach_points"][0]),
                     "reach_xz": [_xz(p) for p in goals[j].get("reach_points", [])],
                     "found_by": found_by[j]} for j in range(len(goals))],
          "found_events": found_events, "steps": steps, "traj_xz": traj}
    with open(save_dir / f"traj_{ep.episode_id}.json", "w") as fh:
        json.dump(td, fh)
