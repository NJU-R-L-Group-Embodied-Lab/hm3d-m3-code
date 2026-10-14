"""MP3D stage frame vs Habitat camera.

Matterport glbs are Z-up / Y-forward. Habitat is Y-up / −Z-forward. Official
Habitat MP3D defaults are ``up=[0,0,1]``, ``front=[0,1,0]``. An incomplete
``*.stage_config.json`` (semantic/navmesh only) overrides those defaults and
leaves the mesh Z-up while the agent walks a Y-up navmesh — RGB looks rotated
90° (ceilings / side walls) and depth stamps miss the floors.

Fix the frame in ``data/qual_scenes/mp3d/<id>/<id>.stage_config.json``. Do not
``flipud`` MP3D RGB: after the stage remap the framebuffer is already CV-up,
same as HM3D. ``HabitatSimRGBSensor`` still does not flip; it does not need to.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def scene_needs_rgb_flip(scene_id: str) -> bool:
    """Kept for call sites. Always false once the MP3D stage frame is set."""
    del scene_id
    return False


def flip_visual_obs(obs: Any) -> Any:
    if obs is None:
        return obs
    if isinstance(obs, (list, tuple)):
        return [flip_visual_obs(x) for x in obs]
    if not isinstance(obs, dict):
        return obs
    out = dict(obs)
    for key in ("rgb", "depth"):
        val = out.get(key)
        if val is None:
            continue
        arr = np.asarray(val)
        if arr.ndim >= 2:
            out[key] = np.ascontiguousarray(np.flipud(arr))
    return out


def wrap_sim_visual_obs(sim: Any, scene_id: str) -> Any:
    """No-op. Stage ``up``/``front`` is the MP3D correction, not a pixel flip."""
    del scene_id
    return sim
