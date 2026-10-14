"""Load HM3D-v2 ObjectNav source goals as the raw material for goal-sets.

The shipped dataset stores, per scene content file
(``data/datasets/objectnav_hm3d_v2/<split>/content/<scene>.json.gz``), a
``goals_by_category`` map: ``"<scene>.basis.glb_<category>" -> [goal, ...]``.
Each goal carries a 3-D ``position``, an ``object_id``/``object_name`` and a list
of ``view_points`` (each ``{agent_state:{position,rotation}, iou}``).

We reuse these real object instances (DM3 reuses HM3D too; our value-add is the
generation/stratification layer on top, not new object annotations) and group
them by scene so the generator can compose multi-object goal-sets of *distinct*
categories.
"""
from __future__ import annotations

import glob
import gzip
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class GoalInstance:
    """One object instance usable as a navigation target."""

    category: str
    object_id: object  # int (HM3D-v2) or str like "freezer_2" (OVON)
    object_name: str
    position: List[float]
    view_points: List[dict] = field(default_factory=list)  # [{position,rotation,iou}]


@dataclass
class SceneGoals:
    scene_key: str                 # e.g. "00877-4ok3usBNeis"
    scene_id: str                  # relative glb, e.g. "hm3d_v0.2/val/00877-.../*.basis.glb"
    scene_dataset_config: str      # value from the source episodes
    navmesh_path: str              # absolute/relative path to the .navmesh
    by_category: Dict[str, List[GoalInstance]]

    @property
    def categories(self) -> List[str]:
        return sorted(self.by_category.keys())


def _coerce_id(v):
    """HM3D-v2 stores int object_id; OVON stores str (e.g. ``freezer_2``)."""
    try:
        return int(v)
    except (ValueError, TypeError):
        return v


def _category_of(key: str) -> str:
    # keys look like "<scene>.basis.glb_<category>"; category may contain '_'.
    marker = ".glb_"
    idx = key.find(marker)
    return key[idx + len(marker):] if idx >= 0 else key


def _navmesh_for(scenes_dir: str, scene_id: str) -> str:
    """Resolve the ``.basis.navmesh`` for a scene, tolerant of dataset-version
    path skew. HM3D-v2 ObjectNav stores ``hm3d_v0.2/val/<scene>/...`` while OVON
    stores ``hm3d/val//<scene>/...`` (note the doubled slash + ``hm3d`` vs
    ``hm3d_v0.2``); both point at the same HM3D-1.0 scene tree on disk.
    """
    rel = scene_id
    if rel.endswith(".glb"):
        rel = rel[: -len(".glb")] + ".navmesh"
    rel = rel.replace("//", "/")
    candidates = [
        os.path.join(scenes_dir, rel),
        os.path.join(scenes_dir, rel.replace("hm3d/val/", "hm3d_v0.2/val/")),
        os.path.join(scenes_dir, rel.replace("hm3d/", "hm3d_v0.2/")),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # Last resort: glob by the scene stem (e.g. "4ok3usBNeis.basis.navmesh").
    stem = os.path.basename(rel)
    hits = glob.glob(os.path.join(scenes_dir, "**", stem), recursive=True)
    return hits[0] if hits else candidates[0]


def load_scene_goals(content_file: str, scenes_dir: str) -> Optional[SceneGoals]:
    """Parse one ``content/<scene>.json.gz`` into a :class:`SceneGoals`.

    Returns ``None`` if the file has no usable goals (e.g. empty categories).
    """
    with gzip.open(content_file, "rt") as f:
        data = json.load(f)

    gbc = data.get("goals_by_category", {})
    if not gbc:
        return None

    eps = data.get("episodes", [])
    if not eps:
        return None
    ep0 = eps[0]
    scene_id = ep0["scene_id"]
    scene_dataset_config = ep0.get("scene_dataset_config", "")
    # scene_key like "00877-4ok3usBNeis" from ".../00877-4ok3usBNeis/*.basis.glb"
    scene_key = scene_id.split("/")[-2] if "/" in scene_id else os.path.basename(scene_id).split(".")[0]

    by_category: Dict[str, List[GoalInstance]] = {}
    for key, goals in gbc.items():
        cat = _category_of(key)
        insts: List[GoalInstance] = []
        for g in goals:
            vps = []
            for vp in g.get("view_points", []) or []:
                st = vp.get("agent_state", {})
                vps.append(
                    {
                        "position": st.get("position"),
                        "rotation": st.get("rotation"),
                        "iou": float(vp.get("iou", 0.0)),
                    }
                )
            insts.append(
                GoalInstance(
                    category=cat,
                    object_id=_coerce_id(g.get("object_id", -1)),
                    object_name=g.get("object_name", ""),
                    position=list(g["position"]),
                    view_points=vps,
                )
            )
        if insts:
            by_category[cat] = insts

    if not by_category:
        return None

    return SceneGoals(
        scene_key=scene_key,
        scene_id=scene_id,
        scene_dataset_config=scene_dataset_config,
        navmesh_path=_navmesh_for(scenes_dir, scene_id),
        by_category=by_category,
    )


def iter_scene_files(content_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(content_dir, "*.json.gz")))
    # Skip macOS AppleDouble sidecars (._*.json.gz) left by some tar extracts.
    return [f for f in files if not os.path.basename(f).startswith("._")]
