"""P2a: render IIN goal images (Habitat Simulator).

Default / locobot / stretch: teleport to the stored ``iin_viewpoint``.
``--camera-profile official`` follows InstanceImageNav (Krantz et al. /
GOAT OVIIN): issuer camera independent of the eval robot — 512x512,
r in {0.5,1,1.5,2} m every 10 deg, h ~ U(0.8,1.5), HFoV ~ U(60,120),
look-at ±5 deg, keep if c_o > 0.7 and c_f > 0.0232*OSA+0.02.

This is the only P2 step that needs a Simulator + GPU/EGL; P1 geometry and
:mod:`benchmark.iin_verify` do not.

    python -m benchmark.render_iin \
        --episodes data/datasets/ma_objectnav/val/ma_stratified_100_seed42.json \
        --out-dir  data/datasets/ma_objectnav/val/iin_images

With ``--verify`` it also renders, per goal, a *positive* view (small pose offset
= same instance) and a *negative* view (another instance of the same category)
and runs :mod:`benchmark.iin_verify`, reporting same/different separation on real
renders — the P2 render→verify integration check.

REQUIRES a render-capable environment (working EGL/GPU). If EGL context creation
fails ("unable to find EGL device for CUDA device 0"), run on a GPU/EGL-healthy
image (see ``sim-code/docker``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from . import config as _cfg
from .config import MAX_VIEWPOINTS_PER_GOAL
from .goal_sources import GoalInstance, SceneGoals

# --------------------------------------------------------------------------
# HM3D semantic-annotation goal instances
#
# Build SceneGoals from a Habitat semantic mesh (MP3D) or a raw glb.
#
# Used when a house is not in the OVON / ObjectNav-HM3D content dump. Object
# centroids come from ``sim.semantic_scene``; viewpoints are a navigable ring
# around the snap, so ``generate.try_make_episode`` can run unchanged.
# --------------------------------------------------------------------------
# Structural / stuff classes that are not ON/IIN/TN targets.
STRUCTURAL_CATS = {
    "void", "wall", "floor", "ceiling", "door", "window", "curtain",
    "stairs", "column", "beam", "railing", "blinds", "board_panel",
    "lighting", "objects", "misc", "shower", "towel", "clothes",
    "unknown", "unlabeled", "none",
}

MIN_AABB_M = 0.12
MAX_AABB_M = 8.0


def normalize_cat(name: str) -> str:
    return str(name or "").strip().lower().replace(" ", "_").replace("-", "_")


def is_goal_category(name: str) -> bool:
    cat = normalize_cat(name)
    if not cat or cat in STRUCTURAL_CATS:
        return False
    if cat.startswith("wall") or cat.startswith("ceil"):
        return False
    return True


def scene_dataset_config_for(scene_glb: str, scenes_dir: str = "") -> str:
    """HM3D val annotated config, or empty so Habitat loads a raw glb."""
    sid = (scene_glb or "").replace("\\", "/")
    use_annotated = ("/val/" in sid or "/minival/" in sid) and "/mp3d/" not in sid
    if not use_annotated:
        return ""
    if scenes_dir:
        return os.path.join(scenes_dir, "hm3d_v0.2", "hm3d_annotated_basis.scene_dataset_config.json")
    return "data/scene_datasets/hm3d-1.0/hm3d_v0.2/hm3d_annotated_basis.scene_dataset_config.json"


def yaw_look_at(src: Sequence[float], dst: Sequence[float]) -> float:
    """Habitat yaw so forward (−Z) points src → dst."""
    d = np.asarray(dst, dtype=np.float64) - np.asarray(src, dtype=np.float64)
    return math.atan2(-float(d[0]), -float(d[2]))


def quat_xyzw_from_yaw(yaw: float) -> List[float]:
    return [0.0, math.sin(yaw / 2.0), 0.0, math.cos(yaw / 2.0)]


def ring_offsets(
    radii_m: Sequence[float] = (0.9, 1.4, 1.9),
    n_angles: int = 12,
) -> List[Tuple[float, float]]:
    """(dx, dz) offsets on concentric rings in the XZ plane."""
    out: List[Tuple[float, float]] = []
    for r in radii_m:
        for k in range(int(n_angles)):
            ang = 2.0 * math.pi * k / float(n_angles)
            out.append((float(r) * math.sin(ang), float(r) * math.cos(ang)))
    return out


def propose_ring_points(center: Sequence[float], floor_y: Optional[float] = None) -> List[List[float]]:
    cy = float(center[1] if floor_y is None else floor_y)
    pts = []
    for dx, dz in ring_offsets():
        pts.append([float(center[0]) + dx, cy, float(center[2]) + dz])
    return pts


def viewpoints_from_ring(nav, center: Sequence[float], floor_y: Optional[float] = None) -> List[dict]:
    """Snap ring samples and keep those that can see the object on the navmesh."""
    vps: List[dict] = []
    for p in propose_ring_points(center, floor_y=floor_y):
        snapped = nav.snap(p)
        if snapped is None or not nav.is_navigable(snapped):
            continue
        snapped_c = nav.snap(center)
        if snapped_c is None or not math.isfinite(nav.geodesic(snapped, snapped_c)):
            continue
        yaw = yaw_look_at(snapped, center)
        vps.append({
            "position": [float(x) for x in snapped],
            "rotation": quat_xyzw_from_yaw(yaw),
            "iou": 0.5,
        })
        if len(vps) >= MAX_VIEWPOINTS_PER_GOAL:
            break
    return vps


def _vec3(v) -> List[float]:
    if callable(v):
        v = v()
    if hasattr(v, "x"):
        return [float(v.x), float(v.y), float(v.z)]
    vals = [float(x) for x in np.asarray(v).reshape(-1)]
    if len(vals) < 3:
        raise ValueError("need xyz")
    return vals[:3]


def instances_from_records(records: Iterable[dict]) -> Dict[str, List[GoalInstance]]:
    """Group already-normalized object dicts into SceneGoals.by_category."""
    by_cat: Dict[str, List[GoalInstance]] = {}
    for rec in records:
        cat = normalize_cat(rec.get("category") or "")
        if not is_goal_category(cat):
            continue
        pos = [float(x) for x in rec["position"]]
        inst = GoalInstance(
            category=cat,
            object_id=rec.get("object_id", cat),
            object_name=str(rec.get("object_name") or cat),
            position=pos,
            view_points=list(rec.get("view_points") or []),
        )
        by_cat.setdefault(cat, []).append(inst)
    return by_cat


def records_from_semantic_scene(sim) -> List[dict]:
    """Read Habitat ``semantic_scene.objects`` into plain dicts (no navmesh)."""
    scene = getattr(sim, "semantic_scene", None)
    objects = list(getattr(scene, "objects", None) or [])
    out: List[dict] = []
    for obj in objects:
        if obj is None:
            continue
        cat_obj = getattr(obj, "category", None)
        cat = ""
        if cat_obj is not None:
            cat = cat_obj.name() if callable(getattr(cat_obj, "name", None)) else str(cat_obj)
        cat = normalize_cat(cat)
        if not is_goal_category(cat):
            continue
        aabb = getattr(obj, "aabb", None)
        if aabb is None:
            continue
        center = _vec3(getattr(aabb, "center", None))
        sizes = _vec3(getattr(aabb, "sizes", None))
        extent = max(sizes)
        if extent < MIN_AABB_M or extent > MAX_AABB_M:
            continue
        oid = getattr(obj, "semantic_id", None)
        if oid is None:
            oid = getattr(obj, "object_id", None)
        if oid is None:
            oid = getattr(obj, "id", cat)
        out.append({
            "category": cat,
            "object_id": oid,
            "object_name": cat,
            "position": center,
            "sizes": sizes,
        })
    return out


def attach_viewpoints(nav, records: Sequence[dict]) -> List[dict]:
    filled = []
    for rec in records:
        pos = rec["position"]
        snapped = nav.snap(pos)
        floor_y = float(snapped[1]) if snapped is not None else None
        vps = viewpoints_from_ring(nav, pos, floor_y=floor_y)
        if not vps:
            continue
        item = dict(rec)
        item["view_points"] = vps
        filled.append(item)
    return filled


def scene_key_from_glb(scene_glb: str) -> str:
    parent = os.path.basename(os.path.dirname(os.path.abspath(scene_glb)))
    stem = os.path.basename(scene_glb).split(".")[0]
    if parent and (parent.startswith("00") or parent == stem or "-" in parent):
        return parent
    return stem


def make_scene_goals(scene_glb: str, nav, records: Sequence[dict]) -> Optional[SceneGoals]:
    recs = attach_viewpoints(nav, records)
    by_cat = instances_from_records(recs)
    if len(by_cat) < 3:
        return None
    navmesh = scene_glb[: -len(".glb")] + ".navmesh" if scene_glb.endswith(".glb") else scene_glb
    return SceneGoals(
        scene_key=scene_key_from_glb(scene_glb),
        scene_id=scene_glb,
        scene_dataset_config=scene_dataset_config_for(scene_glb),
        navmesh_path=navmesh if os.path.exists(navmesh) else getattr(nav, "navmesh_path", navmesh),
        by_category=by_cat,
    )


def open_semantic_sim(scene_glb: str, gpu_device_id: int = 0):
    """Minimal habitat-sim with semantic mesh, no Habitat-lab task config."""
    import habitat_sim as hs

    backend = hs.SimulatorConfiguration()
    backend.scene_id = os.path.abspath(scene_glb)
    sds = scene_dataset_config_for(scene_glb)
    if sds:
        backend.scene_dataset_config_file = sds
    backend.enable_physics = False
    backend.gpu_device_id = int(gpu_device_id)
    if hasattr(backend, "load_semantic_mesh"):
        backend.load_semantic_mesh = True
    if hasattr(backend, "force_separate_semantic_scene_graph"):
        backend.force_separate_semantic_scene_graph = True
    spec = hs.SensorSpec()
    spec.uuid = "rgb"
    spec.sensor_type = hs.SensorType.COLOR
    spec.resolution = [64, 64]
    spec.position = [0.0, 0.88, 0.0]
    agent = hs.agent.AgentConfiguration()
    agent.sensor_specifications = [spec]
    return hs.Simulator(hs.Configuration(backend, [agent]))


def load_semantic_scene_goals(scene_glb: str, nav, gpu_device_id: int = 0) -> Optional[SceneGoals]:
    sim = open_semantic_sim(scene_glb, gpu_device_id=gpu_device_id)
    try:
        records = records_from_semantic_scene(sim)
    finally:
        sim.close()
    return make_scene_goals(scene_glb, nav, records)


# --------------------------------------------------------------------------
# GOAT-Bench goal camera sampling
#
# GOAT-Bench / Krantz viewpoint grids for IIN and LanguageNav (TN).
#
# Habitat-free: polar camera candidates, coverage gates, reproducible per-goal RNG.
# IIN matches Krantz 2211.15876 / GOAT OVIIN. TN matches GOAT Appendix B
# (Stretch 512x512, r in {1.0, 1.5, 2.0}, 10 deg, keep max frame coverage).
# --------------------------------------------------------------------------
def goal_rng(goal: dict) -> np.random.Generator:
    key = str(goal.get("object_id") or goal.get("position") or goal.get("object_category") or "x")
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return np.random.default_rng(int(digest[:8], 16))


def _n_theta(step_deg: float) -> int:
    return int(round(360.0 / float(step_deg)))


def iter_polar_xz(center, radii: Sequence[float], step_deg: float) -> Iterator[Tuple[float, float, float, float]]:
    """Yield (x, z, radius, theta_deg) on the horizontal ring around center."""
    c = np.asarray(center, dtype=np.float64)
    n_theta = _n_theta(step_deg)
    for radius in radii:
        r = float(radius)
        for j in range(n_theta):
            theta_deg = float(j * step_deg)
            theta = math.radians(theta_deg)
            yield (
                float(c[0] + r * math.cos(theta)),
                float(c[2] + r * math.sin(theta)),
                r,
                theta_deg,
            )


def iter_iin_cameras(center, floor_y: float, rng: np.random.Generator) -> Iterator[dict]:
    """Krantz / GOAT OVIIN issuer cameras: 512 later, h and HFoV sampled per pose."""
    lo_h, hi_h = _cfg.IIN_OFFICIAL_HEIGHT_RANGE_M
    lo_f, hi_f = _cfg.IIN_OFFICIAL_HFOV_RANGE_DEG
    jitter = float(_cfg.IIN_OFFICIAL_LOOKAT_JITTER_DEG)
    for x, z, radius, theta_deg in iter_polar_xz(
        center, _cfg.IIN_OFFICIAL_RADII_M, _cfg.IIN_OFFICIAL_THETA_STEP_DEG
    ):
        yield {
            "position": np.array(
                [x, float(floor_y) + float(rng.uniform(lo_h, hi_h)), z],
                dtype=np.float64,
            ),
            "hfov": float(rng.uniform(lo_f, hi_f)),
            "radius_m": radius,
            "theta_deg": theta_deg,
            "yaw_jitter_deg": float(rng.uniform(-jitter, jitter)),
            "pitch_jitter_deg": float(rng.uniform(-jitter, jitter)),
            "kind": "iin",
        }


def iter_tn_cameras(center, floor_y: float) -> Iterator[dict]:
    """GOAT LanguageNav: Stretch height 1.31 m, r in {1, 1.5, 2}, every 10 deg."""
    height = float(_cfg.TN_CAMERA_GOAT["height_m"])
    hfov = float(_cfg.TN_CAMERA_GOAT["hfov"])
    for x, z, radius, theta_deg in iter_polar_xz(
        center, _cfg.TN_GOAT_RADII_M, _cfg.TN_GOAT_THETA_STEP_DEG
    ):
        yield {
            "position": np.array([x, float(floor_y) + height, z], dtype=np.float64),
            "hfov": hfov,
            "radius_m": radius,
            "theta_deg": theta_deg,
            "yaw_jitter_deg": 0.0,
            "pitch_jitter_deg": 0.0,
            "kind": "tn",
        }


def count_iin_grid() -> int:
    return len(_cfg.IIN_OFFICIAL_RADII_M) * _n_theta(_cfg.IIN_OFFICIAL_THETA_STEP_DEG)


def count_tn_grid() -> int:
    return len(_cfg.TN_GOAT_RADII_M) * _n_theta(_cfg.TN_GOAT_THETA_STEP_DEG)


def frame_coverage(sem: np.ndarray, semantic_id: int) -> float:
    if semantic_id < 0:
        return 0.0
    flat = np.asarray(sem).reshape(-1)
    if flat.size == 0:
        return 0.0
    return float((flat == int(semantic_id)).mean())


def cf_threshold(osa: float) -> float:
    return float(_cfg.IIN_OFFICIAL_CF_SLOPE) * float(osa) + float(_cfg.IIN_OFFICIAL_CF_INTERCEPT)


def _hull_area(points: np.ndarray) -> Optional[float]:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] < 4:
        return None
    try:
        from scipy.spatial import ConvexHull

        return float(ConvexHull(pts).area)
    except Exception:  # noqa: BLE001
        return None


def object_coverage(view_pts: np.ndarray, all_pts: np.ndarray) -> Optional[float]:
    """c_o = view convex-hull area / all-candidate hull area (Krantz)."""
    osa = _hull_area(all_pts)
    if osa is None or osa <= 1e-8:
        if all_pts is None or len(all_pts) == 0:
            return None
        return float(len(view_pts)) / float(len(all_pts))
    view_a = _hull_area(view_pts)
    if view_a is None:
        return float(len(view_pts)) / float(max(len(all_pts), 1))
    return float(min(1.0, view_a / osa))


def observable_surface_area(all_pts: np.ndarray) -> Optional[float]:
    return _hull_area(all_pts)


def iin_gates(c_o: Optional[float], c_f: Optional[float], osa: Optional[float]) -> bool:
    if c_f is None or c_f <= 0.0:
        return False
    if c_o is None:
        return c_f >= float(_cfg.IIN_OFFICIAL_MIN_FRAME_COVERAGE)
    if c_o <= float(_cfg.IIN_OFFICIAL_CO_MIN):
        return False
    thresh = cf_threshold(0.0 if osa is None else osa)
    return c_f > thresh


def backproject_mask(
    depth: np.ndarray,
    sem: np.ndarray,
    semantic_id: int,
    cam_pos,
    rot_xyzw,
    camera: dict,
    stride: int = 4,
) -> np.ndarray:
    """Habitat RGBD: +X right, +Y up, -Z forward. Returns (N, 3) world points."""
    if semantic_id < 0:
        return np.zeros((0, 3), dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float32)
    sem = np.asarray(sem)
    if depth.ndim > 2:
        depth = depth.reshape(depth.shape[0], depth.shape[1])
    h, w = depth.shape[:2]
    sem = sem.reshape(h, w)
    ys, xs = np.where(sem == int(semantic_id))
    if xs.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    xs = xs[:: max(1, int(stride))]
    ys = ys[:: max(1, int(stride))]
    z = depth[ys, xs]
    ok = np.isfinite(z) & (z > 0.12) & (z < 8.0)
    xs, ys, z = xs[ok], ys[ok], z[ok]
    if xs.size == 0:
        return np.zeros((0, 3), dtype=np.float64)
    fx = (0.5 * w) / math.tan(math.radians(float(camera["hfov"])) * 0.5)
    fy = fx
    cx, cy = 0.5 * w, 0.5 * h
    x = (xs.astype(np.float64) - cx) * z / fx
    y = (cy - ys.astype(np.float64)) * z / fy
    local = np.stack([x, y, -z.astype(np.float64)], axis=1)
    R = _quat_xyzw_to_R(rot_xyzw)
    return (R @ local.T).T + np.asarray(cam_pos, dtype=np.float64)




def nearby_objects(
    sem: np.ndarray,
    depth: np.ndarray,
    id_to_name: Dict[int, str],
    target_sid: int,
    min_coverage: float = None,
    max_depth_m: float = None,
) -> List[dict]:
    """Visible non-target instances with coverage and mean depth (GOAT nearby)."""
    min_coverage = float(
        _cfg.TN_GOAT_NEARBY_MIN_COVERAGE if min_coverage is None else min_coverage
    )
    max_depth_m = float(
        _cfg.TN_GOAT_NEARBY_MAX_DEPTH_M if max_depth_m is None else max_depth_m
    )
    sem = np.asarray(sem)
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim > 2:
        depth = depth.reshape(depth.shape[0], depth.shape[1])
    h, w = depth.shape[:2]
    sem = sem.reshape(h, w)
    total = float(sem.size)
    out = []
    for sid in np.unique(sem):
        sid_i = int(sid)
        mask = sem == sid_i
        cov = float(mask.mean())
        if cov < min_coverage:
            continue
        md = float(np.mean(depth[mask]))
        if not np.isfinite(md):
            continue
        ys, xs = np.where(mask)
        rec = {
            "semantic_id": sid_i,
            "category": id_to_name.get(sid_i, "unknown"),
            "coverage": cov,
            "mean_depth_m": md,
            "bbox": [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1],
            "is_target": sid_i == int(target_sid),
        }
        if rec["is_target"] or md <= max_depth_m:
            out.append(rec)
    out.sort(key=lambda r: (-r["is_target"], -r["coverage"]))
    return out


def pick_iin_index(records: Sequence[dict]) -> int:
    """Among coverage-valid views pick highest c_f; else highest c_f overall."""
    if not records:
        raise ValueError("empty IIN candidate list")
    valid = [i for i, r in enumerate(records) if r.get("ok")]
    pool = valid or list(range(len(records)))
    return max(
        pool,
        key=lambda i: (
            float(records[i].get("frame_coverage") or 0.0),
            float(records[i].get("object_coverage") or 0.0),
        ),
    )


def pick_tn_index(records: Sequence[dict]) -> int:
    if not records:
        raise ValueError("empty TN candidate list")
    return max(range(len(records)), key=lambda i: float(records[i].get("frame_coverage") or 0.0))


def _yaw_look_at(src: np.ndarray, dst: np.ndarray) -> float:
    """Yaw around +Y so Habitat forward (-Z) points src -> dst."""
    d = np.asarray(dst, dtype=np.float64) - np.asarray(src, dtype=np.float64)
    return math.atan2(-float(d[0]), -float(d[2]))


def _quat_xyzw_from_yaw(yaw: float) -> List[float]:
    return [0.0, math.sin(yaw / 2.0), 0.0, math.cos(yaw / 2.0)]


def _sensor_pos(agent_pos, height_m: float) -> np.ndarray:
    p = np.asarray(agent_pos, dtype=np.float64).copy()
    p[1] = float(p[1]) + float(height_m)
    return p


def _project_object(sensor_pos, yaw: float, target, camera: dict) -> Optional[Tuple[float, float, float]]:
    """Pixel (u, v) and forward depth of target in a yaw-only pinhole."""
    d = np.asarray(target, dtype=np.float64) - np.asarray(sensor_pos, dtype=np.float64)
    c, s = math.cos(yaw), math.sin(yaw)
    cam_x = c * d[0] - s * d[2]
    cam_z = -s * d[0] - c * d[2]
    if cam_z <= 0.08:
        return None
    w = int(camera["width_px"])
    h = int(camera["height_px"])
    fx = (0.5 * w) / math.tan(math.radians(float(camera["hfov"])) * 0.5)
    u = 0.5 * w + fx * cam_x / cam_z
    v = 0.5 * h - fx * d[1] / cam_z
    return float(u), float(v), float(cam_z)


def _wall_facing_stats(rgb, depth=None, object_depth_m=None) -> dict:
    """Flag frames that look like a blank / close wall (no instance).

    ``flat_rgb``: almost no texture (the mirror-into-wall case).
    ``close_plane``: depth is a near, flat surface.
    ``occluded``: median depth is much closer than the 3D object.
    """
    gray = np.asarray(rgb)[:, :, :3].astype(np.float32).mean(axis=2)
    gy, gx = np.gradient(gray)
    edge = float(np.hypot(gx, gy).mean())
    std = float(gray.std())
    out = {
        "rgb_std": round(std, 3),
        "rgb_edge": round(edge, 4),
        "flat_rgb": bool(std < 8.0 and edge < 0.6),
        "close_plane": False,
        "occluded": False,
        "depth_med": None,
        "depth_std": None,
        "object_depth_m": None if object_depth_m is None else float(object_depth_m),
    }
    if depth is not None:
        d = np.asarray(depth, dtype=np.float32).reshape(-1)
        d = d[np.isfinite(d) & (d > 1e-4) & (d < 20.0)]
        if d.size:
            med = float(np.median(d))
            dstd = float(d.std())
            out["depth_med"] = round(med, 4)
            out["depth_std"] = round(dstd, 4)
            out["close_plane"] = bool(med < 0.70 and dstd < 0.20)
            if object_depth_m is not None and float(object_depth_m) > 0.2:
                out["occluded"] = bool(med < 0.5 * float(object_depth_m))
    out["wall_facing"] = bool(out["flat_rgb"] or out["close_plane"] or out["occluded"])
    return out


def stretch512_issue_tags(wall: dict) -> List[str]:
    """Tight failure tags for stretch512 IIN (not the loose wall_facing union).

    Matches the train-2000 audit: blank wall, camera-in-mesh, near-blank
    ceiling/glass, severe fill-frame, look-at miss, and a hard occluder.
    """
    w = wall or {}
    tags: List[str] = []
    std = float(w.get("rgb_std") or 0.0)
    edge = float(w.get("rgb_edge") or 0.0)
    dmed = w.get("depth_med")
    obj = w.get("object_depth_m")
    if w.get("flat_rgb") or (std < 8.0 and edge < 0.6):
        tags.append("flat_rgb")
    if dmed is not None and float(dmed) < 0.15:
        tags.append("inside_mesh")
    if "flat_rgb" not in tags and std < 15.0 and edge < 1.2:
        tags.append("near_blank")
    if dmed is not None and 0.15 <= float(dmed) < 0.25 and std >= 8.0:
        tags.append("too_close")
    if obj is not None and float(obj) >= 4.0:
        tags.append("extreme_far")
    if (
        dmed is not None
        and obj is not None
        and float(obj) > 0.8
        and float(dmed) < 0.22
        and float(dmed) < 0.20 * float(obj)
        and "inside_mesh" not in tags
        and "too_close" not in tags
    ):
        tags.append("hard_occlude")
    return tags


def _score_stretch512_view(wall: dict, centroid_in: bool = True) -> float:
    """Higher is better. Blank / in-mesh views must lose to a textured room."""
    w = wall or {}
    tags = stretch512_issue_tags(w)
    score = 12.0
    penalty = {
        "flat_rgb": 100.0,
        "inside_mesh": 80.0,
        "near_blank": 45.0,
        "too_close": 35.0,
        "extreme_far": 40.0,
        "hard_occlude": 25.0,
    }
    for t in tags:
        score -= penalty.get(t, 10.0)
    score += min(float(w.get("rgb_std") or 0.0), 70.0) * 0.25
    score += min(float(w.get("rgb_edge") or 0.0), 10.0) * 1.5
    dmed = w.get("depth_med")
    if dmed is not None:
        d = float(dmed)
        if 0.85 <= d <= 2.6:
            score += 18.0
        elif d < 0.35:
            score -= 20.0
    if centroid_in:
        score += 8.0
    return float(score)


def _horiz_m(a, b) -> float:
    return float(math.hypot(float(a[0]) - float(b[0]), float(a[2]) - float(b[2])))


def _standoff_agent(pathfinder, agent_pos, obj_pos, standoff_m: float, ladder=()):
    """Move the camera back along object→view, still facing the object.

    ``ladder`` extra distances are tried if the requested snap collapses
    inward (common for closets / hanging clothes). Empty ladder keeps the
    old single-snap behavior used by stretch_v2.
    """
    if standoff_m <= 0:
        return np.asarray(agent_pos, dtype=np.float32)
    a = np.asarray(agent_pos, dtype=np.float64)
    o = np.asarray(obj_pos, dtype=np.float64)
    hz = np.array([a[0] - o[0], 0.0, a[2] - o[2]], dtype=np.float64)
    dist = float(np.linalg.norm(hz))
    if dist < 1e-3:
        hz = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        dist = 1.0
    hz = hz / dist
    wants = [float(standoff_m)]
    for extra in ladder or ():
        x = float(extra)
        if x > 0 and all(abs(x - w) > 1e-6 for w in wants):
            wants.append(x)
    guard = bool(ladder)
    best = None
    best_d = _horiz_m(a, o)
    for want in wants:
        target = np.array(
            [o[0] + hz[0] * want, a[1], o[2] + hz[2] * want], dtype=np.float64
        )
        snapped = pathfinder.snap_point(target.astype(np.float32))
        if snapped is None or not np.all(np.isfinite(np.asarray(snapped))):
            continue
        if hasattr(pathfinder, "is_navigable") and not pathfinder.is_navigable(snapped):
            continue
        snapped = np.asarray(snapped, dtype=np.float32)
        got = _horiz_m(snapped, o)
        if got > best_d:
            best, best_d = snapped, got
        if guard and got < 0.75 * want:
            continue
        return snapped
    if best is not None:
        return best
    return a.astype(np.float32)


def _iter_orbit_agents(pathfinder, obj_pos, floor_y: float, radii, thetas_deg):
    """Navigable same-floor points around the object. Skips collapsed snaps."""
    o = np.asarray(obj_pos, dtype=np.float64)
    for radius in radii:
        r = float(radius)
        if r <= 0.2:
            continue
        for theta_deg in thetas_deg:
            rad = math.radians(float(theta_deg))
            target = np.array(
                [o[0] + math.sin(rad) * r, float(floor_y), o[2] + math.cos(rad) * r],
                dtype=np.float64,
            )
            snapped = pathfinder.snap_point(target.astype(np.float32))
            if snapped is None or not np.all(np.isfinite(np.asarray(snapped))):
                continue
            if hasattr(pathfinder, "is_navigable") and not pathfinder.is_navigable(snapped):
                continue
            snapped = np.asarray(snapped, dtype=np.float32)
            if abs(float(snapped[1]) - float(floor_y)) > 0.85:
                continue
            got = _horiz_m(snapped, o)
            if got < 0.75 * r or got < 0.55:
                continue
            yield snapped, r, float(theta_deg), got


def _eval_stretch512_pose(sim, camera, agent_pos, obj_pos, pitch_clip_deg=50.0):
    sensor = _sensor_pos(agent_pos, camera["height_m"])
    rotation = _look_at_xyzw(sensor, obj_pos, pitch_clip_deg=pitch_clip_deg)
    obs = _observe_at(sim, agent_pos, rotation)
    img = np.asarray(obs["rgb"])[:, :, :3]
    depth = obs.get("depth")
    obj_depth = float(np.linalg.norm(np.asarray(obj_pos, dtype=np.float64) - sensor))
    wall = _wall_facing_stats(img, depth, obj_depth)
    hits = _project_points([obj_pos], sensor, rotation, camera)
    uvz = hits[0] if hits else None
    w = int(camera["width_px"])
    h = int(camera["height_px"])
    centroid_in = bool(
        uvz is not None and 0.0 <= float(uvz[0]) < w and 0.0 <= float(uvz[1]) < h
    )
    if depth is not None:
        d = np.asarray(depth, dtype=np.float32).reshape(-1)
        near = float(np.mean((d > 1e-4) & (d < 0.20))) if d.size else 0.0
    else:
        near = 0.0
    score = _score_stretch512_view(wall, centroid_in=centroid_in)
    if near > 0.18:
        score -= 50.0
    return {
        "img": img,
        "wall": wall,
        "score": score,
        "issues": stretch512_issue_tags(wall),
        "obj_depth": obj_depth,
        "centroid_in": centroid_in,
        "agent_pos": [float(x) for x in np.asarray(agent_pos).reshape(-1).tolist()],
        "rotation": [float(x) for x in rotation],
        "near_frac": round(near, 4),
    }


def _render_stretch512_goal(sim, camera, g, args):
    """Default standoff view, then orbit-rescue if the frame is a known failure."""
    vp = g["iin_viewpoint"]
    obj = g.get("position")
    agent_pos = np.asarray(vp["position"], dtype=np.float32)
    if args.standoff_m > 0 and obj is not None:
        agent_pos = _standoff_agent(
            sim.pathfinder,
            agent_pos,
            obj,
            float(args.standoff_m),
            ladder=(2.4, 3.0),
        )
    if obj is None:
        rotation = vp.get("rotation")
        obs = _observe_at(sim, agent_pos, rotation)
        img = np.asarray(obs["rgb"])[:, :, :3]
        wall = _wall_facing_stats(img, obs.get("depth"), None)
        return img, {
            "wall": wall,
            "issues": stretch512_issue_tags(wall),
            "score": _score_stretch512_view(wall),
            "rescued": False,
            "n_eval": 1,
            "agent_pos": [float(x) for x in np.asarray(agent_pos).tolist()],
            "standoff_m": float(args.standoff_m),
        }
    default = _eval_stretch512_pose(sim, camera, agent_pos, obj, pitch_clip_deg=None)
    best = dict(default)
    best["source"] = "default"
    n_eval = 1
    do_rescue = bool(getattr(args, "rescue", True)) and bool(default["issues"])
    if do_rescue:
        floor_y = float(np.asarray(vp["position"], dtype=np.float64)[1])
        radii = (1.8, 2.4, 3.0, 3.8)
        thetas = tuple(range(0, 360, 30))
        for snapped, radius, theta, got in _iter_orbit_agents(
            sim.pathfinder, obj, floor_y, radii, thetas
        ):
            cand = _eval_stretch512_pose(
                sim, camera, snapped, obj, pitch_clip_deg=50.0
            )
            n_eval += 1
            cand["source"] = "orbit"
            cand["orbit_radius"] = radius
            cand["orbit_theta"] = theta
            cand["orbit_got"] = got
            if cand["score"] > best["score"]:
                best = cand
    img = best.pop("img")
    rec = {
        "wall": best["wall"],
        "issues": best["issues"],
        "score": float(best["score"]),
        "rescued": best.get("source") == "orbit",
        "rescue_attempted": do_rescue,
        "n_eval": n_eval,
        "source": best.get("source") or "default",
        "centroid_in": bool(best.get("centroid_in")),
        "agent_pos": best["agent_pos"],
        "standoff_m": float(args.standoff_m),
        "orbit_radius": best.get("orbit_radius"),
        "orbit_theta": best.get("orbit_theta"),
    }
    return img, rec


def _crop_around(
    img: np.ndarray,
    uvz,
    camera: dict,
    obj_m: float = 0.7,
    keep_square: bool = False,
) -> Tuple[np.ndarray, List[int], bool]:
    """Crop a window around the projected object. Returns crop, box, in_frame.

    ``keep_square=True`` shifts a square window into the frame instead of
    clamping x/y independently (which turns floor objects into flat strips).
    """
    h, w = img.shape[:2]
    if uvz is None:
        return img, [0, 0, w, h], False
    u, v, depth = uvz
    fx = (0.5 * int(camera["width_px"])) / math.tan(math.radians(float(camera["hfov"])) * 0.5)
    half = int(max(32.0, min(0.48 * min(w, h), fx * obj_m / max(depth, 0.2))))
    pad = 8
    in_frame = (-pad <= u < w + pad) and (-pad <= v < h + pad)
    if keep_square:
        side = int(max(24, min(2 * half, w, h)))
        x1 = int(round(u - 0.5 * side))
        y1 = int(round(v - 0.5 * side))
        x1 = max(0, min(x1, w - side))
        y1 = max(0, min(y1, h - side))
        return img[y1 : y1 + side, x1 : x1 + side], [x1, y1, x1 + side, y1 + side], in_frame
    x1 = int(round(u - half))
    y1 = int(round(v - half))
    x2 = int(round(u + half))
    y2 = int(round(v + half))
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 - x1 < 24 or y2 - y1 < 24:
        return img, [0, 0, w, h], False
    return img[y1:y2, x1:x2], [x1, y1, x2, y2], in_frame


def _to_square_size(img: np.ndarray, size: int) -> np.ndarray:
    """Cover-crop to ``size×size``: scale the short side, center-crop the long side.

    Stretch object crops can be wide or short after the frame clamp. Letterbox
    padding used to add black bars; those bars waste SuperGlue pixels and are
    not scene content. Cover keeps only real RGB.
    """
    if size <= 0:
        return np.asarray(img)[:, :, :3]
    from PIL import Image

    im = Image.fromarray(np.asarray(img)[:, :, :3])
    w, h = im.size
    if w < 1 or h < 1:
        return np.zeros((int(size), int(size), 3), dtype=np.uint8)
    if w <= h:
        nw, nh = int(size), max(int(size), int(round(h * float(size) / w)))
    else:
        nh, nw = int(size), max(int(size), int(round(w * float(size) / h)))
    resample = getattr(Image, "Resampling", Image).LANCZOS
    im = im.resize((nw, nh), resample)
    left = max(0, (nw - int(size)) // 2)
    top = max(0, (nh - int(size)) // 2)
    im = im.crop((left, top, left + int(size), top + int(size)))
    if im.size != (int(size), int(size)):
        im = im.resize((int(size), int(size)), resample)
    return np.asarray(im)


# ------------------------------------------------------------------ sim helpers
def _make_sensor(hs, uuid: str, sensor_type, camera: dict):
    spec = hs.SensorSpec()
    spec.uuid = uuid
    spec.sensor_type = sensor_type
    spec.resolution = [int(camera["height_px"]), int(camera["width_px"])]
    spec.position = [0.0, float(camera["height_m"]), 0.0]
    try:
        spec.parameters["hfov"] = str(int(round(float(camera["hfov"]))))
    except Exception:  # noqa: BLE001 - older builds set hfov differently; default is acceptable
        pass
    if hasattr(spec, "hfov"):
        try:
            spec.hfov = float(camera["hfov"])
        except Exception:  # noqa: BLE001
            pass
    return spec


def _make_sim(
    scene_glb: str,
    scene_dataset_config: str,
    camera: dict,
    gpu_device_id: int = 0,
    with_semantic: bool = False,
):
    import habitat_sim as hs

    backend = hs.SimulatorConfiguration()
    backend.scene_id = scene_glb
    if scene_dataset_config:
        backend.scene_dataset_config_file = scene_dataset_config
    backend.enable_physics = False
    backend.gpu_device_id = int(gpu_device_id)
    if with_semantic and hasattr(backend, "load_semantic_mesh"):
        backend.load_semantic_mesh = True

    specs = [_make_sensor(hs, "rgb", hs.SensorType.COLOR, camera)]
    if with_semantic and hasattr(hs.SensorType, "SEMANTIC"):
        specs.append(_make_sensor(hs, "semantic", hs.SensorType.SEMANTIC, camera))
    if camera.get("with_depth") and hasattr(hs.SensorType, "DEPTH"):
        specs.append(_make_sensor(hs, "depth", hs.SensorType.DEPTH, camera))

    agent = hs.agent.AgentConfiguration()
    agent.sensor_specifications = specs
    try:
        return hs.Simulator(hs.Configuration(backend, [agent]))
    except Exception:
        if not with_semantic:
            raise
        agent.sensor_specifications = specs[:1]
        return hs.Simulator(hs.Configuration(backend, [agent]))


def _render_at(sim, position, rotation_xyzw) -> np.ndarray:
    obs = _observe_at(sim, position, rotation_xyzw)
    return np.asarray(obs["rgb"])[:, :, :3]


def _observe_at(sim, position, rotation_xyzw) -> dict:
    import habitat_sim as hs
    from habitat_sim.utils.common import quat_from_coeffs

    st = hs.agent.AgentState()
    st.position = np.asarray(position, dtype=np.float32)
    if rotation_xyzw is not None:
        st.rotation = quat_from_coeffs(np.asarray(rotation_xyzw, dtype=np.float32))
    sim.get_agent(0).set_state(st)
    return sim.get_sensor_observations()


def _scene_glb(scenes_dir: str, scene_id: str) -> str:
    # Same path skew as ma_search._resolve_scene, but do not import that
    # module (it pulls GroundingDINO via vlnce_baselines).
    import glob

    if os.path.isabs(scene_id) or os.path.exists(scene_id):
        return scene_id
    rel = scene_id.replace("//", "/")
    for cand in (
        os.path.join(scenes_dir, rel),
        os.path.join(scenes_dir, rel.replace("hm3d/val/", "hm3d_v0.2/val/")),
        os.path.join(scenes_dir, rel.replace("hm3d/", "hm3d_v0.2/")),
    ):
        if os.path.exists(cand):
            return cand
    hits = glob.glob(os.path.join(scenes_dir, "**", os.path.basename(rel)), recursive=True)
    return hits[0] if hits else os.path.join(scenes_dir, rel)


def _scene_dataset_config(scenes_dir: str, scene_glb: str = "") -> str:

    return scene_dataset_config_for(scene_glb, scenes_dir)


def _look_at_jittered(src, dst, yaw_deg: float = 0.0, pitch_deg: float = 0.0) -> List[float]:
    base = _look_at_xyzw(src, dst)
    if abs(float(yaw_deg)) < 1e-6 and abs(float(pitch_deg)) < 1e-6:
        return base
    try:
        from habitat_sim.utils.common import (
            quat_from_angle_axis,
            quat_from_coeffs,
            quat_to_coeffs,
        )

        rot = quat_from_coeffs(np.asarray(base, dtype=np.float32))
        rot = quat_from_angle_axis(
            math.radians(float(yaw_deg)), np.array([0.0, 1.0, 0.0], dtype=np.float32)
        ) * rot
        rot = rot * quat_from_angle_axis(
            math.radians(float(pitch_deg)), np.array([1.0, 0.0, 0.0], dtype=np.float32)
        )
        return [float(x) for x in np.asarray(quat_to_coeffs(rot)).reshape(-1)[:4]]
    except Exception:  # noqa: BLE001
        return base


def _look_at_xyzw(src, dst, pitch_clip_deg: Optional[float] = None) -> List[float]:
    """Yaw then pitch, gravity-up. Avoids the roll of a raw two-vector look-at."""
    import habitat_sim as hs
    from habitat_sim.utils.common import (
        quat_from_angle_axis,
        quat_from_two_vectors,
        quat_to_coeffs,
    )

    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    delta = dst - src
    horiz = np.array([delta[0], 0.0, delta[2]], dtype=np.float64)
    hn = float(np.linalg.norm(horiz))
    if hn < 1e-6:
        horiz = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        hn = 1.0
    else:
        horiz = horiz / hn
    front = getattr(getattr(hs, "geo", None), "FRONT", None)
    if front is None:
        front = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    rot = quat_from_two_vectors(np.asarray(front, dtype=np.float64), horiz)
    # Camera above the object -> look down. Matches PoseSampler.sample_camera_poses.
    pn = float(np.linalg.norm(src - dst))
    if pn > 1e-6:
        phi = (math.pi / 2.0) - math.acos(max(-1.0, min(1.0, float((src - dst)[1]) / pn)))
        if pitch_clip_deg is not None:
            lim = math.radians(float(pitch_clip_deg))
            phi = max(-lim, min(lim, phi))
        right = getattr(getattr(hs, "geo", None), "RIGHT", None)
        if right is None:
            right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        rot = rot * quat_from_angle_axis(-phi, np.asarray(right, dtype=np.float64))
    coeffs = np.asarray(quat_to_coeffs(rot), dtype=np.float64).reshape(-1)
    return [float(x) for x in coeffs[:4]]


def _quat_xyzw_to_R(q) -> np.ndarray:
    x, y, z, w = [float(v) for v in q]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _project_points(points, cam_pos, rot_xyzw, camera: dict):
    """Pinhole project world points. Habitat cam: +X right, +Y up, -Z forward."""
    R = _quat_xyzw_to_R(rot_xyzw)
    w = int(camera["width_px"])
    h = int(camera["height_px"])
    fx = (0.5 * w) / math.tan(math.radians(float(camera["hfov"])) * 0.5)
    fy = fx
    cx, cy = 0.5 * w, 0.5 * h
    out = []
    cam = np.asarray(cam_pos, dtype=np.float64)
    for pt in points:
        local = R.T @ (np.asarray(pt, dtype=np.float64) - cam)
        depth = -float(local[2])
        if depth <= 0.08:
            out.append(None)
            continue
        u = cx + fx * float(local[0]) / depth
        v = cy - fy * float(local[1]) / depth
        out.append((u, v, depth))
    return out


def _aabb_corners(center, sizes) -> List[np.ndarray]:
    c = np.asarray(center, dtype=np.float64)
    half = 0.5 * np.asarray(sizes, dtype=np.float64)
    pts = []
    for sx in (-1.0, 1.0):
        for sy in (-1.0, 1.0):
            for sz in (-1.0, 1.0):
                pts.append(c + half * np.array([sx, sy, sz], dtype=np.float64))
    return pts


def _in_frame_frac(uvzs, camera: dict, pad: float = 4.0) -> float:
    w = int(camera["width_px"])
    h = int(camera["height_px"])
    if not uvzs:
        return 0.0
    inside = 0
    for uvz in uvzs:
        if uvz is None:
            continue
        u, v, _ = uvz
        if (-pad <= u < w + pad) and (-pad <= v < h + pad):
            inside += 1
    return inside / float(len(uvzs))


def _find_semantic_object(sim, object_id, category, position):
    scene = getattr(sim, "semantic_scene", None)
    objs = getattr(scene, "objects", None) if scene is not None else None
    if not objs:
        return None
    oid = str(object_id or "")
    tail = oid.split("_")[-1]
    pos = np.asarray(position, dtype=np.float64)
    exact, tail_hits, cat_hits = [], [], []
    cat_key = str(category or "").replace(" ", "_").lower()

    def _cat_name(obj) -> str:
        try:
            cat = getattr(obj, "category", None)
            if cat is None:
                return ""
            name = cat.name() if callable(getattr(cat, "name", None)) else str(cat)
            return str(name or "").replace(" ", "_").lower()
        except Exception:  # noqa: BLE001
            return ""

    def _dist(obj) -> float:
        try:
            return float(np.linalg.norm(np.asarray(obj.aabb.center, dtype=np.float64) - pos))
        except Exception:  # noqa: BLE001
            return 1e9

    for obj in objs:
        if obj is None:
            continue
        oid_s = str(getattr(obj, "id", "") or "")
        if oid and (oid_s == oid or oid_s.endswith("/" + oid)):
            exact.append(obj)
        if tail.isdigit() and (oid_s == tail or oid_s.endswith("_" + tail)):
            tail_hits.append(obj)
        if cat_key and _cat_name(obj) == cat_key:
            cat_hits.append(obj)
    if exact:
        return min(exact, key=_dist)
    if tail_hits:
        return min(tail_hits, key=_dist)
    if cat_hits:
        best = min(cat_hits, key=_dist)
        return best if _dist(best) < 1.5 else None
    nearest = min((o for o in objs if o is not None), key=_dist, default=None)
    return nearest if nearest is not None and _dist(nearest) < 0.6 else None


_CAT_SIZE_PRIORS = {
    "rug": (2.2, 0.12, 2.2),
    "carpet": (2.2, 0.12, 2.2),
    "stair": (2.4, 2.4, 2.4),
    "refrigerator": (0.9, 1.9, 0.9),
    "dishwasher": (0.8, 0.9, 0.8),
    "dresser": (1.4, 1.2, 0.7),
    "nightstand": (0.6, 0.7, 0.6),
    "mirror": (0.8, 1.2, 0.15),
    "picture": (0.7, 0.6, 0.12),
    "photo": (0.5, 0.4, 0.08),
    "pillow": (0.6, 0.25, 0.45),
    "book": (0.25, 0.25, 0.15),
    "microwave": (0.55, 0.35, 0.4),
    "blinds": (1.4, 1.4, 0.15),
    "parapet": (1.6, 1.0, 0.2),
    "radiator": (1.0, 0.7, 0.25),
    "statue": (0.5, 1.2, 0.5),
    "plant": (0.6, 1.0, 0.6),
    "flowerpot": (0.35, 0.45, 0.35),
    "hanging clothes": (1.0, 1.4, 0.5),
    "shower glass": (1.2, 2.0, 0.1),
    "window glass": (1.4, 1.4, 0.1),
    "calendar": (0.4, 0.5, 0.08),
    "printer": (0.5, 0.4, 0.4),
    "speaker": (0.3, 0.5, 0.3),
    "ottoman": (0.7, 0.45, 0.7),
    "coffee machine": (0.35, 0.4, 0.3),
    "christmas tree": (1.0, 1.8, 1.0),
    "exercise bike": (1.2, 1.2, 0.6),
    "hanger": (0.5, 1.2, 0.3),
    "glass": (0.3, 0.8, 0.3),
}


def _estimate_object_box(goal: dict, sem_obj) -> Tuple[np.ndarray, np.ndarray]:
    if sem_obj is not None:
        return (
            np.asarray(sem_obj.aabb.center, dtype=np.float64),
            np.asarray(sem_obj.aabb.sizes, dtype=np.float64),
        )
    pos = np.asarray(goal["position"], dtype=np.float64)
    prior = np.asarray(
        _CAT_SIZE_PRIORS.get(str(goal.get("object_category") or ""), (0.9, 0.9, 0.9)),
        dtype=np.float64,
    )
    pts = goal.get("instance_reach_points") or goal.get("reach_points") or []
    if pts:
        arr = np.asarray(pts, dtype=np.float64)
        span = arr.max(axis=0) - arr.min(axis=0)
        sizes = np.array(
            [
                max(float(prior[0]), float(span[0]) + 0.35),
                max(float(prior[1]), abs(float(pos[1]) - float(np.median(arr[:, 1]))) * 2.0, 0.4),
                max(float(prior[2]), float(span[2]) + 0.35),
            ],
            dtype=np.float64,
        )
        return pos, sizes
    return pos, prior


def _floor_y(pathfinder, obj_pos, goal=None) -> float:
    if goal:
        vp = goal.get("iin_viewpoint") or {}
        if vp.get("position"):
            return float(np.asarray(vp["position"])[1])
        pts = goal.get("instance_reach_points") or goal.get("reach_points") or []
        if pts:
            return float(np.median([float(p[1]) for p in pts]))
    p = np.asarray(obj_pos, dtype=np.float32)
    try:
        snapped = pathfinder.snap_point(p)
        if snapped is not None and np.all(np.isfinite(np.asarray(snapped))):
            return float(np.asarray(snapped)[1])
    except Exception:  # noqa: BLE001
        pass
    return float(p[1])


def _hfov_for_object(sizes, radius: float) -> int:
    half = 0.5 * float(np.linalg.norm(np.asarray(sizes, dtype=np.float64)))
    ang = 2.0 * math.degrees(math.atan2(half, max(float(radius), 0.2)))
    raw = max(75.0, min(120.0, ang / 0.50))
    return min(_cfg.IIN_OFFICIAL_HFOV_BINS, key=lambda b: abs(float(b) - raw))


def _apply_hfov(sim, hfov: float, uuids=("rgb", "semantic", "depth")) -> float:
    applied = None
    sensors = getattr(sim, "_sensors", {}) or {}
    for uid in uuids:
        sensor = sensors.get(uid)
        if sensor is None:
            continue
        for attr in ("hfov", "HFOV"):
            if hasattr(sensor, attr):
                try:
                    setattr(sensor, attr, float(hfov))
                    applied = float(getattr(sensor, attr))
                except Exception:  # noqa: BLE001
                    pass
        spec = getattr(sensor, "_spec", None) or getattr(sensor, "spec", None)
        if spec is not None:
            try:
                spec.parameters["hfov"] = str(int(round(float(hfov))))
            except Exception:  # noqa: BLE001
                pass
            if hasattr(spec, "hfov"):
                try:
                    spec.hfov = float(hfov)
                    applied = float(hfov)
                except Exception:  # noqa: BLE001
                    pass
    return float(hfov if applied is None else applied)


def _cast_clear(sim, origin, dest, slack: float = 0.15) -> bool:
    o = np.asarray(origin, dtype=np.float64)
    d = np.asarray(dest, dtype=np.float64) - o
    dist = float(np.linalg.norm(d))
    if dist < 0.2:
        return False
    direction = d / dist
    try:
        import habitat_sim as hs

        ray = hs.geo.Ray()
        ray.origin = o.astype(np.float32)
        ray.direction = direction.astype(np.float32)
        hit = sim.cast_ray(ray, max_distance=float(dist))
        if hit is None:
            return True
        hits = getattr(hit, "hits", None)
        if not hits:
            return True
        first = hits[0]
        rd = float(getattr(first, "ray_distance", dist))
        return rd >= dist - slack
    except Exception:  # noqa: BLE001
        return True


def _near_reach_xz(pos, reach_pts, max_dist: float = 1.8) -> bool:
    if not reach_pts:
        return True
    x, z = float(pos[0]), float(pos[2])
    best = 1e9
    for p in reach_pts:
        best = min(best, math.hypot(x - float(p[0]), z - float(p[2])))
    return best <= max_dist


def _sample_official_cameras(center, sizes, floor_y: float, reach_pts=None, rng=None):
    """Krantz / GOAT OVIIN polar grid. ``sizes`` / ``reach_pts`` unused on the
    main grid; reach points are a last-resort fallback in ``_pick_official_view``.
    """
    del sizes, reach_pts
    rng = np.random.default_rng(0) if rng is None else rng
    for cand in iter_iin_cameras(center, floor_y, rng):
        yield cand


def _score_official(in_frame_frac: float, frame_cov: Optional[float], centroid_in: bool) -> float:
    cov = 0.10 if frame_cov is None else float(frame_cov)
    return (
        5.0 * float(in_frame_frac)
        + 2.0 * min(cov, 0.40)
        - 1.5 * max(0.0, cov - 0.55)
        + (0.5 if centroid_in else 0.0)
    )


def _id_to_name(sim) -> Dict[int, str]:
    out: Dict[int, str] = {}
    scene = getattr(sim, "semantic_scene", None)
    objs = getattr(scene, "objects", None) if scene is not None else None
    if not objs:
        return out
    for obj in objs:
        if obj is None:
            continue
        try:
            sid = int(getattr(obj, "semantic_id", -1))
        except Exception:  # noqa: BLE001
            continue
        cat = getattr(obj, "category", None)
        name = ""
        try:
            name = cat.name() if cat is not None and callable(getattr(cat, "name", None)) else str(cat or "")
        except Exception:  # noqa: BLE001
            name = ""
        out[sid] = str(name or "unknown").replace(" ", "_").lower()
    return out


def _eval_polar_candidate(sim, base_camera, cand, center, corners, semantic_id):
    pos = np.asarray(cand["position"], dtype=np.float64)
    camera = dict(base_camera)
    camera["hfov"] = float(cand["hfov"])
    _apply_hfov(sim, camera["hfov"])
    rot = _look_at_jittered(
        pos, center, cand.get("yaw_jitter_deg") or 0.0, cand.get("pitch_jitter_deg") or 0.0
    )
    obs = _observe_at(sim, pos, rot)
    img = np.asarray(obs["rgb"])[:, :, :3]
    uvzs = _project_points(corners, pos, rot, camera)
    in_frac = _in_frame_frac(uvzs[1:], camera)
    centroid_in = bool(
        uvzs[0] is not None
        and 0.0 <= uvzs[0][0] < camera["width_px"]
        and 0.0 <= uvzs[0][1] < camera["height_px"]
    )
    depth = obs.get("depth")
    if depth is not None:
        depth = np.asarray(depth, dtype=np.float32)
        if float((depth < 0.25).mean()) > 0.12:
            return None
    sem = obs.get("semantic")
    frame_cov = None
    pts = np.zeros((0, 3), dtype=np.float64)
    if sem is not None and semantic_id >= 0:
        frame_cov = frame_coverage(sem, semantic_id)
        if frame_cov <= 0.0:
            return None
        if depth is not None:
            pts = backproject_mask(depth, sem, semantic_id, pos, rot, camera)
    return {
        "img": img,
        "obs": obs,
        "in_frame_frac": in_frac,
        "frame_coverage": frame_cov,
        "object_coverage": None,
        "centroid_in": centroid_in,
        "hfov": float(camera["hfov"]),
        "radius_m": cand.get("radius_m"),
        "theta_deg": cand.get("theta_deg"),
        "height_m": float(pos[1]),
        "cam_pos": [float(x) for x in pos.tolist()],
        "rotation": rot,
        "points": pts,
        "camera": camera,
    }


def _pick_official_view(sim, base_camera: dict, goal: dict) -> Tuple[np.ndarray, dict]:
    """Krantz / GOAT OVIIN: polar grid, coverage gates, one best 512x512 view."""
    obj_pos = np.asarray(goal["position"], dtype=np.float64)
    sem_obj = _find_semantic_object(
        sim, goal.get("object_id"), goal.get("object_category"), obj_pos
    )
    center, sizes = _estimate_object_box(goal, sem_obj)
    semantic_id = int(getattr(sem_obj, "semantic_id", -1)) if sem_obj is not None else -1
    floor_y = _floor_y(sim.pathfinder, center, goal)
    corners = [center] + _aabb_corners(center, sizes)
    half_diag = 0.5 * float(np.linalg.norm(sizes))
    rng = goal_rng(goal)

    records = []
    n_eval = 0
    for cand in iter_iin_cameras(center, floor_y, rng):
        pos = cand["position"]
        if float(np.linalg.norm(pos - center)) < max(0.35, 0.45 * half_diag):
            continue
        if pos[1] < floor_y + 0.25:
            continue
        rec = _eval_polar_candidate(sim, base_camera, cand, center, corners, semantic_id)
        if rec is None:
            continue
        rec["height_m"] = float(pos[1] - floor_y)
        n_eval += 1
        records.append(rec)

    all_pts = (
        np.concatenate([r["points"] for r in records if len(r["points"])], axis=0)
        if any(len(r["points"]) for r in records)
        else np.zeros((0, 3), dtype=np.float64)
    )
    osa = observable_surface_area(all_pts)
    n_ok = 0
    for rec in records:
        c_o = object_coverage(rec["points"], all_pts) if len(all_pts) else None
        rec["object_coverage"] = c_o
        rec["ok"] = iin_gates(c_o, rec["frame_coverage"], osa)
        rec["score"] = _score_official(rec["in_frame_frac"], rec["frame_coverage"], rec["centroid_in"])
        if rec["ok"]:
            n_ok += 1

    best = None
    if records:
        idx = pick_iin_index(records)
        best = records[idx]

    if best is None:
        vp = goal.get("iin_viewpoint") or {}
        agent_pos = np.asarray(vp.get("position") or (center + np.array([1.5, 0.0, 0.0])))
        cam_pos = _sensor_pos(agent_pos, 1.2)
        rot = _look_at_xyzw(cam_pos, center)
        camera = dict(base_camera)
        camera["hfov"] = 90
        _apply_hfov(sim, 90)
        img = _render_at(sim, cam_pos, rot)
        uvzs = _project_points(corners, cam_pos, rot, camera)
        in_frac = _in_frame_frac(uvzs[1:], camera)
        centroid_in = bool(
            uvzs[0] is not None
            and 0.0 <= uvzs[0][0] < camera["width_px"]
            and 0.0 <= uvzs[0][1] < camera["height_px"]
        )
        best = {
            "score": _score_official(in_frac, None, centroid_in),
            "ok": in_frac >= float(_cfg.IIN_OFFICIAL_MIN_OBJECT_IN_FRAME),
            "in_frame_frac": in_frac,
            "frame_coverage": None,
            "object_coverage": None,
            "centroid_in": centroid_in,
            "hfov": 90,
            "radius_m": None,
            "theta_deg": None,
            "height_m": 1.2,
            "cam_pos": [float(x) for x in np.asarray(cam_pos).tolist()],
            "rotation": rot,
            "img": img,
            "fallback": True,
        }

    img = best.pop("img")
    best.pop("points", None)
    best.pop("obs", None)
    cam_out = dict(base_camera, hfov=float(best["hfov"]))
    cam_out.pop("with_depth", None)
    extra = {
        "camera": cam_out,
        "look_at_object": True,
        "in_frame": bool(best.get("ok") or best.get("centroid_in")),
        "in_frame_frac": float(best["in_frame_frac"]),
        "frame_coverage": None if best.get("frame_coverage") is None else float(best["frame_coverage"]),
        "centroid_in": bool(best.get("centroid_in")),
        "crop_box": None,
        "depth_m": float(np.linalg.norm(np.asarray(best["cam_pos"]) - center)),
        "standoff_m": 0.0,
        "agent_pos": best["cam_pos"],
        "rotation": best["rotation"],
        "official": {
            "ok": bool(best.get("ok")),
            "score": float(best.get("score") or 0.0),
            "radius_m": best.get("radius_m"),
            "theta_deg": best.get("theta_deg"),
            "height_m": best.get("height_m"),
            "object_coverage": best.get("object_coverage"),
            "osa": None if osa is None else float(osa),
            "n_eval": n_eval,
            "n_ok": n_ok,
            "semantic_id": semantic_id,
            "aabb_sizes": [float(x) for x in sizes.tolist()],
            "fallback": bool(best.get("fallback")),
            "protocol": "krantz_goat_oviin",
        },
    }
    return img, extra


def pick_languagenav_view(sim, base_camera: dict, goal: dict) -> Tuple[np.ndarray, dict]:
    """GOAT LanguageNav: Stretch 512 grid, keep the max frame-coverage view."""
    obj_pos = np.asarray(goal["position"], dtype=np.float64)
    sem_obj = _find_semantic_object(
        sim, goal.get("object_id"), goal.get("object_category"), obj_pos
    )
    center, sizes = _estimate_object_box(goal, sem_obj)
    semantic_id = int(getattr(sem_obj, "semantic_id", -1)) if sem_obj is not None else -1
    floor_y = _floor_y(sim.pathfinder, center, goal)
    corners = [center] + _aabb_corners(center, sizes)
    names = _id_to_name(sim)

    records = []
    n_eval = 0
    for cand in iter_tn_cameras(center, floor_y):
        pos = np.asarray(cand["position"], dtype=np.float64)
        snapped = None
        try:
            snapped = sim.pathfinder.snap_point(
                np.array([pos[0], floor_y, pos[2]], dtype=np.float32)
            )
        except Exception:  # noqa: BLE001
            snapped = None
        if snapped is not None and np.all(np.isfinite(np.asarray(snapped))):
            if abs(float(snapped[0]) - pos[0]) > 0.45 or abs(float(snapped[2]) - pos[2]) > 0.45:
                continue
            pos = np.array([float(snapped[0]), pos[1], float(snapped[2])], dtype=np.float64)
            cand = dict(cand, position=pos)
        rec = _eval_polar_candidate(sim, base_camera, cand, center, corners, semantic_id)
        if rec is None:
            continue
        rec["height_m"] = float(_cfg.TN_CAMERA_GOAT["height_m"])
        n_eval += 1
        records.append(rec)

    best = None
    if records:
        idx = pick_tn_index(records)
        best = records[idx]

    if best is None:
        vp = goal.get("iin_viewpoint") or {}
        agent_pos = np.asarray(vp.get("position") or (center + np.array([1.5, 0.0, 0.0])))
        cam_pos = np.array(
            [float(agent_pos[0]), float(floor_y) + float(_cfg.TN_CAMERA_GOAT["height_m"]), float(agent_pos[2])],
            dtype=np.float64,
        )
        rot = _look_at_xyzw(cam_pos, center)
        camera = dict(base_camera)
        camera["hfov"] = float(_cfg.TN_CAMERA_GOAT["hfov"])
        _apply_hfov(sim, camera["hfov"])
        obs = _observe_at(sim, cam_pos, rot)
        img = np.asarray(obs["rgb"])[:, :, :3]
        best = {
            "img": img,
            "obs": obs,
            "ok": False,
            "in_frame_frac": 0.0,
            "frame_coverage": None,
            "centroid_in": False,
            "hfov": camera["hfov"],
            "radius_m": None,
            "theta_deg": None,
            "height_m": float(_cfg.TN_CAMERA_GOAT["height_m"]),
            "cam_pos": [float(x) for x in cam_pos.tolist()],
            "rotation": rot,
            "fallback": True,
        }

    nearby = []
    obs = best.pop("obs", None)
    if obs is not None and obs.get("semantic") is not None and obs.get("depth") is not None:
        nearby = nearby_objects(obs["semantic"], obs["depth"], names, semantic_id)
    img = best.pop("img")
    best.pop("points", None)
    extra = {
        "camera": dict(base_camera, hfov=float(best["hfov"])),
        "look_at_object": True,
        "in_frame": bool((best.get("frame_coverage") or 0.0) > 0.0 or best.get("centroid_in")),
        "in_frame_frac": float(best.get("in_frame_frac") or 0.0),
        "frame_coverage": None if best.get("frame_coverage") is None else float(best["frame_coverage"]),
        "centroid_in": bool(best.get("centroid_in")),
        "agent_pos": best["cam_pos"],
        "rotation": best["rotation"],
        "nearby_objects": nearby,
        "languagenav": {
            "ok": bool((best.get("frame_coverage") or 0.0) > 0.0),
            "radius_m": best.get("radius_m"),
            "theta_deg": best.get("theta_deg"),
            "height_m": best.get("height_m"),
            "n_eval": n_eval,
            "semantic_id": semantic_id,
            "fallback": bool(best.get("fallback")),
            "protocol": "goat_languagenav",
        },
    }
    extra["camera"].pop("with_depth", None)
    return img, extra


# ------------------------------------------------------------------ main render
def render(args) -> dict:
    import imageio

    with open(args.episodes) as f:
        data = json.load(f)
    episodes = data["episodes"]
    if args.limit:
        episodes = episodes[: args.limit]
    only_keys = None
    if getattr(args, "only_goals", None):
        raw = json.loads(open(args.only_goals).read())
        items = raw.get("keys") or raw.get("items") or raw
        only_keys = set()
        for it in items:
            if isinstance(it, dict):
                only_keys.add((str(it["episode_id"]), int(it["goal_index"])))
            else:
                only_keys.add((str(it[0]), int(it[1])))

    # group IIN goals by scene so each scene's sim is built once
    by_scene: Dict[str, List[tuple]] = defaultdict(list)
    n_skip_exist = 0
    for ep in episodes:
        for g in ep["goals_set"]:
            if g["modality"] != "IIN" or not g.get("iin_viewpoint"):
                continue
            if only_keys is not None and (ep["episode_id"], int(g["goal_index"])) not in only_keys:
                continue
            fname = f"{ep['episode_id']}__g{g['goal_index']}_{g['object_category']}.png"
            fpath = os.path.join(args.out_dir, fname)
            if args.skip_existing and os.path.isfile(fpath):
                n_skip_exist += 1
                continue
            by_scene[ep["scene_id"]].append((ep["episode_id"], ep, g))

    if args.n_shards > 1:
        scenes = sorted(by_scene)
        keep = {s for i, s in enumerate(scenes) if i % args.n_shards == args.shard_index}
        by_scene = {s: v for s, v in by_scene.items() if s in keep}

    os.makedirs(args.out_dir, exist_ok=True)
    manifest: List[dict] = []
    n_render = 0
    print(
        "[render] skip_existing=%d shard=%d/%d scenes=%d todo=%d"
        % (
            n_skip_exist,
            args.shard_index,
            args.n_shards,
            len(by_scene),
            sum(len(v) for v in by_scene.values()),
        ),
        flush=True,
    )
    verifier = None
    if args.verify:
        from .iin_verify import InstanceVerifier
        verifier = InstanceVerifier(backend=args.verify_backend)

    for scene_id, items in by_scene.items():
        glb = _scene_glb(args.scenes_dir, scene_id)
        sds = _scene_dataset_config(args.scenes_dir, glb)
        if args.camera_profile == "stretch512":
            cam = dict(_cfg.IIN_CAMERA_STRETCH_512)
            cam["with_depth"] = True
        elif args.camera_profile == "stretch":
            cam = dict(_cfg.IIN_CAMERA_STRETCH)
        elif args.camera_profile == "locobot":
            cam = dict(_cfg.IIN_CAMERA)
        elif args.camera_profile == "official":
            cam = dict(_cfg.IIN_CAMERA_OFFICIAL)
        else:
            cam = items[0][2]["iin_viewpoint"]["camera"]
        if args.camera_profile == "official":
            cam = dict(cam)
            cam["with_depth"] = True
        sim = _make_sim(
            glb, sds, cam, args.gpu_device_id, with_semantic=args.camera_profile == "official"
        )
        try:
            # cache same-category instance positions in this scene for negatives
            cat_positions: Dict[str, list] = defaultdict(list)
            for _, ep, g in items:
                cat_positions[g["object_category"]].append(g)

            for eid, ep, g in items:
                if args.camera_profile == "official":
                    img, extra = _pick_official_view(sim, cam, g)
                    camera = extra["camera"]
                    fname = f"{eid}__g{g['goal_index']}_{g['object_category']}.png"
                    fpath = os.path.join(args.out_dir, fname)
                    if args.out_size:
                        img = _to_square_size(img, int(args.out_size))
                    imageio.imwrite(fpath, img)
                    rec = {
                        "episode_id": eid,
                        "goal_index": g["goal_index"],
                        "object_category": g["object_category"],
                        "image": fpath,
                        **extra,
                    }
                    print(
                        "[official] %s g%s %s hfov=%s in_frame=%.2f cov=%s ok=%s"
                        % (
                            eid,
                            g["goal_index"],
                            g["object_category"],
                            extra["camera"]["hfov"],
                            extra["in_frame_frac"],
                            extra["frame_coverage"],
                            extra["official"]["ok"],
                        ),
                        flush=True,
                    )
                    n_render += 1
                    if verifier is not None:
                        rec["verify"] = _verify_goal(sim, verifier, g, cat_positions, imageio, args)
                    manifest.append(rec)
                    continue
                if args.camera_profile == "stretch512":
                    img, extra = _render_stretch512_goal(sim, cam, g, args)
                    fname = f"{eid}__g{g['goal_index']}_{g['object_category']}.png"
                    fpath = os.path.join(args.out_dir, fname)
                    if args.out_size:
                        img = _to_square_size(img, int(args.out_size))
                    imageio.imwrite(fpath, img)
                    rec = {
                        "episode_id": eid,
                        "goal_index": g["goal_index"],
                        "object_category": g["object_category"],
                        "image": fpath,
                        "out_size": int(args.out_size) if args.out_size else None,
                        "camera": {k: cam[k] for k in cam if k != "with_depth"},
                        "look_at_object": True,
                        "look_at_pitch": True,
                        "square_crop": True,
                        "in_frame": bool(extra.get("centroid_in", True)),
                        "crop_box": None,
                        "depth_m": (extra.get("wall") or {}).get("object_depth_m"),
                        "standoff_m": extra.get("standoff_m"),
                        "agent_pos": extra.get("agent_pos"),
                        "wall": extra.get("wall"),
                        "issues": extra.get("issues") or [],
                        "score": extra.get("score"),
                        "rescued": bool(extra.get("rescued")),
                        "rescue_attempted": bool(extra.get("rescue_attempted")),
                        "n_eval": extra.get("n_eval"),
                        "source": extra.get("source"),
                        "orbit_radius": extra.get("orbit_radius"),
                        "orbit_theta": extra.get("orbit_theta"),
                    }
                    print(
                        "[stretch512] %s g%s %s issues=%s score=%.1f rescued=%s n_eval=%s"
                        % (
                            eid,
                            g["goal_index"],
                            g["object_category"],
                            extra.get("issues"),
                            float(extra.get("score") or 0.0),
                            extra.get("rescued"),
                            extra.get("n_eval"),
                        ),
                        flush=True,
                    )
                    n_render += 1
                    if verifier is not None:
                        rec["verify"] = _verify_goal(sim, verifier, g, cat_positions, imageio, args)
                    manifest.append(rec)
                    continue
                vp = g["iin_viewpoint"]
                camera = dict(cam)
                agent_pos = vp["position"]
                if args.standoff_m > 0 and g.get("position") is not None:
                    ladder = (2.4, 3.0) if args.camera_profile == "stretch512" else ()
                    agent_pos = _standoff_agent(
                        sim.pathfinder,
                        agent_pos,
                        g["position"],
                        float(args.standoff_m),
                        ladder=ladder,
                    )
                rotation = vp.get("rotation")
                yaw = None
                use_pitch = bool(args.look_at_pitch) or args.camera_profile == "stretch512"
                keep_square = bool(args.square_crop) or args.camera_profile == "stretch512"
                if args.look_at_object and g.get("position") is not None:
                    sensor = _sensor_pos(agent_pos, camera["height_m"])
                    if use_pitch:
                        rotation = _look_at_xyzw(sensor, g["position"])
                    else:
                        yaw = _yaw_look_at(sensor, g["position"])
                        rotation = _quat_xyzw_from_yaw(yaw)
                obs = _observe_at(sim, agent_pos, rotation)
                img = np.asarray(obs["rgb"])[:, :, :3]
                depth_obs = obs.get("depth")
                uvz = None
                in_frame = True
                crop_box = None
                do_crop = bool(args.crop_object) and args.camera_profile != "stretch512"
                if do_crop and g.get("position") is not None:
                    sensor = _sensor_pos(agent_pos, camera["height_m"])
                    if use_pitch and rotation is not None:
                        hits = _project_points([g["position"]], sensor, rotation, camera)
                        uvz = hits[0] if hits else None
                    else:
                        if yaw is None and rotation is not None:
                            q = np.asarray(rotation, dtype=np.float64)
                            yaw = 2.0 * math.atan2(float(q[1]), float(q[3]))
                        elif yaw is None:
                            yaw = _yaw_look_at(sensor, g["position"])
                        uvz = _project_object(sensor, yaw, g["position"], camera)
                    img, crop_box, in_frame = _crop_around(
                        img, uvz, camera, keep_square=keep_square
                    )
                fname = f"{eid}__g{g['goal_index']}_{g['object_category']}.png"
                fpath = os.path.join(args.out_dir, fname)
                obj_depth = None
                if g.get("position") is not None:
                    sensor = _sensor_pos(agent_pos, camera["height_m"])
                    obj_depth = float(
                        np.linalg.norm(
                            np.asarray(g["position"], dtype=np.float64) - sensor
                        )
                    )
                wall = _wall_facing_stats(img, depth_obs, obj_depth)
                if args.out_size:
                    img = _to_square_size(img, int(args.out_size))
                imageio.imwrite(fpath, img)
                rec = {
                    "episode_id": eid,
                    "goal_index": g["goal_index"],
                    "object_category": g["object_category"],
                    "image": fpath,
                    "out_size": int(args.out_size) if args.out_size else None,
                    "camera": {k: camera[k] for k in camera if k != "with_depth"},
                    "look_at_object": bool(args.look_at_object),
                    "look_at_pitch": bool(use_pitch),
                    "square_crop": bool(keep_square),
                    "in_frame": bool(in_frame),
                    "crop_box": crop_box,
                    "depth_m": None if uvz is None else float(uvz[2]),
                    "standoff_m": float(args.standoff_m),
                    "agent_pos": [float(x) for x in np.asarray(agent_pos).tolist()],
                    "wall": wall,
                }
                n_render += 1

                if verifier is not None:
                    rec["verify"] = _verify_goal(sim, verifier, g, cat_positions, imageio, args)
                manifest.append(rec)
        finally:
            sim.close()
        print(f"[render] {scene_id}: {len(items)} IIN goals rendered")

    man_name = "manifest.json"
    if args.n_shards > 1:
        man_name = "manifest.shard%02d.json" % args.shard_index
    out_manifest = os.path.join(args.out_dir, man_name)
    with open(out_manifest, "w") as f:
        json.dump({"n_rendered": n_render, "items": manifest}, f, indent=2)
    print(f"[render] wrote {n_render} goal images + manifest -> {out_manifest}")
    return {"n_rendered": n_render, "manifest": out_manifest}


def _verify_goal(sim, verifier, g, cat_positions, imageio, args) -> dict:
    """Render a positive (offset pose, same instance) and a negative (other
    instance of the same category) view, then verify against the goal image."""
    vp = g["iin_viewpoint"]
    goal_img = _render_at(sim, vp["position"], vp["rotation"])

    # positive: small lateral offset, same instance -> should verify SAME
    pos = np.asarray(vp["position"], dtype=np.float32) + np.array([0.25, 0.0, 0.25], np.float32)
    snapped = sim.pathfinder.snap_point(pos)
    if not np.all(np.isfinite(np.asarray(snapped))):
        snapped = vp["position"]
    pos_img = _render_at(sim, snapped, vp["rotation"])
    v_pos = verifier.verify(pos_img, goal_img)

    out = {"positive": {"inliers": v_pos.n_inliers, "is_same": v_pos.is_same, "score": v_pos.score}}

    # negative: a DIFFERENT physical instance of the same category (distinct
    # object_id) — same goal entry or same object_id would trivially match.
    others = [o for o in cat_positions[g["object_category"]]
              if o.get("object_id") != g.get("object_id")]
    if others:
        ovp = others[0]["iin_viewpoint"]
        neg_img = _render_at(sim, ovp["position"], ovp["rotation"])
        v_neg = verifier.verify(neg_img, goal_img)
        out["negative"] = {"inliers": v_neg.n_inliers, "is_same": v_neg.is_same, "score": v_neg.score}
    return out


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Render IIN goal images from iin_viewpoint")
    p.add_argument("--episodes", required=True, help="stratified/candidate episode json")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--scenes-dir", default="data/scene_datasets/hm3d-1.0")
    p.add_argument("--limit", type=int, default=0, help="0 = all episodes")
    p.add_argument("--gpu-device-id", type=int, default=0)
    p.add_argument("--verify", action="store_true", help="run render->verify integration check")
    p.add_argument("--verify-backend", default="sift", choices=["sift", "lightglue"])
    p.add_argument(
        "--camera-profile",
        default="episode",
        choices=["episode", "locobot", "stretch", "stretch512", "official"],
        help="episode=iin_viewpoint.camera; stretch=Stretch 360x640; "
        "stretch512=same HFoV/height at 512x512 + pitch look-at; official=IIN issuer",
    )
    p.add_argument(
        "--look-at-object",
        action="store_true",
        help="aim the camera at goal.position (yaw only unless --look-at-pitch / stretch512)",
    )
    p.add_argument(
        "--look-at-pitch",
        action="store_true",
        help="yaw+pitch look-at (implied by stretch512). Needed for floor objects.",
    )
    p.add_argument(
        "--crop-object",
        action="store_true",
        help="crop the render around the projected object center",
    )
    p.add_argument(
        "--square-crop",
        action="store_true",
        help="keep the object crop square by shifting into frame (implied by stretch512)",
    )
    p.add_argument(
        "--standoff-m",
        type=float,
        default=0.0,
        help="pull the camera back to this floor distance from the object",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="do not re-render PNGs that already exist in --out-dir",
    )
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument(
        "--out-size",
        type=int,
        default=0,
        help="if >0, cover-crop/resize the saved PNG to size×size (no black bars)",
    )
    p.add_argument(
        "--only-goals",
        default="",
        help="JSON with keys=[[episode_id, goal_index], ...] to render a subset",
    )
    p.add_argument(
        "--rescue",
        dest="rescue",
        action="store_true",
        default=True,
        help="stretch512: orbit-search if the default view is a known failure",
    )
    p.add_argument(
        "--no-rescue",
        dest="rescue",
        action="store_false",
        help="disable stretch512 orbit rescue",
    )
    return p


def main(argv: Optional[list] = None) -> None:
    render(build_argparser().parse_args(argv))


if __name__ == "__main__":
    main()
