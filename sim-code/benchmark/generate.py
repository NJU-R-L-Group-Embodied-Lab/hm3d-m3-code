"""P1 episode generator: distributed starts + joint reachability + difficulty labels.

For each HM3D scene we compose multi-object goal-sets of *distinct* categories
(one real instance per category), tag each goal with a modality (ON/IIN/TN —
IIN/TN content rendered in P2), sample K agent start points with a controlled
geodesic spacing, and verify *joint reachability* (every goal reachable by at
least one agent). We then label difficulty (spread / spacing / modality mix /
allocation-critical) and emit a candidate-episode pool, from which
:mod:`benchmark.sample_official` draws the released splits.

Run inside the lavira container (habitat env)::

    python -m benchmark.generate --episodes-per-scene 20 --num-agents 2 \
        --out data/datasets/ma_objectnav/val/candidates.json

Use ``--dry-scenes N`` for a quick smoke run over the first N scenes.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import config, routing
from .goal_sources import GoalInstance, SceneGoals, iter_scene_files, load_scene_goals


# --------------------------------------------------------------------------
# Standalone navmesh geometry (HabitatNav / BaseNav)
#
# Standalone navmesh geometry for P1 — no Simulator, no GPU, no sensors.
#
# Habitat-sim 0.1.7 exposes ``PathFinder.load_nav_mesh(path)``, which loads a
# scene's ``.navmesh`` directly and answers every geometric query P1 needs
# (geodesic distance, snap-to-navmesh, navigability, island radius, random
# navigable points). Rendering goal images for IIN is a Simulator concern and is
# left to P2.
#
# This module also defines :class:`BaseNav`, the minimal interface the rest of the
# pipeline (routing / generation) depends on, so :mod:`tests.selftests.benchmark_selftest` can
# substitute a pure-Python mock and exercise everything without Habitat or data.
# --------------------------------------------------------------------------
Point = Sequence[float]


class BaseNav:
    """Geometry interface used by routing/generation (see HabitatNav / MockNav)."""

    navigable_area: float

    def geodesic(self, a: Point, b: Point) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def shortest_path(self, a: Point, b: Point) -> Tuple[float, list]:
        """Geodesic distance and the path polyline points from ``a`` to ``b``.

        Default (used by mocks): straight segment ``[a, b]`` with the geodesic
        length. ``HabitatNav`` overrides with the real navmesh polyline.
        """
        d = self.geodesic(a, b)
        return d, [np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)]

    def geodesic_to_set(self, a: Point, pts: Sequence[Point]) -> Tuple[float, int]:
        """Min geodesic from ``a`` to any point in ``pts``; returns (dist, index).

        index is -1 when ``pts`` is empty or none are reachable.
        """
        best_d, best_i = INF, -1
        for i, p in enumerate(pts):
            d = self.geodesic(a, p)
            if d < best_d:
                best_d, best_i = d, i
        return best_d, best_i

    def snap(self, p: Point) -> Optional[np.ndarray]:  # pragma: no cover - interface
        raise NotImplementedError

    def is_navigable(self, p: Point) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def island_radius(self, p: Point) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def random_point(self) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class HabitatNav(BaseNav):
    """Wraps ``habitat_sim.PathFinder`` loaded from a scene ``.navmesh`` file."""

    def __init__(self, navmesh_path: str, seed: int = 0):
        import habitat_sim  # imported lazily so the module is importable without habitat

        self._hs = habitat_sim
        self.pf = habitat_sim.PathFinder()
        self.pf.load_nav_mesh(navmesh_path)
        if not self.pf.is_loaded:
            raise RuntimeError(f"navmesh failed to load: {navmesh_path}")
        self.pf.seed(int(seed))
        self.navigable_area = float(self.pf.navigable_area)

    def geodesic(self, a: Point, b: Point) -> float:
        sp = self._hs.ShortestPath()
        sp.requested_start = np.asarray(a, dtype=np.float32)
        sp.requested_end = np.asarray(b, dtype=np.float32)
        if not self.pf.find_path(sp):
            return INF
        d = float(sp.geodesic_distance)
        return d if math.isfinite(d) else INF

    def shortest_path(self, a: Point, b: Point) -> Tuple[float, list]:
        sp = self._hs.ShortestPath()
        sp.requested_start = np.asarray(a, dtype=np.float32)
        sp.requested_end = np.asarray(b, dtype=np.float32)
        if not self.pf.find_path(sp):
            return INF, []
        d = float(sp.geodesic_distance)
        pts = [np.asarray(p, dtype=np.float32) for p in sp.points]
        return (d if math.isfinite(d) else INF), pts

    def snap(self, p: Point) -> Optional[np.ndarray]:
        q = self.pf.snap_point(np.asarray(p, dtype=np.float32))
        q = np.asarray(q, dtype=np.float32)
        if not np.all(np.isfinite(q)):
            return None
        return q

    def is_navigable(self, p: Point) -> bool:
        return bool(self.pf.is_navigable(np.asarray(p, dtype=np.float32)))

    def island_radius(self, p: Point) -> float:
        return float(self.pf.island_radius(np.asarray(p, dtype=np.float32)))

    def random_point(self) -> np.ndarray:
        return np.asarray(self.pf.get_random_navigable_point(), dtype=np.float32)

INF = float("inf")

SPACING_WINDOWS = {
    "any": (0.0, INF),
    "coloc": (0.0, 1.0),
    "near": (1.0, 5.0),
    "far": (5.0, INF),
}


# ----------------------------------------------------------------------------- geometry helpers


def select_goal_geometry(nav, goal: GoalInstance):
    """Resolve a goal to navigable points.

    Returns a dict with the routing representative (highest-IoU navigable
    viewpoint), the full navigable reach-set (for the runner's success check),
    and the chosen viewpoint pose for P2 IIN rendering. ``None`` if no navigable
    viewpoint and the object position itself is off-navmesh.
    """
    vps = sorted(goal.view_points, key=lambda v: v.get("iou", 0.0), reverse=True)
    reach: List[List[float]] = []
    iin_vp = None
    for vp in vps:
        pos = vp.get("position")
        if pos is None:
            continue
        snapped = nav.snap(pos)
        if snapped is None or not nav.is_navigable(snapped):
            continue
        if iin_vp is None:
            # Full camera spec for P2 rendering, aligned to HM3D-IIN: the floor
            # pose (position+yaw) plus the calibrated pinhole intrinsics; the
            # render camera sits at height_m above ``position``.
            iin_vp = {
                "position": [float(x) for x in snapped],
                "rotation": vp.get("rotation"),
                "iou": vp.get("iou"),
                "camera": dict(config.IIN_CAMERA),
            }
        reach.append([float(x) for x in snapped])
        if len(reach) >= config.MAX_VIEWPOINTS_PER_GOAL:
            break

    if not reach:
        snapped = nav.snap(goal.position)
        if snapped is None or not nav.is_navigable(snapped):
            return None
        reach = [[float(x) for x in snapped]]

    return {
        "rep": reach[0],            # routing point
        "reach_points": reach,      # success set
        "iin_viewpoint": iin_vp,    # P2 rendering source
    }


def same_class_distractors(nav, scene: SceneGoals, category: str, chosen: GoalInstance,
                          floor_y=None, floor_h=None):
    """Other navigable instances of the *same* category (MultiON 2.0 distractors).

    For IIN/TN goals these are confounders the agent must not stop at; recorded so
    the runner's verification gate and the false-positive metric can use them.
    """
    out = []
    for inst in scene.by_category.get(category, []):
        if inst.object_id == chosen.object_id:
            continue
        snapped = nav.snap(inst.position)
        if snapped is None or not nav.is_navigable(snapped):
            continue
        if floor_y is not None and floor_h is not None:
            if abs(float(snapped[1]) - float(floor_y)) >= float(floor_h):
                continue
        out.append({"object_id": inst.object_id,
                    "position": [float(x) for x in inst.position]})
        if len(out) >= config.MAX_DISTRACTORS_PER_GOAL:
            break
    return out


def category_reach_points(nav, scene: SceneGoals, category: str,
                         floor_y=None, floor_h=None) -> List[List[float]]:
    """Union navigable viewpoints for category-level ObjectNav success."""

    points: List[List[float]] = []
    seen = set()
    for inst in scene.by_category.get(category, []):
        geom = select_goal_geometry(nav, inst)
        if geom is None:
            continue
        for point in geom["reach_points"]:
            if floor_y is not None and floor_h is not None:
                if abs(float(point[1]) - float(floor_y)) >= float(floor_h):
                    continue
            key = tuple(round(float(x), 4) for x in point)
            if key in seen:
                continue
            seen.add(key)
            points.append([float(x) for x in point])
    return points


def _yaw_quat(rng) -> List[float]:
    a = float(rng.uniform(0.0, 2.0 * math.pi))
    return [0.0, math.sin(a / 2.0), 0.0, math.cos(a / 2.0)]


def sample_distributed_starts(nav, K: int, window, rng, attempts: int,
                             y_center=None, y_tol=None):
    """Rejection-sample K navigable starts whose pairwise geodesics lie in ``window``.

    ``y_center`` / ``y_tol`` (metres) keep every start on one floor band.
    """
    lo, hi = window
    starts: List[np.ndarray] = []
    for _ in range(attempts):
        if len(starts) == K:
            break
        p = nav.random_point()
        if nav.island_radius(p) < config.START_MIN_ISLAND_RADIUS:
            continue
        if y_center is not None and y_tol is not None:
            if abs(float(p[1]) - float(y_center)) >= float(y_tol):
                continue
        ok = True
        for q in starts:
            d = nav.geodesic(p, q)
            if not (lo <= d <= hi) or not math.isfinite(d):
                ok = False
                break
        if ok:
            starts.append(p)
    if len(starts) < K:
        return None
    return starts


def _geom_floor_y(geom) -> Optional[float]:
    if not geom:
        return None
    rep = geom.get("rep")
    if rep is None:
        return None
    return float(rep[1])


def build_floor_catalog(nav, scene: SceneGoals):
    """Per-category navigable instances tagged by viewpoint / snap height."""
    catalog = {}
    for cat, insts in scene.by_category.items():
        rows = []
        for inst in insts:
            geom = select_goal_geometry(nav, inst)
            fy = _geom_floor_y(geom)
            if geom is None or fy is None:
                continue
            rows.append((inst, geom, fy))
        if rows:
            catalog[cat] = rows
    return catalog


def _floor_centroids(ys: Sequence[float], h: float) -> List[float]:
    cents: List[float] = []
    for y in sorted(float(v) for v in ys):
        if not cents or (y - cents[-1]) >= h:
            cents.append(y)
    return cents


def floors_with_enough_cats(catalog, M: int, h: float):
    """Floors that have at least ``M`` distinct categories with a viewpoint."""
    all_y = [fy for rows in catalog.values() for *_, fy in rows]
    if not all_y:
        return []
    cents = _floor_centroids(all_y, h)
    out = []
    for c in cents:
        by_cat = {}
        for cat, rows in catalog.items():
            keep = [(inst, geom, fy) for inst, geom, fy in rows if abs(fy - c) < h]
            if keep:
                by_cat[cat] = keep
        if len(by_cat) >= M:
            out.append((c, by_cat))
    return out


def assign_modalities(M: int, mode: str, rng) -> List[str]:
    if mode == "all_on":
        return ["ON"] * M
    if mode == "all_alt":
        return [str(rng.choice(["IIN", "TN"])) for _ in range(M)]
    if mode == "cycle":
        return [config.MODALITIES[i % len(config.MODALITIES)] for i in range(M)]
    if mode == "mixed3":
        if M < 3:
            raise ValueError("mixed3 requires M>=3, got %d" % M)
        tags = ["ON", "IIN", "TN"] + [
            str(rng.choice(config.MODALITIES)) for _ in range(M - 3)
        ]
        rng.shuffle(tags)
        return tags
    # "mixed": random tags, but guarantee at least one ON and one non-ON when M>=2
    tags = [str(rng.choice(config.MODALITIES)) for _ in range(M)]
    if M >= 2:
        if "ON" not in tags:
            tags[int(rng.integers(M))] = "ON"
        if all(t == "ON" for t in tags):
            idx = int(rng.integers(M))
            tags[idx] = str(rng.choice(["IIN", "TN"]))
    return tags


def start_start_matrix(nav, starts: Sequence) -> List[List[float]]:
    K = len(starts)
    ss = [[0.0] * K for _ in range(K)]
    for i in range(K):
        for j in range(i + 1, K):
            d = nav.geodesic(starts[i], starts[j])
            ss[i][j] = ss[j][i] = (d if math.isfinite(d) else None)
    return ss


# ----------------------------------------------------------------------------- episode assembly


def try_make_episode(nav, scene: SceneGoals, *, K: int, M: int, modality_mode: str,
                     spacing_mode: str, success_distance: float, rng, max_tries: int,
                     same_floor: bool = False, catalog=None):
    h = config.FLOOR_HEIGHT_M
    if same_floor:
        catalog = catalog if catalog is not None else build_floor_catalog(nav, scene)
        floors = floors_with_enough_cats(catalog, M, h)
        if not floors:
            return None
    else:
        cats = scene.categories
        if len(cats) < M:
            return None

    for _ in range(max_tries):
        floor_y = None
        if same_floor:
            floor_y, by_cat = floors[int(rng.integers(len(floors)))]
            chosen_cats = list(rng.choice(list(by_cat.keys()), size=M, replace=False))
        else:
            chosen_cats = list(rng.choice(scene.categories, size=M, replace=False))

        goals_meta = []
        rep_pts = []
        ok = True
        for c in chosen_cats:
            if same_floor:
                inst, geom, _fy = by_cat[c][int(rng.integers(len(by_cat[c])))]
                geom = dict(geom)
            else:
                inst = scene.by_category[c][int(rng.integers(len(scene.by_category[c])))]
                geom = select_goal_geometry(nav, inst)
                if geom is None:
                    ok = False
                    break
            geom["distractors"] = same_class_distractors(
                nav, scene, c, inst, floor_y=floor_y, floor_h=h if same_floor else None,
            )
            geom["category_reach_points"] = category_reach_points(
                nav, scene, c, floor_y=floor_y, floor_h=h if same_floor else None,
            )
            goals_meta.append((inst, geom))
            rep_pts.append(geom["rep"])
        if not ok:
            continue

        bucket = spacing_mode
        if spacing_mode == "mixed":
            bucket = str(rng.choice(["coloc", "near", "far"]))
        window = SPACING_WINDOWS.get(bucket, SPACING_WINDOWS["any"])
        starts = sample_distributed_starts(
            nav, K, window, rng, config.START_SAMPLE_ATTEMPTS,
            y_center=floor_y, y_tol=h if same_floor else None,
        )
        if starts is None:
            continue

        if same_floor:
            ys = [float(s[1]) for s in starts] + [float(rp[1]) for rp in rep_pts]
            if not config.heights_same_floor(ys, h):
                continue

        # joint reachability: every goal's rep reachable from >= 1 start
        reachable = True
        for rp in rep_pts:
            dmin, _ = min(((nav.geodesic(s, rp), 0) for s in starts), default=(INF, 0))
            if not math.isfinite(dmin):
                reachable = False
                break
        if not reachable:
            continue

        modalities = assign_modalities(M, modality_mode, rng)
        rsum = routing.route_summary(nav, starts, rep_pts)
        if not rsum["all_reachable"]:
            continue
        ss = start_start_matrix(nav, starts)
        start_y = [float(s[1]) for s in starts]
        # Viewpoint / snap height for floor labels (object centroid can sit on a wall).
        goal_y = [float(geom["rep"][1]) for (_inst, geom) in goals_meta]
        diff = config.assemble(
            num_agents=K,
            modality_tags=modalities,
            goal_goal=rsum["goal_goal"],
            start_start=ss,
            navigable_area=nav.navigable_area,
            allocation_critical=rsum["allocation_critical"],
            start_y=start_y,
            goal_y=goal_y,
            start_goal=rsum["start_goal"],
        )
        diff["same_floor"] = bool(same_floor)
        budget_geo = (config.BUDGET_ALPHA * rsum["oracle_makespan"]) if rsum["oracle_makespan"] else None
        return _build_episode(scene, starts, rng, goals_meta, modalities, rsum, ss,
                              diff, budget_geo, success_distance)
    return None


def _build_episode(scene, starts, rng, goals_meta, modalities, rsum, ss, diff, budget_geo, success_distance) -> dict:
    agents = []
    for i, s in enumerate(starts):
        agents.append({
            "agent_id": i,
            "start_position": [float(x) for x in s],
            "start_rotation": _yaw_quat(rng),
        })
    goals_set = []
    for gi, ((inst, geom), modality) in enumerate(zip(goals_meta, modalities)):
        instance_reach = geom["reach_points"]
        evaluation_reach = (
            geom.get("category_reach_points") or instance_reach
            if modality == "ON"
            else instance_reach
        )
        goals_set.append({
            "goal_index": gi,
            "object_category": inst.category,
            "modality": modality,            # ON | IIN | TN  (IIN/TN payload filled in P2)
            "object_id": inst.object_id,
            "object_name": inst.object_name,
            "position": [float(x) for x in inst.position],
            "rep_point": geom["rep"],
            "reach_points": evaluation_reach,
            "instance_reach_points": instance_reach,
            "reach_semantics": "category_union" if modality == "ON" else "target_instance",
            "iin_viewpoint": geom["iin_viewpoint"],   # for P2 image render
            "tn_text": None,                          # filled in P2
            "distractors": geom.get("distractors", []),  # same-class confounders
        })

    stratum = config.stratum_key(diff)
    # Back-compat mirror so a single-goal/any-start loader can still read it.
    ep = {
        "scene_id": scene.scene_id,
        "scene_dataset_config": scene.scene_dataset_config,
        "num_agents": len(starts),
        "agents": agents,
        "goals_set": goals_set,
        "difficulty": diff,
        "stratum": stratum,
        "routing": {
            "oracle_makespan": rsum["oracle_makespan"],
            "oracle_assignment": rsum["oracle_assignment"],
            "oracle_exact": rsum["oracle_exact"],
            "geometric_makespan": rsum["geometric_makespan"],
            "balanced_makespan": rsum["balanced_makespan"],
            "allocation_gap": rsum["allocation_gap"],
            "start_goal_geo": rsum["start_goal"],
            "goal_goal_geo": rsum["goal_goal"],
            "start_start_geo": ss,
        },
        "budget_geo": (round(budget_geo, 3) if budget_geo else None),
        "success_distance": success_distance,
        # --- back-compat mirror (agent 0 + first goal) ---
        "start_position": agents[0]["start_position"],
        "start_rotation": agents[0]["start_rotation"],
        "object_category": goals_set[0]["object_category"],
        "goals": [{"position": g["position"]} for g in goals_set],
    }
    return ep


# ----------------------------------------------------------------------------- driver


def generate(args) -> dict:
    files = iter_scene_files(args.content_dir)
    if args.dry_scenes:
        files = files[: args.dry_scenes]
    elif args.max_scenes:
        files = files[: args.max_scenes]

    m_values = [int(x) for x in str(args.m_values).split(",") if x.strip()]
    k_raw = [int(x) for x in str(getattr(args, "k_values", "") or "").split(",") if x.strip()]
    k_values = k_raw if k_raw else [int(args.num_agents)]
    base_rng = np.random.default_rng(args.seed)

    episodes: List[dict] = []
    n_scene = 0
    eid = 0
    for si, f in enumerate(files):
        scene = load_scene_goals(f, args.scenes_dir)
        if scene is None:
            print(f"[gen] skip (no goals): {os.path.basename(f)}")
            continue
        if not os.path.exists(scene.navmesh_path):
            print(f"[gen] skip (no navmesh): {scene.navmesh_path}")
            continue
        try:
            nav = HabitatNav(scene.navmesh_path, seed=args.seed + si)
        except Exception as e:  # noqa: BLE001
            print(f"[gen] skip (navmesh load failed {e}): {scene.scene_key}")
            continue

        scene_rng = np.random.default_rng(args.seed + 1000 * (si + 1))
        catalog = build_floor_catalog(nav, scene) if getattr(args, "same_floor", False) else None
        made = 0
        for _attempt in range(args.episodes_per_scene):
            K = int(scene_rng.choice(k_values))
            M = int(scene_rng.choice(m_values))
            ep = try_make_episode(
                nav, scene, K=K, M=M, modality_mode=args.modality_mode,
                spacing_mode=args.start_spacing, success_distance=args.success_distance,
                rng=scene_rng, max_tries=args.max_tries,
                same_floor=bool(getattr(args, "same_floor", False)), catalog=catalog,
            )
            if ep is None:
                continue
            ep["episode_id"] = f"{scene.scene_key}_{eid:05d}"
            ep["scene_key"] = scene.scene_key
            episodes.append(ep)
            eid += 1
            made += 1
        n_scene += 1
        print(f"[gen] {scene.scene_key}: {made}/{args.episodes_per_scene} episodes "
              f"(cats={len(scene.categories)}, area={nav.navigable_area:.1f})")

    meta = {
        "generator": "benchmark.generate",
        "version": "p1",
        "seed": args.seed,
        "num_agents": args.num_agents,
        "k_values": k_values,
        "m_values": m_values,
        "modality_mode": args.modality_mode,
        "start_spacing": args.start_spacing,
        "success_distance": args.success_distance,
        "budget_alpha": config.BUDGET_ALPHA,
        "scenes_used": n_scene,
        "n_episodes": len(episodes),
        "same_floor": bool(getattr(args, "same_floor", False)),
        "floor_height_m": config.FLOOR_HEIGHT_M,
    }
    return {"meta": meta, "episodes": episodes}


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="HM3D-M3 episode generator")
    p.add_argument("--content-dir", default="data/datasets/objectnav_hm3d_v2/val/content")
    p.add_argument("--scenes-dir", default="data/scene_datasets/hm3d-1.0")
    p.add_argument("--out", default="data/datasets/ma_objectnav/val/candidates.json")
    p.add_argument("--num-agents", type=int, default=2)
    p.add_argument(
        "--k-values",
        default="",
        help="If set (e.g. 2,3,4), sample K per episode uniformly; else use --num-agents.",
    )
    p.add_argument("--m-values", default="3,4,5")
    p.add_argument("--episodes-per-scene", type=int, default=20)
    p.add_argument("--max-scenes", type=int, default=0, help="0 = all scenes")
    p.add_argument("--dry-scenes", type=int, default=0, help="quick smoke: first N scenes")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--modality-mode",
        default="mixed",
        choices=["all_on", "all_alt", "cycle", "mixed", "mixed3"],
        help="mixed3 = at least one ON, IIN, and TN (requires M>=3).",
    )
    p.add_argument("--start-spacing", default="mixed", choices=["any", "coloc", "near", "far", "mixed"])
    p.add_argument("--success-distance", type=float, default=1.0)
    p.add_argument("--max-tries", type=int, default=80, help="resampling tries per episode")
    p.add_argument(
        "--same-floor",
        action="store_true",
        help="Keep all starts and goal viewpoints in one floor band (FLOOR_HEIGHT_M).",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_argparser().parse_args(argv)
    out = generate(args)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"[gen] wrote {out['meta']['n_episodes']} episodes -> {args.out}")


if __name__ == "__main__":
    main()
