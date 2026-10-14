"""Freeze official val (n=1000) and mini (n=100 subset) from a generated pool.

Sampling is scene-stratified; K and M stay as drawn in the pool
(Unif{2,3,4} × Unif{3,4,5}). Rejects a seed if any (K,M) cell has fewer than
``--min-cell`` episodes or any used scene is empty.

Mini can be re-drawn from a frozen val with a 9-cell KM quota so the cheap
audit set matches Val's (K,M) mix. Default quota is Val-rounded
``13,12,10,13,10,10,12,10,10`` (K2M3 … K4M5).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import config

KM_CELLS = ((2, 3), (2, 4), (2, 5), (3, 3), (3, 4), (3, 5), (4, 3), (4, 4), (4, 5))
# Rounded Val samefloor seed1123 mix (351/334/315 × M).
DEFAULT_MINI_KM_QUOTA = (13, 12, 10, 13, 10, 10, 12, 10, 10)


def _km(ep: dict) -> Tuple[int, int]:
    return int(ep["num_agents"]), int(len(ep.get("goals_set") or []))


def _scene(ep: dict) -> str:
    return str(ep.get("scene_key") or ep.get("episode_id", "").rsplit("_", 1)[0])


def _floor_ys(ep: dict) -> List[float]:
    ys = [float(a["start_position"][1]) for a in (ep.get("agents") or []) if a.get("start_position")]
    for g in ep.get("goals_set") or []:
        vp = (g.get("iin_viewpoint") or {}).get("position") or g.get("rep_point")
        if vp:
            ys.append(float(vp[1]))
        elif g.get("position"):
            ys.append(float(g["position"][1]))
    return ys


def usability_errors(ep: dict, require_same_floor: bool = False) -> List[str]:
    errs = []
    k = int(ep.get("num_agents") or 0)
    agents = ep.get("agents") or []
    goals = ep.get("goals_set") or []
    if k < 2 or k != len(agents):
        errs.append("bad_agents")
    if len(goals) < 3:
        errs.append("too_few_goals")
    cats = [g.get("object_category") for g in goals]
    if len(set(cats)) != len(cats):
        errs.append("dup_category")
    for i, g in enumerate(goals):
        if not (g.get("reach_points") or g.get("rep_point") or g.get("position")):
            errs.append("goal_%d_no_reach" % i)
        if str(g.get("modality", "ON")).upper() == "IIN" and not g.get("iin_viewpoint"):
            errs.append("goal_%d_iin_no_vp" % i)
    sg = np.asarray((ep.get("routing") or {}).get("start_goal_geo") or [], dtype=np.float64)
    if sg.size == 0 or sg.shape != (k, len(goals)):
        errs.append("bad_start_goal")
    elif not np.all(np.isfinite(sg).any(axis=0)):
        errs.append("not_joint_reachable")
    if not ep.get("scene_id"):
        errs.append("no_scene")
    if ep.get("success_distance") is None:
        errs.append("no_success_distance")
    if require_same_floor and not config.heights_same_floor(_floor_ys(ep), config.FLOOR_HEIGHT_M):
        errs.append("not_same_floor")
    return errs


def filter_usable(episodes: Sequence[dict], require_same_floor: bool = False) -> Tuple[List[dict], int]:
    keep = []
    n_bad = 0
    for ep in episodes:
        if usability_errors(ep, require_same_floor=require_same_floor):
            n_bad += 1
        else:
            keep.append(ep)
    return keep, n_bad


def histogram(episodes: Sequence[dict]) -> dict:
    km = Counter(_km(e) for e in episodes)
    scenes = Counter(_scene(e) for e in episodes)
    return {
        "n": len(episodes),
        "by_K": dict(Counter(int(e["num_agents"]) for e in episodes)),
        "by_M": dict(Counter(len(e.get("goals_set") or []) for e in episodes)),
        "by_KM": {f"K{k}-M{m}": n for (k, m), n in sorted(km.items())},
        "n_scenes": len(scenes),
        "per_scene_min": min(scenes.values()) if scenes else 0,
        "per_scene_max": max(scenes.values()) if scenes else 0,
        "empty_km_cells": [
            f"K{k}-M{m}"
            for k in (2, 3, 4)
            for m in (3, 4, 5)
            if km.get((k, m), 0) == 0
        ],
        "min_cell": min((km.get((k, m), 0) for k in (2, 3, 4) for m in (3, 4, 5)), default=0),
    }


def _take_per_scene(pool: Sequence[dict], n: int, rng: np.random.Generator) -> List[dict]:
    by_s: Dict[str, List[dict]] = defaultdict(list)
    for ep in pool:
        by_s[_scene(ep)].append(ep)
    scenes = sorted(by_s)
    for s in scenes:
        rng.shuffle(by_s[s])
    if not scenes:
        return []
    base, rem = divmod(n, len(scenes))
    quota = {s: base + (1 if i < rem else 0) for i, s in enumerate(scenes)}
    chosen: List[dict] = []
    leftover: List[dict] = []
    for s in scenes:
        take = min(quota[s], len(by_s[s]))
        chosen.extend(by_s[s][:take])
        leftover.extend(by_s[s][take:])
    need = n - len(chosen)
    if need > 0:
        rng.shuffle(leftover)
        chosen.extend(leftover[:need])
    rng.shuffle(chosen)
    return chosen[:n]


def parse_km_quota(spec: str) -> Dict[Tuple[int, int], int]:
    parts = [int(x.strip()) for x in (spec or "").split(",") if x.strip()]
    if len(parts) != len(KM_CELLS):
        raise ValueError("mini KM quota must be 9 ints K2M3..K4M5, got %r" % spec)
    if any(n < 0 for n in parts):
        raise ValueError("mini KM quota must be non-negative")
    return {km: n for km, n in zip(KM_CELLS, parts)}


def _pick_km(
    avail_s: Dict[Tuple[int, int], List[dict]],
    rem_km: Dict[Tuple[int, int], int],
    rng: np.random.Generator,
) -> Optional[Tuple[int, int]]:
    cands = [km for km, eps in avail_s.items() if eps and rem_km.get(km, 0) > 0]
    if not cands:
        return None
    weights = np.asarray([rem_km[km] for km in cands], dtype=np.float64)
    weights = weights / weights.sum()
    return cands[int(rng.choice(len(cands), p=weights))]


def _take_km_quota(
    pool: Sequence[dict],
    quota: Dict[Tuple[int, int], int],
    rng: np.random.Generator,
    *,
    min_per_scene: int = 2,
    max_per_scene: int = 3,
    max_tries: int = 20000,
) -> List[dict]:
    """Scene-covered subset with exact (K,M) counts. Mini ⊂ pool."""
    n = int(sum(quota.values()))
    by_s: Dict[str, Dict[Tuple[int, int], List[dict]]] = defaultdict(lambda: defaultdict(list))
    for ep in pool:
        by_s[_scene(ep)][_km(ep)].append(ep)
    scenes = sorted(by_s)
    if not scenes:
        raise SystemExit("empty pool for KM quota")
    need_min = min_per_scene * len(scenes)
    if n < need_min:
        raise SystemExit("n=%d < min_per_scene*%d=%d" % (n, len(scenes), need_min))
    if n > max_per_scene * len(scenes):
        raise SystemExit("n=%d > max_per_scene*%d=%d" % (n, len(scenes), max_per_scene * len(scenes)))

    for _attempt in range(max_tries):
        avail = {
            s: {km: list(eps) for km, eps in by_s[s].items()}
            for s in scenes
        }
        for s in scenes:
            for km in list(avail[s]):
                rng.shuffle(avail[s][km])
                if not avail[s][km]:
                    del avail[s][km]
        rem_km = {km: int(quota.get(km, 0)) for km in KM_CELLS}
        scount = {s: 0 for s in scenes}
        chosen: List[dict] = []
        failed = False

        order = list(scenes)
        rng.shuffle(order)
        for s in order:
            for _ in range(min_per_scene):
                km = _pick_km(avail[s], rem_km, rng)
                if km is None:
                    failed = True
                    break
                chosen.append(avail[s][km].pop())
                if not avail[s][km]:
                    del avail[s][km]
                rem_km[km] -= 1
                scount[s] += 1
            if failed:
                break
        if failed:
            continue

        while sum(rem_km.values()) > 0:
            need = [km for km in KM_CELLS if rem_km[km] > 0]

            def _slots(km: Tuple[int, int]) -> int:
                return sum(
                    len(avail[s].get(km, []))
                    for s in scenes
                    if scount[s] < max_per_scene
                )

            need.sort(key=_slots)
            placed = False
            for km in need:
                cands = [
                    s for s in scenes
                    if scount[s] < max_per_scene and avail[s].get(km)
                ]
                if not cands:
                    continue
                rng.shuffle(cands)
                cands.sort(key=lambda s: (scount[s], -len(avail[s].get(km, []))))
                s = cands[0]
                chosen.append(avail[s][km].pop())
                if not avail[s][km]:
                    del avail[s][km]
                rem_km[km] -= 1
                scount[s] += 1
                placed = True
                break
            if not placed:
                failed = True
                break
        if failed:
            continue

        if len(chosen) != n or min(scount.values()) < min_per_scene:
            continue
        rng.shuffle(chosen)
        return chosen

    raise SystemExit(
        "could not draw KM-quota mini in %d tries (quota=%s scenes=%d)"
        % (max_tries, quota, len(scenes))
    )


def accept(hist: dict, min_cell: int, n_scenes_expect: int = 0) -> List[str]:
    reasons = []
    if hist["min_cell"] < min_cell:
        reasons.append("min_cell=%d < %d" % (hist["min_cell"], min_cell))
    if hist["empty_km_cells"]:
        reasons.append("empty " + ",".join(hist["empty_km_cells"]))
    if n_scenes_expect and hist["n_scenes"] < n_scenes_expect:
        reasons.append("scenes=%d < %d" % (hist["n_scenes"], n_scenes_expect))
    if hist["per_scene_min"] < 1:
        reasons.append("a scene has 0 episodes")
    return reasons


def _draw_mini(val: Sequence[dict], args, quota) -> List[dict]:
    mini_rng = np.random.default_rng(args.seed + 1)
    if quota is None:
        return _take_per_scene(val, args.n_mini, mini_rng)
    if int(sum(quota.values())) != int(args.n_mini):
        raise SystemExit("mini KM quota sum %d != n_mini %d" % (sum(quota.values()), args.n_mini))
    return _take_km_quota(
        val,
        quota,
        mini_rng,
        min_per_scene=args.min_per_scene,
        max_per_scene=args.max_per_scene,
        max_tries=args.quota_tries,
    )


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pool", default=None)
    p.add_argument("--val-in", default=None, help="Frozen val JSON; resample mini only")
    p.add_argument("--val-out", default=None)
    p.add_argument("--mini-out", required=True)
    p.add_argument("--n-val", type=int, default=1000)
    p.add_argument("--n-mini", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-cell", type=int, default=70)
    p.add_argument("--min-scenes", type=int, default=36)
    p.add_argument("--require-same-floor", action="store_true")
    p.add_argument(
        "--mini-km-quota",
        default=None,
        help="9 ints K2M3..K4M5 (default off). Example: 13,12,10,13,10,10,12,10,10",
    )
    p.add_argument("--min-per-scene", type=int, default=2)
    p.add_argument("--max-per-scene", type=int, default=3)
    p.add_argument("--quota-tries", type=int, default=20000)
    args = p.parse_args(argv)

    quota = parse_km_quota(args.mini_km_quota) if args.mini_km_quota else None

    def _dump(path, episodes, extra, source):
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        meta = {
            "generator": "benchmark.sample_official",
            "source_pool": source,
            "seed": args.seed,
            "k_dist": "Unif{2,3,4}",
            "m_dist": "Unif{3,4,5}",
            "n_episodes": len(episodes),
            "histogram": extra,
            "mini_km_quota": (
                {("K%d-M%d" % km): n for km, n in quota.items()} if quota else None
            ),
            "protocol": {
                "max_env_step": 500,
                "inter_agent": "invisible_penetrable",
                "frozen": True,
                "same_floor": bool(args.require_same_floor),
                "floor_height_m": config.FLOOR_HEIGHT_M,
                "mini_sample": "km_quota" if quota else "scene_stratified",
            },
        }
        with open(path, "w") as f:
            json.dump({"meta": meta, "episodes": episodes}, f)
        print("[official] wrote", path, "n=", len(episodes))

    if args.val_in:
        raw = json.load(open(args.val_in))
        val = raw["episodes"] if isinstance(raw, dict) else raw
        usable, n_bad = filter_usable(val, require_same_floor=args.require_same_floor)
        print("[official] val-in=%d usable=%d dropped=%d" % (len(val), len(usable), n_bad))
        if n_bad:
            raise SystemExit("frozen val has %d unusable episodes" % n_bad)
        if args.n_val and len(usable) != args.n_val:
            print("[official] warn val n=%d != --n-val %d" % (len(usable), args.n_val))
        mini = _draw_mini(usable, args, quota)
        mini_hist = histogram(mini)
        _dump(args.mini_out, mini, mini_hist, args.val_in)
        print("[official] mini", json.dumps(mini_hist))
        return

    if not args.pool or not args.val_out:
        raise SystemExit("need --pool and --val-out, or --val-in to resample mini")

    raw = json.load(open(args.pool))
    pool = raw["episodes"] if isinstance(raw, dict) else raw
    usable, n_bad = filter_usable(pool, require_same_floor=args.require_same_floor)
    print("[official] pool=%d usable=%d dropped=%d" % (len(pool), len(usable), n_bad))
    if len(usable) < args.n_val:
        raise SystemExit("usable pool %d < n_val %d" % (len(usable), args.n_val))

    rng = np.random.default_rng(args.seed)
    val = _take_per_scene(usable, args.n_val, rng)
    hist = histogram(val)
    reasons = accept(hist, args.min_cell, args.min_scenes)
    if reasons:
        raise SystemExit("val reject: " + "; ".join(reasons) + " hist=" + json.dumps(hist))
    mini = _draw_mini(val, args, quota)
    mini_hist = histogram(mini)

    _dump(args.val_out, val, hist, args.pool)
    _dump(args.mini_out, mini, mini_hist, args.pool)
    print("[official] val", json.dumps(hist))
    print("[official] mini", json.dumps(mini_hist))


if __name__ == "__main__":
    main()
