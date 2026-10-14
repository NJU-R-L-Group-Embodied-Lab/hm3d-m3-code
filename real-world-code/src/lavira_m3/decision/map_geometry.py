"""Map extents, world bounds and traversable-cell queries."""

from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

from lavira_m3.profiles import AGENTS

from .config import (
    MAP_EXTENT_PAD_M,
    MAP_GOAL_INSET_M,
    WORLD_HALF_M,
)


def fused_map_aabb(scene, pad_m: float = 0.0) -> Optional[Tuple[float, float, float, float]]:
    mmap = (scene or {}).get("map") if isinstance(scene, dict) else None
    if not isinstance(mmap, dict):
        mmap = scene if isinstance(scene, dict) else {}
    explicit = mmap.get("aabb")
    if isinstance(explicit, (list, tuple)) and len(explicit) >= 4:
        try:
            minx, miny, maxx, maxy = [float(v) for v in explicit[:4]]
        except (TypeError, ValueError):
            minx = None
        else:
            if (
                all(math.isfinite(v) for v in (minx, miny, maxx, maxy))
                and maxx > minx
                and maxy > miny
            ):
                pad = float(pad_m)
                return (minx - pad, miny - pad, maxx + pad, maxy + pad)
    res = float(mmap.get("resolution") or 0.1)
    origin = mmap.get("origin") or [0.0, 0.0]
    ox, oy = float(origin[0]), float(origin[1])
    width, height = mmap.get("width"), mmap.get("height")
    try:
        width = int(width) if width not in (None, "") else 0
        height = int(height) if height not in (None, "") else 0
    except (TypeError, ValueError):
        width = height = 0
    if width > 0 and height > 0 and math.isfinite(res) and res > 0:
        pad = float(pad_m)
        return (ox - pad, oy - pad, ox + width * res + pad, oy + height * res + pad)
    cells = mmap.get("fused") or []
    if not cells:
        return None
    ixs, iys = [], []
    for cell in cells:
        if not isinstance(cell, (list, tuple)) or len(cell) < 2:
            continue
        try:
            ixs.append(int(cell[0]))
            iys.append(int(cell[1]))
        except (TypeError, ValueError):
            continue
    if not ixs:
        return None
    pad = float(pad_m)
    return (
        ox + min(ixs) * res - pad,
        oy + min(iys) * res - pad,
        ox + (max(ixs) + 1) * res + pad,
        oy + (max(iys) + 1) * res + pad,
    )


def planning_map_aabb(scene) -> Optional[Tuple[float, float, float, float]]:
    """Current global observation rectangle, including the unknown border."""
    mmap = (scene or {}).get("map") if isinstance(scene, dict) else None
    if not isinstance(mmap, dict):
        mmap = scene if isinstance(scene, dict) else {}
    explicit = mmap.get("aabb")
    if isinstance(explicit, (list, tuple)) and len(explicit) >= 4:
        return fused_map_aabb(scene, pad_m=0.0)
    try:
        width = int(mmap.get("width") or 0)
        height = int(mmap.get("height") or 0)
    except (TypeError, ValueError):
        width = height = 0
    if width > 0 and height > 0:
        return fused_map_aabb(scene, pad_m=0.0)
    return fused_map_aabb(scene, pad_m=MAP_EXTENT_PAD_M)


def _clip_segment_aabb(x0, y0, x1, y1, aabb):
    minx, miny, maxx, maxy = [float(v) for v in aabb]
    dx, dy = float(x1) - float(x0), float(y1) - float(y0)
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - minx), (dx, maxx - x0), (-dy, y0 - miny), (dy, maxy - y0)):
        if abs(p) <= 1e-12:
            if q < 0:
                return None
            continue
        t = q / p
        if p < 0:
            if t > t1:
                return None
            t0 = max(t0, t)
        else:
            if t < t0:
                return None
            t1 = min(t1, t)
    return t0, t1


def clip_goal_to_map_aabb(
    rx: float,
    ry: float,
    gx: float,
    gy: float,
    aabb: Optional[Sequence[float]],
    inset_m: float = MAP_GOAL_INSET_M,
) -> Tuple[Tuple[float, float], bool]:
    """Keep an in-map goal. Out-of-map: clip robot→VA to the box, then inset 1 cell."""
    if aabb is None:
        return (float(gx), float(gy)), False
    if point_in_aabb(gx, gy, aabb):
        return (float(gx), float(gy)), False
    hit = _clip_segment_aabb(rx, ry, gx, gy, aabb)
    if hit is None:
        return (float(gx), float(gy)), False
    t0, t1 = hit
    span = math.hypot(float(gx) - float(rx), float(gy) - float(ry))
    if span < 1e-9:
        return (float(gx), float(gy)), False
    step = max(float(inset_m), 0.0) / span
    if point_in_aabb(rx, ry, aabb):
        t = max(t0, t1 - step)
    else:
        t = min(t1, t0 + step)
    return (float(rx) + t * (float(gx) - float(rx)), float(ry) + t * (float(gy) - float(ry))), True


def in_world_xy(x: float, y: float) -> bool:
    return -WORLD_HALF_M <= float(x) < WORLD_HALF_M and -WORLD_HALF_M <= float(y) < WORLD_HALF_M


def bound_goal_to_map(
    gx: float,
    gy: float,
    rx: float,
    ry: float,
    scene,
    inset_m: float = MAP_GOAL_INSET_M,
) -> Tuple[Tuple[float, float], bool]:
    """Keep an in-canvas goal. Outside the 100 m map: reject (True) for a new LA+VA."""
    del rx, ry, scene, inset_m
    if not in_world_xy(gx, gy):
        return (float(gx), float(gy)), True
    return (float(gx), float(gy)), False


def point_in_aabb(x: float, y: float, aabb: Sequence[float]) -> bool:
    minx, miny, maxx, maxy = [float(v) for v in aabb]
    return minx <= float(x) <= maxx and miny <= float(y) <= maxy


PLAN_INFLATE_M = {n: p["inflate"] for n, p in AGENTS.items()}


def _scene_occupied_cells(scene) -> Tuple[float, float, float, set]:
    mmap = (scene or {}).get("map") if isinstance(scene, dict) else None
    if not isinstance(mmap, dict):
        mmap = scene if isinstance(scene, dict) else {}
    res = float(mmap.get("resolution") or 0.1) or 0.1
    origin = mmap.get("origin") or [0.0, 0.0]
    ox, oy = float(origin[0]), float(origin[1])
    cells = set()
    for cell in mmap.get("fused") or []:
        if not isinstance(cell, (list, tuple)) or len(cell) < 2:
            continue
        try:
            cells.add((int(cell[0]), int(cell[1])))
        except (TypeError, ValueError):
            continue
    return res, ox, oy, cells


def _inflate_occupied_cells(occupied, res: float, inflate_m: float) -> set:
    lethal = set(occupied)
    if not occupied or inflate_m <= 0 or res <= 0:
        return lethal
    r_cells = float(inflate_m) / float(res)
    r2 = r_cells * r_cells
    reach = int(math.ceil(r_cells))
    for ix, iy in occupied:
        for dy in range(-reach, reach + 1):
            for dx in range(-reach, reach + 1):
                if dx * dx + dy * dy <= r2:
                    lethal.add((ix + dx, iy + dy))
    return lethal


def snap_to_nearest_free(
    x: float,
    y: float,
    scene,
    inflate_m: float,
    max_m: float = 3.0,
    reject=None,
) -> Tuple[Tuple[float, float], bool]:
    """Move an occupied/inflated VA point to the nearest free cell to that original point."""
    x, y = float(x), float(y)
    res, ox, oy, occupied = _scene_occupied_cells(scene)
    if not occupied:
        return (x, y), False
    lethal = _inflate_occupied_cells(occupied, res, inflate_m)

    def cell_of(wx: float, wy: float) -> Tuple[int, int]:
        return (int(math.floor((wx - ox) / res)), int(math.floor((wy - oy) / res)))

    def center(ix: int, iy: int) -> Tuple[float, float]:
        return (ox + (ix + 0.5) * res, oy + (iy + 0.5) * res)

    def is_blocked(ix: int, iy: int, wx: float, wy: float) -> bool:
        if (ix, iy) in lethal:
            return True
        return bool(reject(wx, wy)) if reject else False

    ix0, iy0 = cell_of(x, y)
    if not is_blocked(ix0, iy0, x, y):
        return (x, y), False
    best = None
    best_d = None
    reach = max(1, int(math.ceil(float(max_m) / res)))
    for radius in range(1, reach + 1):
        found_ring = False
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if max(abs(dx), abs(dy)) != radius:
                    continue
                ix, iy = ix0 + dx, iy0 + dy
                cx, cy = center(ix, iy)
                dist = math.hypot(cx - x, cy - y)
                if dist > float(max_m) + 1e-6:
                    continue
                if is_blocked(ix, iy, cx, cy):
                    continue
                found_ring = True
                if best_d is None or dist < best_d - 1e-9:
                    best, best_d = (cx, cy), dist
        if found_ring and best is not None:
            return best, True
    return (x, y), False
