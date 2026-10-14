"""Path validation, progress selection and polyline simplification."""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from .map_geometry import point_in_aabb


def polyline_length(path: Sequence[Sequence[float]]) -> float:
    acc = 0.0
    prev = None
    for pt in path or []:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            continue
        try:
            xy = (float(pt[0]), float(pt[1]))
        except (TypeError, ValueError):
            continue
        if prev is not None:
            acc += math.hypot(xy[0] - prev[0], xy[1] - prev[1])
        prev = xy
    return acc


def path_outside_frac(path: Sequence[Sequence[float]], aabb: Sequence[float]) -> float:
    n = 0
    out = 0
    for pt in path or []:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            continue
        try:
            x, y = float(pt[0]), float(pt[1])
        except (TypeError, ValueError):
            continue
        n += 1
        if not point_in_aabb(x, y, aabb):
            out += 1
    if n == 0:
        return 0.0
    return out / float(n)


def path_is_bad_loop(
    path: Sequence[Sequence[float]],
    rx: float,
    ry: float,
    gx: float,
    gy: float,
    aabb: Optional[Sequence[float]],
) -> bool:
    pts = [pt for pt in (path or []) if isinstance(pt, (list, tuple)) and len(pt) >= 2]
    if len(pts) < 3:
        return False
    plen = polyline_length(pts)
    straight = math.hypot(float(gx) - float(rx), float(gy) - float(ry))
    if aabb is not None and path_outside_frac(pts, aabb) > 0.25:
        return True
    if straight > 0.5 and plen > max(6.0, 3.0 * straight):
        return True
    return False


def point_along_path(
    path: Sequence[Sequence[float]],
    rx: float,
    ry: float,
    dist_m: float,
) -> Optional[Tuple[float, float]]:
    if path is None or dist_m <= 0:
        return None
    pts: List[Tuple[float, float]] = []
    for pt in path:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            continue
        try:
            px, py = float(pt[0]), float(pt[1])
        except (TypeError, ValueError):
            continue
        if math.isfinite(px) and math.isfinite(py):
            pts.append((px, py))
    if len(pts) < 2:
        return None
    acc = 0.0
    prev = (float(rx), float(ry))
    for pt in pts:
        seg = math.hypot(pt[0] - prev[0], pt[1] - prev[1])
        if seg < 1e-4:
            prev = pt
            continue
        if acc + seg >= dist_m:
            t = (dist_m - acc) / seg
            return (prev[0] + t * (pt[0] - prev[0]), prev[1] + t * (pt[1] - prev[1]))
        acc += seg
        prev = pt
    if acc >= 0.70:
        return pts[-1]
    return None


def in_map_progress_point(
    path: Sequence[Sequence[float]],
    rx: float,
    ry: float,
    aabb: Optional[Sequence[float]],
    dist_min: float,
    dist_max: float,
) -> Optional[Tuple[float, float]]:
    """Walk dist_min..dist_max along the planner path, stopping before leaving the map AABB."""
    pts: List[Tuple[float, float]] = []
    for pt in path or []:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            continue
        try:
            px, py = float(pt[0]), float(pt[1])
        except (TypeError, ValueError):
            continue
        if math.isfinite(px) and math.isfinite(py):
            pts.append((px, py))
    if len(pts) < 2:
        return None
    acc = 0.0
    prev = (float(rx), float(ry))
    last_ok = None
    for pt in pts:
        if aabb is not None and not point_in_aabb(pt[0], pt[1], aabb):
            break
        seg = math.hypot(pt[0] - prev[0], pt[1] - prev[1])
        if seg < 1e-4:
            prev = pt
            continue
        remain = float(dist_max) - acc
        if remain <= 0:
            break
        if seg >= remain:
            t = remain / seg
            cand = (prev[0] + t * (pt[0] - prev[0]), prev[1] + t * (pt[1] - prev[1]))
            if aabb is not None and not point_in_aabb(cand[0], cand[1], aabb):
                break
            if acc + remain >= float(dist_min) - 1e-6:
                return cand
            break
        acc += seg
        prev = pt
        if acc >= float(dist_min) - 1e-6:
            last_ok = pt
        if acc >= float(dist_max) - 1e-6:
            return pt
    if last_ok is not None and math.hypot(last_ok[0] - rx, last_ok[1] - ry) >= 0.70:
        return last_ok
    return None


PATH_THIN_EPS_M = 0.15


def _point_seg_dist(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    dx, dy = bx - ax, by - ay
    den = dx * dx + dy * dy
    if den <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / den))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _rdp_keep(points: List[List[float]], epsilon_m: float) -> List[List[float]]:
    if len(points) < 3:
        return list(points)
    ax, ay = points[0]
    bx, by = points[-1]
    farthest = 0
    best = -1.0
    for i in range(1, len(points) - 1):
        d = _point_seg_dist(points[i][0], points[i][1], ax, ay, bx, by)
        if d > best:
            best, farthest = d, i
    if best > epsilon_m and farthest > 0:
        left = _rdp_keep(points[: farthest + 1], epsilon_m)
        right = _rdp_keep(points[farthest:], epsilon_m)
        return left[:-1] + right
    return [points[0], points[-1]]


def thin_planner_path(path, epsilon_m: float = PATH_THIN_EPS_M) -> List[List[float]]:
    """Aloha-style skeleton: keep start, end, and corners. Never invent points."""
    cleaned: List[List[float]] = []
    for pt in path or []:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            continue
        try:
            x, y = float(pt[0]), float(pt[1])
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        if cleaned and math.hypot(x - cleaned[-1][0], y - cleaned[-1][1]) < 1e-3:
            continue
        cleaned.append([x, y])
    if len(cleaned) <= 3:
        return cleaned
    kept = _rdp_keep(cleaned, max(0.02, float(epsilon_m)))
    return kept if len(kept) >= 2 else cleaned
