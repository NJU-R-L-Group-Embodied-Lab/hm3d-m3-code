"""Image boxes, measured depth and fixed-camera projection geometry."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import (
    VA_PROJECT_MAX_M,
    la_action_dirs,
)


def bbox_to_xyxy_px(bbox_2d: Sequence[float], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in bbox_2d]
    mx = max(abs(x1), abs(x2), abs(y1), abs(y2))
    if mx <= 1.5:
        sx, sy = float(width), float(height)
    elif mx <= 999.0 + 1e-6:
        sx, sy = width / 999.0, height / 999.0
    else:
        sx, sy = 1.0, 1.0
    x1, x2 = x1 * sx, x2 * sx
    y1, y2 = y1 * sy, y2 * sy
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    xlo = int(np.clip(math.floor(x1), 0, width - 1))
    xhi = int(np.clip(math.ceil(x2), xlo + 1, width))
    ylo = int(np.clip(math.floor(y1), 0, height - 1))
    yhi = int(np.clip(math.ceil(y2), ylo + 1, height))
    return xlo, ylo, xhi, yhi


VA_BBOX_MIN_FRAC = 0.004  # 0.4% of the image; 23x29 fire-hydrant noise is ~0.07%


def usable_va_bbox(bbox_2d: Optional[Sequence[float]], width: int = 0, height: int = 0) -> bool:
    """False for missing/degenerate/tiny boxes such as [0,0,0,0] (do not project)."""
    if not isinstance(bbox_2d, (list, tuple)) or len(bbox_2d) != 4:
        return False
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox_2d]
    except (TypeError, ValueError):
        return False
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1e-6:
        return False
    if abs(x2 - x1) < 2.0 and abs(y2 - y1) < 2.0:
        return False
    if int(width) >= 2 and int(height) >= 2:
        px = bbox_to_xyxy_px(bbox_2d, int(width), int(height))
        area = float((px[2] - px[0]) * (px[3] - px[1]))
        if area < VA_BBOX_MIN_FRAC * float(width) * float(height):
            return False
    return True


def bbox_iou(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> float:
    """IoU in the shared 0..999 (or same-unit) bbox contract. 0 if either box is unusable."""
    if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)):
        return 0.0
    if len(a) != 4 or len(b) != 4:
        return 0.0
    try:
        ax1, ay1, ax2, ay2 = [float(v) for v in a]
        bx1, by1, bx2, by2 = [float(v) for v in b]
    except (TypeError, ValueError):
        return 0.0
    if ax2 < ax1:
        ax1, ax2 = ax2, ax1
    if ay2 < ay1:
        ay1, ay2 = ay2, ay1
    if bx2 < bx1:
        bx1, bx2 = bx2, bx1
    if by2 < by1:
        by1, by2 = by2, by1
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 1e-9:
        return 0.0
    return inter / union


def fallback_va_bbox(width: int = 0, height: int = 0) -> List[float]:
    """Simulator default: pixel [W/4, H/4, 3W/4, 3H/4] in the 0..999 contract."""
    del width, height
    return [999.0 / 4.0, 999.0 / 4.0, 3.0 * 999.0 / 4.0, 3.0 * 999.0 / 4.0]


def normalize_va_target(value: Optional[object]) -> str:
    """none/null/n/a/empty is only a missing label, not a skip signal."""
    if value is None:
        return "unknown target"
    text = str(value).strip()
    if not text or text.lower() in ("none", "null", "n/a", "na"):
        return "unknown target"
    return text


def annotate_bbox_bgr(
    image, bbox_2d: Optional[Sequence[float]], color=(38, 92, 196), thickness: int = 3
):
    """Draw VA box on a BGR (or gray) frame. Returns a contiguous copy."""
    vis = np.ascontiguousarray(np.asarray(image))
    if vis.ndim == 2:
        vis = np.stack([vis, vis, vis], axis=-1)
    if vis.ndim != 3 or vis.shape[2] < 3:
        return vis
    if vis.dtype != np.uint8:
        vis = np.clip(vis, 0, 255).astype(np.uint8)
    if bbox_2d is None or len(bbox_2d) != 4:
        return vis
    h, w = vis.shape[:2]
    x1, y1, x2, y2 = bbox_to_xyxy_px(bbox_2d, w, h)
    t = max(1, int(thickness))
    vis[y1 : min(h, y1 + t), x1:x2] = color
    vis[max(0, y2 - t) : y2, x1:x2] = color
    vis[y1:y2, x1 : min(w, x1 + t)] = color
    vis[y1:y2, max(0, x2 - t) : x2] = color
    ug, vg = bbox_ground_px(bbox_2d, w, h)
    r = 4
    vis[max(0, vg - r) : min(h, vg + r + 1), max(0, ug - r) : min(w, ug + r + 1)] = color
    return vis


def bbox_center_px(bbox_2d: Sequence[float], width: int, height: int) -> Tuple[int, int]:
    xlo, ylo, xhi, yhi = bbox_to_xyxy_px(bbox_2d, width, height)
    u = int(np.clip(0.5 * (xlo + xhi - 1), 0, width - 1))
    v = int(np.clip(0.5 * (ylo + yhi - 1), 0, height - 1))
    return u, v


def bbox_ground_px(bbox_2d: Sequence[float], width: int, height: int) -> Tuple[int, int]:
    """Center of the whole VA box (used by Aloha / GO1 / GO2 projection)."""
    return bbox_center_px(bbox_2d, width, height)


def body_step_along_pixel(
    u: int,
    v: int,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    cam_yaw: float,
    dist_m: float,
) -> Tuple[float, float]:
    """Unit pixel ray in body frame, scaled to dist_m."""
    xb, yb = pixel_to_body(u, v, 1.0, fx, fy, cx, cy, cam_yaw)
    n = math.hypot(xb, yb)
    if n < 1e-9:
        return dist_m * math.cos(cam_yaw), dist_m * math.sin(cam_yaw)
    return xb / n * dist_m, yb / n * dist_m


def _depth_meters(depth) -> Optional[np.ndarray]:
    if depth is None:
        return None
    d = np.asarray(depth)
    if d.size == 0:
        return None
    if d.ndim == 3:
        d = d[..., 0]
    d = d.astype(np.float32)
    finite = d[np.isfinite(d) & (d > 0)]
    if finite.size and float(np.median(finite)) > 20.0:
        d = d * 0.001
    return d


def depth_at(depth: np.ndarray, u: int, v: int, win: int = 4) -> Optional[float]:
    d = _depth_meters(depth)
    if d is None:
        return None
    h, w = d.shape[:2]
    x0, x1 = max(0, u - win), min(w, u + win + 1)
    y0, y1 = max(0, v - win), min(h, v + win + 1)
    crop = d[y0:y1, x0:x1]
    valid = crop[np.isfinite(crop) & (crop > 0.15) & (crop < 8.0)]
    if valid.size < 4:
        return None
    return float(np.median(valid))


def bbox_inset_median_depth(
    depth, bbox_2d: Sequence[float], width: int, height: int
) -> Optional[float]:
    """Inset-median depth in meters for a VA/DCS bbox. None if observation is insufficient."""
    d = _depth_meters(depth)
    if d is None:
        return None
    h, w = d.shape[:2]
    xlo, ylo, xhi, yhi = bbox_to_xyxy_px(bbox_2d, width, height)
    if xhi - xlo < 2 or yhi - ylo < 2:
        return None
    xpad = max(1, (xhi - xlo) // 6)
    ypad = max(1, (yhi - ylo) // 6)
    crop = d[ylo + ypad : yhi - ypad, xlo + xpad : xhi - xpad]
    valid = crop[np.isfinite(crop) & (crop > 0.15) & (crop < 8.0)]
    if valid.size < 16:
        return None
    return float(np.median(valid))


def bbox_bottom_strip_depth(
    depth, bbox_2d: Sequence[float], width: int, height: int
) -> Optional[float]:
    """Median depth on the bottom quarter of the VA box (same band as bbox_ground_px)."""
    d = _depth_meters(depth)
    if d is None:
        return None
    h, w = d.shape[:2]
    xlo, ylo, xhi, yhi = bbox_to_xyxy_px(bbox_2d, width, height)
    if xhi - xlo < 2 or yhi - ylo < 2:
        return None
    ylo_q = int(ylo + 0.65 * (yhi - ylo))
    xpad = max(1, (xhi - xlo) // 8)
    crop = d[max(0, ylo_q) : yhi, xlo + xpad : max(xlo + xpad + 1, xhi - xpad)]
    valid = crop[np.isfinite(crop) & (crop > 0.15) & (crop < 8.0)]
    if valid.size < 8:
        return None
    return float(np.median(valid))


def va_observe_depth(
    depth,
    bbox_2d: Sequence[float],
    width: int,
    height: int,
    u: int,
    v: int,
) -> Tuple[Optional[float], str]:
    """Live RA depth: point window → wider window → box bottom strip → inset median."""
    z = depth_at(depth, u, v, win=4)
    if z is not None:
        return z, "point"
    z = depth_at(depth, u, v, win=12)
    if z is not None:
        return z, "point_wide"
    z = bbox_bottom_strip_depth(depth, bbox_2d, width, height)
    if z is not None:
        return z, "bottom_strip"
    z = bbox_inset_median_depth(depth, bbox_2d, width, height)
    if z is not None:
        return z, "inset"
    return None, ""


def clip_va_project_depth(
    z: Optional[float],
    max_m: float = VA_PROJECT_MAX_M,
) -> Tuple[Optional[float], bool]:
    """Clip measured VA depth to max_m before pixel projection. None stays None."""
    if z is None:
        return None, False
    val = float(z)
    limit = float(max_m)
    if val > limit:
        return limit, True
    return val, False


def pixel_to_body(
    u: int,
    v: int,
    depth_m: float,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    cam_yaw: float = 0.0,
) -> Tuple[float, float]:
    """Camera optical: z forward, x right, y down → body x forward, y left."""
    x_cam = (u - cx) / fx * depth_m
    z_cam = depth_m
    x_fwd = z_cam
    y_left = -x_cam
    c, s = math.cos(cam_yaw), math.sin(cam_yaw)
    return x_fwd * c - y_left * s, x_fwd * s + y_left * c


def view_center_mean_depth(depth) -> Optional[float]:
    """Mean valid depth (m) on the image center third. None if not enough samples."""
    d = _depth_meters(depth)
    if d is None:
        return None
    h, w = d.shape[:2]
    if h < 3 or w < 3:
        return None
    crop = d[h // 3 : (2 * h) // 3, w // 3 : (2 * w) // 3]
    valid = crop[np.isfinite(crop) & (crop > 0.15) & (crop < 8.0)]
    if valid.size < 8:
        return None
    return float(np.mean(valid))


def view_mask_blocked(
    frames: Sequence[Dict],
    dirs: Sequence[str],
    min_depth_m: float,
) -> Tuple[List[str], Dict[str, Optional[float]]]:
    """Directions whose center-region mean depth is closer than min_depth_m. Missing depth is not blocked."""
    by_name = {}
    for fr in frames or []:
        name = str((fr or {}).get("name") or "")
        if name:
            by_name[name] = fr
    depths: Dict[str, Optional[float]] = {}
    blocked: List[str] = []
    for name in dirs:
        fr = by_name.get(name) or {}
        if fr.get("center_depth_m") is not None and fr.get("depth") is None:
            try:
                z = float(fr.get("center_depth_m"))
            except (TypeError, ValueError):
                z = None
            if z is not None and not math.isfinite(z):
                z = None
        else:
            z = view_center_mean_depth(fr.get("depth"))
            if z is None and fr.get("center_depth_m") is not None:
                try:
                    z = float(fr.get("center_depth_m"))
                except (TypeError, ValueError):
                    z = None
                if z is not None and not math.isfinite(z):
                    z = None
        depths[name] = z
        if z is not None and z < float(min_depth_m):
            blocked.append(name)
    return blocked, depths


def legal_la_directions(
    dirs: Sequence[str],
    blocked: Sequence[str],
    exclude_rear: bool = False,
) -> List[str]:
    source = la_action_dirs(dirs) if exclude_rear else [str(d) for d in dirs]
    blocked_set = set(blocked)
    legal = [d for d in source if d not in blocked_set]
    if legal:
        return legal
    for cand in ("front", "left", "right", "left_front", "right_front"):
        if cand in dirs:
            return [cand]
    remain = source
    return list(remain[:1]) if remain else []


def blocked_prompt_note(
    blocked: Sequence[str],
    depths: Dict[str, Optional[float]],
    min_depth_m: float,
) -> str:
    if not blocked:
        return ""
    parts = []
    for name in blocked:
        z = depths.get(name)
        if z is None:
            parts.append(str(name))
        else:
            parts.append("%s (%.2f m)" % (name, z))
    return (
        "\nNear-range mask: these views have center-region mean depth under "
        "%.2f m and are omitted from the action list (images are still shown): %s. "
        "Prefer a listed direction.\n" % (float(min_depth_m), ", ".join(parts))
    )


CAM_YAW = {
    "front": 0.0,
    "left_front": math.radians(45),
    "left": math.radians(90),
    "left_back": math.radians(135),
    "behind": math.pi,
    "right_back": math.radians(-135),
    "right": math.radians(-90),
    "right_front": math.radians(-45),
}
