"""Body geometry, waypoint distances and navigation completion rules."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

from lavira_m3.profiles import AGENTS

IN_PLACE_XY_M = 0.05


def pose_xy(pose) -> Optional[Tuple[float, float]]:
    if callable(pose):
        try:
            pose = pose()
        except Exception:
            return None
    if pose is None:
        return None
    if isinstance(pose, dict):
        pos = pose.get("position") or pose.get("xy")
        if isinstance(pos, (list, tuple)) and len(pos) >= 2:
            try:
                return float(pos[0]), float(pos[1])
            except (TypeError, ValueError):
                return None
        if pose.get("x") is not None and pose.get("y") is not None:
            try:
                return float(pose["x"]), float(pose["y"])
            except (TypeError, ValueError):
                return None
        return None
    if isinstance(pose, (list, tuple)) and len(pose) >= 2:
        try:
            return float(pose[0]), float(pose[1])
        except (TypeError, ValueError):
            return None
    return None


def xy_dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(float(ax) - float(bx), float(ay) - float(by))


def start_goal_dist(pose, gx: float, gy: float) -> Optional[float]:
    xy = pose_xy(pose)
    if xy is None:
        return None
    return xy_dist(xy[0], xy[1], gx, gy)


def already_within_waypoint(pose, gx: float, gy: float, reach_m: Optional[float] = None) -> bool:
    from .config import waypoint_reach_m

    dist = start_goal_dist(pose, gx, gy)
    if dist is None:
        return False
    reach = float(waypoint_reach_m() if reach_m is None else reach_m)
    return dist > IN_PLACE_XY_M and dist <= reach


def waypoint_reached(
    pose,
    gx: float,
    gy: float,
    start_dist: Optional[float] = None,
    reach_m: Optional[float] = None,
) -> bool:
    """Stop translation once inside reach_m (0.20 m). In-place yaw (start_dist <= 5 cm) never trips this."""
    from .config import waypoint_reach_m

    xy = pose_xy(pose)
    if xy is None:
        return False
    reach = float(waypoint_reach_m() if reach_m is None else reach_m)
    if reach <= 0:
        return False
    if start_dist is not None and start_dist <= IN_PLACE_XY_M:
        return False
    return xy_dist(xy[0], xy[1], gx, gy) <= reach


def reach_stop_message(pose, gx: float, gy: float, reach_m: Optional[float] = None) -> str:
    from .config import waypoint_reach_m

    xy = pose_xy(pose)
    reach = float(waypoint_reach_m() if reach_m is None else reach_m)
    dist = xy_dist(xy[0], xy[1], gx, gy) if xy else float("nan")
    return "reached %.2f m (limit %.2f m)" % (dist, reach)


COMPLETION_ORIGINAL = "original"


COMPLETION_INTERMEDIATE = "intermediate"


COMPLETION_BLOCKED = "blocked"


SAME_GOAL_M = 0.05


MIN_PROGRESS_M = 0.10


def _finite_xy(pt) -> Optional[Tuple[float, float]]:
    xy = pose_xy(pt)
    if xy is None:
        return None
    if not (math.isfinite(xy[0]) and math.isfinite(xy[1])):
        return None
    return xy


def classify_nav_completion(
    start_xy,
    pose_xy,
    requested_xy,
    accepted_xy=None,
    reach_m: Optional[float] = None,
    same_goal_m: float = SAME_GOAL_M,
    min_progress_m: float = MIN_PROGRESS_M,
) -> Optional[str]:
    """Same task: original A vs planner-accepted B vs how the wait ended.

    original: robot is inside reach of A.
    intermediate: B is not A, robot is at B, and it actually moved toward A.
    blocked: B is not A, robot is at B, but almost no motion (B already at the feet).
    None: not finished.
    """
    pose = _finite_xy(pose_xy)
    requested = _finite_xy(requested_xy)
    if pose is None or requested is None:
        return None
    try:
        reach = float(0.75 if reach_m is None else reach_m)
    except (TypeError, ValueError):
        return None
    if reach <= 0:
        return None
    dist_a = xy_dist(pose[0], pose[1], requested[0], requested[1])
    if dist_a <= reach:
        return COMPLETION_ORIGINAL
    accepted = _finite_xy(accepted_xy)
    if accepted is None:
        return None
    if xy_dist(accepted[0], accepted[1], requested[0], requested[1]) <= same_goal_m:
        return None
    if xy_dist(pose[0], pose[1], accepted[0], accepted[1]) > reach:
        return None
    start = _finite_xy(start_xy)
    progress = 0.0
    if start is not None:
        progress = xy_dist(start[0], start[1], requested[0], requested[1]) - dist_a
    if progress >= min_progress_m:
        return COMPLETION_INTERMEDIATE
    return COMPLETION_BLOCKED


def nav_completion_fields(kind: Optional[str], requested_xy, accepted_xy=None) -> Dict[str, object]:
    requested = _finite_xy(requested_xy)
    accepted = _finite_xy(accepted_xy)
    return {
        "completion": kind,
        "requested_goal": list(requested) if requested else None,
        "accepted_goal": list(accepted) if accepted else None,
    }


def nav_completion_message(
    kind: Optional[str],
    pose_xy,
    requested_xy,
    accepted_xy=None,
    reach_m: Optional[float] = None,
) -> str:
    requested = _finite_xy(requested_xy)
    accepted = _finite_xy(accepted_xy)
    if kind == COMPLETION_ORIGINAL and requested is not None:
        return reach_stop_message(pose_xy, requested[0], requested[1], reach_m)
    if kind == COMPLETION_INTERMEDIATE and requested is not None and accepted is not None:
        pose = _finite_xy(pose_xy)
        dist = xy_dist(pose[0], pose[1], requested[0], requested[1]) if pose else float("nan")
        return "reached intermediate (%.2f, %.2f); original (%.2f, %.2f) still %.2f m away" % (
            accepted[0],
            accepted[1],
            requested[0],
            requested[1],
            dist,
        )
    if kind == COMPLETION_BLOCKED and requested is not None and accepted is not None:
        return (
            "planner accepted (%.2f, %.2f) already at the robot; original (%.2f, %.2f) not reached"
            % (accepted[0], accepted[1], requested[0], requested[1])
        )
    return "navigation %s" % (kind or "pending")


def clamp_map_step(
    xy: Tuple[float, float],
    rx: float,
    ry: float,
    max_m: float,
) -> Tuple[Tuple[float, float], float, bool]:
    """Keep the heading toward xy, but never ask Nav2 for more than max_m."""
    dx, dy = float(xy[0]) - float(rx), float(xy[1]) - float(ry)
    dist = math.hypot(dx, dy)
    if dist <= max_m or dist < 1e-6:
        return (float(xy[0]), float(xy[1])), dist, False
    scale = max_m / dist
    return (float(rx) + dx * scale, float(ry) + dy * scale), dist, True


def push_outside_footprint(
    xy: Tuple[float, float],
    rx: float,
    ry: float,
    ryaw: float,
    rear: float = 0.55,
    front: float = 0.55,
    half: float = 0.50,
    margin: float = 0.25,
) -> Tuple[Tuple[float, float], float, bool]:
    """If xy sits in the robot rectangle (+margin), slide it out along the same ray."""
    x, y = float(xy[0]), float(xy[1])
    dx, dy = x - float(rx), y - float(ry)
    dist = math.hypot(dx, dy)
    c, s = math.cos(ryaw), math.sin(ryaw)
    need_r, need_f, need_h = rear + margin, front + margin, half + margin

    def inside(px: float, py: float) -> bool:
        fx = c * (px - rx) + s * (py - ry)
        ly = -s * (px - rx) + c * (py - ry)
        return -need_r <= fx <= need_f and abs(ly) <= need_h

    if not inside(x, y):
        return (x, y), dist, False
    if dist < 1e-3:
        ux, uy = c, s
    else:
        ux, uy = dx / dist, dy / dist
    t = 0.05
    limit = need_f + need_r + need_h + 0.4
    while t <= limit:
        px, py = rx + ux * t, ry + uy * t
        if not inside(px, py):
            return (px, py), dist, True
        t += 0.05
    return (rx + ux * limit, ry + uy * limit), dist, True


def body_to_map(xb: float, yb: float, rx: float, ry: float, ryaw: float) -> Tuple[float, float]:
    c, s = math.cos(ryaw), math.sin(ryaw)
    return rx + xb * c - yb * s, ry + xb * s + yb * c


def heading_yaw(
    x: float, y: float, rx: float, ry: float, current_yaw: float, min_step: float = 0.05
) -> float:
    dx, dy = x - rx, y - ry
    if math.hypot(dx, dy) < min_step:
        return current_yaw
    return math.atan2(dy, dx)


BODY_ENVELOPE = {n: tuple(p["envelope"]) for n, p in AGENTS.items()}


def body_envelope(name: str) -> Tuple[float, float, float]:
    return BODY_ENVELOPE[str(name)]


def heading_step(rx: float, ry: float, ryaw: float, dist_m: float) -> Tuple[float, float]:
    return (float(rx) + float(dist_m) * math.cos(ryaw), float(ry) + float(dist_m) * math.sin(ryaw))


def ray_step(
    rx: float,
    ry: float,
    x: float,
    y: float,
    dist_m: float,
    yaw_if_coincident: float = 0.0,
) -> Tuple[float, float]:
    """Point dist_m along the robot→xy ray. Coincident xy uses heading."""
    dx, dy = float(x) - float(rx), float(y) - float(ry)
    n = math.hypot(dx, dy)
    if n < 1e-6:
        return heading_step(rx, ry, yaw_if_coincident, dist_m)
    scale = float(dist_m) / n
    return float(rx) + dx * scale, float(ry) + dy * scale


def inside_body_envelope(
    x: float,
    y: float,
    rx: float,
    ry: float,
    ryaw: float,
    name: str,
    pad: float = 0.0,
) -> bool:
    rear, front, half = body_envelope(name)
    c, s = math.cos(ryaw), math.sin(ryaw)
    fx = c * (float(x) - float(rx)) + s * (float(y) - float(ry))
    ly = -s * (float(x) - float(rx)) + c * (float(y) - float(ry))
    return -(rear + pad) <= fx <= (front + pad) and abs(ly) <= (half + pad)
