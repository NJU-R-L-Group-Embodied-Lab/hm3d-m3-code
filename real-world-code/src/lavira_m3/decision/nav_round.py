"""One bounded navigation phase, shared by the three robot clients."""

from __future__ import annotations

import math
import time

from .map_geometry import (
    PLAN_INFLATE_M,
    _inflate_occupied_cells,
    _scene_occupied_cells,
    fused_map_aabb,
    in_world_xy,
)
from .navigation_geometry import pose_xy

ROUND_S = 30.0
PLAN_S = 6.0
STALL_S = 6.0
CANCEL_RESERVE_S = 1.0
HOP_M = 1.5


def remaining(client, limit):
    deadline = getattr(client, "round_deadline", None)
    if deadline is None:
        return max(0.0, float(limit))
    return max(0.0, min(float(limit), deadline - CANCEL_RESERVE_S - time.monotonic()))


class MotionProgress:
    """Measure odometry displacement; nonzero velocity commands are not progress."""

    def __init__(self, pose):
        self.anchor = pose_xy(pose)
        self.last_progress = time.monotonic()

    def stalled(self, pose, timeout=STALL_S):
        xy = pose_xy(pose)
        if xy is not None and all(math.isfinite(v) for v in xy):
            if self.anchor is None:
                self.anchor = xy
            elif math.hypot(xy[0] - self.anchor[0], xy[1] - self.anchor[1]) >= 0.10:
                self.anchor = xy
                self.last_progress = time.monotonic()
        return time.monotonic() - self.last_progress >= timeout


class TravelBudget:
    """Accumulate planar odom path length. Yaw-only spin and jitter do not count."""

    def __init__(self, budget_m=None, jitter_m=None, source="odom_ekf"):
        from .config import travel_budget_m, travel_jitter_m

        self.budget_m = float(budget_m if budget_m is not None else travel_budget_m())
        self.jitter_m = float(jitter_m if jitter_m is not None else travel_jitter_m())
        self.source = str(source or "odom_ekf")
        self.travel_m = 0.0
        self.last = None

    def add(self, pose):
        xy = pose_xy(pose)
        if xy is None or not all(math.isfinite(v) for v in xy):
            return self.travel_m
        if self.last is None:
            self.last = xy
            return self.travel_m
        step = math.hypot(xy[0] - self.last[0], xy[1] - self.last[1])
        if step >= self.jitter_m:
            self.travel_m += step
            self.last = xy
        return self.travel_m

    def exhausted(self, pose=None):
        if pose is not None:
            self.add(pose)
        return self.budget_m > 0.0 and self.travel_m + 1e-9 >= self.budget_m


def xy_yaw_from_odom_msg(msg):
    """nav_msgs/Odometry → (x, y, yaw). Used by /odom_ekf subscribers."""
    p, q = msg.pose.pose.position, msg.pose.pose.orientation
    yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
    return (float(p.x), float(p.y), float(yaw))


def sample_travel_pose(client, fallback=None):
    fn = getattr(client, "travel_pose_fn", None)
    if callable(fn):
        try:
            pose = fn()
            if pose is not None:
                return pose
        except Exception:
            pass
    return fallback


def travel_budget_hit(client, travel, fallback_pose=None, rotation_only=False):
    if rotation_only or travel is None:
        return False
    return travel.exhausted(sample_travel_pose(client, fallback_pose))


def short_hop_candidates(pose, goal, scene, robot):
    """Conservative straight corridors, nearest heading first; EGO still validates."""
    if pose is None or not isinstance(scene, dict) or robot not in PLAN_INFLATE_M:
        return []
    # A fresh HTTP response alone does not establish sensor freshness.
    lidar = ((scene.get("devices") or {}).get(robot) or {}).get("lidar") or {}
    if lidar.get("live") is not True:
        return []
    try:
        if abs(time.time() - float(scene.get("t", 0))) > 3.0:
            return []
        res, ox, oy, occupied = _scene_occupied_cells(scene)
        bounds = fused_map_aabb(scene)
        px, py = float(pose[0]), float(pose[1])
        heading = math.atan2(float(goal[1]) - py, float(goal[0]) - px)
        if (
            not bounds
            or res <= 0
            or not all(math.isfinite(v) for v in (res, ox, oy, px, py, heading))
        ):
            return []
    except (TypeError, ValueError, IndexError):
        return []
    # Use the planning-map hard radius exactly; no additional grid inflation.
    lethal = _inflate_occupied_cells(occupied, res, PLAN_INFLATE_M[robot])
    steps = max(1, int(math.ceil(HOP_M / (res / 4))))
    candidates = []
    for degrees in (0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180):
        angle = heading + math.radians(degrees)
        dx, dy = HOP_M * math.cos(angle), HOP_M * math.sin(angle)
        for i in range(steps + 1):
            x, y = px + dx * i / steps, py + dy * i / steps
            cell = (int(math.floor((x - ox) / res)), int(math.floor((y - oy) / res)))
            if (
                not in_world_xy(x, y)
                or not (bounds[0] <= x <= bounds[2] and bounds[1] <= y <= bounds[3])
                or cell in lethal
            ):
                break
        else:
            candidates.append((px + dx, py + dy))
    return candidates[:3]
