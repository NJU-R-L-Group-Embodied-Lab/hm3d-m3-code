"""L1: polynomial decoding and latched command acceptance, independent of ROS."""

import math
from dataclasses import dataclass, field


def sample_trajectory(msg):
    durations = list(msg.duration)
    coefficients = [list(msg.coef_x), list(msg.coef_y), list(msg.coef_z)]
    if (
        msg.order != 5
        or not 1 <= len(durations) <= 200
        or any(len(axis) != 6 * len(durations) for axis in coefficients)
        or any(not math.isfinite(d) or not 0 < d <= 180 for d in durations)
        or sum(durations) > 300
        or any(not math.isfinite(v) for axis in coefficients for v in axis)
    ):
        raise ValueError("invalid EGO polynomial")
    if getattr(msg, "rotation_only", False):
        yaw = list(msg.coef_yaw)
        if len(durations) != 1 or len(yaw) != 6 or not all(math.isfinite(v) for v in yaw):
            raise ValueError("invalid EGO yaw polynomial")
        if any(v != 0 for axis in coefficients for v in axis[:-1]):
            raise ValueError("rotation trajectory must have stationary XYZ")
    path = []
    for piece, duration in enumerate(durations):
        steps = max(1, math.ceil(duration / 0.1))
        for index in range(steps + 1):
            if piece and index == 0:
                continue
            point = []
            for axis in coefficients:
                value = 0.0
                for coefficient in axis[piece * 6 : piece * 6 + 6]:
                    value = value * (index * duration / steps) + coefficient
                point.append(value)
            if any(not math.isfinite(v) for v in point) or any(abs(v) >= 50 for v in point[:2]):
                raise ValueError("trajectory leaves world bounds")
            path.append(point[:2])
    return path


class CommandGate:
    inputs = (
        "odometry",
        "raw_localization",
        "imu",
        "wheel",
        "heartbeat",
        "shared_map",
        "peer_state",
    )
    limits = (0.5, 0.0, 0.0, 0.0, 0.0, 0.25)

    def __init__(self):
        self.received = {name: -math.inf for name in self.inputs}
        self.active = None
        self.execute = False
        self.reason = ""
        self.trajectory_until = 0
        self.last_command = None

    def begin(self, request_id, execute, now):
        self.active, self.execute = request_id, execute
        self.reason = ""
        self.trajectory_until = 0
        self.last_command = None

    def stop(self, reason):
        self.active, self.execute, self.reason = None, False, reason

    def observe(self, source, now):
        self.received[source] = now

    def stale(self, now):
        return next(
            (
                name
                for name, received in self.received.items()
                if not 0 <= now - received <= (5.0 if name in ("shared_map", "peer_state") else 0.5)
            ),
            "",
        )

    def expired(self, now, ros_now):
        if self.stale(now):
            return "stale_" + self.stale(now)
        if self.trajectory_until and ros_now > self.trajectory_until:
            return "trajectory_expired"
        if self.last_command is not None and now - self.last_command > 0.3:
            return "command_timeout"
        return ""

    def accept(self, request_id, values, now, ros_now):
        if request_id != self.active or not self.execute or not self.trajectory_until:
            return False
        reason = self.expired(now, ros_now)
        if len(values) != 6 or any(
            not math.isfinite(v) or abs(v) > limit + 1e-6 for v, limit in zip(values, self.limits)
        ):
            reason = "invalid_command"
        if reason:
            self.stop(reason)
            return False
        self.last_command = now
        return True


# Existing Aloha reach thresholds: path_tracker.ARRIVE_XY and navigation.EXPLORATION_ARRIVE_M.
HTTP_REACH_M = 0.40
EXPLORATION_REACH_M = 0.25
COMPLETION_ORIGINAL = "original"
COMPLETION_INTERMEDIATE = "intermediate"
COMPLETION_BLOCKED = "blocked"
SAME_GOAL_M = 0.05
MIN_PROGRESS_M = 0.10


def classify_nav_completion(
    start_xy,
    pose_xy,
    requested_xy,
    accepted_xy=None,
    reach_m=HTTP_REACH_M,
    same_goal_m=SAME_GOAL_M,
    min_progress_m=MIN_PROGRESS_M,
):
    """original / intermediate / blocked / None for requested A vs accepted B."""
    try:
        px, py = float(pose_xy[0]), float(pose_xy[1])
        ax, ay = float(requested_xy[0]), float(requested_xy[1])
        reach = float(reach_m)
    except (TypeError, ValueError, IndexError):
        return None
    if not all(math.isfinite(v) for v in (px, py, ax, ay, reach)) or reach <= 0:
        return None
    dist_a = math.hypot(px - ax, py - ay)
    if dist_a <= reach:
        return COMPLETION_ORIGINAL
    if accepted_xy is None:
        return None
    try:
        bx, by = float(accepted_xy[0]), float(accepted_xy[1])
    except (TypeError, ValueError, IndexError):
        return None
    if not all(math.isfinite(v) for v in (bx, by)):
        return None
    if math.hypot(bx - ax, by - ay) <= same_goal_m:
        return None
    if math.hypot(px - bx, py - by) > reach:
        return None
    progress = 0.0
    if start_xy is not None:
        try:
            sx, sy = float(start_xy[0]), float(start_xy[1])
        except (TypeError, ValueError, IndexError):
            sx = sy = None
        if sx is not None and math.isfinite(sx) and math.isfinite(sy):
            progress = math.hypot(sx - ax, sy - ay) - dist_a
    if progress >= min_progress_m:
        return COMPLETION_INTERMEDIATE
    return COMPLETION_BLOCKED


@dataclass
class NavResult:
    ok: bool
    state: str
    message: str
    execution_id: object = None
    target: object = None
    frame: object = None
    extra: dict = field(default_factory=dict)
