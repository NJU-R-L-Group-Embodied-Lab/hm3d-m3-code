"""L1: polynomial decoding and latched command acceptance, independent of ROS."""

import math


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


def straight_preview(start, goal):
    """Map overlay / loop-trim polyline. Never talks to EGO."""
    if start is None or goal is None or len(start) < 2 or len(goal) < 2:
        return []
    try:
        sx, sy = float(start[0]), float(start[1])
        gx, gy = float(goal[0]), float(goal[1])
    except (TypeError, ValueError):
        return []
    if not all(math.isfinite(v) for v in (sx, sy, gx, gy)):
        return []
    if math.hypot(gx - sx, gy - sy) < 1e-3:
        return [[sx, sy]]
    return [[sx, sy], [gx, gy]]


class CommandGate:
    inputs = ("odometry", "localization", "cloud", "heartbeat", "shared_map", "peer_state")
    limits = (0.4, 0.15, 0.0, 0.0, 0.0, 0.8)

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

    def promote(self):
        if not self.active:
            return False
        self.execute = True
        return True

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
