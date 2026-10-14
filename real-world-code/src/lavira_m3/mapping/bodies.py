"""Oriented footprints for self / teammate cloud punch-out. Sizes differ per robot."""

from __future__ import annotations

import math

from lavira_m3.profiles import AGENTS

# Length along body X (forward), width along body Y. Metres.
FOOTPRINT = {n: tuple(p["footprint"]) for n, p in AGENTS.items()}
POSE_LIVE_SEC = 5.0


def yaw_from_quat(q):
    x, y, z, w = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def footprint(name):
    return FOOTPRINT[name]


def inside_box(px, py, x, y, yaw, length, width):
    dx, dy = px - x, py - y
    c, s = math.cos(yaw), math.sin(yaw)
    lx = c * dx + s * dy
    ly = -s * dx + c * dy
    return abs(lx) <= length * 0.5 and abs(ly) <= width * 0.5


def inside_robot(px, py, name, x, y, yaw):
    length, width = footprint(name)
    return inside_box(px, py, x, y, yaw, length, width)


def raster_footprint(x, y, yaw, length, width, res=0.1):
    """World-xy cell centers that fall inside an oriented box."""
    half = 0.5 * math.hypot(length, width)
    ix0 = int(math.floor((x - half) / res))
    ix1 = int(math.floor((x + half) / res))
    iy0 = int(math.floor((y - half) / res))
    iy1 = int(math.floor((y + half) / res))
    out = []
    for ix in range(ix0, ix1 + 1):
        for iy in range(iy0, iy1 + 1):
            px = (ix + 0.5) * res
            py = (iy + 0.5) * res
            if inside_box(px, py, x, y, yaw, length, width):
                out.append((px, py))
    return out


# rear, front, half-width in metres; own and teammate point-cloud masks.
FILTER_RANGES = {
    n: {"own": tuple(p["own_filter"]), "peer": tuple(p["peer_filter"])} for n, p in AGENTS.items()
}


def inside_filter(px, py, name, x, y, yaw, z=None, own=False):
    """Yaw-oriented XY exclusion at every height; z is intentionally ignored."""
    if name not in FILTER_RANGES:
        return inside_robot(px, py, name, x, y, yaw)
    aft, fore, half = FILTER_RANGES[name]["own" if own else "peer"]
    dx, dy = px - x, py - y
    c, s = math.cos(yaw), math.sin(yaw)
    forward, lateral = c * dx + s * dy, -s * dx + c * dy
    return -aft <= forward <= fore and abs(lateral) <= half
