"""Aloha lidar/body → chassis base_link. Same numbers as mid360 tf_bridge."""

from __future__ import annotations

import math

# base_link -> livox/body, measured by tape: lidar is ahead of the chassis front, offset left, yaw +90 deg
BASE_TO_LIVOX = (0.290, 0.070, 0.735, 1.57079632679)


def _T(x, y, z, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return (
        (c, -s, 0.0, x),
        (s, c, 0.0, y),
        (0.0, 0.0, 1.0, z),
        (0.0, 0.0, 0.0, 1.0),
    )


def _mul(a, b):
    return tuple(
        tuple(sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)) for i in range(4)
    )


def _inv_yaw(x, y, z, yaw):
    c, s = math.cos(yaw), math.sin(yaw)
    return (-c * x - s * y, s * x - c * y, -z, -yaw)


def body_xy_yaw_to_base(x, y, yaw, extr=BASE_TO_LIVOX):
    """World pose of FAST-LIO body → world pose of base_link."""
    bx, by, bz, byaw = extr
    livox_base = _inv_yaw(bx, by, bz, byaw)
    world_body = _T(x, y, 0.0, yaw)
    lb = _T(*livox_base[:3], livox_base[3])
    world_base = _mul(world_body, lb)
    return world_base[0][3], world_base[1][3], math.atan2(world_base[1][0], world_base[0][0])


def quat_yaw(q):
    x, y, z, w = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
