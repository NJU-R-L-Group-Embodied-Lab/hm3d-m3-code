"""Fixed 100 m × 100 m canvas. Go1 birth (0, 0) is world origin.

Cell IDs are world-absolute: ix = floor(x / 0.1) in [-500, 499].
Do not remap IDs when the observed AABB moves. Out-of-canvas points are dropped.
"""

from __future__ import annotations

RES_M = 0.1
HALF_M = 50.0
ORIGIN = (0.0, 0.0, 0.0)
IX_MIN = -500
IX_MAX = 499


def in_world_xy(x, y):
    return -HALF_M <= float(x) < HALF_M and -HALF_M <= float(y) < HALF_M


def in_world_ix(ix, iy):
    return IX_MIN <= int(ix) <= IX_MAX and IX_MIN <= int(iy) <= IX_MAX


def world_aabb():
    return (-HALF_M, -HALF_M, HALF_M, HALF_M)


def clip_ix_window(x0, y0, x1, y1):
    return (
        max(IX_MIN, int(x0)),
        max(IX_MIN, int(y0)),
        min(IX_MAX + 1, int(x1)),
        min(IX_MAX + 1, int(y1)),
    )
