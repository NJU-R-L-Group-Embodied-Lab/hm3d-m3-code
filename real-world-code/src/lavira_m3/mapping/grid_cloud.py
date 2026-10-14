"""OccupancyGrid lethal cells → thin XYZ points for EGO grid_map/cloud."""

from __future__ import annotations

import math
import struct

LETHAL = 100
POINT_Z = (0.05,)


def lethal_xy(data, width, height, resolution, origin_xy):
    ox, oy = float(origin_xy[0]), float(origin_xy[1])
    res = float(resolution)
    if width <= 0 or height <= 0 or res <= 0 or len(data) < width * height:
        return []
    points = []
    for iy in range(height):
        row = iy * width
        y = oy + (iy + 0.5) * res
        for ix in range(width):
            if int(data[row + ix]) < LETHAL:
                continue
            points.append((ox + (ix + 0.5) * res, y))
    return points


def xyz_from_xy(xy, heights=POINT_Z):
    pts = []
    for x, y in xy:
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        for z in heights:
            pts.append((float(x), float(y), float(z)))
    return pts


def pack_xyz(points):
    return b"".join(struct.pack("<fff", x, y, z) for x, y, z in points)
