"""Compose complete fused and local obstacle extents before planner inflation."""

import math

import numpy as np

from .world import clip_ix_window, in_world_xy


def compose_grid(known_xy, occupied_xy, peers_xy, accumulated, resolution, frame, bounds=None):
    """Return a stable world-aligned grid; free local cells never erase shared hits."""
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError("invalid map resolution")
    points = list(occupied_xy) + list(peers_xy)
    extent = list(known_xy) + points
    if accumulated is not None:
        msg = accumulated
        info = msg.info
        if msg.header.frame_id != frame or len(msg.data) != info.width * info.height:
            raise ValueError("accumulated map frame or dimensions mismatch")
        if not math.isfinite(info.resolution) or info.resolution <= 0:
            raise ValueError("invalid accumulated resolution")
        q = info.origin.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        c, s = math.cos(yaw), math.sin(yaw)
        ox, oy = info.origin.position.x, info.origin.position.y
        # Include the complete local window, even if no obstacle is currently hit.
        extent.extend(
            (ox + c * x - s * y, oy + s * x + c * y)
            for x, y in (
                (0, 0),
                (info.width * info.resolution, 0),
                (0, info.height * info.resolution),
                (info.width * info.resolution, info.height * info.resolution),
            )
        )
        yy, xx = np.where(np.asarray(msg.data).reshape(info.height, info.width) >= 100)
        lx, ly = (xx + 0.5) * info.resolution, (yy + 0.5) * info.resolution
        points.extend(zip(ox + c * lx - s * ly, oy + s * lx + c * ly))
    extent.extend(points)
    if any(not (math.isfinite(x) and math.isfinite(y)) for x, y in extent):
        raise ValueError("non-finite map coordinate")
    if extent:
        xs, ys = zip(*extent)
        pad = math.ceil(5.0 / resolution)
        new = clip_ix_window(
            math.floor(min(xs) / resolution) - pad,
            math.floor(min(ys) / resolution) - pad,
            math.ceil(max(xs) / resolution) + pad + 1,
            math.ceil(max(ys) / resolution) + pad + 1,
        )
    else:
        new = clip_ix_window(0, 0, 10, 10)
    if bounds is not None:
        new = clip_ix_window(
            min(bounds[0], new[0]),
            min(bounds[1], new[1]),
            max(bounds[2], new[2]),
            max(bounds[3], new[3]),
        )
    x0, y0, x1, y1 = new
    width, height = x1 - x0, y1 - y0
    # Bound allocation explicitly: never silently crop a map to make it fit.
    if width * height > 16_000_000:
        raise ValueError("complete planning map exceeds 16 million cells")
    data = np.zeros((height, width), dtype=np.int8)
    for x, y in points:
        if not in_world_xy(x, y):
            continue
        ix, iy = math.floor(x / resolution) - x0, math.floor(y / resolution) - y0
        if 0 <= ix < width and 0 <= iy < height:
            data[iy, ix] = 100
    return {
        "width": width,
        "height": height,
        "origin": (x0 * resolution, y0 * resolution),
        "resolution": resolution,
        "data": data.ravel().tolist(),
        "bounds": new,
    }
