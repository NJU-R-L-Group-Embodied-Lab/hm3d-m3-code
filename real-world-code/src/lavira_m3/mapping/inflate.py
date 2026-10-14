"""Build the point-planning hard band and a passable soft band."""

from __future__ import annotations

import math

from lavira_m3.profiles import AGENTS

# Hard lethal only. EGO has no soft-cost band, so keep soft at 0.
HARD_INFLATE_M = 0.80
SOFT_INFLATE_M = 0.0
INFLATE_M = HARD_INFLATE_M
PLAN_INFLATE = {n: p["inflate"] for n, p in AGENTS.items()}
PLAN_SOFT_INFLATE = {n: p["soft_inflate"] for n, p in AGENTS.items()}
SOFT_COST_NEAR = 99
SOFT_COST_FAR = 20


def inflate_lethal(data, width, height, resolution, radius_m=INFLATE_M):
    if width <= 0 or height <= 0 or resolution <= 0:
        raise ValueError("invalid grid")
    if radius_m < 0:
        raise ValueError("invalid inflate radius")
    out = list(data)
    if radius_m == 0:
        return out
    r_cells = radius_m / resolution
    r2 = r_cells * r_cells
    reach = int(math.ceil(r_cells))
    seeds = [(i % width, i // width) for i, value in enumerate(data) if value >= 100]
    for ox, oy in seeds:
        for dy in range(-reach, reach + 1):
            for dx in range(-reach, reach + 1):
                if dx * dx + dy * dy > r2 + 1e-9:
                    continue
                x, y = ox + dx, oy + dy
                if 0 <= x < width and 0 <= y < height:
                    index = y * width + x
                    if out[index] < 100:
                        out[index] = 100
    return out


def inflate_layers(data, width, height, resolution, hard_m=HARD_INFLATE_M, soft_m=SOFT_INFLATE_M):
    """Return hard lethal cells and a combined hard+soft cost layer."""
    hard = inflate_lethal(data, width, height, resolution, hard_m)
    combined = list(hard)
    total = float(hard_m) + float(soft_m)
    if total <= hard_m:
        return hard, combined
    outer = inflate_lethal(data, width, height, resolution, total)
    seeds = [(i % width, i // width) for i, value in enumerate(data) if value >= 100]
    hard_cells = float(hard_m) / float(resolution)
    soft_cells = float(soft_m) / float(resolution)
    total_cells = hard_cells + soft_cells
    for index, value in enumerate(outer):
        if value < 100 or combined[index] >= 100:
            continue
        x, y = index % width, index // width
        distance_cells = min(math.hypot(x - ox, y - oy) for ox, oy in seeds)
        if distance_cells <= hard_cells + 1e-9:
            continue
        fraction = max(0.0, min(1.0, (total_cells - distance_cells) / soft_cells))
        combined[index] = int(round(SOFT_COST_FAR + (SOFT_COST_NEAR - SOFT_COST_FAR) * fraction))
    return hard, combined


def open_inflate_start(planning, thin, width, height, resolution, ox, oy, x, y):
    """Allow a restart from an inflate halo without opening the rest of the halo.

    Planning stays lethal on inflate. If the pose is on raw occupied (thin>=100),
    do nothing. If it is only in the inflate halo, clear the shortest halo-only
    walk to the nearest free cell so A* can leave. Other halo cells stay 100.
    """
    if (
        planning is None
        or thin is None
        or width <= 0
        or height <= 0
        or resolution <= 0
        or len(planning) < width * height
        or len(thin) < width * height
    ):
        return planning
    if not all(math.isfinite(v) for v in (resolution, ox, oy, x, y)):
        return planning
    ix = int(math.floor((float(x) - float(ox)) / resolution))
    iy = int(math.floor((float(y) - float(oy)) / resolution))
    if not (0 <= ix < width and 0 <= iy < height):
        return planning
    start = iy * width + ix
    if thin[start] >= 100 or planning[start] != 100:
        return planning
    from collections import deque

    prev = {(ix, iy): None}
    queue = deque([(ix, iy)])
    last_halo = None
    # Edge-connected cells form a continuous corridor without diagonal corner cuts.
    steps = ((1, 0), (-1, 0), (0, 1), (0, -1))
    while queue:
        cx, cy = queue.popleft()
        for dx, dy in steps:
            nx, ny = cx + dx, cy + dy
            if not (0 <= nx < width and 0 <= ny < height) or (nx, ny) in prev:
                continue
            index = ny * width + nx
            if thin[index] >= 100:
                continue
            if planning[index] == 0:
                last_halo = (cx, cy)
                queue.clear()
                break
            if planning[index] != 100:
                continue
            prev[(nx, ny)] = (cx, cy)
            queue.append((nx, ny))
    if last_halo is None:
        return planning
    cell = last_halo
    while cell is not None:
        planning[cell[1] * width + cell[0]] = 0
        cell = prev[cell]
    return planning
