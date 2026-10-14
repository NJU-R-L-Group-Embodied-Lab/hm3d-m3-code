"""Build the per-robot planning layers the dogs pull over HTTP."""

from __future__ import annotations

from .inflate import PLAN_INFLATE, PLAN_SOFT_INFLATE
from .observed_grid import full_grids
from .peers import PeerCache, occupied_cells
from .protocol import Record

ROBOTS = tuple(PLAN_INFLATE)


def _abs_cells(data, width, x0, y0):
    cells = []
    for index, value in enumerate(data):
        if value >= 100:
            cells.append([index % width + x0, index // width + y0])
    return cells


def compact_overlay(geometry, thin_env, view, planning):
    width, _height, resolution, ox, oy = geometry
    x0 = int(round(ox / resolution))
    y0 = int(round(oy / resolution))
    occupied = {(ix + x0, iy + y0) for ix, iy in occupied_cells(view, geometry)}
    for ix, iy in _abs_cells(thin_env, width, x0, y0):
        occupied.add((ix, iy))
    halo = []
    for ix, iy in _abs_cells(planning, width, x0, y0):
        if (ix, iy) not in occupied:
            halo.append([ix, iy])
    return [list(cell) for cell in occupied], halo


def layers_for_robot(status, fused_cells, meta, robot, now=None):
    if robot not in PLAN_INFLATE:
        raise ValueError("unknown planning robot")
    inflate_m = PLAN_INFLATE[robot]
    cache = PeerCache()
    cache.observe(status, now=now)
    view = cache.view(robot, now=now)
    rows = [Record(ix, iy, 0, st) for (ix, iy), st in fused_cells.items()]
    geometry, thin_env, peers, environment, planning = full_grids(
        meta, rows, {"peer_view": view}, inflate_m=inflate_m, soft_m=PLAN_SOFT_INFLATE[robot]
    )
    width, height, resolution, ox, oy = geometry
    x0 = int(round(ox / resolution))
    y0 = int(round(oy / resolution))
    occupied, halo = compact_overlay(geometry, thin_env, view, planning)
    soft = []
    for index, value in enumerate(planning):
        if 0 < value < 100:
            soft.append([index % width + x0, index // width + y0])
    return {
        "available": True,
        "robot": robot,
        "inflate_m": inflate_m,
        "hard_inflate_m": inflate_m,
        "soft_inflate_m": PLAN_SOFT_INFLATE[robot],
        "width": int(width),
        "height": int(height),
        "resolution": float(resolution),
        "origin": [float(ox), float(oy)],
        "peers": peers,
        "environment": environment,
        "planning": planning,
        "occupied": occupied,
        "inflate": halo,
        "hard": halo,
        "soft": soft,
        "peer_count": sum(1 for value in peers if value >= 100),
    }
