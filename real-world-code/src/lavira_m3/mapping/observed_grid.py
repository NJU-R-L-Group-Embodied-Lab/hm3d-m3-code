"""Only fused robot observations become environment cells; no prebuilt-map input."""

import math

from .inflate import HARD_INFLATE_M, SOFT_INFLATE_M, inflate_layers, open_inflate_start
from .peers import bounds, occupied_cells
from .world import clip_ix_window, in_world_ix

# Border around observed cells. STEP_MAX is 5 m; 5 m (50 cells) keeps a
# 5 m VA on the rectangle so the planner can search. Unseen cells are 0 (free).
UNKNOWN_BORDER_M = 5.0
UNKNOWN_COST = 0


def full_grids(meta, rows, poses, inflate_m=None, soft_m=None):
    if inflate_m is None:
        inflate_m = HARD_INFLATE_M
    if soft_m is None:
        soft_m = SOFT_INFLATE_M
    inflate_m = float(inflate_m)
    if meta is not None and (
        meta.frame != "map" or abs(meta.resolution - 0.1) > 1e-8 or tuple(meta.origin) != (0, 0, 0)
    ):
        raise ValueError("unexpected observation grid contract")
    cells = {}
    for r in rows:
        if r.iz != 0 or r.state not in (1, 2) or type(r.ix) is not int or type(r.iy) is not int:
            raise ValueError("invalid observed cell")
        if not in_world_ix(r.ix, r.iy):
            continue
        cells[r.ix, r.iy] = max(cells.get((r.ix, r.iy), 0), r.state)
    if meta is None and cells:
        raise ValueError("observations require metadata")
    # Body poses only grow the unknown extent. Teammate envelopes are stamped separately.
    view = poses.get("peer_view", {})
    extent = list(cells)
    for p in ([view["own"]] if view.get("own") else []) + view.get("peers", []):
        a, b, c, d = bounds(p)
        extent.extend(
            ((math.floor(a / 0.1), math.floor(b / 0.1)), (math.ceil(c / 0.1), math.ceil(d / 0.1)))
        )
    pad = max(int(math.ceil(UNKNOWN_BORDER_M / 0.1)), int(math.ceil(inflate_m / 0.1)) + 2)
    if extent:
        xs, ys = zip(*extent)
        x0, y0, x1, y1 = clip_ix_window(
            min(xs) - pad, min(ys) - pad, max(xs) + pad + 1, max(ys) + pad + 1
        )
    else:
        x0, y0, x1, y1 = clip_ix_window(-5, -5, 5, 5)
    width, height = x1 - x0, y1 - y0
    if not 0 < width * height <= 2_000_000:
        raise ValueError("observed map exceeds two million cells")
    geometry = (width, height, 0.1, x0 * 0.1, y0 * 0.1)
    environment = [UNKNOWN_COST] * (width * height)
    for (ix, iy), state in cells.items():
        environment[(iy - y0) * width + ix - x0] = 100 if state == 2 else 0
    peers = [UNKNOWN_COST] * len(environment)
    for ix, iy in occupied_cells(view, geometry):
        peers[iy * width + ix] = 100
    thin_env = list(environment)
    thin_peers = list(peers)
    _, environment = inflate_layers(environment, width, height, 0.1, inflate_m, soft_m)
    _, peers = inflate_layers(peers, width, height, 0.1, inflate_m, soft_m)
    planning = [max(a, b) for a, b in zip(environment, peers)]
    own = view.get("own") or {}
    try:
        px, py = float(own["x"]), float(own["y"])
    except (KeyError, TypeError, ValueError):
        px = py = None
    if px is not None and math.isfinite(px) and math.isfinite(py):
        thin_occ = [100 if a >= 100 or b >= 100 else 0 for a, b in zip(thin_env, thin_peers)]
        open_inflate_start(planning, thin_occ, width, height, 0.1, x0 * 0.1, y0 * 0.1, px, py)
    return geometry, thin_env, peers, environment, planning
