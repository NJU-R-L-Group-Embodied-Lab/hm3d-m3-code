"""Per-robot 2D voxels. Confirmed-free decay like grid_accum; punch own / peer bodies."""

from __future__ import annotations

import math

from .bodies import inside_filter
from .components import remove_small_components
from .occupancy import HAG_MAX, HAG_MIN, SENSOR_AGL, OccupancyEvidence, height_agl
from .protocol import Meta, Record, now_ns
from .world import in_world_ix, in_world_xy

LIVE_R = 5.0
RAY_MAX = 30.0
RAY_MIN = 0.12
POSE_JUMP_M = 0.45
POSE_JUMP_YAW = 0.70
POSE_SLIP_MPS = 2.5


def lidar_agl(name, override=None):
    return float(override) if override is not None else SENSOR_AGL[name]


def _pose_xy_yaw(body):
    if body is None:
        return 0.0, 0.0, 0.0
    if isinstance(body, dict):
        return float(body["x"]), float(body["y"]), float(body.get("yaw") or 0.0)
    if len(body) == 2:
        return float(body[0]), float(body[1]), 0.0
    return float(body[0]), float(body[1]), float(body[2])


class Builder2D:
    def __init__(
        self, source_id, resolution=0.1, zmin=HAG_MIN, zmax=HAG_MAX, lidar_agl=None, decay_sec=30.0
    ):
        self.source_id = source_id
        self.zmin = zmin
        self.zmax = zmax
        self.lidar_agl = lidar_agl if lidar_agl is not None else None
        self.decay_sec = float(decay_sec)
        self.meta = Meta(
            resolution=resolution,
            map_id="%s-%s" % (source_id[:8], now_ns()),
            frame="map",
            mode=1,
        )
        self.cells = {}
        self.pending = {}
        self.evidence = OccupancyEvidence(resolution, decay_sec)
        self._last_body = None
        self._last_stamp = None

    def reset(self):
        self.cells.clear()
        self.pending.clear()
        self.evidence = OccupancyEvidence(self.meta.resolution, self.decay_sec)
        self.meta.map_id = "%s-%s" % (self.source_id[:8], now_ns())
        self.meta.timestamp = 0
        self.meta.pose_valid = False
        self._last_body = None
        self._last_stamp = None

    def _pose_jumped(self, own, stamp):
        prev, prev_t = self._last_body, self._last_stamp
        if prev is None or prev_t is None:
            return False
        dt = stamp - prev_t
        if dt <= 0:
            return False
        dist = math.hypot(own[0] - prev[0], own[1] - prev[1])
        dyaw = abs(math.atan2(math.sin(own[2] - prev[2]), math.cos(own[2] - prev[2])))
        return dist >= POSE_JUMP_M or dyaw >= POSE_JUMP_YAW or dist / dt >= POSE_SLIP_MPS

    def _bodies(self, own, peers):
        bx, by, byaw = own
        out = [(self.source_id, bx, by, byaw)]
        for p in peers or ():
            if isinstance(p, dict):
                name = p.get("name") or p.get("id")
                if not name or name == self.source_id:
                    continue
                out.append((name, float(p["x"]), float(p["y"]), float(p.get("yaw") or 0.0)))
        return out

    def _blocked(self, x, y, bodies, z=None):
        return any(
            inside_filter(x, y, name, px, py, yaw, z, own=(name == self.source_id))
            for name, px, py, yaw in bodies
        )

    def _drop(self, key):
        if key in self.cells:
            self.pending[key] = 0
        self.cells.pop(key, None)
        self.evidence.discard_xy({key})

    def scan(self, points_xyz, sensor_xyz, stamp_ns, body_xy, peers=None):
        sx, sy, sz = sensor_xyz
        own = _pose_xy_yaw(body_xy)
        bx, by, _ = own
        bodies = self._bodies(own, peers)
        stamp = stamp_ns * 1e-9 if stamp_ns > 1e8 else float(stamp_ns)
        if self._pose_jumped(own, stamp):
            return False
        self._last_body = own
        self._last_stamp = stamp
        points = []
        rays = []
        for x, y, z in points_xyz:
            if not all(math.isfinite(v) for v in (x, y, z)):
                continue
            if not RAY_MIN <= math.dist((x, y, z), sensor_xyz) <= RAY_MAX:
                continue
            # Real returns retain their clearing rays even outside the marking slice.
            rays.append((x, y, z))
            if not in_world_xy(x, y):
                continue
            hag = height_agl(z, sz, self.source_id, self.lidar_agl)
            if not self.zmin <= hag <= self.zmax:
                continue
            if self._blocked(x, y, bodies, z) or math.hypot(x - bx, y - by) > LIVE_R:
                continue
            if not RAY_MIN <= math.hypot(x - sx, y - sy) <= RAY_MAX:
                continue
            points.append((x, y, z))
        if not self.evidence.update(points, sensor_xyz, stamp, ray_endpoints=rays):
            return True
        previous = self.cells
        occupied = {key[:2] for key in self.evidence.voxels if in_world_ix(key[0], key[1])}
        self.cells = dict.fromkeys(occupied, 2)
        self._punch_bodies(bodies)
        # Filter the shared-grid output, retaining evidence so small clusters can grow.
        remove_small_components(self.cells)
        for key in previous.keys() - self.cells.keys():
            self.pending[key] = 0
        for key in self.cells.keys() - previous.keys():
            self.pending[key] = 2
        self.meta.timestamp = stamp_ns
        return True

    def _punch_bodies(self, bodies):
        res = self.meta.resolution
        origin = self.meta.origin
        drop = []
        xy_keys = set(self.cells) | {k[:2] for k in self.evidence.candidates}
        for ix, iy in xy_keys:
            x = origin[0] + (ix + 0.5) * res
            y = origin[1] + (iy + 0.5) * res
            if self._blocked(x, y, bodies):
                drop.append((ix, iy))
        for key in drop:
            self._drop(key)

    def set_pose(self, xyz, quat_xyzw):
        self.meta.pose_valid = True
        self.meta.position = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
        self.meta.orientation = tuple(float(q) for q in quat_xyzw)

    def snapshot_rows(self):
        return [Record(ix, iy, 0, st) for (ix, iy), st in self.cells.items() if st in (1, 2)]

    def delta_rows(self):
        rows = [Record(ix, iy, 0, st) for (ix, iy), st in self.pending.items()]
        self.pending.clear()
        return rows
