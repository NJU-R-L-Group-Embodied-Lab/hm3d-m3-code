"""L1: shared 3-D occupancy evidence; no ROS, transport or robot control."""

import math
from collections import deque

import numpy as np

from lavira_m3.profiles import AGENTS

HAG_MIN, HAG_MAX = 0.05, 1.10
CONFIRM_HITS, CONFIRM_WINDOW, CONFIRM_SPAN = 3, 1.2, 0.4
CONFIRM_INTERVAL = 0.25
CLEAR_SECONDS, FREE_GAP = 30.0, 0.5
# Heights refer to the observation origin supplied by each robot adapter.
SENSOR_AGL = {n: p["lidar_height"] for n, p in AGENTS.items()}


def height_agl(z, sensor_z, name, mount=None):
    return z - sensor_z + (SENSOR_AGL[name] if mount is None else mount)


def _ray_free_voxels(origin, endpoints, resolution, occupied):
    """Conservative sampled 3-D rays, ending short of each measured surface.

    Only existing occupied voxels need empty evidence. No ray is invented for
    a missing return; endpoint voxels are vetoed by the caller's hit set.
    """
    if not occupied or len(endpoints) == 0:
        return set()
    keys = np.asarray(list(occupied), dtype=np.int64)
    row_type = np.dtype((np.void, keys.dtype.itemsize * 3))
    existing = np.ascontiguousarray(keys).view(row_type).ravel()
    result = set()
    # Rays need only be sampled as far as the furthest tracked obstacle.
    reach = (
        float(np.linalg.norm((keys + 0.5) * resolution - origin, axis=1).max())
        + math.sqrt(3) * resolution
    )
    for start in range(0, len(endpoints), 128):
        delta = np.asarray(endpoints[start : start + 128], dtype=float) - origin
        length = np.linalg.norm(delta, axis=1)
        valid = length > 2 * resolution
        delta = delta[valid]
        length = length[valid]
        if not len(length):
            continue
        # Half-voxel samples may miss a grazing voxel, conservatively retaining it.
        distance = np.arange(resolution * 0.5, min(float(length.max()), reach), resolution * 0.5)
        usable = distance[None, :] < length[:, None] - math.sqrt(3) * resolution
        samples = origin + delta[:, None, :] * (distance[None, :, None] / length[:, None, None])
        traversed = np.floor(samples[usable] / resolution).astype(np.int64)
        rows = np.ascontiguousarray(traversed).view(row_type).ravel()
        matched = keys[np.isin(existing, rows)]
        result.update(map(tuple, matched.tolist()))
    return result


class OccupancyEvidence:
    def __init__(
        self,
        resolution=0.1,
        clear_seconds=CLEAR_SECONDS,
        confirm_hits=CONFIRM_HITS,
        confirm_window=CONFIRM_WINDOW,
        max_points_per_cell=20,
    ):
        if (
            resolution <= 0
            or clear_seconds <= 0
            or confirm_hits < 2
            or confirm_window < CONFIRM_SPAN
        ):
            raise ValueError("invalid occupancy policy")
        self.resolution, self.clear_seconds = resolution, clear_seconds
        self.confirm_hits, self.confirm_window = confirm_hits, confirm_window
        self.max_points_per_cell = max_points_per_cell
        self.voxels = {}
        self.candidates = {}
        self.cleared = deque(maxlen=128)
        self.stamp = None

    def update(self, endpoints, origin, stamp, ray_endpoints=None):
        if not math.isfinite(stamp) or (self.stamp is not None and stamp <= self.stamp):
            return False
        points = np.asarray(endpoints, dtype=float).reshape(-1, 3)
        points = points[np.isfinite(points).all(axis=1)]
        if not np.isfinite(origin).all():
            raise ValueError("invalid ray origin")
        self.stamp = stamp
        keys, first = (
            np.unique(
                np.floor(points / self.resolution).astype(np.int64), axis=0, return_index=True
            )
            if len(points)
            else (np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=np.int64))
        )
        points = points[first]
        hits = {tuple(key): point for key, point in zip(keys, points)}
        rays = (
            points
            if ray_endpoints is None
            else np.asarray(ray_endpoints, dtype=float).reshape(-1, 3)
        )
        rays = rays[np.isfinite(rays).all(axis=1)]
        ray_keys, first = (
            np.unique(np.floor(rays / self.resolution).astype(np.int64), axis=0, return_index=True)
            if len(rays)
            else (np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=np.int64))
        )
        rays = rays[first]
        ray_hits = {tuple(key) for key in ray_keys}
        free = (
            _ray_free_voxels(np.asarray(origin), rays, self.resolution, self.voxels)
            - ray_hits
            - hits.keys()
        )
        for key, entry in list(self.voxels.items()):
            if key in hits:
                entry.update(point=hits[key], last_hit=stamp, free_s=0.0, last_free=None)
            elif key in ray_hits:
                # A real surface return vetoes clearing even if it is not a marking point.
                entry.update(last_hit=stamp, free_s=0.0, last_free=None)
            elif key in free:
                last = entry["last_free"]
                if last is not None and 0 < stamp - last <= FREE_GAP:
                    entry["free_s"] += stamp - last
                entry["last_free"] = stamp
                if entry["free_s"] >= self.clear_seconds:
                    self.cleared.append(
                        dict(
                            xyz=tuple((k + 0.5) * self.resolution for k in key),
                            stamp=stamp,
                            last_hit=entry["last_hit"],
                            free_s=entry["free_s"],
                        )
                    )
                    del self.voxels[key]
            else:
                entry["last_free"] = None
        for key, times in list(self.candidates.items()):
            while times and stamp - times[0] > self.confirm_window:
                times.popleft()
            if not times:
                del self.candidates[key]
        counts = {}
        for key in self.voxels:
            counts[key[:2]] = counts.get(key[:2], 0) + 1
        for key, point in hits.items():
            if key in self.voxels:
                continue
            times = self.candidates.setdefault(key, deque())
            # Normalize evidence across the 10 Hz accumulator and 3–4 Hz senders.
            if not times or stamp - times[-1] >= CONFIRM_INTERVAL - 1e-6:
                times.append(stamp)
            if len(times) >= self.confirm_hits and stamp - times[0] >= CONFIRM_SPAN - 1e-6:
                if counts.get(key[:2], 0) >= self.max_points_per_cell:
                    continue
                self.voxels[key] = dict(point=point, last_hit=stamp, free_s=0.0, last_free=None)
                counts[key[:2]] = counts.get(key[:2], 0) + 1
                del self.candidates[key]
        return True

    def discard_xy(self, keys):
        for collection in (self.voxels, self.candidates):
            for key in list(collection):
                if key[:2] in keys:
                    del collection[key]

    def diagnostics(self, watch_xy):
        rows = []
        for xy in watch_xy:
            nearby = []
            for key in self.voxels.keys() | self.candidates.keys():
                center = tuple((k + 0.5) * self.resolution for k in key)
                if abs(center[0] - xy[0]) > 0.16 or abs(center[1] - xy[1]) > 0.16:
                    continue
                entry = self.voxels.get(key)
                nearby.append(
                    dict(
                        xyz=center,
                        confirmed=entry is not None,
                        last_hit=entry["last_hit"] if entry else self.candidates[key][-1],
                        free_s=entry["free_s"] if entry else 0.0,
                        last_free=entry["last_free"] if entry else None,
                        candidate_hits=len(self.candidates.get(key, ())),
                    )
                )
            removed = [
                e
                for e in self.cleared
                if abs(e["xyz"][0] - xy[0]) <= 0.16 and abs(e["xyz"][1] - xy[1]) <= 0.16
            ]
            rows.append(dict(xy=xy, voxels=nearby, cleared=removed))
        return dict(
            stamp=self.stamp,
            confirmed=len(self.voxels),
            candidates=len(self.candidates),
            policy=dict(
                confirm_hits=self.confirm_hits,
                window_s=self.confirm_window,
                min_span_s=CONFIRM_SPAN,
                min_interval_s=CONFIRM_INTERVAL,
                clear_s=self.clear_seconds,
            ),
            watch=rows,
        )
