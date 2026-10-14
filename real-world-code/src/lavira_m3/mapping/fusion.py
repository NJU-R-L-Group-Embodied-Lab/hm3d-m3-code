"""Configured-source 2D union. Occupied wins. Stale pose keeps last map."""

from __future__ import annotations

import math
import time
from collections import deque

from lavira_m3.profiles import AGENTS

from .bodies import POSE_LIVE_SEC, footprint, raster_footprint, yaw_from_quat
from .world import HALF_M, in_world_ix


class Store:
    def __init__(self):
        self.ready = False
        self.meta = None
        self.cells = {}
        self.seq = 0
        self.map_id = ""

    def apply(self, meta, rows):
        if meta.type == 1:
            if self.ready and self.map_id == meta.map_id and meta.seq <= self.seq:
                return False
            nxt = {}
            for r in rows:
                if r.iz != 0 or not in_world_ix(r.ix, r.iy):
                    continue
                if r.state in (1, 2):
                    nxt[(r.ix, r.iy)] = r.state
            self.cells = nxt
            self.ready = True
            self.meta = meta
            self.seq = meta.seq
            self.map_id = meta.map_id
            return True
        if (
            not self.ready
            or self.map_id != meta.map_id
            or self.seq != meta.base
            or meta.seq != meta.base + 1
        ):
            return False
        for r in rows:
            if r.iz != 0 or not in_world_ix(r.ix, r.iy):
                continue
            key = (r.ix, r.iy)
            if r.state == 0:
                self.cells.pop(key, None)
            else:
                self.cells[key] = r.state
        self.meta = meta
        self.seq = meta.seq
        return True


class Fusion2D:
    NAMES = tuple(AGENTS)

    def __init__(self):
        self.sources = {n: Store() for n in self.NAMES}
        self.fused = Store()
        self.epoch = str(time.time_ns())
        self.stream_id = self.epoch
        self.history = deque(maxlen=64)
        self.poses = {}
        self.heard = {}
        self.map_heard = {}
        self.last_reject = ""

    def clear(self):
        self.__init__()

    def _touch(self, name, meta=None, pose=None):
        now = time.time()
        self.heard[name] = now
        rec = pose
        if rec is None and meta is not None and meta.pose_valid:
            rec = {
                "position": list(meta.position),
                "orientation": list(meta.orientation),
            }
        if rec is None:
            return
        q = rec.get("orientation") or [0.0, 0.0, 0.0, 1.0]
        pos = rec.get("position") or [0.0, 0.0, 0.0]
        self.poses[name] = {
            "position": [float(pos[0]), float(pos[1]), float(pos[2] if len(pos) > 2 else 0.0)],
            "orientation": [float(q[0]), float(q[1]), float(q[2]), float(q[3])],
            "yaw": yaw_from_quat(q),
            "t": now,
        }
        if rec.get("status"):
            self.poses[name]["status"] = str(rec["status"])

    def set_pose(self, name, position, orientation, status=None):
        if name not in self.sources:
            return False
        pose = {"position": position, "orientation": orientation}
        if status:
            pose["status"] = status
        self._touch(name, pose=pose)
        return True

    def apply(self, name, meta, rows):
        if name not in self.sources:
            return False
        previous = self.poses.get(name)
        if previous and meta.pose_valid and self.sources[name].map_id == meta.map_id:
            distance = math.hypot(
                meta.position[0] - previous["position"][0],
                meta.position[1] - previous["position"][1],
            )
            yaw = yaw_from_quat(meta.orientation) - previous["yaw"]
            if distance >= 0.45 or abs(math.atan2(math.sin(yaw), math.cos(yaw))) >= 0.70:
                self.last_reject = "pose_jump"
                return False
        self.last_reject = ""
        if not self.sources[name].apply(meta, rows):
            return False
        self._touch(name, meta=meta)
        self.map_heard[name] = time.time()
        self._rebuild()
        return True

    def _rebuild(self):
        merged = {}
        last_meta = None
        for src in self.sources.values():
            if not src.ready:
                continue
            last_meta = src.meta
            for k, st in src.cells.items():
                prev = merged.get(k)
                if prev is None or st > prev:
                    merged[k] = st
        # Keep occupied components of at least three cells (8-connectivity).
        unseen = {k for k, state in merged.items() if state == 2}
        while unseen:
            component = [unseen.pop()]
            for x, y in component:
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        neighbor = (x + dx, y + dy)
                        if neighbor in unseen:
                            unseen.remove(neighbor)
                            component.append(neighbor)
            if len(component) < 3:
                for key in component:
                    del merged[key]
        changes = [
            [x, y, merged.get((x, y), 0)]
            for x, y in self.fused.cells.keys() | merged.keys()
            if self.fused.cells.get((x, y), 0) != merged.get((x, y), 0)
        ]
        self.fused.cells = merged
        self.fused.ready = bool(merged) or any(s.ready for s in self.sources.values())
        if last_meta is not None:
            self.fused.meta = last_meta
        self.fused.seq += 1
        self.history.append((self.fused.seq, changes))

    def updates(self, map_id=None, since=-1):
        seq = self.fused.seq
        delta = (
            map_id == self.stream_id
            and 0 <= since <= seq
            and (since == seq or self.history and since >= self.history[0][0] - 1)
        )
        cells = {}
        if delta:
            for revision, changes in self.history:
                if revision > since:
                    for x, y, state in changes:
                        cells[x, y] = state
        else:
            cells = self.fused.cells
        meta = self.fused.meta
        return dict(
            type="delta" if delta else "snapshot",
            map_id=self.stream_id,
            seq=seq,
            base=since if delta else None,
            ready=self.fused.ready,
            resolution=meta.resolution if meta else 0.1,
            origin=list(meta.origin) if meta else [0, 0, 0],
            frame=meta.frame if meta else "map",
            cells=[[x, y, state] for (x, y), state in cells.items()],
        )

    def occupied_xy(self):
        """Rasterized footprints of all live configured agents."""
        res = self.fused.meta.resolution if self.fused.meta else 0.1
        now = time.time()
        out = []
        for name in self.NAMES:
            view = self._pose_view(name, now)
            if not view.get("live") or "position" not in view:
                continue
            length, width = footprint(name)
            out.extend(
                raster_footprint(
                    view["position"][0],
                    view["position"][1],
                    view.get("yaw") or 0.0,
                    length,
                    width,
                    res,
                )
            )
        return res, out

    def _pose_view(self, name, now):
        pose = self.poses.get(name)
        heard = self.heard.get(name)
        age = None if heard is None else now - heard
        live = age is not None and age <= POSE_LIVE_SEC
        if pose is None:
            return {"age_s": age, "live": live}
        out = dict(pose)
        out["age_s"] = age
        out["live"] = live
        return out

    def live_peers(self, except_name=None):
        now = time.time()
        out = []
        for name in self.NAMES:
            if name == except_name:
                continue
            view = self._pose_view(name, now)
            if not view.get("live") or "position" not in view:
                continue
            pos = view["position"]
            out.append(
                {
                    "name": name,
                    "x": pos[0],
                    "y": pos[1],
                    "yaw": view.get("yaw") or 0.0,
                }
            )
        return out

    def source_cells(self, name, now=None, require_live=True):
        src = self.sources.get(name)
        if src is None or not src.ready:
            return []
        now = time.time() if now is None else now
        if require_live and not self._pose_view(name, now).get("live"):
            return []
        return [[ix, iy] for (ix, iy), st in src.cells.items() if st == 2]

    def overlap_cells(self, now=None):
        """Cells occupied by two or more live robots."""
        hits = {}
        for name in self.NAMES:
            for ix, iy in self.source_cells(name, now):
                hits[(ix, iy)] = hits.get((ix, iy), 0) + 1
        return [[ix, iy] for (ix, iy), count in hits.items() if count >= 2]

    def scene(self):
        now = time.time()
        sources = {n: self.source_cells(n, now, require_live=False) for n in self.NAMES}
        counts = {}
        for cells in sources.values():
            for cell in cells:
                key = tuple(cell)
                counts[key] = counts.get(key, 0) + 1
        meta = self.fused.meta
        res = meta.resolution if meta else 0.1
        origin = list(meta.origin[:2]) if meta else [0.0, 0.0]
        fused = [[ix, iy] for (ix, iy), st in self.fused.cells.items() if st == 2]
        aabb = None
        if fused:
            from .observed_grid import UNKNOWN_BORDER_M

            ixs = [int(cell[0]) for cell in fused]
            iys = [int(cell[1]) for cell in fused]
            pad = float(UNKNOWN_BORDER_M)
            aabb = [
                max(-HALF_M, origin[0] + min(ixs) * res - pad),
                max(-HALF_M, origin[1] + min(iys) * res - pad),
                min(HALF_M, origin[0] + (max(ixs) + 1) * res + pad),
                min(HALF_M, origin[1] + (max(iys) + 1) * res + pad),
            ]
        return {
            "resolution": res,
            "origin": origin,
            "frame": meta.frame if meta else "map",
            "aabb": aabb,
            "fused": fused,
            "overlap": [list(key) for key, count in counts.items() if count >= 2],
            "sources": sources,
        }

    def status(self):
        now = time.time()
        occ = sum(1 for s in self.fused.cells.values() if s == 2)
        free = sum(1 for s in self.fused.cells.values() if s == 1)
        sources = {}
        for n, s in self.sources.items():
            sources[n] = {
                "ready": s.ready,
                "seq": s.seq,
                "occupied": sum(1 for v in s.cells.values() if v == 2),
                "pose": self._pose_view(n, now),
            }
        return {
            "source": "fused",
            "ready": self.fused.ready,
            "resolution": self.fused.meta.resolution if self.fused.meta else 0.1,
            "occupied": occ,
            "free": free,
            "epoch": self.epoch,
            "sources": sources,
        }
