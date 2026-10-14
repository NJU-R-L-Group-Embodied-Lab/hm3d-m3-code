"""Incremental fused-map client. Same contract GO2 receiver_ros2 uses."""

from __future__ import annotations

import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .protocol import Meta


class MapDownload:
    def __init__(self):
        self.cells = {}
        self.map_id = None
        self.seq = -1
        self.ready = False
        self.resolution = 0.1
        self.origin = (0.0, 0.0, 0.0)
        self.frame = "map"
        self.resync = True

    def query(self):
        if self.resync or not self.map_id:
            return None, -1
        return self.map_id, self.seq

    def meta(self):
        origin = self.origin
        if len(origin) == 2:
            origin = (float(origin[0]), float(origin[1]), 0.0)
        return Meta(
            resolution=float(self.resolution),
            origin=tuple(float(v) for v in origin),
            frame=self.frame or "map",
        )

    def _remember_meta(self, update):
        if update.get("resolution") is not None:
            self.resolution = float(update["resolution"])
        if update.get("origin") is not None:
            origin = list(update["origin"])
            if len(origin) == 2:
                origin.append(0.0)
            self.origin = tuple(float(v) for v in origin[:3])
        if update.get("frame"):
            self.frame = str(update["frame"])
        if "ready" in update:
            self.ready = bool(update["ready"])

    def apply(self, update):
        if not update or update.get("type") not in ("snapshot", "delta"):
            self.resync = True
            return False
        if update["type"] == "snapshot":
            nxt = {}
            for x, y, state in update.get("cells") or []:
                state = int(state)
                if state:
                    nxt[(int(x), int(y))] = state
            self.cells = nxt
            self.map_id = update.get("map_id")
            self.seq = int(update.get("seq", 0))
            self._remember_meta(update)
            self.ready = bool(update.get("ready", True))
            self.resync = False
            return True
        if update.get("map_id") != self.map_id or update.get("base") != self.seq:
            self.resync = True
            return False
        for x, y, state in update.get("cells") or []:
            key = (int(x), int(y))
            state = int(state)
            if state == 0:
                self.cells.pop(key, None)
            else:
                self.cells[key] = state
        self.seq = int(update["seq"])
        self._remember_meta(update)
        return True

    def pull(self, host, port, timeout=3.0):
        last = False
        for _ in range(2):
            map_id, since = self.query()
            query = {"since": str(since)}
            if map_id:
                query["map_id"] = map_id
            req = Request(
                "http://%s:%s/v1/map-updates?%s" % (host, port, urlencode(query)),
                headers={"Cache-Control": "no-store"},
            )
            with urlopen(req, timeout=timeout) as response:
                update = json.loads(response.read().decode())
            last = self.apply(update)
            if last or not self.resync:
                return last
        return last
