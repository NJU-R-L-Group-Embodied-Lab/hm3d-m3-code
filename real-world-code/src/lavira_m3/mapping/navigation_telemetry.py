"""Bounded, read-only telemetry cache. Never calls a navigation/control API."""

import copy
import math
import threading
import time
from collections import deque

from lavira_m3.profiles import AGENTS

ROBOTS = tuple(AGENTS)


class NavigationTelemetry:
    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.lock = threading.Lock()
        self.records = {}
        self.interest = {}
        self.retired = {n: deque(maxlen=16) for n in ROBOTS}

    def ingest(self, robot, body):
        if robot not in ROBOTS or not isinstance(body, dict):
            raise ValueError("unknown robot or invalid packet")
        session, seq = body.get("session"), body.get("seq")
        if (
            not isinstance(session, str)
            or not 1 <= len(session) <= 64
            or not isinstance(seq, int)
            or seq < 0
        ):
            raise ValueError("invalid session/sequence")
        state = body.get("state")
        if not isinstance(state, dict) or len(str(state)) > 8000:
            raise ValueError("invalid state")
        ages = state.get("ages", {})
        if not isinstance(ages, dict) or any(
            type(v) not in (int, float) or v < 0 for v in ages.values()
        ):
            raise ValueError("invalid source ages")
        if type(state.get("trajectory_remaining_s", 0)) not in (int, float):
            raise ValueError("invalid trajectory expiry")

        def check(value):
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("non-finite telemetry")
            if isinstance(value, dict):
                for item in value.values():
                    check(item)
            if isinstance(value, list):
                for item in value:
                    check(item)

        check(body)
        for key, limit in (("path", 500), ("reference", 120), ("prediction", 120)):
            points = body.get(key, [])
            if not isinstance(points, list) or len(points) > limit:
                raise ValueError("invalid " + key)
            for p in points:
                if (
                    not isinstance(p, list)
                    or len(p) != 2
                    or any(type(x) not in (int, float) for x in p)
                ):
                    raise ValueError("invalid point")
        pose = state.get("pose")
        if pose is not None and (
            not isinstance(pose, list)
            or len(pose) != 3
            or any(type(x) not in (int, float) for x in pose)
        ):
            raise ValueError("invalid pose")
        now = self.clock()
        with self.lock:
            old = self.records.get(robot)
            if session in self.retired[robot]:
                return {"ok": False, "error": "retired session"}
            if old and old["session"] == session and seq <= old["seq"]:
                return {"ok": False, "error": "out-of-order packet"}
            fresh = old is None or old["session"] != session
            if fresh and old:
                self.retired[robot].append(old["session"])
            revision = str(body.get("revision", ""))[:100]
            same_revision = not fresh and old["revision"] == revision
            has_path = "path" in body or (same_revision and old["has_path"])
            rec = dict(
                session=session,
                seq=seq,
                state=copy.deepcopy(state),
                received=now,
                revision=revision,
                has_path=has_path,
                path=copy.deepcopy(body.get("path", old["path"] if same_revision else [])),
                reference=copy.deepcopy(body.get("reference", [])),
                prediction=copy.deepcopy(body.get("prediction", [])),
                trail=deque(maxlen=150) if fresh else old["trail"],
            )
            if pose and state.get("ages", {}).get("odom", 999) < 1:
                rec["trail"].append((now, pose[:2]))
            self.records[robot] = rec
            return {
                "ok": True,
                "detail": self.interest.get(robot, 0) > now,
                "need_path": not has_path,
            }

    def snapshot(self, selected="", cursors=None, include_all=False):
        now = self.clock()
        out = {}
        cursors = cursors or {}
        with self.lock:
            if selected in ROBOTS:
                self.interest[selected] = now + 2
            for robot in ROBOTS:
                rec = self.records.get(robot)
                if not rec:
                    out[robot] = {"live": False, "state": {"phase": "offline"}}
                    continue
                age = max(0, now - rec["received"])
                state = copy.deepcopy(rec["state"])
                state["ages"] = {
                    k: round(v + age, 3)
                    for k, v in state.get("ages", {}).items()
                    if type(v) in (int, float)
                }
                state["trajectory_remaining_s"] = max(
                    0, float(state.get("trajectory_remaining_s", 0)) - age
                )
                live = age <= 1.5
                cursor = (
                    rec["session"] + ":" + rec["revision"] + ("" if rec["has_path"] else ":pending")
                )
                item = dict(
                    live=live,
                    age_s=round(age, 3),
                    session=rec["session"],
                    state=state,
                    cursor=cursor,
                )
                if cursors.get(robot) != cursor:
                    item["path"] = rec["path"]
                if robot == selected or include_all:
                    item.update(
                        reference=rec["reference"] if live else [],
                        prediction=rec["prediction"] if live else [],
                        trail=[p for t, p in rec["trail"] if now - t <= 30],
                    )
                out[robot] = item
        return {"robots": out, "selected": selected, "poll_ms": 500}
