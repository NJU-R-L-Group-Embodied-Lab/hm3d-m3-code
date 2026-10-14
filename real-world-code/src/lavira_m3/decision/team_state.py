"""Initial team observations, leader lease and assignment state."""

from __future__ import annotations

import time
from typing import Any, Dict, List

from .config import t0_leader_name, t0_robots


class T0State:
    def __init__(self) -> None:
        self.expected = list(t0_robots())
        self.online: Dict[str, float] = {}
        self.observes: Dict[str, Dict[str, Any]] = {}
        self.assignment: Dict[str, Dict[str, Any]] = {}
        self.team_reasoning = ""
        self.panoramic_observation: Any = {}
        self.map_understanding = ""
        self.image_goal_scenes: Dict[str, str] = {}
        self.la_time_ms = None
        self.raw = ""
        self.lock_robot = ""
        self.lock_until = 0.0
        self.failed = ""

    def reset(self, expected=None, keep_online: bool = False) -> None:
        if expected is not None:
            self.expected = list(expected)
        self.observes = {}
        self.assignment = {}
        self.team_reasoning = ""
        self.panoramic_observation = {}
        self.map_understanding = ""
        self.image_goal_scenes = {}
        self.la_time_ms = None
        self.raw = ""
        self.lock_robot = ""
        self.lock_until = 0.0
        self.failed = ""
        if keep_online:
            self.online = {k: v for k, v in self.online.items() if k in self.expected}
        else:
            self.online = {}

    def mark_online(self, robot: str) -> None:
        if robot in self.expected:
            self.online[robot] = time.time()

    def online_names(self) -> List[str]:
        return [n for n in self.expected if n in self.online]

    def offline_names(self) -> List[str]:
        return [n for n in self.expected if n not in self.online]

    def missing_names(self) -> List[str]:
        return [n for n in self.online_names() if n not in self.observes]

    def ready(self) -> bool:
        required = self.online_names()
        return bool(required) and all(name in self.observes for name in required)

    def leader(self) -> str:
        designated = t0_leader_name()
        online = self.online_names()
        if designated in online:
            return designated
        return online[0] if online else designated

    def lock_holder(self) -> str:
        if self.lock_robot and time.time() < self.lock_until:
            return self.lock_robot
        return ""

    def public(self) -> Dict[str, Any]:
        assignments = {}
        for name, rec in self.assignment.items():
            if not isinstance(rec, dict):
                continue
            assignments[name] = {
                key: rec.get(key) for key in ("action", "claim_goal_id", "region") if key in rec
            }
        return {
            "expected": list(self.expected),
            "online": self.online_names(),
            "offline": self.offline_names(),
            "uploaded": sorted(self.observes),
            "missing": self.missing_names(),
            "ready": self.ready(),
            "assigned": bool(self.assignment),
            "locked_by": self.lock_holder(),
            "failed": self.failed,
            "leader": self.leader(),
            "panoramic_observation": self.panoramic_observation,
            "map_understanding": self.map_understanding,
            "image_goal_scenes": dict(self.image_goal_scenes),
            "team_reasoning": self.team_reasoning,
            "la_time_ms": self.la_time_ms,
            "assignments": assignments,
            "raw": self.raw,
        }
