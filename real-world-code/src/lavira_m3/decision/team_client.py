"""HTTP client + text builders for Goal Board and Progress Analysis."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener

log = logging.getLogger("lavira_m3.board")


def _opener():
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
    os.environ["no_proxy"] = os.environ["NO_PROXY"]
    return build_opener(ProxyHandler({}))


def _join_url(hub: str, spec: str) -> str:
    if spec.startswith("http://") or spec.startswith("https://"):
        return spec
    return hub.rstrip("/") + (spec if spec.startswith("/") else "/" + spec)


class TeamClient:
    def __init__(self, hub: str, robot: str):
        self.hub = hub.rstrip("/")
        self.robot = robot
        self._op = _opener()

    def _json(
        self, method: str, path: str, body: Optional[Dict] = None, timeout: float = 3.0
    ) -> Dict:
        data = None if body is None else json.dumps(body).encode()
        req = Request(
            self.hub + path,
            data=data,
            method=method,
            headers={"Content-Type": "application/json", "Cache-Control": "no-store"},
        )
        try:
            with self._op.open(req, timeout=timeout) as r:
                raw = r.read().decode() or "{}"
            parsed = json.loads(raw)
            if not isinstance(parsed, dict):
                return {"ok": False, "error": "board returned non-object"}
            parsed.setdefault("ok", True)
            return parsed
        except HTTPError as exc:
            raw = exc.read() or b"{}"
            try:
                parsed = json.loads(raw.decode() or "{}")
            except json.JSONDecodeError:
                parsed = {}
            if not isinstance(parsed, dict):
                parsed = {}
            parsed.setdefault("ok", False)
            parsed.setdefault("error", str(exc))
            log.warning("board %s %s failed: %s", method, path, parsed.get("error"))
            return parsed
        except (URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            log.warning("board %s %s failed: %s", method, path, exc)
            return {"ok": False, "error": str(exc)}

    def init_goals(self, goals: List[Dict]) -> Dict:
        return self._json("POST", "/api/board/init", {"goals": goals, "robot": self.robot})

    def snapshot(self) -> Dict:
        data = self._json("GET", "/api/board")
        if data.get("ok") is False:
            return data
        data.setdefault("goals", [])
        data.setdefault("found_by", [])
        data.setdefault("progress", {})
        data.setdefault("step", 0)
        return data

    def post_progress(self, text: str, step: int) -> Dict:
        return self._json(
            "POST",
            "/api/board/progress",
            {"robot": self.robot, "text": text, "step": step},
        )

    def post_trace(self, payload: Dict[str, Any]) -> Dict:
        body = dict(payload or {})
        body["robot"] = self.robot
        return self._json("POST", "/api/board/trace", body, timeout=12.0)

    def post_history(self, payload: Dict[str, Any]) -> Dict:
        body = dict(payload or {})
        body["robot"] = self.robot
        return self._json("POST", "/api/board/history", body, timeout=12.0)

    def post_found(
        self,
        goal_id: str,
        verification: Optional[Dict] = None,
        xy: Optional[Any] = None,
    ) -> Dict:
        body: Dict[str, Any] = {
            "robot": self.robot,
            "goal_id": goal_id,
            "verification": verification or {},
        }
        if xy is not None:
            body["xy"] = list(xy) if isinstance(xy, (list, tuple)) else xy
        return self._json("POST", "/api/board/found", body)

    def fetch_image(self, spec: str):
        if not spec:
            return None
        import io

        from PIL import Image

        if not spec.startswith("http") and not spec.startswith("/"):
            return Image.open(spec).convert("RGB")
        req = Request(
            _join_url(self.hub, spec),
            method="GET",
            headers={"Cache-Control": "no-store"},
        )
        with self._op.open(req, timeout=8) as resp:
            raw = resp.read()
        return Image.open(io.BytesIO(raw)).convert("RGB")

    def t0_hello(self) -> Dict:
        return self._json("POST", "/api/t0/hello", {"robot": self.robot}, timeout=5.0)

    def t0_observe(self, pose, dirs, frames) -> Dict:
        return self._json(
            "POST",
            "/api/t0/observe",
            {"robot": self.robot, "pose": pose, "dirs": list(dirs or []), "frames": frames},
            timeout=60.0,
        )

    def t0_status(self) -> Dict:
        return self._json("GET", "/api/t0/status?robot=" + self.robot, timeout=5.0)

    def t0_lock(self) -> Dict:
        return self._json("POST", "/api/t0/lock", {"robot": self.robot}, timeout=5.0)

    def t0_bundle(self) -> Dict:
        return self._json("GET", "/api/t0/bundle?robot=" + self.robot, timeout=60.0)

    def t0_assign(self, assignments: Dict, team_fields, raw: str = "") -> Dict:
        if isinstance(team_fields, str):
            team_fields = {"team_reasoning": team_fields}
        team_fields = dict(team_fields or {})
        return self._json(
            "POST",
            "/api/t0/assign",
            {
                "robot": self.robot,
                "assignments": assignments,
                "team_reasoning": team_fields.get("team_reasoning") or "",
                "panoramic_observation": team_fields.get("panoramic_observation") or {},
                "map_understanding": team_fields.get("map_understanding") or "",
                "image_goal_scenes": team_fields.get("image_goal_scenes") or {},
                "la_time_ms": team_fields.get("la_time_ms"),
                "raw": raw,
            },
            timeout=8.0,
        )

    def t0_fail(self, error: str) -> Dict:
        return self._json(
            "POST", "/api/t0/fail", {"robot": self.robot, "error": error}, timeout=5.0
        )

    def prompt_blocks(self) -> Dict[str, Any]:
        data = self._json("GET", "/api/board/prompt?robot=" + self.robot)
        available = data.get("ok") is not False and not data.get("error")
        if not available:
            return {
                "board": "",
                "progress": "",
                "protocol": "",
                "step": "0",
                "available": False,
                "error": data.get("error") or "board unavailable",
            }
        return {
            "board": data.get("board") or "",
            "progress": data.get("progress") or "",
            "t0_sector": data.get("t0_sector") or "",
            "teammate_live": data.get("teammate_live") or "",
            "protocol": data.get("protocol") or "",
            "step": str(data.get("step", 0)),
            "available": True,
            "error": "",
        }
