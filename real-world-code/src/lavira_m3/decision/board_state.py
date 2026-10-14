"""Synchronized goal, progress, observation and history state."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from lavira_m3.profiles import AGENTS

from .board_format import (
    build_board_text,
    build_teammate_progress,
    goal_public_id,
    parse_goal_index,
)
from .config import ROBOT_IDS, t0_lock_s, t0_robots
from .prompts_real import (
    build_t0_sector_block,
    build_teammate_spread_note,
    normalize_image_goal_scenes,
)
from .team_state import T0State

log = logging.getLogger("lavira_m3.board_hub")


RUNTIME_STALE_S = 30.0


SHARE_CAM_MAP = {
    n: tuple(("rear" if d == "behind" else d, d) for d in p["directions"])
    for n, p in AGENTS.items()
}


START_SEARCH_WAV = SYSTEMS_ONLINE_WAV = ALL_FOUND_WAV = ""


def play_start_search_wav(path: str = "") -> bool:
    """No host audio side effects."""
    return False


def _tts_muted() -> bool:
    return os.environ.get("LAVIRA_MUTE_START_WAV", "").strip().lower() in ("1", "true", "yes", "on")


def _audio_device_unused():
    """Audio device configuration is outside the navigation service."""
    return None


_GOAL_KIND = {"ON": "Object goal", "TN": "Text goal", "IIN": "Image goal"}


_KIND_KEY = {"ON": "object", "TN": "text", "IIN": "image"}


_FINDER_SPEAK = {name: name for name in AGENTS}


def found_announce_text(robot: str, modality: str = "") -> str:
    kind = _GOAL_KIND.get(str(modality or "").upper(), "Goal")
    who = _FINDER_SPEAK.get(str(robot), str(robot))
    return f"{kind} was found by {who}."


def found_announce_wav(robot: str, modality: str = "") -> str:
    return ""


def play_found_wav(path: str, wait: bool = False) -> bool:
    """No host audio side effects."""
    return False


def play_systems_online_wav() -> bool:
    return False


def play_all_found_wav() -> bool:
    return False


def announce_found(robot: str, goal_id: str, category: str, modality: str = "") -> bool:
    """No host audio side effects."""
    return False


def _announce_found_then_complete(
    robot: str, goal_id: str, category: str, modality: str, all_done: bool
) -> None:
    return


def list_local_m3_processes():
    """Fleet liveness comes from timestamped traces, not local process names."""
    return []


def _decode_ref_b64(text: str) -> Optional[bytes]:
    raw = (text or "").strip()
    if not raw:
        return None
    if "," in raw and raw.lower().startswith("data:"):
        raw = raw.split(",", 1)[1]
    try:
        data = base64.b64decode(raw)
    except Exception:
        return None
    if not data or len(data) > 4 * 1024 * 1024:
        return None
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(data)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except Exception:
        return None


class BoardStore:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.goals: List[dict] = []
        self.found_by: List[int] = []
        self.found_xy: List[Optional[List[float]]] = []
        self.progress: Dict[str, Dict[str, Any]] = {}
        self.traces: Dict[str, Dict[str, Any]] = {}
        self.histories: Dict[str, List[Dict[str, Any]]] = {}
        self.history_blobs: Dict[str, bytes] = {}
        self.refs: Dict[str, bytes] = {}
        self.step = 0
        self.started = False
        self.t0 = T0State()
        self._said_systems_online = False
        self._said_all_found = False
        self._share_ingest_at = 0.0
        self._share_fetch = None

    def _map2d_url(self) -> str:
        return os.environ.get("MAP2D_URL") or os.environ.get("FLEET_HUB") or "http://127.0.0.1:8770"

    def _share_get(self, url: str):
        fetch = self._share_fetch
        if fetch is not None:
            return fetch(url)
        req = urllib.request.Request(url, headers={"Cache-Control": "no-store"})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=2) as resp:
            return resp.status, resp.read()

    def ingest_share_cameras(self) -> None:
        """If the webpage already has a dog's four JPEGs, give those frames to T0."""
        now = time.time()
        if now - self._share_ingest_at < 1.0:
            return
        self._share_ingest_at = now
        if not self.started or self.t0.assignment:
            return
        try:
            code, raw = self._share_get(self._map2d_url().rstrip("/") + "/v1/scene")
            scene = json.loads(raw.decode() or "{}") if code == 200 else {}
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, UnicodeDecodeError):
            return
        poses = (scene or {}).get("poses") or {}
        for robot, mapping in SHARE_CAM_MAP.items():
            if robot not in self.t0.expected or robot in self.t0.observes:
                continue
            pose = poses.get(robot) or {}
            if not pose.get("live") or not pose.get("position"):
                continue
            jpegs = {}
            try:
                for cam, _t0 in mapping:
                    code, data = self._share_get(
                        "%s/v1/camera-image/%s/%s" % (self._map2d_url().rstrip("/"), robot, cam)
                    )
                    if (
                        code != 200
                        or not data.startswith(b"\xff\xd8")
                        or not data.endswith(b"\xff\xd9")
                    ):
                        jpegs = {}
                        break
                    jpegs[cam] = data
            except (urllib.error.URLError, TimeoutError, OSError):
                continue
            if len(jpegs) != len(mapping):
                continue
            frames = [
                {"name": t0_name, "jpeg_b64": base64.b64encode(jpegs[cam]).decode("ascii")}
                for cam, t0_name in mapping
            ]
            pos = pose["position"]
            posted = self.t0_observe(
                robot,
                [float(pos[0]), float(pos[1]), float(pose.get("yaw") or 0.0)],
                [t0_name for _cam, t0_name in mapping],
                frames,
            )
            if posted.get("ok"):
                log.info("T0 ingested %s from map2d webpage cameras", robot)
            else:
                log.warning("T0 webpage ingest %s rejected: %s", robot, posted.get("error"))

    def snapshot(self) -> Dict[str, Any]:
        return {
            "ok": True,
            "started": bool(self.started),
            "goals": list(self.goals),
            "found_by": list(self.found_by),
            "found_xy": [list(p) if p else None for p in self.found_xy],
            "progress": {k: dict(v) for k, v in self.progress.items()},
            "traces": {k: dict(v) for k, v in self.traces.items()},
            "history": {k: [dict(item) for item in v] for k, v in self.histories.items()},
            "step": int(self.step),
            "t0": self.t0.public(),
        }

    def init_goals(self, goals: List[dict], robot: str) -> Dict[str, Any]:
        if not isinstance(goals, list) or not goals:
            return {"ok": False, "error": "goals must be a non-empty list"}
        normalized = []
        incoming_refs: Dict[str, bytes] = {}
        for i, raw in enumerate(goals):
            item = dict(raw or {})
            item.setdefault("goal_index", i)
            modality = str(item.get("modality") or "ON").upper()
            if modality not in ("ON", "TN", "IIN"):
                modality = "ON"
            item["modality"] = modality
            item.setdefault("object_category", item.get("name") or item.get("category") or "object")
            gid = goal_public_id(item, i)
            blob = _decode_ref_b64(str(item.pop("reference_image_b64", "") or ""))
            if blob:
                incoming_refs[gid] = blob
                item["reference_image"] = "/api/board/ref/" + gid
            elif modality == "IIN" and str(item.get("reference_image") or "").startswith(
                "/api/board/ref/"
            ):
                pass
            elif modality != "IIN":
                item.pop("reference_image", None)
            if modality == "TN":
                item.setdefault("tn_text", item.get("description") or "")
            normalized.append(item)
        with self.lock:
            force = bool(robot == "web")
            existing_found = any(int(v) >= 0 for v in self.found_by)
            if existing_found and self.goals and not force:
                if len(normalized) != len(self.goals):
                    return {
                        "ok": False,
                        "error": "board already has confirmed FOUND; refusing to replace goals",
                    }
                old_scenes = {
                    int(g.get("goal_index", i)): str(g.get("scene_text") or "").strip()
                    for i, g in enumerate(self.goals)
                    if str(g.get("scene_text") or "").strip()
                }
                for item in normalized:
                    if str(item.get("scene_text") or "").strip():
                        continue
                    try:
                        prev = old_scenes.get(int(item.get("goal_index", -999)))
                    except (TypeError, ValueError):
                        prev = None
                    if prev:
                        item["scene_text"] = prev
                self.goals = normalized
                self.refs.update(incoming_refs)
                self._pad_found_xy(len(normalized))
                return self.snapshot()
            self.goals = normalized
            self.found_by = [-1] * len(normalized)
            self.found_xy = [None] * len(normalized)
            self.refs = incoming_refs
            self.progress = {}
            self.started = force
            self._said_systems_online = False
            self._said_all_found = False
            self.t0.reset(expected=list(self.t0.expected), keep_online=True)
            snap = self.snapshot()
        return snap

    def reset_board(self, t0_robots_arg: object = None) -> Dict[str, Any]:
        expected = t0_robots(t0_robots_arg) if t0_robots_arg not in (None, "") else None
        with self.lock:
            self.goals = []
            self.found_by = []
            self.found_xy = []
            self.progress = {}
            self.traces = {}
            self.histories = {}
            self.history_blobs = {}
            self.refs = {}
            self.step = 0
            self.started = False
            self._said_systems_online = False
            self._said_all_found = False
            self.t0.reset(expected=expected)
            return self.snapshot()

    def runtime_busy(self) -> str:
        """Block clear only while the joint T0 VLM call is in flight."""
        holder = self.t0.lock_holder()
        if holder:
            return "T0 joint LA in progress (locked by %s); not clearing" % holder
        return ""

    def clear_runtime_cache(self) -> Dict[str, Any]:
        """Drop traces/progress/T0/FOUND. Keep submitted goals, started, and IIN refs."""
        reason = self.runtime_busy()
        if reason:
            snap = self.snapshot()
            snap["ok"] = False
            snap["error"] = reason
            return snap
        with self.lock:
            self.found_by = [-1] * len(self.goals)
            self.found_xy = [None] * len(self.goals)
            self.progress = {}
            self.traces = {}
            self.histories = {}
            self.history_blobs = {}
            self.step = 0
            self._said_systems_online = False
            self._said_all_found = False
            self.t0.reset(keep_online=True)
            snap = self.snapshot()
        snap["cleared"] = "runtime"
        return snap

    def get_ref(self, goal_id: str) -> Optional[bytes]:
        with self.lock:
            return self.refs.get(str(goal_id or "").strip())

    def t0_hello(self, robot: str) -> Dict[str, Any]:
        if robot not in ROBOT_IDS:
            return {"ok": False, "error": f"unknown robot {robot}"}
        with self.lock:
            if robot not in self.t0.expected:
                return {"ok": False, "error": f"{robot} not in T0_ROBOTS"}
            self.t0.mark_online(robot)
            pub = {"ok": True, **self.t0.public()}
        return pub

    def t0_observe(
        self, robot: str, pose: Any, dirs: List[str], frames: List[Dict]
    ) -> Dict[str, Any]:
        if robot not in ROBOT_IDS:
            return {"ok": False, "error": f"unknown robot {robot}"}
        if not self.started:
            return {"ok": False, "error": "board not started; wait for webpage submit"}
        if robot not in self.t0.expected:
            return {"ok": False, "error": f"{robot} not in T0_ROBOTS"}
        clean_frames = []
        for fr in frames or []:
            name = str((fr or {}).get("name") or "").strip()
            b64 = str((fr or {}).get("jpeg_b64") or "")
            if not name:
                continue
            if b64 and len(b64) > 800_000:
                return {"ok": False, "error": f"{robot} view {name} image too large"}
            item = {"name": name, "jpeg_b64": b64}
            raw_z = (fr or {}).get("center_depth_m")
            if raw_z is not None:
                try:
                    z = float(raw_z)
                except (TypeError, ValueError):
                    z = None
                if z is not None and z == z and abs(z) != float("inf"):
                    item["center_depth_m"] = z
            clean_frames.append(item)
        have = {fr["name"] for fr in clean_frames if fr.get("jpeg_b64")}
        need = [str(name).strip() for name in (dirs or []) if str(name).strip()]
        missing_views = [name for name in need if name not in have]
        if missing_views:
            return {"ok": False, "error": f"{robot} T0 observe missing views {missing_views}"}
        if not have:
            return {"ok": False, "error": f"{robot} T0 observe has no images"}
        with self.lock:
            if self.t0.assignment:
                return {"ok": True, **self.t0.public(), "ignored": "already assigned"}
            self.t0.mark_online(robot)
            self.t0.observes[robot] = {
                "robot": robot,
                "pose": pose,
                "dirs": list(dirs or []),
                "frames": clean_frames,
                "uploaded_at": time.time(),
            }
            self.t0.failed = ""
            return {"ok": True, **self.t0.public()}

    def t0_status(self, robot: str = "") -> Dict[str, Any]:
        self.ingest_share_cameras()
        with self.lock:
            out = {"ok": True, **self.t0.public()}
            if robot and robot in self.t0.assignment:
                out["la"] = dict(self.t0.assignment[robot])
                out["team_reasoning"] = self.t0.team_reasoning
            return out

    def t0_lock(self, robot: str) -> Dict[str, Any]:
        self.ingest_share_cameras()
        with self.lock:
            if robot != self.t0.leader():
                return {"ok": False, "error": "only the T0 leader may lock", **self.t0.public()}
            if self.t0.assignment:
                return {"ok": True, "acquired": False, **self.t0.public()}
            if not self.t0.ready():
                return {"ok": False, "error": "T0 observes are incomplete", **self.t0.public()}
            holder = self.t0.lock_holder()
            if holder and holder != robot:
                return {"ok": False, "error": f"locked by {holder}", **self.t0.public()}
            self.t0.lock_robot = robot
            self.t0.lock_until = time.time() + t0_lock_s()
            return {"ok": True, "acquired": True, **self.t0.public()}

    def t0_bundle(self, robot: str) -> Dict[str, Any]:
        with self.lock:
            if self.t0.lock_holder() != robot:
                return {"ok": False, "error": "lock required"}
            return {
                "ok": True,
                "board": build_board_text(self.goals, self.found_by),
                "goals": list(self.goals),
                "observes": {k: dict(v) for k, v in self.t0.observes.items()},
                "robots": self.t0.online_names(),
                **self.t0.public(),
            }

    def t0_assign(
        self,
        robot: str,
        assignments: Dict[str, Any],
        team_reasoning: str,
        raw: str,
        panoramic_observation: Any = None,
        map_understanding: str = "",
        la_time_ms=None,
        image_goal_scenes: Any = None,
    ) -> Dict[str, Any]:
        if not isinstance(assignments, dict) or not assignments:
            return {"ok": False, "error": "assignments must be a non-empty object"}
        with self.lock:
            if robot != self.t0.leader():
                return {"ok": False, "error": "only the T0 leader may assign", **self.t0.public()}
            holder = self.t0.lock_holder()
            if holder and holder != robot:
                return {"ok": False, "error": "lock required"}
            cleaned = {}
            allowed = set(self.t0.online_names()) or set(self.t0.expected)
            for name, rec in assignments.items():
                if name in allowed and isinstance(rec, dict):
                    cleaned[name] = {
                        key: rec.get(key)
                        for key in ("action", "claim_goal_id", "region")
                        if key in rec
                    }
            if not cleaned:
                return {"ok": False, "error": "no valid robot assignments"}
            self.t0.assignment = cleaned
            self.t0.team_reasoning = str(team_reasoning or "")
            self.t0.panoramic_observation = (
                panoramic_observation if panoramic_observation is not None else {}
            )
            self.t0.map_understanding = str(map_understanding or "")
            scenes = normalize_image_goal_scenes(image_goal_scenes)
            if scenes:
                self.t0.image_goal_scenes = scenes
            for i, goal in enumerate(self.goals):
                if str(goal.get("modality") or "").upper() != "IIN":
                    continue
                gid = goal_public_id(goal, i)
                text = scenes.get(gid) or scenes.get("goal_%s" % goal.get("goal_index", i))
                if text:
                    goal["scene_text"] = text
            try:
                self.t0.la_time_ms = None if la_time_ms in (None, "") else float(la_time_ms)
            except (TypeError, ValueError):
                self.t0.la_time_ms = None
            self.t0.raw = str(raw or "")
            self.t0.observes = {}
            self.t0.lock_robot = ""
            self.t0.lock_until = 0.0
            self.t0.failed = ""
            return {"ok": True, **self.t0.public()}

    def t0_fail(self, robot: str, error: str) -> Dict[str, Any]:
        with self.lock:
            if robot == self.t0.leader():
                self.t0.failed = str(error or "T0 joint LA failed")
                self.t0.lock_robot = ""
                self.t0.lock_until = 0.0
            return {"ok": True, **self.t0.public()}

    def post_progress(self, robot: str, text: str, step: int) -> Dict[str, Any]:
        if not robot:
            return {"ok": False, "error": "robot required"}
        with self.lock:
            self.progress[robot] = {"text": str(text or ""), "step": int(step)}
            self.step = max(self.step, int(step))
            return self.snapshot()

    def post_trace(self, robot: str, body: Dict[str, Any]) -> Dict[str, Any]:
        if not robot:
            return {"ok": False, "error": "robot required"}
        rec = {
            "robot": robot,
            "step": body.get("step"),
            "la": dict(body.get("la") or {}),
            "la_time_ms": body.get("la_time_ms"),
            "va": dict(body.get("va") or {}),
            "va_time_ms": body.get("va_time_ms"),
            "ra_time_ms": body.get("ra_time_ms"),
            "trace_created_ns": body.get("trace_created_ns"),
            "nav": dict(body.get("nav") or {}),
            "waypoint": body.get("waypoint"),
            "va_map": body.get("va_map"),
            "va_pixel": body.get("va_pixel"),
            "va_bbox": body.get("va_bbox"),
            "va_path": body.get("va_path"),
            "va_image": body.get("va_image"),
            "front_image": body.get("front_image"),
            "image_size": body.get("image_size"),
            "direction": body.get("direction"),
            "found_goal_id": body.get("found_goal_id"),
            "la_raw": body.get("la_raw"),
            "va_raw": body.get("va_raw"),
            "updated_at": time.time(),
        }
        with self.lock:
            prev = dict(self.traces.get(robot) or {})
            # A timed-out older POST can finish after a newer worker request.
            incoming_ns = rec.get("trace_created_ns")
            previous_ns = prev.get("trace_created_ns")
            if (
                isinstance(incoming_ns, int)
                and isinstance(previous_ns, int)
                and incoming_ns < previous_ns
            ):
                return self.snapshot()
            if rec["step"] is None:
                rec["step"] = prev.get("step")
            if body.get("clear_va"):
                rec["va"] = {}
                rec["va_time_ms"] = None
                rec["va_bbox"] = None
                rec["va_pixel"] = None
                rec["va_map"] = None
                rec["va_path"] = None
                rec["va_image"] = ""
                rec["image_size"] = None
                rec["found_goal_id"] = None
            for key in (
                "la",
                "la_time_ms",
                "va",
                "va_time_ms",
                "ra_time_ms",
                "nav",
                "waypoint",
                "va_map",
                "va_pixel",
                "va_bbox",
                "va_path",
                "va_image",
                "front_image",
                "image_size",
                "direction",
                "found_goal_id",
                "la_raw",
                "va_raw",
            ):
                if key == "ra_time_ms" and key in body:
                    continue  # Explicit null clears the previous cycle's timing.
                if body.get("clear_va") and key in (
                    "va",
                    "va_time_ms",
                    "va_bbox",
                    "va_pixel",
                    "va_map",
                    "va_path",
                    "va_image",
                    "image_size",
                    "found_goal_id",
                ):
                    continue
                if rec.get(key) in (None, {}, [], ""):
                    rec[key] = prev.get(key)
            self.traces[robot] = rec
            return self.snapshot()

    def _history_key(self, robot: str, waypoint_id: int) -> str:
        return "%s/%s" % (robot, waypoint_id)

    def post_history(self, robot: str, body: Dict[str, Any]) -> Dict[str, Any]:
        if robot not in ROBOT_IDS:
            return {"ok": False, "error": f"unknown robot {robot}"}
        try:
            waypoint_id = int(body.get("waypoint_id"))
        except (TypeError, ValueError):
            return {"ok": False, "error": "waypoint_id required"}
        rec = {
            "waypoint_id": waypoint_id,
            "step": body.get("step"),
            "pose": body.get("pose"),
            "action": str(body.get("action") or ""),
            "target": str(body.get("target") or ""),
            "xy": body.get("xy"),
            "nav_ok": body.get("nav_ok"),
            "front_image": "/api/board/history/%s/%s" % (robot, waypoint_id),
        }
        blob = _decode_ref_b64(str(body.get("front_image") or ""))
        with self.lock:
            items = list(self.histories.get(robot) or [])
            replaced = False
            for i, old in enumerate(items):
                if int(old.get("waypoint_id", -1)) == waypoint_id:
                    merged = dict(old)
                    for key, value in rec.items():
                        if key == "front_image":
                            continue
                        if value not in (None, "", []):
                            merged[key] = value
                    merged["front_image"] = rec["front_image"]
                    items[i] = merged
                    replaced = True
                    break
            if not replaced:
                items.append(rec)
            items.sort(key=lambda item: int(item.get("waypoint_id") or 0))
            self.histories[robot] = items
            if blob:
                self.history_blobs[self._history_key(robot, waypoint_id)] = blob
            return self.snapshot()

    def get_history_image(self, robot: str, waypoint_id: int) -> Optional[bytes]:
        with self.lock:
            return self.history_blobs.get(self._history_key(robot, waypoint_id))

    def _pad_found_xy(self, n: int) -> None:
        cur = list(getattr(self, "found_xy", None) or [])
        if len(cur) < n:
            cur.extend([None] * (n - len(cur)))
        elif len(cur) > n:
            cur = cur[:n]
        self.found_xy = cur

    def _parse_found_xy(self, raw: Any) -> Optional[List[float]]:
        if not isinstance(raw, (list, tuple)) or len(raw) < 2:
            return None
        try:
            x, y = float(raw[0]), float(raw[1])
        except (TypeError, ValueError):
            return None
        if x != x or y != y:
            return None
        return [x, y]

    def post_found(self, robot: str, goal_id: str, xy: Any = None) -> Dict[str, Any]:
        agent_id = ROBOT_IDS.get(robot)
        if agent_id is None:
            return {"ok": False, "error": f"unknown robot {robot}"}
        with self.lock:
            idx = parse_goal_index(goal_id, self.goals)
            if idx is None:
                return {"ok": False, "error": f"unknown goal_id {goal_id}"}
            if self.found_by[idx] >= 0:
                return self.snapshot()
            self.found_by[idx] = int(agent_id)
            self._pad_found_xy(len(self.goals))
            parsed_xy = self._parse_found_xy(xy)
            if parsed_xy is not None and idx < len(self.found_xy):
                self.found_xy[idx] = parsed_xy
            snap = self.snapshot()
        return snap

    def prompt(self, robot: str) -> Dict[str, Any]:
        with self.lock:
            snap = self.snapshot()
            board = build_board_text(self.goals, self.found_by)
            progress = build_teammate_progress(self.progress, robot, self.step)
            spread = build_teammate_spread_note(robot, self.traces, self.histories)
            t0_sector = build_t0_sector_block(robot, self.t0.public())
        snap.update(
            {
                "board": board,
                "progress": progress,
                "t0_sector": t0_sector,
                "teammate_live": spread,
                "protocol": "async_board_pa",
                "robot": robot,
            }
        )
        return snap


STORE = BoardStore()
