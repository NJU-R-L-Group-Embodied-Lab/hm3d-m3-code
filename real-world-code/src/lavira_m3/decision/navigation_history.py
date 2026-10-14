"""Navigation history serialization and local image/manifest storage."""

from __future__ import annotations

import base64
import json
import logging
import os
from typing import Any, Dict, List, Optional

from .config import history_root
from .model_messages import encode_image

log = logging.getLogger(__name__)
TRACE_IMAGE_DIM = 480


def history_rgb_period_s() -> float:
    """Body-front RGB period while the chassis is moving. Default 5 s."""
    try:
        return max(0.05, float(os.environ.get("M3_HISTORY_RGB_PERIOD_S") or 5.0))
    except (TypeError, ValueError):
        return 5.0


def front_frame(frames: List[Dict]) -> Optional[Dict[str, Any]]:
    for fr in frames or []:
        if str((fr or {}).get("name") or "") == "front":
            return fr
    return None


def history_public(rec: Dict[str, Any]) -> Dict[str, Any]:
    pose = rec.get("pose")
    if isinstance(pose, tuple):
        pose = list(pose)
    xy = rec.get("reached_xy") or rec.get("xy")
    if isinstance(xy, tuple):
        xy = list(xy)
    return {
        "waypoint_id": rec.get("waypoint_id"),
        "step": rec.get("step"),
        "pose": pose,
        "action": rec.get("action") or "",
        "target": rec.get("target") or "",
        "xy": xy,
        "nav_ok": rec.get("nav_ok"),
        "kind": rec.get("kind") or "nav",
        "t": rec.get("t"),
        "is_waypoint": bool(rec.get("is_waypoint")),
        "leave_direction": rec.get("leave_direction") or "",
        "has_pano": bool(waypoint_has_rgb(rec)),
    }


def copy_view_for_archive(fr: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "name": str((fr or {}).get("name") or ""),
        "fx": (fr or {}).get("fx"),
        "fy": (fr or {}).get("fy"),
        "cx": (fr or {}).get("cx"),
        "cy": (fr or {}).get("cy"),
        "camera": (fr or {}).get("camera"),
    }
    for key in ("rgb", "depth"):
        val = (fr or {}).get(key)
        if val is None:
            out[key] = None
            continue
        try:
            out[key] = val.copy()
        except Exception:
            out[key] = val
    return out


def waypoint_has_rgb(rec: Optional[Dict[str, Any]]) -> bool:
    if not rec:
        return False
    for fr in rec.get("pano") or []:
        if fr.get("rgb") is not None:
            return True
    return False


def history_directory(robot_name: str, session: str) -> str:
    folder = os.path.join(history_root(), robot_name, session)
    os.makedirs(folder, exist_ok=True)
    return folder


def save_front_image(rec: Dict[str, Any], robot_name: str, session: str) -> bool:
    if rec.get("rgb") is None:
        return False
    try:
        jpeg = encode_image(rec["rgb"], TRACE_IMAGE_DIM)
        wid = int(rec.get("waypoint_id") or 0)
        filename = os.path.join(
            history_directory(robot_name, session), "waypoint_%03d_front.jpg" % wid
        )
        with open(filename, "wb") as stream:
            stream.write(base64.b64decode(jpeg))
        rec["disk_path"] = filename
        return True
    except Exception:
        log.exception("save front history failed waypoint=%s", rec.get("waypoint_id"))
        return False


def save_history_manifest(records: List[Dict], robot_name: str, session: str) -> None:
    try:
        folder = history_directory(robot_name, session)
        payload = {
            "robot": robot_name,
            "session": session,
            "waypoints": [history_public(rec) for rec in records],
        }
        with open(os.path.join(folder, "manifest.json"), "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
    except Exception:
        log.exception("write history manifest failed")
