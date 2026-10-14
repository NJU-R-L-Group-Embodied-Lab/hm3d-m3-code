"""Validate navigation results and serialize decision trace values."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from ..contracts import NavResult


def navigation_succeeded(result: Any) -> bool:
    if isinstance(result, NavResult):
        return bool(result.ok)
    return bool(result)


def planner_path_from_result(result: Any) -> Optional[List[List[float]]]:
    """Keep only a real planner polyline. Never invent a start–goal chord."""
    extra = getattr(result, "extra", None) if result is not None else None
    path = None
    if isinstance(result, NavResult):
        path = (result.extra or {}).get("path")
    elif isinstance(result, dict):
        path = result.get("path")
        extra = result
    if not isinstance(path, list) or len(path) < 3:
        if isinstance(extra, dict) and isinstance(extra.get("path"), list):
            path = extra.get("path")
    cleaned: List[List[float]] = []
    for pt in path or []:
        if not isinstance(pt, (list, tuple)) or len(pt) < 2:
            return None
        try:
            x, y = float(pt[0]), float(pt[1])
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        cleaned.append([x, y])
    return cleaned if len(cleaned) >= 3 else None


def trace_polyline(path: Any) -> Optional[List[List[float]]]:
    cleaned = planner_path_from_result(path)
    if not cleaned and isinstance(path, list):
        cleaned = planner_path_from_result({"path": path})
    if not cleaned:
        return None
    return cleaned


def json_public(data: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for key, value in (data or {}).items():
        if key.startswith("_"):
            continue
        out[key] = value
    return out


def parse_map_xy(raw: Any) -> Optional[Tuple[float, float]]:
    if isinstance(raw, dict):
        raw = [raw.get("x"), raw.get("y")]
    if not isinstance(raw, (list, tuple)) or len(raw) < 2:
        return None
    try:
        x, y = float(raw[0]), float(raw[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return (x, y)
