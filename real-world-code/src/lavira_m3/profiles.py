"""Validated fleet data. Agent identifiers never imply a hardware model."""

import json
import math
import os
from pathlib import Path

CARDINAL_VIEWS = ("front", "left", "behind", "right")
SURROUND_VIEWS = ("front", "left_front", "left", "left_back", "right_back", "right", "right_front")
VIEW_YAW = {
    "front": 0,
    "left_front": 45,
    "left": 90,
    "left_back": 135,
    "behind": 180,
    "rear": 180,
    "right_back": -135,
    "right": -90,
    "right_front": -45,
}
DEFAULT_AGENT = {
    "footprint": [0.8, 0.4],
    "envelope": [0.4, 0.4, 0.2],
    "own_filter": [0.4, 0.4, 0.2],
    "peer_filter": [0.5, 0.5, 0.3],
    "lidar_height": 0.3,
    "inflate": 0.35,
    "soft_inflate": 0.0,
    "arrival_tolerance": 0.75,
    "claim_distance": 3.0,
    "directions": list(CARDINAL_VIEWS),
    "navigation_backend": "ros2",
}


def validate_agents(raw):
    if not isinstance(raw, dict) or not raw:
        raise ValueError("agents must be a nonempty object")
    result = {}
    for name, values in raw.items():
        if (
            not isinstance(name, str)
            or not name
            or len(name) > 31
            or any(
                c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
                for c in name
            )
        ):
            raise ValueError("invalid agent identifier")
        if not isinstance(values, dict):
            raise ValueError("agent profile must be an object")
        unknown = set(values) - set(DEFAULT_AGENT)
        if unknown:
            raise ValueError("unknown agent fields: " + ", ".join(sorted(unknown)))
        profile = dict(DEFAULT_AGENT, **values)
        if "footprint" in values and "envelope" not in values:
            length, width = values["footprint"]
            profile["envelope"] = [length / 2, length / 2, width / 2]
        for field, size in (
            ("footprint", 2),
            ("envelope", 3),
            ("own_filter", 3),
            ("peer_filter", 3),
        ):
            numbers = profile[field]
            if len(numbers) != size or any(not math.isfinite(v) or v < 0 for v in numbers):
                raise ValueError("invalid " + field)
        if min(profile["footprint"]) <= 0 or profile["envelope"][2] <= 0:
            raise ValueError("body dimensions must be positive")
        for field in (
            "lidar_height",
            "inflate",
            "soft_inflate",
            "arrival_tolerance",
            "claim_distance",
        ):
            if not math.isfinite(profile[field]) or profile[field] < 0:
                raise ValueError("invalid " + field)
        if profile["arrival_tolerance"] <= 0 or profile["claim_distance"] <= 0:
            raise ValueError("distance limits must be positive")
        views = profile["directions"]
        if not views or len(set(views)) != len(views) or any(v not in VIEW_YAW for v in views):
            raise ValueError("invalid camera directions")
        if profile["navigation_backend"] not in ("ros1", "ros2", "http"):
            raise ValueError("invalid navigation backend")
        result[name] = profile
    return result


def load_config():
    path = os.environ.get("NAV_CONFIG")
    data = json.loads(Path(path).read_text()) if path else {}
    if not isinstance(data, dict):
        raise ValueError("configuration must be an object")
    return data, validate_agents(data.get("agents", {"agent_0": {}}))


CONFIG, AGENTS = load_config()


def profile(name):
    try:
        return AGENTS[name]
    except KeyError:
        raise ValueError("unknown agent: " + str(name)) from None


def backend(name):
    return os.environ.get("NAV_ROS", profile(name)["navigation_backend"])
