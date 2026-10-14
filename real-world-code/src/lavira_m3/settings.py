"""Runtime configuration shared by both middleware ports."""

import math
import os
from pathlib import Path

from .profiles import AGENTS, CONFIG, profile

AGENT = os.environ.get("NAV_AGENT", next(iter(AGENTS)))
PROFILE = profile(AGENT)
FRAME = CONFIG.get("frame", "map")
BODY_FRAME = CONFIG.get("body_frame", "base_link")
TOPICS = dict(
    odometry="/odom",
    localization="/Odometry",
    imu="/imu/data",
    cloud="/points",
    command="/cmd_vel",
    map="/map2d/planning",
)
TOPICS.update(CONFIG.get("topics", {}))
SAFETY = dict(
    map_timeout=5.0,
    odom_timeout=0.2,
    command_timeout=0.15,
    max_speed=0.4,
    max_yaw_rate=0.8,
    arrival_tolerance=PROFILE["arrival_tolerance"],
)
SAFETY.update(CONFIG.get("safety", {}))
if any(not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in SAFETY.values()):
    raise ValueError("safety limits must be finite and positive")
if FRAME != "map" or BODY_FRAME != "base_link":
    raise ValueError(
        "Native navigation uses canonical map/base_link; transform external sensors into these frames"
    )
if any(not isinstance(v, str) or not v for v in (FRAME, BODY_FRAME, *TOPICS.values())):
    raise ValueError("frames and topics must be nonempty strings")
SERVICES = dict(map="http://127.0.0.1:8770", board="http://127.0.0.1:8770")
SERVICES.update(CONFIG.get("services", {}))
RUNTIME = Path(os.environ.get("NAV_RUNTIME_DIR", ".runtime"))


def lock_path(kind):
    directory = RUNTIME / "locks"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / (AGENT + "-" + kind + ".lock")
