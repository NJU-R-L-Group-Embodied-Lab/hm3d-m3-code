"""Shared runtime config. Secrets only from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from lavira_m3.profiles import AGENTS, profile
from lavira_m3.profiles import CARDINAL_VIEWS as CARDINAL_VIEWS
from lavira_m3.profiles import SURROUND_VIEWS as SURROUND_VIEWS
from lavira_m3.profiles import VIEW_YAW as VIEW_YAW

BOARD_DEFAULT = "http://127.0.0.1:8770"
BOARD_UPSTREAM_DEFAULT = "http://127.0.0.1:8770"
NAV_DEFAULT = "http://127.0.0.1:8770"
VA_URL_DEFAULT = "http://127.0.0.1:8000/v1"
VA_MODEL_DEFAULT = "Qwen3.8-27B"
DEEPSEEK_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-flash"
GEMINI_URL = ""  # No third-party proxy default.
GEMINI_MODEL = "gemini-3.8-flash"
ALIYUN_VA_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
ALIYUN_VA_MODEL = "qwen3.5-27b"
ALIYUN_LA_MODEL = "qwen3.8-flash"


ROBOT_IDS = {name: index for index, name in enumerate(AGENTS)}
ID_TO_ROBOT = {index: name for name, index in ROBOT_IDS.items()}

REAR_LA_DIRS = ("behind", "left_back", "right_back")


def la_action_dirs(dirs: Sequence[str]) -> List[str]:
    """Post-T0 async LA action list: keep cameras for views, drop rear-facing actions."""
    return [d for d in dirs if str(d) not in REAR_LA_DIRS]


# Shared by Aloha / GO1 / GO2.
# No depth → 1.5 m along the pixel ray. Depth > 5 m → clip to 5 m (VA only).
# Valid 0–5 m depth keeps the measured range (do not lift to 1.5 m).
# One waypoint walk cap is odometric path length from /odom_ekf (TRAVEL_BUDGET_M).
# Do not pre-clip the waypoint by straight-line hypot. STEP_MAX_M is backtrack / loop-repair only.
# Aloha Nav2/HTTP default stop is ≤ 0.20 m. GO1/GO2 arrive at ≤ 0.75 m.
# Measured VA depth 0–0.75 m skips chassis motion and starts the next plan.
WAYPOINT_REACH_M = 0.20
ROS1_REACH_M = 0.75
ROS2_REACH_M = 0.75
HTTP_REACH_M = 0.75
EXPLORATION_REACH_M = 0.25  # no-depth search stop; keep tighter than HTTP_REACH_M
VA_CLOSE_PUSH_M = 0.75  # <0.2 m VA is pushed onto this circle along the robot–VA ray
STEP_MAX_M = 10.0
TRAVEL_BUDGET_M = 5.0  # real XY path length per waypoint; cancel as intermediate
TRAVEL_JITTER_M = 0.02  # ignore odom chatter below this
ODOM_EKF_TOPIC = "/odom_ekf"
VA_PROJECT_MAX_M = 5.0  # VA depth clip before projection; not the walk cap
STEP_MIN_M = 1.5
SHORT_HOP_M = 1.5  # actual XY travel below this reuses last panorama next cycle
MAP_EXTENT_PAD_M = 10.0  # observation grid unknown border; covers STEP_MAX
MAP_GOAL_INSET_M = 0.1  # one cell inside the current map AABB
WORLD_HALF_M = 50.0  # fixed canvas [-50, 50); Go1 birth is (0, 0)
WORLD_RES_M = 0.1
VIEW_MASK_MIN_DEPTH_M = 1.5  # LA omits a view when center-region mean depth is closer
PANO_MAX_DIM = 256
VA_MAX_DIM = 640
FOUND_MAX_DEPTH_M = 3.0  # all robots: LA claim visual distance
HTTP_FOUND_MAX_DEPTH_M = 3.0  # Aloha LA claim; same 3 m as GO1/GO2
VA_PREVIEW_DEPTH_M = 2.0  # static T0 map point only when real depth is missing
T0_WAIT_S = 120  # wait for teammate T0 observes; then error, never solo LA
T0_LOCK_S = 90
T0_MAP_DIM = 512
VIEW_WAIT_S = 10  # dog cameras: wait for all 4 frames before failing the gate


def claim_max_m_for(name: str) -> float:
    name = str(name)[6:] if str(name).startswith("agent_agent_") else str(name)
    return float(profile(name)["claim_distance"]) if name in AGENTS else FOUND_MAX_DEPTH_M


def t0_joint_enabled() -> bool:
    return os.environ.get("T0_JOINT", "1").strip().lower() not in ("0", "false", "no", "off")


def t0_robots(raw: object = None) -> List[str]:
    if raw is None:
        raw = os.environ.get("T0_ROBOTS", ",".join(AGENTS))
    if isinstance(raw, (list, tuple)):
        names = [str(n).strip() for n in raw if str(n).strip()]
    else:
        names = [n.strip() for n in str(raw).split(",") if n.strip()]
    return [n for n in names if n in ROBOT_IDS] or list(AGENTS)


def t0_leader_name() -> str:
    name = os.environ.get("T0_LEADER", next(iter(AGENTS))).strip()
    robots = t0_robots()
    return name if name in robots else robots[0]


def t0_wait_s() -> float:
    return float(os.environ.get("T0_WAIT_S", str(T0_WAIT_S)))


def t0_lock_s() -> float:
    return float(os.environ.get("T0_LOCK_S", str(T0_LOCK_S)))


def t0_static() -> bool:
    """Preview T0 LA/VA/plan only; do not execute chassis RA translation.

    Aloha in-place yaw for VA still goes through face_direction(execute=True).
    """
    return os.environ.get("T0_STATIC", "0").strip().lower() in ("1", "true", "yes", "on")


def view_wait_s() -> float:
    return float(os.environ.get("VIEW_WAIT_S", str(VIEW_WAIT_S)))


def va_preview_depth_m() -> float:
    return float(os.environ.get("VA_PREVIEW_DEPTH_M", str(VA_PREVIEW_DEPTH_M)))


def view_mask_min_depth_m() -> float:
    return float(os.environ.get("VIEW_MASK_MIN_DEPTH_M", str(VIEW_MASK_MIN_DEPTH_M)))


def short_hop_m() -> float:
    return float(os.environ.get("SHORT_HOP_M", str(SHORT_HOP_M)))


def step_max_m() -> float:
    return float(os.environ.get("STEP_MAX_M", str(STEP_MAX_M)))


def travel_budget_m() -> float:
    return float(os.environ.get("TRAVEL_BUDGET_M", str(TRAVEL_BUDGET_M)))


def travel_jitter_m() -> float:
    return float(os.environ.get("TRAVEL_JITTER_M", str(TRAVEL_JITTER_M)))


def odom_ekf_topic() -> str:
    return os.environ.get("ODOM_EKF_TOPIC", ODOM_EKF_TOPIC) or ODOM_EKF_TOPIC


def waypoint_reach_m() -> float:
    return float(os.environ.get("WAYPOINT_REACH_M", str(WAYPOINT_REACH_M)))


def va_backend() -> str:
    raw = os.environ.get("VA_BACKEND", "aliyun").strip().lower()
    if raw in ("local", "vllm"):
        return "local"
    if raw == "qwen":
        url = os.environ.get("VA_BASE_URL", "")
        if "aliyuncs.com" in url or "compatible-mode" in url:
            return "aliyun"
        return "local"
    if raw in ("aliyun", "dashscope"):
        return "aliyun"
    if raw == "deepseek":
        return "deepseek"
    return raw or "deepseek"


def va_base_url() -> str:
    backend = va_backend()
    if backend == "deepseek":
        return os.environ.get("DEEPSEEK_BASE_URL", DEEPSEEK_URL)
    if backend == "local":
        return os.environ.get("VA_BASE_URL", VA_URL_DEFAULT)
    return os.environ.get("VA_BASE_URL", ALIYUN_VA_URL)


def va_model() -> str:
    raw = os.environ.get("VA_MODEL", "").strip()
    backend = va_backend()
    if backend == "local":
        return raw or VA_MODEL_DEFAULT
    if backend == "deepseek":
        return raw if raw.startswith("deepseek") else DEEPSEEK_MODEL
    return raw or ALIYUN_VA_MODEL


def va_api_key() -> str:
    backend = va_backend()
    if backend == "deepseek":
        return os.environ.get("DEEPSEEK_API_KEY", "")
    if backend == "local":
        return os.environ.get("VA_API_KEY", "sk-no-key-required")
    return (
        os.environ.get("VA_API_KEY")
        or os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("LLM_API_KEY")
        or ""
    )


def la_backend() -> str:
    return (os.environ.get("LA_BACKEND", "deepseek") or "deepseek").strip().lower()


def la_base_url() -> str:
    if la_backend() == "deepseek":
        return os.environ.get("DEEPSEEK_BASE_URL", DEEPSEEK_URL)
    if la_backend() == "gemini":
        return os.environ.get("LA_BASE_URL") or os.environ.get("GEMINI_BASE_URL") or GEMINI_URL
    return os.environ.get("LA_BASE_URL") or os.environ.get("VA_BASE_URL") or ALIYUN_VA_URL


def la_model() -> str:
    raw = os.environ.get("LA_MODEL", "").strip()
    if la_backend() == "deepseek":
        return raw or DEEPSEEK_MODEL
    if la_backend() == "gemini":
        return raw or GEMINI_MODEL
    return raw or ALIYUN_LA_MODEL


def la_api_key() -> str:
    if la_backend() == "deepseek":
        return os.environ.get("DEEPSEEK_API_KEY", "")
    if la_backend() == "gemini":
        return os.environ.get("GEMINI_API_KEY") or os.environ.get("LA_API_KEY") or ""
    return (
        os.environ.get("LA_API_KEY")
        or os.environ.get("VA_API_KEY")
        or os.environ.get("DASHSCOPE_API_KEY")
        or os.environ.get("LLM_API_KEY")
        or ""
    )


def history_root() -> str:
    env = str(os.environ.get("M3_HISTORY_DIR") or "").strip()
    if env:
        return env
    return os.environ.get("NAV_RUNTIME_DIR", ".runtime") + "/history"


@dataclass
class LLMConfig:
    la_backend: str = field(default_factory=la_backend)
    la_base_url: str = field(default_factory=la_base_url)
    la_model: str = field(default_factory=la_model)
    la_api_key: str = field(default_factory=la_api_key, repr=False)
    va_backend: str = field(default_factory=va_backend)
    va_base_url: str = field(default_factory=va_base_url)
    va_model: str = field(default_factory=va_model)
    va_api_key: str = field(default_factory=va_api_key, repr=False)


@dataclass
class RobotConfig:
    name: str
    panorama: str = "cardinal4"  #  # "surround7" | "cardinal4"
    hub: str = field(
        default_factory=lambda: (
            os.environ.get("BOARD_URL") or os.environ.get("FLEET_HUB", BOARD_DEFAULT)
        )
    )
    nav_url: str = field(default_factory=lambda: os.environ.get("MAP2D_NAV_URL", NAV_DEFAULT))
    max_cycles: int = field(default_factory=lambda: int(os.environ.get("M3_MAX_CYCLES", "30")))
    found_max_depth_m: Optional[float] = None

    def __post_init__(self) -> None:
        if self.found_max_depth_m is None:
            self.found_max_depth_m = claim_max_m_for(self.name)

    @property
    def navigation_backend(self):
        return os.environ.get("NAV_ROS", profile(self.name)["navigation_backend"])

    @property
    def arrival_tolerance(self):
        return float(profile(self.name)["arrival_tolerance"])

    @property
    def agent_id(self) -> int:
        return ROBOT_IDS[self.name]

    @property
    def directions(self) -> Tuple[str, ...]:
        return tuple(profile(self.name)["directions"])


def load_goals(path: str) -> List[Dict]:
    import json
    from pathlib import Path

    data = json.loads(Path(path).read_text())
    if isinstance(data, dict) and "goals" in data:
        data = data["goals"]
    goals = []
    for i, g in enumerate(data):
        item = dict(g)
        item.setdefault("goal_index", i)
        item.setdefault("modality", "ON")
        item.setdefault("object_category", item.get("name") or item.get("category") or "object")
        goals.append(item)
    return goals


def board_all_found(snap: Dict) -> bool:
    """True only when the board is reachable and every goal has a finder."""
    if not snap or snap.get("ok") is False or snap.get("error"):
        return False
    goals = snap.get("goals") or []
    found = snap.get("found_by") or []
    if not goals or len(found) != len(goals):
        return False
    return all(v not in ("", None, -1, "none") for v in found)
