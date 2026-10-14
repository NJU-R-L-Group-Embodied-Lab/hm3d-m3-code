import math
import time

from lavira_m3.decision.config import (
    CARDINAL_VIEWS,
    STEP_MAX_M,
    SURROUND_VIEWS,
    board_all_found,
    load_goals,
)
from lavira_m3.decision.navigation_geometry import clamp_map_step
from lavira_m3.decision.prompts_real import parse_backtrack_id, parse_direction
from lavira_m3.decision.vision_geometry import (
    bbox_center_px,
    bbox_ground_px,
    bbox_inset_median_depth,
    bbox_iou,
    bbox_to_xyxy_px,
    clip_va_project_depth,
    fallback_va_bbox,
    legal_la_directions,
    normalize_va_target,
    usable_va_bbox,
    va_observe_depth,
    view_center_mean_depth,
    view_mask_blocked,
)

#!/usr/bin/env python3


def parse_json_object(text):
    import json
    import re

    text = text.strip()
    if "```" in text:
        text = text.split("```", 2)[1]
        text = re.sub(r"^json", "", text, flags=re.I).strip()
    start, end = text.find("{"), text.rfind("}")
    blob = text[start : end + 1].replace("True", "true").replace("False", "false")
    return json.loads(blob)


def test_dirs():
    from lavira_m3.decision.config import la_action_dirs

    assert parse_direction("navigate to left_front", SURROUND_VIEWS) == "left_front"
    assert parse_direction("navigate to behind", CARDINAL_VIEWS) == "behind"
    assert parse_direction("navigate to forward", SURROUND_VIEWS) == "front"
    assert parse_direction("navigate to behind", SURROUND_VIEWS) == "front"
    assert parse_direction("navigate to left_back", SURROUND_VIEWS) == "left_back"
    assert parse_direction("navigate to right_back", SURROUND_VIEWS) == "right_back"
    assert "behind" not in la_action_dirs(CARDINAL_VIEWS)
    assert "left_back" not in la_action_dirs(SURROUND_VIEWS)
    assert "right_back" not in la_action_dirs(SURROUND_VIEWS)
    assert parse_backtrack_id("backtrack to 3") == 3


def test_json():
    data = parse_json_object('```json\n{"stop": False, "action": "navigate to left"}\n```')
    assert data["action"] == "navigate to left"


def test_goals():
    import os

    path = os.path.join(os.path.dirname(__file__), "goals.example.json")
    goals = load_goals(path)
    assert len(goals) == 2
    assert goals[0]["object_category"] == "office chair"


def test_remote_provider_defaults():
    import os

    from lavira_m3.decision.config import (
        ALIYUN_VA_MODEL,
        ALIYUN_VA_URL,
        DEEPSEEK_MODEL,
        DEEPSEEK_URL,
        VA_MODEL_DEFAULT,
        VA_URL_DEFAULT,
        LLMConfig,
    )

    keys = (
        "VA_BACKEND",
        "VA_MODEL",
        "VA_BASE_URL",
        "VA_API_KEY",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "LA_BACKEND",
        "LA_MODEL",
        "LA_BASE_URL",
        "LA_API_KEY",
        "GEMINI_API_KEY",
        "GEMINI_BASE_URL",
        "DASHSCOPE_API_KEY",
    )
    old = {k: os.environ.get(k) for k in keys}
    try:
        for k in keys:
            os.environ.pop(k, None)
        os.environ["DEEPSEEK_API_KEY"] = "sk-ds"
        cfg = LLMConfig()
        assert cfg.la_backend == "deepseek"
        assert cfg.la_model == DEEPSEEK_MODEL
        assert cfg.la_base_url == DEEPSEEK_URL
        assert cfg.la_api_key == "sk-ds"
        assert cfg.va_backend == "aliyun"
        assert cfg.va_model == ALIYUN_VA_MODEL
        assert cfg.va_base_url == ALIYUN_VA_URL
        assert cfg.va_api_key != "sk-ds"

        os.environ["VA_BACKEND"] = "aliyun"
        os.environ["VA_API_KEY"] = "sk-aliyun"
        cfg = LLMConfig()
        assert cfg.va_backend == "aliyun"
        assert cfg.va_model == ALIYUN_VA_MODEL
        assert cfg.va_base_url == ALIYUN_VA_URL
        assert cfg.va_api_key == "sk-aliyun"

        os.environ["VA_BACKEND"] = "deepseek"
        os.environ["DEEPSEEK_API_KEY"] = "sk-test"
        os.environ["DEEPSEEK_BASE_URL"] = DEEPSEEK_URL
        cfg = LLMConfig()
        assert cfg.va_backend == "deepseek"
        assert cfg.va_model == DEEPSEEK_MODEL
        assert cfg.va_base_url == DEEPSEEK_URL
        assert cfg.va_api_key == "sk-test"

        os.environ["VA_MODEL"] = "Qwen3.8-27B"
        cfg = LLMConfig()
        assert cfg.va_model == DEEPSEEK_MODEL

        os.environ["VA_BACKEND"] = "local"
        os.environ["VA_MODEL"] = "Qwen3.8-27B"
        os.environ["VA_BASE_URL"] = VA_URL_DEFAULT
        cfg = LLMConfig()
        assert cfg.va_backend == "local"
        assert cfg.va_model == VA_MODEL_DEFAULT
        assert cfg.va_base_url == VA_URL_DEFAULT
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_generate_va_aliyun_and_gemini_la():
    from unittest.mock import MagicMock, patch

    from lavira_m3.decision.config import LLMConfig
    from lavira_m3.decision.llm_router import LLMRouter

    cfg = LLMConfig(
        la_backend="aliyun",
        la_base_url="https://example.aliyuncs.com/compatible-mode/v1",
        la_model="qwen3.8-flash",
        la_api_key="sk-aliyun",
        va_backend="aliyun",
        va_base_url="https://example.aliyuncs.com/compatible-mode/v1",
        va_model="qwen3.5-27b",
        va_api_key="sk-aliyun",
    )
    with patch("lavira_m3.decision.llm_router.OpenAI") as openai_cls:
        client = MagicMock()
        choice = MagicMock()
        choice.message.content = '{"bbox_2d":[199,151,502,454]}'
        choice.finish_reason = "stop"
        client.chat.completions.create.return_value.choices = [choice]
        openai_cls.return_value = client
        router = LLMRouter(cfg)
        text = router.generate_va([{"role": "user", "content": "x"}])
        assert "199" in text
        va_kwargs = client.chat.completions.create.call_args.kwargs
        assert va_kwargs["extra_body"] == {"enable_thinking": False}
        assert va_kwargs["max_tokens"] == 1024
        assert va_kwargs["model"] == "qwen3.5-27b"
        choice.message.content = '{"action":"navigate to front"}'
        text = router.generate_la([{"role": "user", "content": "go"}], max_tokens=2048)
        assert "navigate to front" in text
        la_kwargs = client.chat.completions.create.call_args.kwargs
        assert la_kwargs["model"] == "qwen3.8-flash"
        assert la_kwargs["extra_body"] == {"enable_thinking": False}
        assert la_kwargs["max_tokens"] == 2048
        assert openai_cls.call_count == 1
        assert openai_cls.call_args.kwargs["base_url"].endswith("/compatible-mode/v1")


def test_generate_va_uses_la_style_deepseek():
    from unittest.mock import MagicMock, patch

    from lavira_m3.decision.config import LLMConfig
    from lavira_m3.decision.llm_router import LLMRouter

    cfg = LLMConfig(
        la_backend="deepseek",
        la_base_url="https://api.deepseek.com",
        la_model="deepseek-flash",
        la_api_key="sk-test",
        va_backend="deepseek",
        va_base_url="https://api.deepseek.com",
        va_model="deepseek-flash",
        va_api_key="sk-test",
    )
    with patch("lavira_m3.decision.llm_router.OpenAI") as openai_cls:
        client = MagicMock()
        choice = MagicMock()
        choice.message.content = '{"bbox_2d":[199,151,502,454]}'
        choice.finish_reason = "stop"
        client.chat.completions.create.return_value.choices = [choice]
        openai_cls.return_value = client
        router = LLMRouter(cfg)
        text = router.generate_va([{"role": "user", "content": "x"}])
        assert "199" in text
        kwargs = client.chat.completions.create.call_args.kwargs
        assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}
        assert kwargs["max_completion_tokens"] == 1024
        assert kwargs["model"] == "deepseek-flash"


def test_usable_va_bbox_rejects_none_zero():
    assert usable_va_bbox([199, 151, 502, 454])
    assert not usable_va_bbox(None)
    assert not usable_va_bbox([0, 0, 0, 0])
    assert not usable_va_bbox([0.0, 0.0, 0.0, 0.0])
    assert not usable_va_bbox([1, 1, 1, 1])
    assert not usable_va_bbox([10, 20, 30])


def test_va_prompt_search_landmark_no_open_floor():
    from lavira_m3.decision.prompts_real import va_prompt

    text = va_prompt("Find printer", "front", "empty", current_step=3, width=640, height=480)
    assert "Never set target to none" not in text
    assert "open floor leading onward" not in text
    assert "open doorway or hallway opening" not in text
    assert "Never choose a glass door" in text
    assert "Every closed door is not enterable" in text
    assert "configured teammate" in text
    assert "configured teammate" in text
    assert "configured teammate" in text
    assert "Thinking budget" not in text
    assert "# Role" in text
    assert "# Context" in text
    assert "# Instructions" in text
    assert "# Constraints" in text
    assert "# Output" in text
    assert "- Step: 3" in text
    assert "- Direction: front" in text
    assert "- Image size: 640x480 pixels" in text
    assert "0..999" in text
    assert "reference image is attached" not in text
    assert "distant searchable landmark" not in text
    assert "under 128 tokens" in text
    assert "under 40 tokens" not in text
    assert "JSON only:" in text
    assert "[0,0,0,0]" in text
    assert "not be too close" not in text
    assert "can be a useful box" in text
    assert "already marked found on the board" in text
    assert "Do not box floor or a doorway instead of that object" in text
    assert "box ONE next place to go" not in text
    assert "cubicle aisle continuation" not in text
    assert "kitchenette / lounge edge" in text
    assert "usually a poor landmark" in text
    assert "Do not box a glass door, a closed door, or another robot" in text
    assert "If an Image-goal scene is given" in text
    scenes = va_prompt(
        "Find printer",
        "front",
        "empty",
        current_step=3,
        width=640,
        height=480,
        image_goal_scenes="Image-goal scenes (from T0, shared by the whole team):\n- goal_0: A kitchen with a wood island.",
    )
    assert "A kitchen with a wood island" in scenes


def test_fallback_va_bbox_is_center_quarter():
    box = fallback_va_bbox(640, 480)
    assert usable_va_bbox(box)
    x1, y1, x2, y2 = bbox_to_xyxy_px(box, 640, 480)
    assert abs(x1 - 160) <= 1
    assert abs(y1 - 120) <= 1
    assert abs(x2 - 480) <= 1
    assert abs(y2 - 360) <= 1
    assert normalize_va_target(None) == "unknown target"
    assert normalize_va_target("none") == "unknown target"
    assert normalize_va_target("n/a") == "unknown target"
    assert normalize_va_target("  ") == "unknown target"
    assert normalize_va_target("hallway opening") == "hallway opening"


def test_bbox_contract():
    u, v = bbox_center_px([0, 0, 999, 999], 640, 480)
    assert abs(u - 319) <= 2
    assert abs(v - 239) <= 2
    u, v = bbox_center_px([0.0, 0.0, 1.0, 1.0], 640, 480)
    assert abs(u - 319) <= 2
    u, v = bbox_center_px([100, 100, 200, 200], 640, 480)
    assert 60 < u < 140


def test_bbox_bottom_quarter():
    u, v = bbox_ground_px([0, 0, 999, 999], 640, 480)
    cu, cv = bbox_center_px([0, 0, 999, 999], 640, 480)
    assert (u, v) == (cu, cv)
    assert abs(u - 319) <= 2
    assert abs(v - 239) <= 2
    u, v = bbox_ground_px([0.0, 0.0, 1.0, 1.0], 640, 480)
    assert abs(u - 319) <= 2
    assert abs(v - 239) <= 2


def test_step_cap_and_history():
    from lavira_m3.decision.config import TRAVEL_BUDGET_M
    from lavira_m3.decision.nav_round import TravelBudget

    assert STEP_MAX_M == 10.0
    assert TRAVEL_BUDGET_M == 5.0
    xy, dist, clipped = clamp_map_step((20.0, 0.0), 0.0, 0.0, STEP_MAX_M)
    assert clipped
    assert abs(dist - 20.0) < 1e-6
    assert abs(xy[0] - 10.0) < 1e-6
    assert abs(xy[1]) < 1e-6
    xy, dist, clipped = clamp_map_step((10.0, 0.0), 0.0, 0.0, STEP_MAX_M)
    assert not clipped
    assert abs(xy[0] - 10.0) < 1e-6
    xy, dist, clipped = clamp_map_step((1.5, 0.0), 0.0, 0.0, STEP_MAX_M)
    assert not clipped
    assert xy == (1.5, 0.0)
    travel = TravelBudget(budget_m=5.0, jitter_m=0.02)
    assert travel.exhausted((0.0, 0.0, 0.0)) is False
    travel.add((0.01, 0.0, 1.5))
    assert travel.travel_m == 0.0
    travel.add((3.0, 4.0, 0.0))
    assert abs(travel.travel_m - 5.0) < 1e-6
    assert travel.exhausted() is True


def test_va_observe_depth_cascade():
    import numpy as np

    empty = np.zeros((48, 64), dtype=np.float32)
    z, src = va_observe_depth(empty, [100, 100, 800, 800], 64, 48, 32, 24)
    assert z is None and src == ""
    filled = np.full((48, 64), 2.4, dtype=np.float32)
    z, src = va_observe_depth(filled, [100, 100, 800, 800], 64, 48, 32, 24)
    assert z is not None and src == "point"
    point_only = np.zeros((48, 64), dtype=np.float32)
    point_only[20:28, 28:36] = 1.8
    z, src = va_observe_depth(point_only, [400, 400, 600, 700], 64, 48, 32, 24)
    assert z is None or src in ("point", "point_wide", "bottom_strip", "inset")
    inset = np.zeros((48, 64), dtype=np.float32)
    inset[8:40, 8:56] = 1.7
    inset[22:26, 30:34] = 0.0
    z, src = va_observe_depth(inset, [100, 100, 900, 900], 64, 48, 32, 24)
    assert z is not None
    assert src in ("point", "point_wide", "bottom_strip", "inset")


def test_clip_va_project_depth():
    z, clipped = clip_va_project_depth(None)
    assert z is None and clipped is False
    z, clipped = clip_va_project_depth(3.2)
    assert abs(z - 3.2) < 1e-9 and clipped is False
    z, clipped = clip_va_project_depth(5.0)
    assert abs(z - 5.0) < 1e-9 and clipped is False
    z, clipped = clip_va_project_depth(5.01)
    assert abs(z - 5.0) < 1e-9 and clipped is True
    z, clipped = clip_va_project_depth(7.4)
    assert abs(z - 5.0) < 1e-9 and clipped is True


def test_project_clips_depth_over_5m():
    import numpy as np

    from lavira_m3.decision.agent_loop import LaviraM3Agent
    from lavira_m3.decision.config import RobotConfig

    class Fake:
        def pose(self):
            return (0.0, 0.0, 0.0)

    view = {
        "rgb": np.zeros((48, 64, 3), dtype=np.uint8),
        "fx": 50.0,
        "fy": 50.0,
        "cx": 32.0,
        "cy": 24.0,
    }
    for name, pano in (("large", "surround7"), ("small_a", "cardinal4"), ("small_b", "cardinal4")):
        agent = LaviraM3Agent(RobotConfig(name=name, panorama=pano), Fake(), [], llm=object())
        agent.runtime.nav = type(
            "N", (), {"nearest_traversable": staticmethod(lambda x, y, max_m=1.0: (x, y))}
        )()
        from unittest.mock import patch

        with patch(
            "lavira_m3.decision.visual_projection.va_observe_depth",
            return_value=(7.2, "point"),
        ), patch("lavira_m3.decision.visual_projection.t0_static", return_value=False):
            xy = agent._project(view, [200, 200, 700, 700], "front")
        assert xy is not None
        assert abs(agent.last_va_depth - 5.0) < 1e-6
        assert abs(math.hypot(xy[0], xy[1]) - 5.0) < 0.15
        with patch(
            "lavira_m3.decision.visual_projection.va_observe_depth",
            return_value=(3.1, "point"),
        ), patch("lavira_m3.decision.visual_projection.t0_static", return_value=False):
            xy = agent._project(view, [200, 200, 700, 700], "front")
        assert abs(agent.last_va_depth - 3.1) < 1e-6
        assert abs(math.hypot(xy[0], xy[1]) - 3.1) < 0.15


def test_search_step_no_depth_not_found():
    import numpy as np

    from lavira_m3.decision.agent_loop import LaviraM3Agent
    from lavira_m3.decision.config import RobotConfig

    class Fake:
        def pose(self):
            return (0.0, 0.0, 0.0)

    agent = LaviraM3Agent(
        RobotConfig(name="small_b", panorama="cardinal4"), Fake(), [], llm=object()
    )
    view = {
        "name": "front",
        "rgb": np.zeros((48, 64, 3), dtype=np.uint8),
        "depth": np.zeros((48, 64), dtype=np.float32),
        "fx": 200.0,
        "fy": 200.0,
        "cx": 32.0,
        "cy": 24.0,
    }
    xy = agent._project(view, [400, 200, 600, 400], "front")
    assert xy is not None
    assert agent.last_va_search_step is True
    assert agent.last_va_depth is None
    dist = (xy[0] ** 2 + xy[1] ** 2) ** 0.5
    assert 1.2 <= dist <= 3.2


def test_view_mask_center_depth():
    import numpy as np

    from lavira_m3.decision.config import VIEW_MASK_MIN_DEPTH_M

    assert VIEW_MASK_MIN_DEPTH_M == 1.5
    near = np.full((48, 64), 0.8, dtype=np.float32)
    far = np.full((48, 64), 3.0, dtype=np.float32)
    empty = np.zeros((48, 64), dtype=np.float32)
    assert view_center_mean_depth(near) is not None and view_center_mean_depth(near) < 1.5
    assert view_center_mean_depth(far) > 1.5
    assert view_center_mean_depth(empty) is None
    frames = [
        {"name": "front", "depth": far},
        {"name": "left", "depth": near},
        {"name": "behind", "depth": empty},
        {"name": "right", "center_depth_m": 0.6},
    ]
    blocked, depths = view_mask_blocked(frames, ["front", "left", "behind", "right"], 1.5)
    assert "left" in blocked and "right" in blocked
    assert "front" not in blocked and "behind" not in blocked
    legal = legal_la_directions(["front", "left", "behind", "right"], blocked)
    assert "left" not in legal and "right" not in legal
    assert "front" in legal
    assert "behind" in legal
    assert legal_la_directions(["front", "left"], ["front", "left"]) == ["front"]
    assert "behind" in legal_la_directions(["front", "left", "behind", "right"], [])
    assert "left_back" in legal_la_directions(list(SURROUND_VIEWS), [])
    assert "behind" not in legal_la_directions(
        ["front", "left", "behind", "right"], [], exclude_rear=True
    )
    assert "left_back" not in legal_la_directions(list(SURROUND_VIEWS), [], exclude_rear=True)


def test_la_prompt_omits_blocked_dirs():
    from lavira_m3.decision.prompts_real import la_prompt

    text = la_prompt(
        "agent_2",
        ["front", "behind"],
        "board",
        "none",
        hardware_dirs=["front", "left", "behind", "right"],
        blocked_note="\nNear-range mask: left (0.60 m), right (0.60 m)\n",
    )
    assert '"navigate to left"' not in text
    assert '"navigate to right"' not in text
    assert '"navigate to front"' in text
    assert '"navigate to behind"' not in text
    assert "rear-facing cameras are not legal actions" in text
    assert "not a legal action" not in text
    assert "backtrack" not in text
    assert "Near-range mask" in text
    assert "four fixed cameras" in text
    assert "# Role" in text
    assert "# Context" in text
    assert "# Instructions" in text
    assert "# Constraints" in text
    assert "# Output" in text
    assert "JSON only:" in text
    assert '"stop"' not in text
    assert "Do not output a stop field" not in text
    assert "prefer leaving that object" in text
    assert "double-check" in text
    assert "Never choose a glass door" in text
    assert "Every closed door is not enterable" in text
    assert "already-open non-glass doorway or open passage" in text
    assert "configured teammate" in text
    assert "configured teammate" in text
    assert "configured teammate" in text
    assert "already marked found on the board" in text
    assert "within about 3 m" in text or "about 3 m" in text
    assert "choose that view's direction and go to it" not in text
    assert "enters another room" not in text
    assert "cubicle / workstation aisles" in text
    assert "Prefer not re-exploring" in text
    assert "Initial Panorama" in text
    assert "Other robots are never a search target" in text
    assert "If Image-goal scenes are given" in text
    with_scene = la_prompt(
        "agent_2",
        ["front", "behind"],
        "board",
        "none",
        image_goal_scenes="Image-goal scenes (from T0, shared by the whole team):\n- goal_0: A kitchen with a wood island.",
    )
    assert "A kitchen with a wood island" in with_scene
    assert "which place an object belongs to" in with_scene


def test_async_la_excludes_rear():
    import numpy as np

    from lavira_m3.contracts import NavResult
    from lavira_m3.decision.agent_loop import LaviraM3Agent
    from lavira_m3.decision.config import RobotConfig

    class Board:
        def prompt_blocks(self):
            return {"available": True, "board": "board", "progress": ""}

        def post_progress(self, *a, **k):
            return {"ok": True}

        def post_trace(self, *a, **k):
            return {"ok": True}

        def post_history(self, *a, **k):
            return {"ok": True}

        def snapshot(self):
            return {
                "ok": True,
                "goals": [{"goal_index": 0, "object_category": "chair", "modality": "ON"}],
                "found_by": [""],
            }

        def fetch_image(self, spec):
            return None

    class RT:
        def pose(self):
            return (0.0, 0.0, 0.0)

        def capture_panorama(self):
            rgb = np.full((18, 24, 3), 40, dtype=np.uint8)
            return [
                {"name": "front", "rgb": rgb},
                {"name": "left", "rgb": rgb},
                {"name": "behind", "rgb": rgb},
                {"name": "right", "rgb": rgb},
            ]

        def capture_direction(self, name):
            rgb = np.full((18, 24, 3), 80, dtype=np.uint8)
            return {"name": name, "rgb": rgb}

        def face_direction(self, name):
            return NavResult(True, "skipped", "already facing")

        def navigate_to(self, x, y, yaw=None, **kwargs):
            return NavResult(True, "skipped", "already at goal")

    class LLM:
        def __init__(self):
            self.la_text = []

        def generate_la(self, messages, max_tokens=2048):
            blob = []
            for m in messages:
                content = m.get("content")
                if isinstance(content, list):
                    for p in content:
                        if isinstance(p, dict) and p.get("type") == "text":
                            blob.append(str(p.get("text") or ""))
            self.la_text.append("\n".join(blob))
            return '{"action":"navigate to behind","progress_analysis":"x","reasoning":"y","claim_goal_id":""}'

        def generate_va(self, messages):
            return '{"bbox_2d":[100,100,200,200],"target":"chair","reasoning":"r"}'

    llm = LLM()
    agent = LaviraM3Agent(
        RobotConfig(name="small_a", panorama="cardinal4"), RT(), [{"goal_index": 0}], llm=llm
    )
    agent.board = Board()
    agent.t0_joint = False
    agent._go = lambda view, bbox, direction, **kwargs: ((0.4, 0.0), NavResult(True, "ok", "moved"))
    agent._last_pano = agent.runtime.capture_panorama()
    assert agent._pick_other_direction("front") != "behind"
    assert agent._clamp_async_la_direction("behind") != "behind"
    assert agent._clamp_async_la_direction("front") == "front"
    assert agent.cycle() is True
    assert '"navigate to behind"' not in llm.la_text[0]
    assert agent._last_la["direction"] != "behind"
    assert agent._last_va["direction"] != "behind"


def test_teammate_spread_note_uses_trace_pose():
    from lavira_m3.decision.prompts_real import build_teammate_spread_note

    note = build_teammate_spread_note(
        "large",
        traces={"small_a": {"waypoint": [5.4, -3.0]}, "small_b": {"va_map": [1.2, 8.0]}},
        histories={},
    )
    assert "agent_1 (small_a): last known map pose (5.40, -3.00)" in note
    assert "agent_2 (small_b): last known map pose (1.20, 8.00)" in note
    assert "agent_0 (large)" not in note
    assert "Prefer not walking toward these poses or waypoints" in note
    assert "configured teammate" in note
    assert "configured teammate" in note
    assert "configured teammate" in note


def test_dcs_prompt_requires_near_stop():
    from lavira_m3.decision.prompts_real import stop_check_prompt

    on = stop_check_prompt(
        "Find remaining unresolved targets:\n- goal_1",
        "goal_1",
        goal={"modality": "ON", "object_category": "printer"},
    )
    tn = stop_check_prompt(
        "Find remaining unresolved targets:\n- goal_2",
        "goal_2",
        goal={
            "modality": "TN",
            "object_category": "office chairs",
            "tn_text": "black rolling office chairs along the cubicle aisle",
        },
    )
    iin = stop_check_prompt(
        "Find remaining unresolved targets:\n- goal_0",
        "goal_0",
        goal={
            "modality": "IIN",
            "object_category": "yellow dog figure",
            "reference_image": "/api/board/ref/goal_0",
        },
    )
    text = tn
    assert "The claim is reported only if you output FOUND" in text
    assert "appears in the current panorama" in text
    assert '"reasoning": "<brief, under 40 tokens>"' in text
    assert '"decision": "FOUND" or "CONTINUE"' in text
    assert '"analysis"' not in text
    assert "bbox_2d" not in text
    assert "about 5 m" not in text
    assert "already large" not in text
    assert "tiny distant blob" not in text
    assert "Do not CONTINUE because it looks small" not in text
    assert "under 3 m" not in text
    assert "about 3 m" not in text
    assert "CONTINUE" in text
    assert '"camera"' not in text
    assert "Modality: ON (name)" in on
    assert "Name: printer" in on
    assert "Modality: TN (text)" in tn
    assert "black rolling office chairs along the cubicle aisle" in tn
    assert "Modality: IIN (image)" in iin
    assert "attached reference image" in iin
    assert "Name: yellow dog figure" in iin


def test_claim_and_dcs_use_3m():
    from lavira_m3.decision.config import RobotConfig, claim_max_m_for
    from lavira_m3.decision.prompts_real import (
        dcs_reject_feedback,
        la_prompt,
        stop_check_prompt,
    )

    assert claim_max_m_for("large") == 3.0
    assert claim_max_m_for("agent_0") == 3.0
    assert claim_max_m_for("small_a") == 3.0
    assert claim_max_m_for("small_b") == 3.0
    assert claim_max_m_for("agent_2") == 3.0
    assert RobotConfig(name="large", panorama="surround7").found_max_depth_m == 3.0
    assert RobotConfig(name="small_a", panorama="cardinal4").found_max_depth_m == 3.0
    assert RobotConfig(name="small_b", panorama="cardinal4").found_max_depth_m == 3.0

    aloha = la_prompt("agent_large", ["front"], "board", "none")
    assert "about 3 m" in aloha
    assert "within about 2 m" not in aloha
    assert "about 5 m" not in aloha
    dog = la_prompt("agent_small_b", ["front"], "board", "none")
    assert "about 3 m" in dog
    assert "within about 2 m" not in dog
    assert "about 5 m" not in dog

    dcs = stop_check_prompt(
        "Find remaining unresolved targets:\n- goal_0",
        "goal_0",
        claim_max_m=3.0,
        goal={"modality": "ON", "object_category": "printer"},
    )
    assert "appears in the current panorama" in dcs
    assert "Name: printer" in dcs
    assert "bbox_2d" not in dcs
    assert "about 5 m" not in dcs
    assert "already large" not in dcs

    assert "not visible, or not that target" in dcs_reject_feedback(3.0)


def test_instruction_skips_found_goals():
    from lavira_m3.decision.agent_loop import LaviraM3Agent
    from lavira_m3.decision.config import RobotConfig

    class Board:
        def snapshot(self):
            return {
                "ok": True,
                "goals": [
                    {"goal_index": 0, "modality": "IIN", "object_category": "fig"},
                    {"goal_index": 1, "modality": "ON", "object_category": "printer"},
                    {"goal_index": 2, "modality": "TN", "object_category": "chairs"},
                ],
                "found_by": [-1, -1, 2],
            }

    agent = LaviraM3Agent(
        RobotConfig(name="small_b", panorama="cardinal4"), object(), [], llm=object()
    )
    agent.board = Board()
    text = agent._instruction()
    assert "goal_2" not in text
    assert "printer" in text
    assert "chairs" not in text
    assert "reference image unavailable" in text


def test_va_gets_iin_reference_image():
    from PIL import Image

    from lavira_m3.decision.agent_loop import LaviraM3Agent
    from lavira_m3.decision.config import RobotConfig
    from lavira_m3.decision.model_messages import encode_image

    ref = Image.new("RGB", (16, 16), (200, 30, 30))

    class Board:
        def snapshot(self):
            return {
                "ok": True,
                "goals": [
                    {
                        "goal_index": 0,
                        "modality": "IIN",
                        "object_category": "dog-figure",
                        "reference_image": "/api/board/ref/goal_0",
                    }
                ],
                "found_by": [-1],
            }

        def fetch_image(self, spec):
            assert spec == "/api/board/ref/goal_0"
            return ref

    captured = {}

    class LLM:
        def generate_va(self, messages):
            captured["messages"] = messages
            return '{"reasoning": "x", "bbox_2d": [10, 20, 30, 40], "target": "dog-figure"}'

    agent = LaviraM3Agent(
        RobotConfig(name="small_b", panorama="cardinal4"), object(), [], llm=LLM()
    )
    agent.board = Board()
    view = {"name": "front", "rgb": __import__("numpy").zeros((8, 8, 3), dtype="uint8")}
    agent._va(view, "front", "")
    blob = captured["messages"][0]["content"]
    texts = [p.get("text") for p in blob if p.get("type") == "text"]
    assert any("reference image" in str(t).lower() for t in texts)
    images = [p for p in blob if p.get("type") == "image_url"]
    assert len(images) >= 2
    assert encode_image(ref) in images[-1]["image_url"]["url"]
    text = agent._instruction()
    assert "match the attached reference image" in text


def test_dcs_gets_claimed_iin_reference_image():
    from PIL import Image

    from lavira_m3.decision.agent_loop import LaviraM3Agent
    from lavira_m3.decision.config import RobotConfig
    from lavira_m3.decision.model_messages import encode_image

    ref = Image.new("RGB", (16, 16), (20, 180, 40))

    class Board:
        def snapshot(self):
            return {
                "ok": True,
                "goals": [
                    {
                        "goal_index": 0,
                        "modality": "IIN",
                        "object_category": "dog-figure",
                        "reference_image": "/api/board/ref/goal_0",
                    }
                ],
                "found_by": [-1],
            }

        def fetch_image(self, spec):
            assert spec == "/api/board/ref/goal_0"
            return ref

    captured = {}

    class LLM:
        def generate_la(self, messages, max_tokens=2048):
            captured["messages"] = messages
            return '{"reasoning": "visible", "decision": "FOUND"}'

    agent = LaviraM3Agent(
        RobotConfig(name="small_b", panorama="cardinal4"), object(), [], llm=LLM()
    )
    agent.board = Board()
    import numpy as np

    frames = [{"name": "front", "rgb": np.zeros((8, 8, 3), dtype=np.uint8)}]
    ok, data = agent._dcs(frames, "goal_0")
    assert ok is True
    blob = captured["messages"][0]["content"]
    texts = [p.get("text") for p in blob if p.get("type") == "text"]
    assert any("Reference image for the claimed target" in str(t) for t in texts)
    assert any("Modality: IIN" in str(t) for t in texts)
    images = [p for p in blob if p.get("type") == "image_url"]
    assert len(images) == 2
    assert encode_image(ref) in images[0]["image_url"]["url"]
    assert data["decision"] == "FOUND"


def test_dcs_on_and_tn_use_name_or_description():
    from lavira_m3.decision.agent_loop import LaviraM3Agent
    from lavira_m3.decision.config import RobotConfig

    class Board:
        def snapshot(self):
            return {
                "ok": True,
                "goals": [
                    {"goal_index": 0, "modality": "ON", "object_category": "printer"},
                    {
                        "goal_index": 1,
                        "modality": "TN",
                        "object_category": "office chairs",
                        "tn_text": "black rolling chairs by the aisle",
                    },
                ],
                "found_by": [-1, -1],
            }

    captured = {}

    class LLM:
        def generate_la(self, messages, max_tokens=2048):
            captured.setdefault("texts", []).append(messages[0]["content"][0]["text"])
            captured.setdefault("nimg", []).append(
                sum(1 for p in messages[0]["content"] if p.get("type") == "image_url")
            )
            return '{"reasoning": "no", "decision": "CONTINUE"}'

    agent = LaviraM3Agent(
        RobotConfig(name="small_b", panorama="cardinal4"), object(), [], llm=LLM()
    )
    agent.board = Board()
    import numpy as np

    frames = [{"name": "front", "rgb": np.zeros((8, 8, 3), dtype=np.uint8)}]
    agent._dcs(frames, "goal_0")
    agent._dcs(frames, "goal_1")
    assert "Name: printer" in captured["texts"][0]
    assert "Modality: ON" in captured["texts"][0]
    assert "black rolling chairs by the aisle" in captured["texts"][1]
    assert "Modality: TN" in captured["texts"][1]
    assert captured["nimg"] == [1, 1]


def test_handle_claim_dcs_only():
    from lavira_m3.contracts import NavResult
    from lavira_m3.decision.agent_loop import LaviraM3Agent
    from lavira_m3.decision.config import RobotConfig

    class Board:
        def __init__(self):
            self.found = [-1, -1]
            self.found_xy = [None, None]
            self.posted = []

        def snapshot(self):
            return {
                "ok": True,
                "goals": [{"goal_index": 0}, {"goal_index": 1}],
                "found_by": list(self.found),
                "found_xy": list(self.found_xy),
            }

        def post_found(self, goal_id, verification=None, xy=None):
            self.posted.append(goal_id)
            idx = int(str(goal_id).split("_")[1])
            self.found[idx] = 2
            if xy is not None:
                self.found_xy[idx] = [float(xy[0]), float(xy[1])]
            snap = self.snapshot()
            snap["ok"] = True
            return snap

        def prompt_blocks(self):
            return {"available": True, "board": "", "progress": ""}

    class RT:
        def pose(self):
            return (0.0, 0.0, 0.0)

        def navigate_to(self, *a, **k):
            return NavResult(True, "succeeded", "ok")

    agent = LaviraM3Agent(
        RobotConfig(name="small_b", panorama="cardinal4"),
        RT(),
        [{"goal_index": 0}, {"goal_index": 1}],
        llm=object(),
    )
    agent.board = Board()
    frames = [{"name": "front", "rgb": object()}]
    agent.last_va_depth = 1.2
    agent.last_va_depth_estimated = False
    agent._dcs = lambda frames, claim: (True, {"decision": "FOUND", "reasoning": "visible"})
    assert agent._handle_claim("goal_0", frames) is False
    assert agent.board.posted == ["goal_0"]
    assert agent.board.found_xy[0] == [0.0, 0.0]
    assert agent._goal_still_open("goal_0") is False
    assert agent._handle_claim("goal_0", frames) is False
    agent._dcs = lambda frames, claim: (True, {"decision": "FOUND", "reasoning": "visible"})
    assert agent._handle_claim("goal_1", frames) is True
    assert agent.board.posted == ["goal_0", "goal_1"]
    agent.board.found = [-1, -1]
    agent.board.found_xy = [None, None]
    agent.board.posted = []
    agent.runtime.pose = lambda: (0.0, 0.0, 0.0)
    agent.last_va_depth = 3.4
    agent._dcs = lambda frames, claim: (True, {"decision": "FOUND", "reasoning": "visible but far"})
    assert agent._handle_claim("goal_0", frames) is False
    assert agent.board.posted == ["goal_0"]
    agent.board.found = [-1, -1]
    agent.board.found_xy = [None, None]
    agent.board.posted = []
    agent.last_va_depth = None
    agent._dcs = lambda frames, claim: (
        True,
        {"decision": "FOUND", "reasoning": "visible no depth"},
    )
    assert agent._handle_claim("goal_0", frames) is False
    assert agent.board.posted == ["goal_0"]
    agent.board.found = [-1, -1]
    agent.board.found_xy = [None, None]
    agent.board.posted = []
    agent._dcs = lambda frames, claim: (False, {"decision": "CONTINUE", "reasoning": "not found"})
    assert agent._handle_claim("goal_0", frames) is False
    assert agent.board.posted == []
    assert "not visible, or not that target" in agent.stop_feedback

    aloha = LaviraM3Agent(
        RobotConfig(name="large", panorama="surround7"),
        RT(),
        [{"goal_index": 0}, {"goal_index": 1}],
        llm=object(),
    )
    aloha.board = Board()
    aloha.last_va_depth_estimated = False
    aloha._dcs = lambda frames, claim: (True, {"decision": "FOUND", "reasoning": "visible"})
    aloha.last_va_depth = 2.5
    assert aloha.robot.found_max_depth_m == 3.0
    assert aloha._handle_claim("goal_0", frames) is False
    assert aloha.board.posted == ["goal_0"]


def test_found_view_posted_to_trace():
    import numpy as np

    from lavira_m3.decision.agent_loop import LaviraM3Agent
    from lavira_m3.decision.config import RobotConfig

    class Board:
        def __init__(self):
            self.found = [-1]
            self.found_xy = [None]
            self.traces = []

        def snapshot(self):
            return {
                "ok": True,
                "goals": [{"goal_index": 0, "object_category": "yellow cup"}],
                "found_by": list(self.found),
                "found_xy": list(self.found_xy),
            }

        def post_found(self, goal_id, verification=None, xy=None):
            self.found[0] = 1
            snap = self.snapshot()
            snap["ok"] = True
            return snap

        def post_trace(self, payload):
            self.traces.append(dict(payload))
            return {"ok": True}

        def prompt_blocks(self):
            return {"available": True, "board": "", "progress": ""}

    class RT:
        def pose(self):
            return (1.0, 2.0, 0.0)

    rgb = np.zeros((48, 64, 3), dtype=np.uint8)
    rgb[10:30, 20:40] = 200
    frames = [
        {"name": "left", "rgb": rgb},
        {"name": "front", "rgb": np.zeros((48, 64, 3), dtype=np.uint8)},
    ]
    agent = LaviraM3Agent(
        RobotConfig(name="small_a", panorama="cardinal4"), RT(), [{"goal_index": 0}], llm=object()
    )
    agent.board = Board()
    agent._dcs = lambda frames, claim: (
        True,
        {
            "decision": "FOUND",
            "reasoning": "visible",
            "bbox_2d": [200, 200, 700, 700],
            "camera": "left",
        },
    )
    assert agent._handle_claim("goal_0", frames, direction="front") is True
    assert agent.board.traces
    rec = agent.board.traces[-1]
    assert rec["direction"] == "left"
    assert rec["found_goal_id"] == "goal_0"
    assert rec["va"]["found"] is True
    assert rec["va"]["camera"] == "left"
    assert rec["va_image"]
    assert rec["va_bbox"] == [200, 200, 700, 700]


def test_claim_allows_second_goal_at_same_pose():
    from lavira_m3.decision.agent_loop import LaviraM3Agent
    from lavira_m3.decision.config import RobotConfig
    from lavira_m3.decision.navigation_results import parse_map_xy
    from lavira_m3.decision.prompts_real import AFTER_FOUND_SWITCH_FEEDBACK

    assert "Do not claim a different goal_id" in AFTER_FOUND_SWITCH_FEEDBACK
    assert "1.5 m" not in AFTER_FOUND_SWITCH_FEEDBACK
    assert parse_map_xy([1.2, -3.4, 0.1]) == (1.2, -3.4)
    assert parse_map_xy(None) is None

    class Board:
        def __init__(self):
            self.found = [-1, 1, -1]
            self.found_xy = [None, [0.2, 0.1], None]
            self.posted = []

        def snapshot(self):
            return {
                "ok": True,
                "goals": [{"goal_index": 0}, {"goal_index": 1}, {"goal_index": 2}],
                "found_by": list(self.found),
                "found_xy": list(self.found_xy),
            }

        def post_found(self, goal_id, verification=None, xy=None):
            self.posted.append(goal_id)
            idx = int(str(goal_id).split("_")[1])
            self.found[idx] = 0
            if xy is not None:
                self.found_xy[idx] = [float(xy[0]), float(xy[1])]
            snap = self.snapshot()
            snap["ok"] = True
            return snap

        def prompt_blocks(self):
            return {"available": True, "board": "", "progress": ""}

    class RT:
        def pose(self):
            return (0.0, 0.0, 0.0)

    agent = LaviraM3Agent(
        RobotConfig(name="small_a", panorama="cardinal4"),
        RT(),
        [{"goal_index": 0}, {"goal_index": 1}, {"goal_index": 2}],
        llm=object(),
    )
    agent.board = Board()
    frames = [{"name": "front", "rgb": object()}]
    agent._dcs = lambda frames, claim: (True, {"decision": "FOUND", "reasoning": "visible"})
    assert agent._handle_claim("goal_0", frames) is False
    assert agent.board.posted == ["goal_0"]
    agent._dcs = lambda frames, claim: (True, {"decision": "FOUND", "reasoning": "same desk"})
    assert agent._handle_claim("goal_2", frames) is True
    assert agent.board.posted == ["goal_0", "goal_2"]


def test_found_depth_gate():
    import numpy as np

    near = np.full((48, 64), 1.2, dtype=np.float32)
    far = np.full((48, 64), 3.5, dtype=np.float32)
    bbox = [100, 100, 800, 800]
    z_near = bbox_inset_median_depth(near, bbox, 64, 48)
    z_far = bbox_inset_median_depth(far, bbox, 64, 48)
    assert z_near is not None and z_near < 2.0
    assert z_far is not None and z_far >= 3.0
    assert bbox_inset_median_depth(None, bbox, 64, 48) is None


def test_interrupt_idle():
    from lavira_m3.decision.interrupt import requested

    assert requested() is False


def test_json_public():
    from lavira_m3.decision.navigation_results import json_public

    data = json_public({"action": "navigate to front", "_raw": "secret", "stop": False})
    assert data["action"] == "navigate to front"
    assert "_raw" not in data


def test_approach_points():
    from lavira_m3.decision.agent_loop import LaviraM3Agent

    pts = LaviraM3Agent._approach_points(None, (3.0, 0.0), (0.0, 0.0, 0.0))
    assert pts[0] == (3.0, 0.0)
    near = min(pts[1:], key=lambda p: abs(p[0]))
    assert 0.70 <= near[0] <= 0.90 and abs(near[1]) < 1e-6


def test_cap_step_keeps_short_depth():
    from lavira_m3.decision.agent_loop import LaviraM3Agent
    from lavira_m3.decision.config import STEP_MIN_M, RobotConfig
    from lavira_m3.decision.navigation_geometry import inside_body_envelope

    class Fake:
        def pose(self):
            return (0.0, 0.0, 0.0)

    agent = LaviraM3Agent(
        RobotConfig(name="small_b", panorama="cardinal4"), Fake(), [], llm=object()
    )
    left_close = (0.0, 0.57)
    assert not inside_body_envelope(left_close[0], left_close[1], 0.0, 0.0, 0.0, "small_b")
    out = agent._cap_step(left_close, (0.0, 0.0, 0.0))
    assert out == (STEP_MIN_M, 0.0)
    assert agent.last_va_heading_step is True
    out = agent._cap_step((0.10, 0.0), (0.0, 0.0, 0.0))
    assert out == (STEP_MIN_M, 0.0)
    out = agent._cap_step((0.0, 2.0), (0.0, 0.0, 0.0))
    assert abs(out[0]) < 1e-6 and abs(out[1] - 2.0) < 1e-6
    out = agent._cap_step((12.0, 0.0), (0.0, 0.0, 0.0))
    assert abs(out[0] - 12.0) < 1e-6 and abs(out[1]) < 1e-6
    assert agent.last_va_heading_step is False
    assert agent._goal_ok((0.25, 0.0), (0.0, 0.0, 0.0)) is False
    assert agent._goal_ok((0.80, 0.0), (0.0, 0.0, 0.0)) is False
    assert agent._goal_ok((STEP_MIN_M, 0.0), (0.0, 0.0, 0.0)) is True


def test_small_b_inside_envelope_keeps_measured_range():
    from lavira_m3.decision.agent_loop import LaviraM3Agent
    from lavira_m3.decision.config import RobotConfig
    from lavira_m3.decision.navigation_geometry import inside_body_envelope

    class Fake:
        def pose(self):
            return (0.0, 0.0, 0.0)

    agent = LaviraM3Agent(
        RobotConfig(name="small_b", panorama="cardinal4"), Fake(), [], llm=object()
    )
    side = (0.10, 0.05)
    assert inside_body_envelope(side[0], side[1], 0.0, 0.0, 0.0, "small_b")
    out = agent._cap_step(side, (0.0, 0.0, 0.0))
    assert math.isclose(math.hypot(*out), 1.5)
    assert agent.last_va_heading_step is True


def test_large_inside_envelope_keeps_measured_range():
    from lavira_m3.decision.agent_loop import LaviraM3Agent
    from lavira_m3.decision.config import RobotConfig

    class Fake:
        def pose(self):
            return (0.0, 0.0, 0.0)

    agent = LaviraM3Agent(RobotConfig(name="large", panorama="surround7"), Fake(), [], llm=object())
    side = (0.20, 0.30)
    out = agent._cap_step(side, (0.0, 0.0, 0.0))
    assert math.isclose(math.hypot(*out), 1.5)
    assert agent.last_va_heading_step is True


def test_small_b_filtered_walk_target_skips_nav():
    from lavira_m3.contracts import NavResult
    from lavira_m3.decision.agent_loop import LaviraM3Agent
    from lavira_m3.decision.config import RobotConfig

    calls = []

    class RT:
        def pose(self):
            return (0.0, 0.0, 0.0)

        def navigate_to(self, x, y, yaw=None, **kwargs):
            calls.append((x, y, kwargs.get("requested_goal")))
            return NavResult(True, "succeeded", "moved")

    agent = LaviraM3Agent(RobotConfig(name="small_b", panorama="cardinal4"), RT(), [], llm=object())
    # Close filtered walk point skips, even if raw VA depth is far.
    agent.last_va_depth = 1.20
    agent.last_va_depth_estimated = False
    agent._project = lambda *a, **k: (0.50, 0.0)
    xy, result = agent._go({}, [10, 20, 30, 40], "front")
    assert xy == (0.50, 0.0)
    assert result.ok is True
    assert "0.75" in result.message
    assert calls == []

    # Far filtered walk point is sent; close VA depth does not skip.
    calls.clear()
    agent.last_va_depth = 0.40
    agent._project = lambda *a, **k: (2.00, 0.0)
    agent._approach_points = lambda xy, pose: [xy]
    xy, result = agent._go({}, [10, 20, 30, 40], "front")
    assert calls
    assert calls[0][0] == 2.00
    assert calls[0][2] == (2.00, 0.0)


def test_project_keeps_measured_depth_under_3m():
    import numpy as np

    from lavira_m3.decision.agent_loop import LaviraM3Agent
    from lavira_m3.decision.config import RobotConfig

    class Fake:
        def pose(self):
            return (0.0, 0.0, 0.0)

    agent = LaviraM3Agent(
        RobotConfig(name="small_b", panorama="cardinal4"), Fake(), [], llm=object()
    )
    view = {
        "name": "front",
        "rgb": np.zeros((48, 64, 3), dtype=np.uint8),
        "depth": np.full((48, 64), 0.90, dtype=np.float32),
        "fx": 200.0,
        "fy": 200.0,
        "cx": 32.0,
        "cy": 24.0,
    }
    xy = agent._project(view, [400, 200, 600, 400], "front")
    assert xy is not None
    assert agent.last_va_search_step is False
    assert agent.last_va_depth is not None and abs(agent.last_va_depth - 0.90) < 0.05
    dist = (xy[0] ** 2 + xy[1] ** 2) ** 0.5
    assert 0.70 <= dist <= 1.10
    capped = agent._cap_step(xy, (0.0, 0.0, 0.0))
    assert abs(capped[0] - xy[0]) < 1e-9 and abs(capped[1] - xy[1]) < 1e-9


def test_classify_nav_completion_original_intermediate_blocked():
    from lavira_m3.decision.navigation_geometry import (
        COMPLETION_BLOCKED,
        COMPLETION_INTERMEDIATE,
        COMPLETION_ORIGINAL,
        classify_nav_completion,
    )

    assert (
        classify_nav_completion((0, 0), (4.9, 0), (5.0, 0), (5.0, 0), 0.75) == COMPLETION_ORIGINAL
    )
    assert (
        classify_nav_completion((0, 0), (1.4, 0), (5.0, 0), (1.5, 0), 0.75)
        == COMPLETION_INTERMEDIATE
    )
    assert (
        classify_nav_completion((0, 0), (0.05, 0), (5.0, 0), (0.0, 0), 0.75) == COMPLETION_BLOCKED
    )
    assert classify_nav_completion((0, 0), (0.0, 0), (5.0, 0), None, 0.75) is None


def test_snap_to_nearest_free():
    from lavira_m3.decision.map_geometry import snap_to_nearest_free

    # A 3x3 occupied block around (1.05, 1.05); inflate 0.25 m.
    scene = {
        "map": {
            "resolution": 0.1,
            "origin": [0.0, 0.0],
            "fused": [[ix, iy] for ix in range(8, 14) for iy in range(8, 14)],
        }
    }
    raw = (1.05, 1.05)
    snapped, did = snap_to_nearest_free(raw[0], raw[1], scene, inflate_m=0.25, max_m=2.0)
    assert did is True
    assert abs(snapped[0] - raw[0]) > 0.2 or abs(snapped[1] - raw[1]) > 0.2
    # Nearest free should be closer to the original than a far corner.
    assert ((snapped[0] - raw[0]) ** 2 + (snapped[1] - raw[1]) ** 2) ** 0.5 < 1.0
    free, did_free = snap_to_nearest_free(3.05, 3.05, scene, inflate_m=0.25)
    assert did_free is False
    assert abs(free[0] - 3.05) < 1e-9 and abs(free[1] - 3.05) < 1e-9


def test_large_go_submits_va_once():
    from lavira_m3.contracts import NavResult
    from lavira_m3.decision.agent_loop import LaviraM3Agent
    from lavira_m3.decision.config import RobotConfig

    calls = []

    class RT:
        def pose(self):
            return (0.0, 0.0, 0.0)

        def navigate_to(self, x, y, yaw=None):
            calls.append((round(x, 2), round(y, 2)))
            return NavResult(True, "succeeded", "ok", extra={"path": [[0.0, 0.0], [x, y]]})

    agent = LaviraM3Agent(RobotConfig(name="large", panorama="surround7"), RT(), [], llm=object())
    agent._project = lambda *a, **k: (2.05, 2.05)
    xy, result = agent._go({}, [10, 20, 30, 40], "front")
    assert result.ok is True
    assert xy == (2.05, 2.05)
    assert calls == [(2.05, 2.05)]


def test_large_go_preview_failed_retries_approach_points():
    from lavira_m3.contracts import NavResult
    from lavira_m3.decision.agent_loop import LaviraM3Agent
    from lavira_m3.decision.config import RobotConfig

    calls = []

    class RT:
        def pose(self):
            return (0.0, 0.0, 0.0)

        def navigate_to(self, x, y, yaw=None):
            calls.append((round(x, 2), round(y, 2)))
            return NavResult(False, "preview_failed", "no reachable target nearby")

        @property
        def nav(self):
            return None

    agent = LaviraM3Agent(RobotConfig(name="large", panorama="surround7"), RT(), [], llm=object())
    agent._project = lambda *a, **k: (3.0, 0.0)
    xy, result = agent._go({}, [10, 20, 30, 40], "front")
    assert result.ok is False
    assert result.state == "preview_failed"
    assert xy is None
    assert len(calls) == 1  # Rejected preview must not submit a second motion goal.
    assert calls[0] == (3.0, 0.0)
    assert all(abs(call[1]) < 1e-6 for call in calls)
    assert all(calls[i][0] < calls[i - 1][0] - 0.10 for i in range(1, len(calls)))


def test_serial_candidate_distance_all_robots():
    from lavira_m3.contracts import NavResult
    from lavira_m3.decision.agent_loop import LaviraM3Agent
    from lavira_m3.decision.config import RobotConfig

    for name, pano in (("large", "surround7"), ("small_a", "cardinal4"), ("small_b", "cardinal4")):
        calls = []

        class RT:
            def __init__(self, recorded_calls):
                self.recorded_calls = recorded_calls

            def pose(self):
                return (0.0, 0.0, 0.0)

            def navigate_to(self, x, y, yaw=None, **kwargs):
                self.recorded_calls.append((round(x, 2), round(y, 2)))
                return NavResult(False, "preview_failed", "blocked")

        agent = LaviraM3Agent(RobotConfig(name=name, panorama=pano), RT(calls), [], llm=object())
        agent._project = lambda *a, **k: (10.0, 0.0)
        agent._go({}, [10, 20, 30, 40], "front")
        assert calls == [(10.0, 0.0)], name


def test_clip_goal_to_map_aabb():
    from lavira_m3.decision.map_geometry import bound_goal_to_map, clip_goal_to_map_aabb

    aabb = (0.0, 0.0, 4.0, 4.0)
    inside, clipped = clip_goal_to_map_aabb(1.0, 1.0, 2.5, 2.0, aabb)
    assert clipped is False
    assert inside == (2.5, 2.0)
    out, clipped = clip_goal_to_map_aabb(1.0, 1.0, 8.0, 1.0, aabb)
    assert clipped is True
    assert abs(out[0] - 3.9) < 1e-6
    assert abs(out[1] - 1.0) < 1e-6
    scene = {
        "map": {
            "resolution": 0.1,
            "origin": [0.0, 0.0],
            "fused": [[i, 0] for i in range(10)],
        }
    }
    # In-canvas goals stay; only the fixed 100 m map rejects a point.
    kept, clipped = bound_goal_to_map(3.0, 0.05, 0.2, 0.05, scene)
    assert clipped is False
    assert kept == (3.0, 0.05)
    far, clipped = bound_goal_to_map(12.0, 0.05, 0.2, 0.05, scene)
    assert clipped is False
    assert far == (12.0, 0.05)
    hit, clipped = bound_goal_to_map(60.0, 0.05, 0.2, 0.05, scene)
    assert clipped is True
    assert hit == (60.0, 0.05)


def test_path_loop_trim():
    from lavira_m3.decision.map_geometry import fused_map_aabb
    from lavira_m3.decision.path_geometry import in_map_progress_point, path_is_bad_loop

    scene = {
        "map": {
            "resolution": 0.1,
            "origin": [0.0, 0.0],
            "fused": [[i, j] for i in range(40) for j in range(40)],
        }
    }
    aabb = fused_map_aabb(scene)
    assert aabb is not None
    loop = [[0.1, 0.1], [1.0, 0.1], [2.2, 0.1], [8.0, 0.1], [8.0, 8.0], [1.5, 1.5]]
    assert path_is_bad_loop(loop, 0.1, 0.1, 1.5, 1.5, aabb)
    step = in_map_progress_point(loop, 0.1, 0.1, aabb, 1.5, 3.0)
    assert step is not None
    dist = ((step[0] - 0.1) ** 2 + (step[1] - 0.1) ** 2) ** 0.5
    assert 1.4 <= dist <= 3.1
    early_exit = [[0.1, 0.1], [0.6, 0.1], [8.0, 0.1]]
    assert in_map_progress_point(early_exit, 0.1, 0.1, aabb, 1.5, 3.0) is None
    short = [[0.1, 0.1], [0.6, 0.2], [1.2, 0.3]]
    assert not path_is_bad_loop(short, 0.1, 0.1, 1.2, 0.3, aabb)


def test_go_trims_loop_path():
    import lavira_m3.decision.agent_loop as al
    from lavira_m3.contracts import NavResult
    from lavira_m3.decision.agent_loop import LaviraM3Agent
    from lavira_m3.decision.config import RobotConfig

    loop = [[0.0, 0.0], [1.0, 0.0], [12.0, 0.0], [12.0, 12.0], [2.0, 0.2]]
    calls = []

    class Nav:
        def preview_plan(self, x, y, yaw, pose):
            return loop

    class RT:
        def __init__(self):
            self.nav = Nav()

        def pose(self):
            return (0.0, 0.0, 0.0)

        def navigate_to(self, x, y, yaw=None):
            calls.append((round(x, 2), round(y, 2)))
            return NavResult(
                True, "succeeded", "ok", extra={"path": [[0.0, 0.0], [x, y], [x + 0.1, y]]}
            )

    orig = al.fetch_scene
    al.fetch_scene = lambda *a, **k: {
        "map": {
            "resolution": 0.1,
            "origin": [0.0, 0.0],
            "fused": [[i, j] for i in range(30) for j in range(10)],
        }
    }
    try:
        agent = LaviraM3Agent(
            RobotConfig(name="small_b", panorama="cardinal4"), RT(), [], llm=object()
        )
        agent.board = type(
            "Board", (), {"post_trace": staticmethod(lambda payload: {"ok": True})}
        )()
        agent._project = lambda *a, **k: (2.0, 0.2)
        xy, result = agent._go({}, [10, 20, 30, 40], "front")
    finally:
        al.fetch_scene = orig
    assert calls
    assert abs(calls[0][0] - 1.5) < 0.25
    assert abs(calls[0][1]) < 0.25
    assert xy is not None


def test_go_traces_path_before_navigate():
    import lavira_m3.decision.agent_loop as al
    from lavira_m3.contracts import NavResult
    from lavira_m3.decision.agent_loop import LaviraM3Agent
    from lavira_m3.decision.config import RobotConfig

    path = [[0.0, 0.0], [0.9, 0.1], [1.8, 0.0]]
    traces = []
    walked = []

    class Nav:
        def preview_plan(self, x, y, yaw, pose):
            return path

    class Board:
        def post_trace(self, payload):
            traces.append(dict(payload))
            if walked:
                raise AssertionError("va_path posted after navigate")
            return {"ok": True}

    class RT:
        def __init__(self):
            self.nav = Nav()

        def pose(self):
            return (0.0, 0.0, 0.0)

        def navigate_to(self, x, y, yaw=None):
            walked.append((x, y))
            assert traces, "expected va_path before walk"
            assert traces[-1].get("va_path") == path
            return NavResult(True, "succeeded", "ok", extra={"path": path})

    orig = al.fetch_scene
    al.fetch_scene = lambda *a, **k: {
        "map": {
            "resolution": 0.1,
            "origin": [0.0, 0.0],
            "fused": [[i, j] for i in range(30) for j in range(10)],
        }
    }
    try:
        agent = LaviraM3Agent(
            RobotConfig(name="small_b", panorama="cardinal4"), RT(), [], llm=object()
        )
        agent.board = Board()
        agent._project = lambda *a, **k: (1.8, 0.0)
        agent._trim_loop_path = lambda xy, pose: xy
        xy, result = agent._go({}, [10, 20, 30, 40], "front")
    finally:
        al.fetch_scene = orig
    assert walked
    assert any(t.get("va_path") == path for t in traces)
    assert result.ok is True
    assert xy is not None


def test_large_go_traces_path_from_navigate():
    from lavira_m3.contracts import NavResult
    from lavira_m3.decision.agent_loop import LaviraM3Agent
    from lavira_m3.decision.config import RobotConfig

    path = [[0.0, 0.0], [0.8, 0.0], [1.5, 0.0]]
    traces = []
    walked = []

    class Board:
        def post_trace(self, payload):
            traces.append(dict(payload))
            return {"ok": True}

    class RT:
        def pose(self):
            return (0.0, 0.0, 0.0)

        def navigate_to(self, x, y, yaw=None):
            walked.append((x, y))
            return NavResult(True, "succeeded", "ok", extra={"path": path})

    agent = LaviraM3Agent(RobotConfig(name="large", panorama="surround7"), RT(), [], llm=object())
    agent.board = Board()
    agent._project = lambda *a, **k: (1.5, 0.0)
    xy, result = agent._go({}, [10, 20, 30, 40], "front")
    deadline = time.time() + 1.0
    while time.time() < deadline and not any(t.get("va_path") == path for t in traces):
        time.sleep(0.02)
    assert walked == [(1.5, 0.0)]
    assert any(t.get("va_path") == path for t in traces)
    assert result.ok is True
    assert xy is not None


def test_thin_planner_path_keeps_corners_and_step():
    from lavira_m3.decision.path_geometry import thin_planner_path

    assert thin_planner_path([]) == []
    short = [[0.0, 0.0], [0.4, 0.0], [0.9, 0.0]]
    assert thin_planner_path(short) == short
    dense = [[i * 0.05, 0.0] for i in range(61)]
    got = thin_planner_path(dense)
    assert got == [[0.0, 0.0], [3.0, 0.0]]
    # 2 m east then 2 m north: keep the corner, not a hypotenuse chord.
    elbow = [[i * 0.05, 0.0] for i in range(41)] + [[2.0, j * 0.05] for j in range(1, 41)]
    cornered = thin_planner_path(elbow)
    assert cornered[0] == [0.0, 0.0]
    assert abs(cornered[-1][0] - 2.0) < 1e-9 and abs(cornered[-1][1] - 2.0) < 1e-9
    assert any(abs(p[0] - 2.0) < 0.08 and abs(p[1]) < 0.08 for p in cornered)
    assert len(cornered) == 3


def test_planner_path_rejects_empty_and_keeps_polyline():
    from lavira_m3.contracts import NavResult
    from lavira_m3.decision.navigation_results import planner_path_from_result

    assert planner_path_from_result(None) is None
    assert planner_path_from_result(NavResult(False, "failed", "x")) is None
    assert planner_path_from_result({"path": [[1.0, 2.0]]}) is None
    assert planner_path_from_result({"path": [[0.0, 0.0], [1.0, 2.0]]}) is None
    long_path = [[0.0, 0.0], [0.4, 0.1], [0.9, 0.3], [1.4, -0.2]]
    got = planner_path_from_result(
        NavResult(False, "not_executable", "envelope", extra={"path": long_path})
    )
    assert got == long_path


def test_static_go_keeps_first_planner_path():
    import os

    from lavira_m3.contracts import NavResult
    from lavira_m3.decision.agent_loop import LaviraM3Agent
    from lavira_m3.decision.config import RobotConfig
    from lavira_m3.decision.navigation_results import planner_path_from_result

    long_path = [[2.49, 0.28], [2.20, -0.10], [1.80, -0.80], [1.37, -1.65]]
    short_path = [[2.49, 0.28], [2.12, -0.35]]
    calls = []

    class RT:
        def pose(self):
            return (2.49, 0.28, -1.64)

        def navigate_to(self, x, y, yaw=None):
            calls.append((round(x, 2), round(y, 2)))
            if len(calls) == 1:
                return NavResult(False, "not_executable", "envelope", extra={"path": long_path})
            return NavResult(False, "not_executable", "envelope", extra={"path": short_path})

    os.environ["T0_STATIC"] = "1"
    try:
        agent = LaviraM3Agent(
            RobotConfig(name="large", panorama="surround7"), RT(), [], llm=object()
        )
        agent._project = lambda *a, **k: (1.37, -1.65)
        xy, result = agent._go({}, [10, 20, 30, 40], "front")
        assert len(calls) == 1
        assert planner_path_from_result(result) == long_path
        assert xy is not None
    finally:
        os.environ.pop("T0_STATIC", None)


def test_wait_board_goals():
    from lavira_m3.decision.agent_loop import wait_board_goals

    class FakeBoard:
        def __init__(self):
            self.n = 0
            self.hub = "http://127.0.0.1:8769"

        def t0_hello(self):
            return {"ok": True}

        def snapshot(self):
            self.n += 1
            if self.n < 2:
                return {
                    "ok": True,
                    "started": False,
                    "goals": [{"goal_index": 0, "object_category": "chair"}],
                }
            return {
                "ok": True,
                "started": True,
                "goals": [{"goal_index": 0, "object_category": "chair"}],
            }

    goals = wait_board_goals(FakeBoard(), poll_s=0.01)
    assert goals[0]["object_category"] == "chair"


def test_empty_board_not_complete():
    assert board_all_found({"ok": False, "goals": [], "found_by": []}) is False
    assert board_all_found({"ok": True, "goals": [], "found_by": []}) is False
    assert board_all_found({"ok": True, "goals": [{}], "found_by": [-1]}) is False
    assert board_all_found({"ok": True, "goals": [{}, {}], "found_by": [0, 1]}) is True


def test_require_all_views():
    from lavira_m3.decision.config import CARDINAL_VIEWS, SURROUND_VIEWS
    from lavira_m3.decision.observations import (
        MissingViewsError,
        require_all_views,
        wait_all_views,
    )

    rgb = object()
    aloha = [{"name": n, "rgb": rgb} for n in SURROUND_VIEWS]
    require_all_views("large", aloha, SURROUND_VIEWS)
    try:
        require_all_views("large", aloha[:-1], SURROUND_VIEWS)
        raise AssertionError("expected MissingViewsError")
    except MissingViewsError as exc:
        assert "right_front" in str(exc)
    dogs = [{"name": n, "rgb": rgb} for n in CARDINAL_VIEWS]
    dogs[2]["rgb"] = None
    try:
        require_all_views("small_a", dogs, CARDINAL_VIEWS)
        raise AssertionError("expected MissingViewsError")
    except MissingViewsError as exc:
        assert "behind" in str(exc)

    hits = {"n": 0}

    def late_right():
        hits["n"] += 1
        frames = [{"name": n, "rgb": rgb} for n in CARDINAL_VIEWS]
        if hits["n"] < 3:
            frames[-1]["rgb"] = None
        return frames

    got = wait_all_views("small_a", late_right, CARDINAL_VIEWS, timeout_s=1.0, poll_s=0.01)
    assert hits["n"] == 3
    assert got[-1]["rgb"] is rgb
    try:
        wait_all_views("small_a", lambda: dogs, CARDINAL_VIEWS, timeout_s=0.05, poll_s=0.01)
        raise AssertionError("expected MissingViewsError after wait")
    except MissingViewsError as exc:
        assert "behind" in str(exc)


def test_motion_history_2s(tmp_path=None):
    import os
    import tempfile
    import time

    import numpy as np

    from lavira_m3.contracts import NavResult
    from lavira_m3.decision.agent_loop import LaviraM3Agent
    from lavira_m3.decision.config import RobotConfig
    from lavira_m3.decision.navigation_history import history_rgb_period_s
    from lavira_m3.decision.prompts_real import HISTORY_SECTION_LABEL, history_text

    folder = tmp_path or tempfile.mkdtemp(prefix="m3hist_")
    os.environ["M3_HISTORY_DIR"] = folder
    os.environ.pop("M3_HISTORY_RGB_PERIOD_S", None)
    assert abs(history_rgb_period_s() - 5.0) < 1e-9
    os.environ["M3_HISTORY_RGB_PERIOD_S"] = "0.12"

    class Board:
        def __init__(self):
            self.posted = []

        def post_history(self, rec):
            self.posted.append(dict(rec))
            return {"ok": True}

        def prompt_blocks(self):
            return {"available": True, "board": "", "progress": ""}

    class RT:
        def __init__(self):
            self.n = 0
            self.navs = 0
            self.faces = 0
            self.fixed_pose = None

        def pose(self):
            if self.fixed_pose is not None:
                return self.fixed_pose
            return (float(self.n), 0.1, 0.0)

        def capture_direction(self, name):
            rgb = np.full((18, 24, 3), 20 + self.n, dtype=np.uint8)
            self.n += 1
            return {"name": name, "rgb": rgb}

        def capture_panorama(self):
            rgb = np.full((18, 24, 3), 200, dtype=np.uint8)
            return [
                {"name": "front", "rgb": rgb},
                {"name": "left", "rgb": rgb},
                {"name": "behind", "rgb": rgb},
                {"name": "right", "rgb": rgb},
            ]

        def face_direction(self, name):
            self.faces += 1
            if name == "front":
                return NavResult(True, "skipped", "already facing")
            time.sleep(0.28)
            return NavResult(True, "succeeded", "turned")

        def navigate_to(self, x, y, yaw=None, **kwargs):
            self.navs += 1
            time.sleep(0.28)
            return NavResult(True, "succeeded", "ok")

    captured = []

    class LLM:
        def generate_la(self, messages):
            captured.append(messages)
            return '{"action":"navigate to front","progress_analysis":"x","reasoning":"y","stop":false,"claim_goal_id":""}'

    dog = LaviraM3Agent(
        RobotConfig(name="small_b", panorama="cardinal4"), RT(), [{"goal_index": 0}], llm=LLM()
    )
    dog.board = Board()
    dog.history_session = "dogstill"
    dog.step = 1
    dog._face_with_history("left")
    assert dog.runtime.faces == 1
    # GO2 never yaws; still body must not write history RGB.
    assert dog.history == []
    dog.runtime.fixed_pose = (0.0, 0.0, 0.0)
    still = dog._navigate_with_history(0.10, 0.0, 0.0)
    assert still.ok
    assert dog.runtime.navs == 1
    assert dog.history == []

    agent = LaviraM3Agent(
        RobotConfig(name="small_a", panorama="cardinal4"), RT(), [{"goal_index": 0}], llm=LLM()
    )
    agent.board = Board()
    agent.history_session = "testsession"
    agent.step = 1
    assert agent.history == []
    before = agent.runtime.n
    result = agent._navigate_with_history(1.0, 0.0, 0.0)
    assert result.ok
    assert agent.runtime.navs == 1
    n = len(agent.history)
    assert n >= 3
    assert all(rec.get("rgb") is not None for rec in agent.history)
    assert agent.history[0]["kind"] == "nav_start"
    assert all(rec.get("kind") == "nav_hz" for rec in agent.history[1:])
    assert agent.runtime.n - before == n
    stop_rgb = agent.runtime.capture_panorama()[0]["rgb"]
    assert not any(rec["rgb"] is stop_rgb for rec in agent.history)
    text = history_text(agent.history)
    assert "Frame 0" in text and "5 s" in text
    assert "no motion RGB yet" in history_text([])
    agent._record(action="navigate to front", target="open floor", xy=(1.0, 0.0), nav_ok=True)
    assert len(agent.history) == n
    assert agent.history[-1]["action"] == "navigate to front"
    assert agent.history[-1].get("rgb") is not None
    agent._la(agent.runtime.capture_panorama())
    content = captured[0][0]["content"]
    labels = [part.get("text", "") for part in content if part.get("type") == "text"]
    images = [part for part in content if part.get("type") == "image_url"]
    assert any("every 5 seconds" in t or "5 s" in t for t in labels)
    assert HISTORY_SECTION_LABEL in labels or any("NAVIGATION HISTORY" in t for t in labels)
    assert any("INITIAL PANORAMA" in t for t in labels)
    assert (
        sum(
            1
            for t in labels
            if t.startswith("Initial ")
            and not t.startswith("Initial Panorama")
            and "NO IMAGE" not in t
        )
        == 4
    )
    assert sum(1 for t in labels if t.startswith("Frame ") and "front" in t) == n
    assert len(images) >= n + 8
    assert agent._origin_views is not None
    assert [rec["name"] for rec in agent._origin_views] == ["front", "left", "behind", "right"]
    assert all(rec.get("rgb") is not None for rec in agent._origin_views)
    disk = os.path.join(folder, "small_a", "testsession")
    assert os.path.isfile(os.path.join(disk, "waypoint_000_front.jpg"))
    assert os.path.isfile(os.path.join(disk, "waypoint_%03d_front.jpg" % (n - 1)))
    assert os.path.isfile(os.path.join(disk, "manifest.json"))
    assert any(p.get("waypoint_id") == 0 and p.get("front_image") for p in agent.board.posted)
    assert not hasattr(agent, "_open_front_waypoint")

    aloha = LaviraM3Agent(
        RobotConfig(name="large", panorama="surround7"), RT(), [{"goal_index": 0}], llm=LLM()
    )
    aloha.board = Board()
    aloha.history_session = "turnsession"
    aloha.step = 1
    skipped = aloha._face_with_history("front")
    assert skipped.state == "skipped"
    assert aloha.history == []
    before_turn = aloha.runtime.n
    turned = aloha._face_with_history("left")
    assert turned.ok
    assert aloha.runtime.faces == 2
    n_turn = len(aloha.history)
    assert n_turn >= 3
    assert aloha.history[0]["kind"] == "turn_start"
    assert all(rec.get("kind") == "turn_hz" for rec in aloha.history[1:])
    assert aloha.runtime.n - before_turn == n_turn


def test_la_initial_panorama_surround7_and_cardinal4():
    import numpy as np

    from lavira_m3.decision.agent_loop import LaviraM3Agent
    from lavira_m3.decision.config import CARDINAL_VIEWS, SURROUND_VIEWS, RobotConfig

    class Board:
        def prompt_blocks(self):
            return {"available": True, "board": "", "progress": ""}

    class RT:
        def __init__(self, names):
            self.names = list(names)
            self.pose_xy = (1.25, -0.5, 0.0)

        def pose(self):
            return self.pose_xy

        def capture_panorama(self):
            rgb = np.full((12, 16, 3), 90, dtype=np.uint8)
            return [{"name": n, "rgb": rgb} for n in self.names]

    captured = []

    class LLM:
        def generate_la(self, messages):
            captured.append(messages)
            return '{"action":"navigate to front","progress_analysis":"x","reasoning":"y","claim_goal_id":""}'

    dog = LaviraM3Agent(
        RobotConfig(name="small_a", panorama="cardinal4"),
        RT(CARDINAL_VIEWS),
        [{"goal_index": 0}],
        llm=LLM(),
    )
    dog.board = Board()
    dog._la(dog.runtime.capture_panorama())
    dog_labels = [p.get("text", "") for p in captured[0][0]["content"] if p.get("type") == "text"]
    dog_images = [p for p in captured[0][0]["content"] if p.get("type") == "image_url"]
    assert [rec["name"] for rec in dog._origin_views] == list(CARDINAL_VIEWS)
    assert (
        sum(
            1
            for t in dog_labels
            if t.startswith("Initial ")
            and not t.startswith("Initial Panorama")
            and "NO IMAGE" not in t
        )
        == 4
    )
    assert any("INITIAL PANORAMA" in t for t in dog_labels)
    assert any("4 views" in t and "front, left, behind, right" in t for t in dog_labels)
    assert len(dog_images) == 8

    captured.clear()
    aloha = LaviraM3Agent(
        RobotConfig(name="large", panorama="surround7"),
        RT(SURROUND_VIEWS),
        [{"goal_index": 0}],
        llm=LLM(),
    )
    aloha.board = Board()
    first = aloha.runtime.capture_panorama()
    first_front = first[0]["rgb"]
    aloha._la(first)
    aloha._la(aloha.runtime.capture_panorama())
    assert [rec["name"] for rec in aloha._origin_views] == list(SURROUND_VIEWS)
    assert aloha._origin_views[0]["rgb"] is first_front
    labels = [p.get("text", "") for p in captured[1][0]["content"] if p.get("type") == "text"]
    images = [p for p in captured[1][0]["content"] if p.get("type") == "image_url"]
    assert (
        sum(
            1
            for t in labels
            if t.startswith("Initial ")
            and not t.startswith("Initial Panorama")
            and "NO IMAGE" not in t
        )
        == 7
    )
    assert any("7 views" in t and "left_front" in t and "right_back" in t for t in labels)
    assert "Prefer not re-exploring" in captured[1][0]["content"][0]["text"]
    assert len(images) == 14


def test_nav_start_history_1hz(tmp_path=None):
    test_motion_history_2s(tmp_path)


def test_llm_call_log_jsonl_and_skip_dcs():
    import json
    import os
    import tempfile
    from unittest.mock import MagicMock, patch

    from lavira_m3.decision.config import LLMConfig
    from lavira_m3.decision.llm_call_log import LlmCallLogger, tokens_per_second
    from lavira_m3.decision.llm_router import LLMRouter

    assert tokens_per_second(180, 2810.4) == 64.048
    assert tokens_per_second(None, 1000) is None
    folder = tempfile.mkdtemp(prefix="m3llm_")
    logger = LlmCallLogger("small_b", folder, "sess")
    assert logger.advertise() is False
    assert logger.topic == "/lavira_m3/small_b/llm_call"
    rec = logger.write(
        kind="va",
        raw='{"bbox_2d":[1,2,3,4]}',
        returned_at=1727000000.123,
        duration_ms=2000.0,
        usage={"prompt_tokens": 100, "completion_tokens": 80, "total_tokens": 180},
        model="deepseek-flash",
        backend="deepseek",
        step=3,
    )
    assert rec["tokens_per_s"] == 40.0
    assert rec["topic"] == "/lavira_m3/small_b/llm_call"
    lines = (
        open(os.path.join(folder, "llm_calls.jsonl"), encoding="utf-8").read().strip().splitlines()
    )
    assert len(lines) == 1
    saved = json.loads(lines[0])
    assert saved["kind"] == "va" and saved["raw"].startswith("{")
    assert saved["duration_ms"] == 2000.0
    assert saved["completion_tokens"] == 80

    cfg = LLMConfig(
        la_backend="deepseek",
        la_base_url="https://api.deepseek.com",
        la_model="deepseek-flash",
        la_api_key="sk-test",
        va_backend="deepseek",
        va_base_url="https://api.deepseek.com",
        va_model="deepseek-flash",
        va_api_key="sk-test",
    )
    with patch("lavira_m3.decision.llm_router.OpenAI") as openai_cls:
        client = MagicMock()
        choice = MagicMock()
        choice.message.content = '{"action":"navigate to front"}'
        choice.finish_reason = "stop"
        resp = client.chat.completions.create.return_value
        resp.choices = [choice]
        resp.usage = MagicMock(prompt_tokens=10, completion_tokens=20, total_tokens=30)
        openai_cls.return_value = client
        router = LLMRouter(cfg)
        logged = LlmCallLogger("large", folder, "sess2")
        router.bind_call_logger(logged)
        router.generate_la([{"role": "user", "content": "dcs"}])
        after_dcs = (
            open(os.path.join(folder, "llm_calls.jsonl"), encoding="utf-8")
            .read()
            .strip()
            .splitlines()
        )
        assert len(after_dcs) == 1
        router.note_record("la", 2)
        router.generate_la([{"role": "user", "content": "la"}])
        after_la = [
            json.loads(x)
            for x in open(os.path.join(folder, "llm_calls.jsonl"), encoding="utf-8")
            if x.strip()
        ]
        assert len(after_la) == 2
        assert after_la[-1]["kind"] == "la"
        assert after_la[-1]["step"] == 2
        assert after_la[-1]["completion_tokens"] == 20


def test_bbox_iou():
    assert bbox_iou([0, 0, 100, 100], [0, 0, 100, 100]) == 1.0
    assert bbox_iou([0, 0, 100, 100], [200, 200, 300, 300]) == 0.0
    overlap = bbox_iou([0, 0, 100, 100], [50, 50, 150, 150])
    assert 0.14 < overlap < 0.15
    assert bbox_iou(None, [0, 0, 10, 10]) == 0.0
    assert bbox_iou([0, 0, 10], [0, 0, 10, 10]) == 0.0


def test_short_hop_prompts():
    from lavira_m3.decision.config import SHORT_HOP_M, short_hop_m
    from lavira_m3.decision.prompts_real import (
        short_hop_la_feedback,
        short_hop_va_feedback,
    )

    assert SHORT_HOP_M == 1.5
    assert short_hop_m() == 1.5
    la = short_hop_la_feedback(
        {"action": "navigate to front", "claim_goal_id": "goal_0", "direction": "front"},
        {"target": "office chair", "bbox_2d": [100, 120, 220, 340], "direction": "front"},
    )
    assert "shorter than 1.5 m" in la
    assert "Previous LA: action=navigate to front" in la
    assert "claim_goal_id=goal_0" in la
    assert "office chair" in la
    assert "[100, 120, 220, 340]" in la
    assert "prefer that same claim_goal_id" in la
    assert "rather than walking away or remapping it" in la
    assert "different legal direction" in la
    same = short_hop_va_feedback(
        {"target": "office chair", "bbox_2d": [100, 120, 220, 340], "direction": "front"},
        "front",
    )
    assert "SAME direction" in same
    assert "IoU" in same
    assert "under 0.3" in same
    other = short_hop_va_feedback(
        {"target": "office chair", "bbox_2d": [100, 120, 220, 340], "direction": "front"},
        "left",
    )
    assert "new direction" in other


def test_short_hop_helpers():
    import numpy as np

    from lavira_m3.contracts import NavResult
    from lavira_m3.decision.agent_loop import LaviraM3Agent
    from lavira_m3.decision.config import RobotConfig

    class RT:
        def __init__(self):
            self.pano = 0
            self.xy = (0.0, 0.0, 0.0)

        def pose(self):
            return self.xy

        def capture_panorama(self):
            self.pano += 1
            rgb = np.zeros((16, 20, 3), dtype=np.uint8)
            return [
                {"name": "front", "rgb": rgb},
                {"name": "left", "rgb": rgb},
                {"name": "behind", "rgb": rgb},
                {"name": "right", "rgb": rgb},
            ]

    agent = LaviraM3Agent(RobotConfig(name="small_b", panorama="cardinal4"), RT(), [], llm=object())
    agent.t0_joint = False
    rgb = np.zeros((16, 20, 3), dtype=np.uint8)
    full = [
        {"name": "front", "rgb": rgb},
        {"name": "left", "rgb": rgb},
        {"name": "behind", "rgb": rgb},
        {"name": "right", "rgb": rgb},
    ]
    agent._last_pano = full
    agent._short_hop = True
    reused = agent._cycle_frames()
    assert agent.runtime.pano == 0
    assert [fr["name"] for fr in reused] == ["front", "left", "behind", "right"]

    agent._last_pano = [{"name": "front", "rgb": rgb}]
    agent._short_hop = True
    recaptured = agent._cycle_frames()
    assert agent.runtime.pano == 1
    assert agent._short_hop is False
    assert len(recaptured) == 4

    agent.t0_joint = True
    agent.t0_consumed = False
    agent._short_hop = True
    agent._last_pano = full
    agent._cycle_frames()
    assert agent.runtime.pano == 2

    agent.t0_joint = False
    agent._short_hop = True
    agent._last_va = {"direction": "front", "bbox_2d": [100, 100, 200, 200], "target": "chair"}
    assert agent._va_repeats_last({"bbox_2d": [110, 110, 190, 190], "target": "desk"}, "front")
    assert agent._va_repeats_last({"bbox_2d": [10, 10, 20, 20], "target": "chair"}, "front")
    assert not agent._va_repeats_last({"bbox_2d": [700, 100, 900, 400], "target": "door"}, "front")
    assert not agent._va_repeats_last({"bbox_2d": [110, 110, 190, 190], "target": "chair"}, "left")

    agent.runtime.xy = (0.2, 0.0, 0.0)
    agent._mark_travel((0.0, 0.0, 0.0), NavResult(True, "succeeded", "ok"))
    assert agent._short_hop is True
    agent.runtime.xy = (3.0, 0.0, 0.0)
    agent._mark_travel((0.0, 0.0, 0.0), NavResult(True, "succeeded", "ok"))
    assert agent._short_hop is False
    agent._mark_travel((0.0, 0.0, 0.0), NavResult(True, "skipped", "already at goal"))
    assert agent._short_hop is True
    agent._clear_short_hop()
    assert agent._short_hop is False
    assert agent._short_travel_streak == 0


def test_backtrack_listing_and_walk():
    import numpy as np

    from lavira_m3.contracts import NavResult
    from lavira_m3.decision.agent_loop import LaviraM3Agent
    from lavira_m3.decision.config import STEP_MAX_M, RobotConfig, step_max_m
    from lavira_m3.decision.prompts_real import (
        AFTER_BACKTRACK_FEEDBACK,
        BACKTRACK_STUCK_FEEDBACK,
        backtrack_prompt_block,
        la_backtrack_replan_prompt,
        la_prompt,
    )
    from lavira_m3.decision.vision_geometry import CAM_YAW

    assert step_max_m() == STEP_MAX_M == 10.0
    empty = la_prompt("agent_small_b", ["front"], "board", "none")
    assert "backtrack" not in empty
    listed = la_prompt(
        "agent_small_b",
        ["front", "left"],
        "board",
        "none",
        extra_actions=["backtrack to 0"],
        backtrack_note='Reached points within 10 m\nAvailable IDs: 0\n- "backtrack to 0"',
    )
    assert '"backtrack to 0"' in listed
    assert "Available IDs: 0" in listed
    assert "do not treat them as required" in listed
    assert "backtrack to <id>" in listed
    note = backtrack_prompt_block([(0, (1.0, 2.0), 3.5)], 10.0)
    assert "Available IDs: 0" in note
    assert "You do not have to" in note
    assert "archived arrival RGB" in note
    assert "must" not in note.lower()
    replan = la_backtrack_replan_prompt("agent_small_b", ["front", "left"], previous_action="front")
    assert "second chance" in replan.lower()
    assert "Do not choose backtrack" in replan
    assert "claim_goal_id" in replan
    assert "live front" in replan
    assert "behind" not in replan or "not legal" in replan
    assert '"backtrack to' not in replan

    class LLM:
        def generate_la(self, messages, max_tokens=2048):
            return '{"action":"navigate to left","reasoning":"retry left","stop":false}'

        def generate_va(self, messages):
            return '{"bbox_2d":[4,4,12,12],"target":"chair","reasoning":"live"}'

    class RT:
        def __init__(self):
            self.xy = (8.0, 0.0, 0.0)
            self.goals = []
            self.fronts = []

        def pose(self):
            return self.xy

        def navigate_to(self, x, y, yaw=None, **kwargs):
            self.goals.append((x, y, yaw))
            self.xy = (float(x), float(y), float(yaw or 0.0))
            return NavResult(True, "succeeded", "ok")

        def capture_panorama(self):
            rgb = np.full((16, 20, 3), 40, dtype=np.uint8)
            return [{"name": n, "rgb": rgb} for n in ("front", "left", "behind", "right")]

        def capture_direction(self, name):
            rgb = np.full((16, 20, 3), 90, dtype=np.uint8)
            self.fronts.append(name)
            return {"name": name, "rgb": rgb, "depth": np.full((16, 20), 1.2, dtype=np.float32)}

        def face_direction(self, name):
            return NavResult(True, "skipped", "already facing")

    rgb = np.full((12, 16, 3), 50, dtype=np.uint8)
    pano = [{"name": "front", "rgb": rgb}, {"name": "left", "rgb": rgb}]
    agent = LaviraM3Agent(RobotConfig(name="small_b", panorama="cardinal4"), RT(), [], llm=LLM())
    agent._navigate_with_history = lambda x, y, yaw=None, **k: agent.runtime.navigate_to(
        x, y, yaw, **k
    )
    agent._save_history_manifest = lambda: None
    agent._clear_va_trace = lambda *a, **k: None
    agent.history = [
        {
            "waypoint_id": 0,
            "reached_xy": (0.0, 0.0),
            "yaw": 0.1,
            "is_waypoint": True,
            "pano": pano,
            "leave_direction": "front",
        },
        {"waypoint_id": 1, "kind": "nav_hz"},
        {"waypoint_id": 2, "reached_xy": (4.0, 0.0), "yaw": 0.2, "is_waypoint": True, "pano": pano},
        {
            "waypoint_id": 3,
            "reached_xy": (20.0, 0.0),
            "yaw": 0.3,
            "is_waypoint": True,
            "pano": pano,
        },
        {"waypoint_id": 4, "reached_xy": (1.0, 0.0), "is_waypoint": True},
        {"waypoint_id": 5, "reached_xy": (8.0, 0.0), "yaw": 0.0, "is_waypoint": True, "pano": pano},
    ]
    items = agent._available_backtrack_items()
    assert [i for i, _xy, _d in items] == [0, 1]
    assert items[0][1] == (0.0, 0.0)
    assert items[1][1] == (4.0, 0.0)
    assert all(i != 4 for i, _xy, _d in items)

    agent._short_travel_streak = 2
    legal = ["front"]
    extras = []
    if items and (agent._short_travel_streak >= 2 or not legal):
        extras.append(BACKTRACK_STUCK_FEEDBACK)
    assert "listed backtrack is available" in extras[0]
    assert "do not have to" in extras[0]

    before = list(agent.history)
    assert agent._backtrack("backtrack to 9") is True
    assert agent.runtime.goals == []
    assert agent.nav_feedback == ""
    assert agent.history == before

    assert agent._backtrack("backtrack to 0") is True
    expect_yaw = 0.1 + float(CAM_YAW["left"])
    assert agent.runtime.goals == [(0.0, 0.0, expect_yaw)]
    reached = [h.get("reached_xy") for h in agent.history if h.get("reached_xy")]
    assert (0.0, 0.0) in reached
    assert (4.0, 0.0) in reached
    assert (20.0, 0.0) in reached
    assert agent.nav_feedback == AFTER_BACKTRACK_FEEDBACK
    assert "same path" in agent.nav_feedback
    assert agent.history[0].get("leave_direction") == "left"
    newest = [h for h in agent.history if h.get("is_waypoint") and h.get("kind") == "waypoint"]
    assert newest
    assert newest[-1]["reached_xy"] == (0.0, 0.0)
    assert newest[-1]["requested_xy"] == (0.0, 0.0)


def test_archive_uses_land_pose_not_commanded_xy():
    import numpy as np

    from lavira_m3.contracts import NavResult
    from lavira_m3.decision.agent_loop import LaviraM3Agent
    from lavira_m3.decision.config import RobotConfig

    class RT:
        def __init__(self):
            self.xy = (-0.82, -0.63, 0.4)

        def pose(self):
            return self.xy

        def capture_panorama(self):
            rgb = np.full((12, 16, 3), 20, dtype=np.uint8)
            return [{"name": "front", "rgb": rgb}]

        def navigate_to(self, x, y, yaw=None, **kwargs):
            return NavResult(True, "succeeded", "ok")

        def face_direction(self, name):
            return NavResult(True, "skipped", "already facing")

    class LLM:
        def generate_la(self, messages, max_tokens=2048):
            return '{"action":"navigate to front","stop":false}'

        def generate_va(self, messages):
            return '{"bbox_2d":[1,1,2,2],"target":"x"}'

    agent = LaviraM3Agent(RobotConfig(name="small_a", panorama="cardinal4"), RT(), [], llm=LLM())
    agent._save_history_manifest = lambda: None
    agent.history = [{"kind": "nav_hz", "xy": (-0.53, -1.24)}]
    agent._nav_leg_start = 0
    agent._archive_arrival_waypoint((-0.53, -1.24), arriving_direction="front")
    agent._record(action="navigate to front", target="chair", xy=(-0.53, -1.24), nav_ok=True)
    wps = [h for h in agent.history if h.get("is_waypoint")]
    assert len(wps) == 1
    assert wps[0]["reached_xy"] == (-0.82, -0.63)
    assert wps[0]["requested_xy"] == (-0.53, -1.24)
    assert wps[0]["xy"] == (-0.82, -0.63)
    assert agent.history[0].get("reached_xy") == (-0.53, -1.24)
    items = agent._available_backtrack_items()
    assert items == []


def test_scb_fail_hides_id_then_relists():
    import numpy as np

    from lavira_m3.contracts import NavResult
    from lavira_m3.decision.agent_loop import LaviraM3Agent
    from lavira_m3.decision.config import RobotConfig
    from lavira_m3.decision.prompts_real import (
        SCB_FAIL_HIDE_CYCLES,
        scb_walk_failed_feedback,
    )

    class RT:
        def __init__(self):
            self.xy = (8.0, 0.0, 0.0)
            self.goals = []

        def pose(self):
            return self.xy

        def navigate_to(self, x, y, yaw=None, **kwargs):
            self.goals.append((x, y, yaw))
            return NavResult(False, "planning_timeout", "no trajectory")

        def capture_panorama(self):
            rgb = np.full((12, 16, 3), 30, dtype=np.uint8)
            return [{"name": "front", "rgb": rgb}]

        def face_direction(self, name):
            return NavResult(True, "skipped", "already facing")

    class LLM:
        def generate_la(self, messages, max_tokens=2048):
            return '{"action":"navigate to front","stop":false}'

        def generate_va(self, messages):
            return '{"bbox_2d":[1,1,2,2],"target":"x"}'

    rgb = np.full((12, 16, 3), 50, dtype=np.uint8)
    pano = [{"name": "front", "rgb": rgb}]
    agent = LaviraM3Agent(RobotConfig(name="small_a", panorama="cardinal4"), RT(), [], llm=LLM())
    agent._navigate_with_history = lambda x, y, yaw=None, **k: agent.runtime.navigate_to(
        x, y, yaw, **k
    )
    agent._save_history_manifest = lambda: None
    agent._clear_va_trace = lambda *a, **k: None
    agent.history = [
        {
            "waypoint_id": 0,
            "reached_xy": (0.0, 0.0),
            "yaw": 0.1,
            "pose": (0.0, 0.0, 0.1),
            "is_waypoint": True,
            "pano": pano,
        },
        {"waypoint_id": 1, "reached_xy": (4.0, 0.0), "yaw": 0.2, "is_waypoint": True, "pano": pano},
    ]
    assert [i for i, _xy, _d in agent._available_backtrack_items()] == [0, 1]
    assert agent._backtrack("backtrack to 0") is True
    assert agent.runtime.goals == [(0.0, 0.0, 0.1)]
    assert "do not immediately repeat" in agent.nav_feedback.lower()
    assert "planning_timeout" in agent.nav_feedback
    assert agent.nav_feedback == scb_walk_failed_feedback(0, "planning_timeout")
    assert agent._scb_fail_hide[0] == SCB_FAIL_HIDE_CYCLES
    assert [i for i, _xy, _d in agent._available_backtrack_items()] == [1]
    for _ in range(SCB_FAIL_HIDE_CYCLES):
        agent._tick_scb_fail_hides()
    assert [i for i, _xy, _d in agent._available_backtrack_items()] == [0, 1]


def test_scb_uses_live_front_and_stop_dcs():
    import numpy as np

    from lavira_m3.contracts import NavResult
    from lavira_m3.decision.agent_loop import LaviraM3Agent
    from lavira_m3.decision.config import RobotConfig

    class Board:
        def __init__(self):
            self.traces = []
            self.found = [""]
            self.goals = [{"goal_index": 0, "object_category": "chair", "modality": "ON"}]

        def snapshot(self):
            return {"ok": True, "goals": list(self.goals), "found_by": list(self.found)}

        def post_trace(self, payload):
            self.traces.append(payload)
            return {"ok": True}

        def post_found(self, *a, **k):
            self.found = ["small_b"]
            return {"ok": True, "goals": list(self.goals), "found_by": ["small_b"]}

        def post_progress(self, *a, **k):
            return {"ok": True}

    class LLM:
        def generate_la(self, messages, max_tokens=2048):
            text = str(messages)
            if (
                "second-chance" in text
                or "second chance" in text.lower()
                or "Previous Action from this waypoint" in text
            ):
                return '{"action":"navigate to front","reasoning":"here","stop":true}'
            return '{"decision":"FOUND","reasoning":"live chair"}'

        def generate_va(self, messages):
            raise AssertionError("VA must not run when live-front DCS finishes the board")

    class RT:
        def __init__(self):
            self.xy = (3.0, 0.0, 0.0)
            self.caps = []

        def pose(self):
            return self.xy

        def navigate_to(self, x, y, yaw=None, **kwargs):
            self.xy = (float(x), float(y), float(yaw or 0.0))
            return NavResult(True, "succeeded", "ok")

        def capture_panorama(self):
            rgb = np.full((16, 20, 3), 10, dtype=np.uint8)
            return [{"name": n, "rgb": rgb} for n in ("front", "left", "behind", "right")]

        def capture_direction(self, name):
            self.caps.append(name)
            rgb = np.full((16, 20, 3), 200, dtype=np.uint8)
            return {"name": name, "rgb": rgb, "depth": np.full((16, 20), 0.8, dtype=np.float32)}

        def face_direction(self, name):
            return NavResult(True, "skipped", "already facing")

    rgb = np.full((12, 16, 3), 40, dtype=np.uint8)
    agent = LaviraM3Agent(RobotConfig(name="small_b", panorama="cardinal4"), RT(), [], llm=LLM())
    agent.board = Board()
    agent._navigate_with_history = lambda x, y, yaw=None, **k: agent.runtime.navigate_to(
        x, y, yaw, **k
    )
    agent._save_history_manifest = lambda: None
    agent.history = [
        {
            "waypoint_id": 0,
            "reached_xy": (0.0, 0.0),
            "yaw": 0.0,
            "pose": (0.0, 0.0, 0.0),
            "is_waypoint": True,
            "pano": [{"name": "front", "rgb": rgb}],
            "leave_direction": "right",
        }
    ]
    prepared = agent._run_scb_then_front_va(
        "backtrack to 0",
        {"action": "backtrack to 0", "claim_goal_id": "goal_0"},
        "pa",
        False,
    )
    assert prepared is None
    assert agent.runtime.caps == ["front"]
    assert any(t.get("clear_va") for t in agent.board.traces)


def test_short_hop_keeps_backtrack():
    import numpy as np

    from lavira_m3.contracts import NavResult
    from lavira_m3.decision.agent_loop import LaviraM3Agent
    from lavira_m3.decision.config import RobotConfig

    class Board:
        def prompt_blocks(self):
            return {"available": True, "board": "board", "progress": ""}

        def post_progress(self, *a, **k):
            return {"ok": True}

        def post_trace(self, *a, **k):
            return {"ok": True}

        def post_history(self, *a, **k):
            return {"ok": True}

        def snapshot(self):
            return {"ok": True, "goals": [], "found_by": []}

        def fetch_image(self, spec):
            return None

    class RT:
        def pose(self):
            return (0.0, 0.0, 0.0)

        def capture_panorama(self):
            rgb = np.full((18, 24, 3), 40, dtype=np.uint8)
            return [
                {"name": "front", "rgb": rgb},
                {"name": "left", "rgb": rgb},
                {"name": "behind", "rgb": rgb},
                {"name": "right", "rgb": rgb},
            ]

        def capture_direction(self, name):
            rgb = np.full((18, 24, 3), 80, dtype=np.uint8)
            return {"name": name, "rgb": rgb}

        def face_direction(self, name):
            return NavResult(True, "skipped", "already facing")

        def navigate_to(self, x, y, yaw=None, **kwargs):
            return NavResult(True, "skipped", "already at goal")

    class LLM:
        def generate_la(self, messages, max_tokens=2048):
            return '{"action":"backtrack to 0","progress_analysis":"hist","reasoning":"r","claim_goal_id":""}'

        def generate_va(self, messages):
            return '{"bbox_2d":[100,100,200,200],"target":"chair","reasoning":"same"}'

    rgb = np.zeros((16, 20, 3), dtype=np.uint8)
    frames = [
        {"name": "front", "rgb": rgb},
        {"name": "left", "rgb": rgb},
        {"name": "behind", "rgb": rgb},
        {"name": "right", "rgb": rgb},
    ]
    agent = LaviraM3Agent(RobotConfig(name="small_b", panorama="cardinal4"), RT(), [], llm=LLM())
    agent.board = Board()
    agent._short_hop = True
    agent._last_va = {"direction": "front", "bbox_2d": [100, 100, 200, 200], "target": "chair"}
    la = {"action": "navigate to front", "progress_analysis": "p"}
    la2, direction, _view, _va = agent._run_va_with_short_hop_guard(
        {"name": "front", "rgb": rgb}, "front", "p", frames, la
    )
    assert direction == "backtrack"
    assert "backtrack to 0" in str(la2.get("action"))


def test_short_hop_cycle_reuses_and_forbids():
    import numpy as np

    from lavira_m3.contracts import NavResult
    from lavira_m3.decision.agent_loop import LaviraM3Agent
    from lavira_m3.decision.config import RobotConfig

    class Board:
        def prompt_blocks(self):
            return {"available": True, "board": "board", "progress": ""}

        def post_progress(self, *a, **k):
            return {"ok": True}

        def post_trace(self, *a, **k):
            return {"ok": True}

        def post_history(self, *a, **k):
            return {"ok": True}

        def snapshot(self):
            return {
                "ok": True,
                "goals": [
                    {"goal_index": 0, "object_category": "chair", "modality": "ON"},
                    {"goal_index": 1, "object_category": "lamp", "modality": "ON"},
                ],
                "found_by": ["", ""],
            }

        def fetch_image(self, spec):
            return None

    class RT:
        def __init__(self):
            self.pano = 0
            self.faces = []
            self.caps = []

        def pose(self):
            return (0.0, 0.0, 0.0)

        def capture_panorama(self):
            self.pano += 1
            rgb = np.full((18, 24, 3), 40 + self.pano, dtype=np.uint8)
            return [
                {"name": "front", "rgb": rgb},
                {"name": "left", "rgb": rgb},
                {"name": "behind", "rgb": rgb},
                {"name": "right", "rgb": rgb},
            ]

        def capture_direction(self, name):
            self.caps.append(name)
            rgb = np.full((18, 24, 3), 80, dtype=np.uint8)
            return {"name": name, "rgb": rgb}

        def face_direction(self, name):
            self.faces.append(name)
            return NavResult(True, "skipped", "already facing")

        def navigate_to(self, x, y, yaw=None, **kwargs):
            return NavResult(True, "skipped", "already at goal")

    class LLM:
        def __init__(self):
            self.la_n = 0
            self.va_n = 0
            self.la_text = []
            self.va_text = []

        def generate_la(self, messages, max_tokens=2048):
            self.la_n += 1
            blob = []
            for m in messages:
                content = m.get("content")
                if isinstance(content, list):
                    for p in content:
                        if isinstance(p, dict) and p.get("type") == "text":
                            blob.append(str(p.get("text") or ""))
            self.la_text.append("\n".join(blob))
            if self.la_n <= 2:
                return '{"action":"navigate to front","progress_analysis":"hist","reasoning":"r","claim_goal_id":""}'
            return '{"action":"navigate to left","progress_analysis":"hist2","reasoning":"r2","claim_goal_id":""}'

        def generate_va(self, messages):
            self.va_n += 1
            blob = []
            for m in messages:
                content = m.get("content")
                if isinstance(content, list):
                    for p in content:
                        if isinstance(p, dict) and p.get("type") == "text":
                            blob.append(str(p.get("text") or ""))
            self.va_text.append("\n".join(blob))
            if self.va_n <= 3:
                return '{"bbox_2d":[100,100,200,200],"target":"chair","reasoning":"same"}'
            return '{"bbox_2d":[700,80,900,360],"target":"open doorway","reasoning":"other"}'

    llm = LLM()
    agent = LaviraM3Agent(
        RobotConfig(name="small_b", panorama="cardinal4"), RT(), [{"goal_index": 0}], llm=llm
    )
    agent.board = Board()
    agent.t0_joint = False
    agent._go = lambda view, bbox, direction, **kwargs: (
        (0.1, 0.0),
        NavResult(True, "skipped", "already at goal"),
    )

    assert agent.cycle() is True
    pano_after_first = agent.runtime.pano
    assert pano_after_first >= 1
    assert agent._short_hop is True
    assert agent._last_va["direction"] == "front"
    first_faces = list(agent.runtime.faces)
    first_caps = list(agent.runtime.caps)

    assert agent.cycle() is True
    assert agent.runtime.pano == pano_after_first + 1
    assert "Previous LA" in llm.la_text[1]
    assert "prefer that same claim_goal_id" in llm.la_text[1]
    assert "short-hop replan" in llm.va_text[1]
    assert llm.va_n >= 4
    assert llm.la_n >= 3
    assert "Prefer a different legal direction" in llm.la_text[2]
    assert agent._last_va["direction"] == "left"
    assert agent._last_va["target"] == "open doorway"
    assert "left" in agent.runtime.faces
    assert first_faces.count("front") == 1
    assert agent.runtime.caps.count("front") == first_caps.count("front")


if __name__ == "__main__":
    test_dirs()
    test_json()
    test_goals()
    test_remote_provider_defaults()
    test_usable_va_bbox_rejects_none_zero()
    test_va_prompt_search_landmark_no_open_floor()
    test_fallback_va_bbox_is_center_quarter()
    test_generate_va_uses_la_style_deepseek()
    test_generate_va_aliyun_and_gemini_la()
    test_bbox_contract()
    test_bbox_bottom_quarter()
    test_step_cap_and_history()
    test_va_observe_depth_cascade()
    test_clip_va_project_depth()
    test_project_clips_depth_over_5m()
    test_search_step_no_depth_not_found()
    test_view_mask_center_depth()
    test_la_prompt_omits_blocked_dirs()
    test_async_la_excludes_rear()
    test_teammate_spread_note_uses_trace_pose()
    test_dcs_prompt_requires_near_stop()
    test_claim_and_dcs_use_3m()
    test_instruction_skips_found_goals()
    test_va_gets_iin_reference_image()
    test_dcs_gets_claimed_iin_reference_image()
    test_dcs_on_and_tn_use_name_or_description()
    test_handle_claim_dcs_only()
    test_found_view_posted_to_trace()
    test_claim_allows_second_goal_at_same_pose()
    test_found_depth_gate()
    test_interrupt_idle()
    test_json_public()
    test_approach_points()
    test_cap_step_keeps_short_depth()
    test_small_b_inside_envelope_keeps_measured_range()
    test_large_inside_envelope_keeps_measured_range()
    test_small_b_filtered_walk_target_skips_nav()
    test_project_keeps_measured_depth_under_3m()
    test_classify_nav_completion_original_intermediate_blocked()
    test_snap_to_nearest_free()
    test_large_go_submits_va_once()
    test_large_go_preview_failed_retries_approach_points()
    test_serial_candidate_distance_all_robots()
    test_clip_goal_to_map_aabb()
    test_path_loop_trim()
    test_go_trims_loop_path()
    test_go_traces_path_before_navigate()
    test_large_go_traces_path_from_navigate()
    test_thin_planner_path_keeps_corners_and_step()
    test_planner_path_rejects_empty_and_keeps_polyline()
    test_static_go_keeps_first_planner_path()
    test_wait_board_goals()
    test_empty_board_not_complete()
    test_require_all_views()
    test_motion_history_2s()
    test_la_initial_panorama_surround7_and_cardinal4()
    test_llm_call_log_jsonl_and_skip_dcs()
    test_bbox_iou()
    test_short_hop_prompts()
    test_short_hop_helpers()
    test_backtrack_listing_and_walk()
    test_archive_uses_land_pose_not_commanded_xy()
    test_scb_fail_hides_id_then_relists()
    test_scb_uses_live_front_and_stop_dcs()
    test_short_hop_keeps_backtrack()
    test_short_hop_cycle_reuses_and_forbids()
    print("offline ok")
