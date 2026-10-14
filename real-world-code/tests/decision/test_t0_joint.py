#!/usr/bin/env python3
"""Offline T0 joint LA tests. No robot motion and no live LLM."""

from __future__ import annotations

import base64
import io
import json
import os
import threading
import time
from http.server import ThreadingHTTPServer

from PIL import Image

from lavira_m3.decision.board_hub import Handler
from lavira_m3.decision.board_state import BoardStore
from lavira_m3.decision.config import CARDINAL_VIEWS, SURROUND_VIEWS
from lavira_m3.decision.prompts_real import (
    T0_MAP2D_CAPTION,
    la_prompt,
    short_hop_la_feedback,
    short_hop_va_feedback,
    stop_check_prompt,
    t0_joint_la_prompt,
    t0_view_caption,
    va_prompt,
)
from lavira_m3.decision.t0_joint import (
    action_is_legal,
    build_t0_messages,
    legal_dirs,
    parse_t0_assignments,
    rasterize_map_2d,
)


def has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def tiny_jpeg_b64(color=(10, 20, 30)) -> str:
    img = Image.new("RGB", (8, 8), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def test_prompts_are_english():
    board = (
        "Targets (team object board; V2-verified FOUND only):\n- goal_0 [ON]: chair — unresolved"
    )
    texts = [
        t0_joint_la_prompt(["large", "small_a", "small_b"], board),
        t0_view_caption("large", "left_front"),
        t0_view_caption("small_a", "behind"),
        T0_MAP2D_CAPTION,
        la_prompt("agent_0", SURROUND_VIEWS, board, "none"),
        va_prompt("Find the chair", "front", "empty"),
        short_hop_la_feedback(
            {"action": "navigate to front", "claim_goal_id": "", "direction": "front"},
            {"target": "chair", "bbox_2d": [10, 20, 30, 40], "direction": "front"},
        ),
        short_hop_va_feedback(
            {"target": "chair", "bbox_2d": [10, 20, 30, 40], "direction": "front"}, "front"
        ),
        stop_check_prompt("Find the chair", "goal_0"),
    ]
    for text in texts:
        assert text.strip(), "empty prompt"
        assert not has_cjk(text), text[:80]
    assert "# Context" in va_prompt("Find the chair", "front", "empty")
    assert "# Role" in va_prompt("Find the chair", "front", "empty")
    t0 = t0_joint_la_prompt(
        ["large", "small_b"],
        board,
        legal_dirs_by_robot={"small_b": ["front", "behind"]},
        blocked_notes={"small_b": "Near-range mask: left (0.60 m)"},
    )
    assert "panoramic_observation" in t0
    assert "Near-range mask: left (0.60 m)" in t0
    assert "center-region mean depth is under 1.5 m" in t0
    assert "map_understanding" in t0
    assert "team_reasoning" in t0
    assert "Do not output progress_analysis" in t0
    assert '  "progress_analysis":' not in t0
    assert '  "reasoning":' not in t0
    assert '"stop"' not in t0
    assert "Do not send any robot to search a goal already marked found" in t0
    assert "switch to a different unresolved goal" not in t0
    assert "The first action may leave toward that sector" in t0
    assert "about 3 m" in t0
    assert "about 5 m" not in t0
    assert "about 2 m (large)" not in t0
    assert "about 3 m (small_a/small_b)" not in t0
    assert "image_goal_scenes" in t0
    assert "Do not invent map coordinates" in t0
    la = la_prompt("agent_0", SURROUND_VIEWS, board, "none")
    assert '"stop"' not in la
    assert "backtrack" not in la
    assert "Do not output a stop field" not in la
    assert "Search only unresolved targets" in la
    assert "already marked found on the board" in la
    assert "choose that view's direction and go to it" not in la
    assert "enters another room" not in la
    assert "cubicle / workstation aisles" in la
    assert "Prefer not re-exploring" in la
    assert "Initial Panorama" in la
    assert "Never choose a glass door" in la
    assert "Every closed door is not enterable" in la
    assert "Never choose a glass door" in t0
    assert "Every closed door is not enterable" in t0
    assert '"navigate to left_back"' not in la
    assert '"navigate to right_back"' not in la
    assert "rear-facing cameras are not legal actions" in la
    dog_la = la_prompt("agent_1", CARDINAL_VIEWS, board, "none")
    assert '"navigate to behind"' not in dog_la
    assert "rear-facing cameras are not legal actions" in dog_la
    assert "not a legal action" not in dog_la
    assert "Other robots are never a search target" in la
    assert "Prefer not to follow, approach, or cluster with a teammate" in la
    assert "configured teammate" in t0
    assert "configured teammate" in t0
    assert "configured teammate" in t0
    assert "Rear-facing cameras are not legal T0 actions" in t0
    assert "T0 may assign a rear-facing first action" not in t0
    assert "Later per-robot LA cannot use those rear cameras" not in t0
    assert "Do not use backtrack at T0" not in t0
    assert '"region"' in t0
    assert "stored on the shared board" in t0
    assert '"navigate to behind"' not in t0
    assert '"navigate to left_back"' not in t0
    assert '"navigate to right_back"' not in t0
    assert '"navigate to front"' in t0
    assert '"navigate to left"' in t0
    assert '"navigate to right"' in t0
    assert "enters another room" not in t0
    assert "choose a door out" not in t0
    assert "Other robots are never a search target" in t0
    assert "these robots is visible" in t0
    assert "Never choose a glass door" in va_prompt("Find the chair", "front", "empty")
    assert "Every closed door is not enterable" in va_prompt("Find the chair", "front", "empty")
    assert "configured teammate" in va_prompt("Find the chair", "front", "empty")
    assert "can be a useful box" in va_prompt("Find the chair", "front", "empty")
    assert "cubicle aisle continuation" not in va_prompt("Find the chair", "front", "empty")
    assert "usually a poor landmark" in va_prompt("Find the chair", "front", "empty")


def test_t0_view_labels():
    cap = t0_view_caption("large", "left_front")
    assert 'agent_0 (large), current RGB view "left_front"' in cap
    assert "Configured views" in cap
    dog = t0_view_caption("small_b", "behind")
    assert 'agent_2 (small_b), current RGB view "behind"' in dog
    assert "Configured views" in dog


def test_action_is_legal():
    assert action_is_legal("navigate to left_front", legal_dirs("large"))
    assert not action_is_legal("navigate to behind", legal_dirs("large"))
    assert not action_is_legal("navigate to left_back", legal_dirs("large"))
    assert not action_is_legal("navigate to right_back", legal_dirs("large"))
    assert not action_is_legal("navigate to behind", legal_dirs("small_b"))
    assert action_is_legal("navigate to front", legal_dirs("small_b"))
    assert action_is_legal("navigate to left", legal_dirs("small_b"))
    assert action_is_legal("navigate to right", legal_dirs("small_b"))
    assert not action_is_legal("navigate to left_front", legal_dirs("small_b"))
    assert not action_is_legal("backtrack to 0", legal_dirs("large"))
    assert "behind" not in legal_dirs("small_a")
    assert "left_back" not in legal_dirs("large")
    assert "right_back" not in legal_dirs("large")
    assert (
        "front" in legal_dirs("small_a")
        and "left" in legal_dirs("small_a")
        and "right" in legal_dirs("small_a")
    )


def test_parse_and_messages():
    raw = """
    {
      "panoramic_observation": {
        "large": "open room ahead",
        "small_a": "left corridor",
        "small_b": "rear room"
      },
      "map_understanding": "three robots face different rooms",
      "team_reasoning": "spread east west north",
      "image_goal_scenes": {
        "goal_1": "A kitchen with white cabinets and a wood island."
      },
      "assignments": {
        "large": {"action": "navigate to front", "stop": false, "claim_goal_id": "",
                   "progress_analysis": "drop me", "reasoning": "drop me"},
        "small_a": {"action": "navigate to left", "stop": false, "claim_goal_id": ""},
        "agent_2": {"action": "navigate to front", "stop": false, "claim_goal_id": ""}
      }
    }
    """
    parsed, team = parse_t0_assignments(raw, ["large", "small_a", "small_b"])
    assert team["team_reasoning"].startswith("spread")
    assert team["image_goal_scenes"]["goal_1"].startswith("A kitchen")
    assert team["map_understanding"].startswith("three robots")
    assert team["panoramic_observation"]["small_a"] == "left corridor"
    assert parsed["large"]["action"] == "navigate to front"
    assert "progress_analysis" not in parsed["large"]
    assert "reasoning" not in parsed["large"]
    assert parsed["small_a"]["action"] == "navigate to left"
    assert parsed["small_a"]["region"] == ""
    assert parsed["small_b"]["action"] == "navigate to front"
    rear, _ = parse_t0_assignments(
        '{"small_b": {"action": "navigate to behind"}}',
        ["small_b"],
    )
    assert rear == {}
    aloha_rear, _ = parse_t0_assignments(
        '{"large": {"action": "navigate to left_back"}}',
        ["large"],
    )
    assert aloha_rear == {}
    aloha_right_back, _ = parse_t0_assignments(
        '{"large": {"action": "navigate to right_back"}}',
        ["large"],
    )
    assert aloha_right_back == {}
    bad, _ = parse_t0_assignments(
        '{"small_a": {"action": "navigate to left_front", "progress_analysis": "x", "reasoning": "y"}}',
        ["small_a"],
    )
    assert bad == {}

    observes = {
        "large": {
            "pose": [1.0, 2.0, 0.2],
            "frames": [{"name": "front", "jpeg_b64": tiny_jpeg_b64()}],
        },
        "small_a": {
            "pose": [2.0, 0.0, 1.5],
            "frames": [{"name": "left", "jpeg_b64": tiny_jpeg_b64((0, 0, 255))}],
        },
        "small_b": {
            "pose": [0.0, 2.5, -1.5],
            "frames": [{"name": "behind", "jpeg_b64": tiny_jpeg_b64((255, 80, 0))}],
        },
    }
    scene = {
        "map": {
            "resolution": 0.1,
            "origin": [0.0, 0.0],
            "fused": [[10, 10], [11, 10]],
            "overlap": [[10, 10]],
        }
    }
    img = rasterize_map_2d(scene, {"large": {"x": 1, "y": 1, "yaw": 0}, "small_a": [2, 0, 1.5]})
    assert img.size[0] == 512
    mug = Image.new("RGB", (8, 8), (200, 40, 40))
    messages = build_t0_messages(
        observes,
        "Targets: goal_0 chair",
        img,
        reference_images=[("goal_1", mug)],
        robots=["large", "small_a", "small_b"],
    )
    content = messages[0]["content"]
    texts = [p["text"] for p in content if p.get("type") == "text"]
    joined = "\n".join(texts)
    assert not has_cjk(joined)
    assert "IMAGE GOALS" in joined
    assert "goal_1 image-goal photo" in joined
    assert 'Image: agent_0 (large), current RGB view "front"' in joined
    assert 'Image: agent_1 (small_a), current RGB view "left"' in joined
    assert "Initial shared occupancy map" in joined
    assert any(p.get("type") == "image_url" for p in content)
    goal_idx = next(i for i, t in enumerate(texts) if "IMAGE GOALS" in t)
    map_idx = next(i for i, t in enumerate(texts) if "Initial shared occupancy map" in t)
    pano_idx = next(i for i, t in enumerate(texts) if "large / agent_0 current panorama" in t)
    assert goal_idx < map_idx < pano_idx


def test_t0_prompt_masks_near_views():
    observes = {
        "large": {
            "pose": [2.5, 0.3, 0.0],
            "dirs": list(SURROUND_VIEWS),
            "frames": [{"name": n, "center_depth_m": 3.2} for n in SURROUND_VIEWS],
        },
        "small_b": {
            "pose": [2.57, 1.75, 1.43],
            "dirs": list(CARDINAL_VIEWS),
            "frames": [
                {"name": "front", "center_depth_m": 2.8},
                {"name": "left", "center_depth_m": 0.57},
                {"name": "behind", "center_depth_m": 2.4},
                {"name": "right", "center_depth_m": 0.80},
            ],
        },
    }
    messages = build_t0_messages(
        observes, "Targets: goal_0 IIN — unresolved", Image.new("RGB", (8, 8), (20, 20, 20))
    )
    texts = [p["text"] for p in messages[0]["content"] if p.get("type") == "text"]
    joined = "\n".join(texts)
    small_b_actions = joined.split("agent_2 (small_b) legal actions:")[1].split("=== ")[0]
    assert '"navigate to left"' not in small_b_actions
    assert '"navigate to behind"' not in small_b_actions
    assert '"navigate to front"' in small_b_actions
    large_actions = joined.split("agent_0 (large) legal actions:")[1].split("agent_2")[0]
    assert '"navigate to left_back"' not in large_actions
    assert '"navigate to right_back"' not in large_actions
    assert '"navigate to front"' in large_actions
    assert '"navigate to left"' in large_actions
    assert '"navigate to right"' in large_actions
    assert "Near-range mask" in joined
    assert "0.57 m" in joined
    assert "omitted from the action list" in joined


def _start_board():
    from lavira_m3.decision import board_state as mod

    mod.STORE = BoardStore()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    time.sleep(0.05)
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def test_t0_board_barrier():
    import json
    from urllib.request import Request, urlopen

    def call(method, url, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=3) as resp:
            return resp.status, json.loads(resp.read().decode() or "{}")

    os.environ["T0_ROBOTS"] = "large,small_a,small_b"
    os.environ["T0_LEADER"] = "large"
    httpd, base = _start_board()
    try:
        code, snap = call(
            "POST",
            base + "/api/board/init",
            {
                "robot": "web",
                "goals": [
                    {"modality": "ON", "object_category": "chair"},
                    {
                        "modality": "IIN",
                        "object_category": "mug",
                        "reference_image_b64": tiny_jpeg_b64(),
                    },
                ],
            },
        )
        assert code == 200 and snap["ok"]
        assert snap["goals"][1]["reference_image"].startswith("/api/board/ref/")
        assert snap["t0"]["assigned"] is False

        for name, pose in (("large", [0, 0, 0]), ("small_a", [1, 0, 0]), ("small_b", [2, 0, 0])):
            code, st = call(
                "POST",
                base + "/api/t0/observe",
                {
                    "robot": name,
                    "pose": pose,
                    "dirs": ["front"],
                    "frames": [{"name": "front", "jpeg_b64": tiny_jpeg_b64()}],
                },
            )
            assert st["ok"]
        assert st["ready"] is True

        code, locked = call("POST", base + "/api/t0/lock", {"robot": "large"})
        assert locked["acquired"] is True
        code, bundle = call("GET", base + "/api/t0/bundle?robot=large")
        assert bundle["ok"] and "large" in bundle["observes"]

        call(
            "POST",
            base + "/api/t0/assign",
            {
                "robot": "large",
                "team_reasoning": "spread",
                "panoramic_observation": {"large": "front room", "small_a": "left hall"},
                "map_understanding": "small_a is west of large",
                "la_time_ms": 3900,
                "image_goal_scenes": {
                    "goal_1": "A kitchen with white cabinets and a wood island.",
                },
                "assignments": {
                    "large": {
                        "action": "navigate to front",
                        "progress_analysis": "empty",
                        "reasoning": "a",
                        "stop": False,
                    },
                    "small_a": {
                        "action": "navigate to left",
                        "progress_analysis": "empty",
                        "reasoning": "b",
                        "stop": False,
                    },
                },
            },
        )
        code, small_a = call("GET", base + "/api/t0/status?robot=small_a")
        assert small_a["assigned"] is True
        assert small_a["la"]["action"] == "navigate to left"
        assert "progress_analysis" not in small_a["la"]
        assert "reasoning" not in small_a["la"]
        code, board = call("GET", base + "/api/board")
        assert board["t0"]["assignments"]["small_a"]["action"] == "navigate to left"
        assert "progress_analysis" not in board["t0"]["assignments"]["small_a"]
        assert board["t0"]["team_reasoning"] == "spread"
        assert board["t0"]["map_understanding"] == "small_a is west of large"
        assert board["t0"]["la_time_ms"] == 3900
        assert small_a["la_time_ms"] == 3900
        assert board["t0"]["panoramic_observation"]["small_a"] == "left hall"
        assert board["t0"]["image_goal_scenes"]["goal_1"].startswith("A kitchen")
        assert board["goals"][1]["scene_text"].startswith("A kitchen")
        code, prompt = call("GET", base + "/api/board/prompt?robot=small_b")
        assert "scene: A kitchen" in prompt.get("board", "")
        assert "large" in board["t0"]["assignments"]
    finally:
        httpd.shutdown()


def test_ingest_share_cameras_feeds_t0():
    os.environ["T0_ROBOTS"] = "large,small_a,small_b"
    store = BoardStore()
    store.init_goals([{"modality": "ON", "object_category": "chair"}], "web")
    jpeg = base64.b64decode(tiny_jpeg_b64())
    scene = {"poses": {"small_a": {"live": True, "position": [1.2, 3.4, 0.3], "yaw": 0.5}}}

    def fetch(url):
        if url.endswith("/v1/scene"):
            return 200, json.dumps(scene).encode()
        for cam in ("front", "left", "rear", "right"):
            if url.endswith("/v1/camera-image/small_a/%s" % cam):
                return 200, jpeg
        return 404, b""

    store._share_fetch = fetch
    store._share_ingest_at = 0.0
    store.ingest_share_cameras()
    rec = store.t0.observes["small_a"]
    assert rec["pose"] == [1.2, 3.4, 0.5]
    assert [fr["name"] for fr in rec["frames"]] == ["front", "left", "behind", "right"]
    assert "small_a" in store.t0.online_names()
    assert "large" not in store.t0.observes


def test_ingest_share_cameras_skips_before_start_or_partial():
    os.environ["T0_ROBOTS"] = "large,small_a,small_b"
    store = BoardStore()
    called = []
    store._share_fetch = lambda url: called.append(url) or (200, b"{}")
    store._share_ingest_at = 0.0
    store.ingest_share_cameras()
    assert called == []
    store.init_goals([{"modality": "ON", "object_category": "chair"}], "web")
    jpeg = base64.b64decode(tiny_jpeg_b64())

    def fetch(url):
        if url.endswith("/v1/scene"):
            return 200, json.dumps(
                {"poses": {"small_a": {"live": True, "position": [0, 0], "yaw": 0}}}
            ).encode()
        if url.endswith("/v1/camera-image/small_a/rear"):
            return 404, b""
        return 200, jpeg

    store._share_fetch = fetch
    store._share_ingest_at = 0.0
    store.ingest_share_cameras()
    assert "small_a" not in store.t0.observes


def test_t0_observe_requires_images():
    os.environ["T0_ROBOTS"] = "large,small_a,small_b"
    store = BoardStore()
    store.init_goals([{"modality": "ON", "object_category": "chair"}], "web")
    bad = store.t0_observe(
        "large",
        [0, 0, 0],
        ["front", "left"],
        [{"name": "front", "jpeg_b64": tiny_jpeg_b64()}],
    )
    assert bad.get("ok") is False
    assert "missing" in str(bad.get("error") or "")


def test_t0_wait_errors_without_teammates():
    os.environ["T0_JOINT"] = "1"
    os.environ["T0_WAIT_S"] = "0.2"
    os.environ["T0_ROBOTS"] = "large,small_a,small_b"
    os.environ["T0_LEADER"] = "large"
    from lavira_m3.contracts import NavResult
    from lavira_m3.decision.agent_loop import LaviraM3Agent, T0BarrierError
    from lavira_m3.decision.config import RobotConfig

    rgb = Image.new("RGB", (8, 8), (20, 40, 60))
    frames = [{"name": n, "rgb": rgb} for n in SURROUND_VIEWS]

    class FakeRuntime:
        def pose(self):
            return (0.0, 0.0, 0.0)

        def capture_panorama(self):
            return frames

        def face_direction(self, name):
            raise AssertionError("no navigation before T0 assignment")

        def capture_direction(self, name):
            raise AssertionError("no VA before T0 assignment")

        def navigate_to(self, x, y, yaw=None):
            raise AssertionError("no navigation before T0 assignment")

        def stop(self):
            return NavResult(True, "stopped", "")

    class MissingBoard:
        def t0_observe(self, pose, dirs, payload):
            return {
                "ok": True,
                "online": ["large", "small_a"],
                "offline": ["small_b"],
                "uploaded": ["large"],
                "missing": ["small_a"],
                "ready": False,
            }

        def t0_status(self):
            return {
                "ok": True,
                "online": ["large", "small_a"],
                "offline": ["small_b"],
                "uploaded": ["large"],
                "missing": ["small_a"],
                "ready": False,
                "assigned": False,
                "failed": "",
                "leader": "large",
            }

    agent = LaviraM3Agent(
        RobotConfig(name="large", panorama="surround7"),
        FakeRuntime(),
        [{"goal_index": 0, "object_category": "chair"}],
        llm=object(),
    )
    agent.board = MissingBoard()
    agent._la = lambda _frames: (_ for _ in ()).throw(AssertionError("independent LA must not run"))
    try:
        agent._t0_or_wait(frames)
        raise AssertionError("expected T0BarrierError")
    except T0BarrierError as exc:
        text = str(exc)
        assert "missing" in text
        assert "small_a" in text


def test_t0_skips_offline_robots():
    os.environ["T0_ROBOTS"] = "large,small_a,small_b"
    os.environ["T0_LEADER"] = "large"
    store = BoardStore()
    store.init_goals([{"modality": "ON", "object_category": "chair"}], "web")
    hello = store.t0_hello("small_a")
    assert hello["ok"]
    assert hello["online"] == ["small_a"]
    assert hello["offline"] == ["large", "small_b"]
    assert hello["ready"] is False
    posted = store.t0_observe(
        "small_a",
        [0, 0, 0],
        ["front"],
        [{"name": "front", "jpeg_b64": tiny_jpeg_b64()}],
    )
    assert posted["ok"]
    assert posted["ready"] is True
    assert posted["missing"] == []
    assert posted["leader"] == "small_a"
    locked = store.t0_lock("large")
    assert locked.get("ok") is False
    locked = store.t0_lock("small_a")
    assert locked.get("acquired") is True

    from lavira_m3.decision.agent_loop import LaviraM3Agent
    from lavira_m3.decision.config import RobotConfig

    rgb = Image.new("RGB", (8, 8), (20, 40, 60))
    frames = [{"name": n, "rgb": rgb} for n in CARDINAL_VIEWS]

    class FakeRuntime:
        def pose(self):
            return (0.0, 0.0, 0.0)

        def stop(self):
            return None

    class SoloBoard:
        def t0_observe(self, pose, dirs, payload):
            return {
                "ok": True,
                "online": ["small_a"],
                "offline": ["large", "small_b"],
                "missing": [],
                "ready": True,
                "assigned": True,
            }

        def t0_status(self):
            return {
                "ok": True,
                "online": ["small_a"],
                "offline": ["large", "small_b"],
                "missing": [],
                "ready": True,
                "assigned": True,
                "leader": "small_a",
                "la": {"action": "navigate to front"},
            }

    agent = LaviraM3Agent(
        RobotConfig(name="small_a", panorama="cardinal4"),
        FakeRuntime(),
        [{"goal_index": 0, "object_category": "chair"}],
        llm=object(),
    )
    agent.board = SoloBoard()
    la = agent._t0_or_wait(frames)
    assert la["action"] == "navigate to front"


def test_t0_wait_returns_assignment():
    os.environ["T0_JOINT"] = "1"
    from lavira_m3.decision.agent_loop import LaviraM3Agent
    from lavira_m3.decision.config import RobotConfig

    rgb = Image.new("RGB", (8, 8), (20, 40, 60))
    frames = [{"name": n, "rgb": rgb} for n in SURROUND_VIEWS]

    class FakeRuntime:
        def pose(self):
            return (0.0, 0.0, 0.0)

        def stop(self):
            return None

    class AssignedBoard:
        def t0_observe(self, pose, dirs, payload):
            return {"ok": True, "ready": True, "assigned": True}

        def t0_status(self):
            return {
                "ok": True,
                "ready": True,
                "assigned": True,
                "la_time_ms": 3900,
                "la": {"action": "navigate to front"},
            }

    agent = LaviraM3Agent(
        RobotConfig(name="large", panorama="surround7"),
        FakeRuntime(),
        [{"goal_index": 0, "object_category": "chair"}],
        llm=object(),
    )
    agent.board = AssignedBoard()
    la = agent._t0_or_wait(frames)
    assert la["action"] == "navigate to front"
    assert la["_t0_la_time_ms"] == 3900


if __name__ == "__main__":
    test_prompts_are_english()
    test_t0_view_labels()
    test_action_is_legal()
    test_parse_and_messages()
    test_t0_prompt_masks_near_views()
    test_t0_board_barrier()
    test_ingest_share_cameras_feeds_t0()
    test_ingest_share_cameras_skips_before_start_or_partial()
    test_t0_observe_requires_images()
    test_t0_wait_errors_without_teammates()
    test_t0_skips_offline_robots()
    test_t0_wait_returns_assignment()
    print("t0 joint ok")
