#!/usr/bin/env python3
"""Offline Goal Board hub tests. No robot motion."""

from __future__ import annotations

import json
import os
import threading
import time
from http.server import ThreadingHTTPServer
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from lavira_m3.decision.board_hub import Handler
from lavira_m3.decision.board_state import BoardStore

os.environ["LAVIRA_MUTE_START_WAV"] = "1"


def _start():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.05)
    return httpd, f"http://127.0.0.1:{port}"


def _json(method, url, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=3) as resp:
        return resp.status, json.loads(resp.read().decode() or "{}")


def test_web_init_plays_start_wav_once():
    from lavira_m3.decision import board_state as mod
    from lavira_m3.decision.board_state import play_found_wav, play_start_search_wav

    assert play_start_search_wav() is False
    assert play_found_wav("/tmp/none.wav") is False
    store = BoardStore()
    with patch.object(mod, "play_start_search_wav", return_value=True) as play:
        snap = store.init_goals([{"modality": "ON", "object_category": "chair"}], "large")
        assert snap.get("started") is False
        time.sleep(0.05)
        assert play.call_count == 0
        snap = store.init_goals([{"modality": "ON", "object_category": "chair"}], "web")
        assert snap.get("started") is True
        time.sleep(0.05)
        assert play.call_count == 0


def test_board_flow():
    from lavira_m3.decision import board_state as mod

    mod.STORE = BoardStore()
    httpd, base = _start()
    try:
        goals = [
            {"goal_index": 0, "modality": "ON", "object_category": "office chair"},
            {
                "goal_index": 1,
                "modality": "TN",
                "object_category": "plant",
                "tn_text": "by the window",
            },
        ]
        with patch("lavira_m3.decision.board_state.play_start_search_wav", return_value=True):
            code, data = _json("POST", base + "/api/board/init", {"goals": goals, "robot": "large"})
        assert code == 200 and data["ok"]
        assert data["found_by"] == [-1, -1]
        assert data.get("started") is False
        code, data = _json("POST", base + "/api/board/init", {"goals": goals, "robot": "web"})
        assert data.get("started") is True

        code, data = _json(
            "POST",
            base + "/api/board/progress",
            {
                "robot": "large",
                "text": "I scanned the hallway.",
                "step": 2,
            },
        )
        assert data["progress"]["large"]["text"].startswith("I scanned")

        code, prompt = _json("GET", base + "/api/board/prompt?robot=small_a")
        assert "goal_0" in prompt["board"]
        assert "unresolved" in prompt["board"]
        assert "I scanned the hallway" in prompt["progress"]
        assert "claimed progress_analysis" in prompt["progress"]
        assert "small_a" not in prompt["progress"]

        own = _json("GET", base + "/api/board/prompt?robot=large")[1]
        assert "I scanned the hallway" not in own["progress"]

        with patch("lavira_m3.decision.board_state.announce_found", return_value=True):
            code, data = _json(
                "POST", base + "/api/board/found", {"robot": "large", "goal_id": "goal_0"}
            )
        assert data["found_by"][0] == 0
        assert data["found_by"][1] == -1

        # Re-init must not wipe FOUND.
        try:
            _json(
                "POST",
                base + "/api/board/init",
                {
                    "goals": goals + [{"goal_index": 2, "object_category": "extra"}],
                    "robot": "small_a",
                },
            )
            raise AssertionError("expected 400 when replacing a board that already has FOUND")
        except HTTPError as exc:
            assert exc.code == 400
        snap = _json("GET", base + "/api/board")[1]
        assert snap["found_by"][0] == 0
        assert len(snap["goals"]) == 2

        code, data = _json(
            "POST",
            base + "/api/board/trace",
            {
                "robot": "large",
                "step": 3,
                "la": {"action": "navigate to front", "stop": False},
                "va": {"target": "plant", "bbox_2d": [10, 20, 30, 40]},
                "nav": {"state": "not_executable", "message": "blocked"},
                "waypoint": [1.2, 3.4],
                "va_map": [2.5, 0.3],
                "va_pixel": [120, 80],
                "va_path": [[2.5, 0.1], [2.5, 0.3]],
                "va_image": "aaaa",
            },
        )
        assert code == 200 and data["ok"]
        assert data["traces"]["large"]["la"]["action"] == "navigate to front"
        assert data["traces"]["large"]["va"]["target"] == "plant"
        assert data["traces"]["large"]["va_map"] == [2.5, 0.3]
        assert data["traces"]["large"]["va_path"][1][1] == 0.3
    finally:
        httpd.shutdown()


def test_empty_board_not_complete():
    store = BoardStore()
    snap = store.snapshot()
    assert snap["ok"] is True
    assert snap["goals"] == []
    assert snap["found_by"] == []


def test_board_reset_clears_leftover():
    from lavira_m3.decision import board_state as mod

    mod.STORE = BoardStore()
    httpd, base = _start()
    try:
        _json(
            "POST",
            base + "/api/board/init",
            {
                "goals": [
                    {"goal_index": 0, "modality": "ON", "object_category": "office chair"},
                    {
                        "goal_index": 1,
                        "modality": "TN",
                        "object_category": "plant",
                        "tn_text": "by the window",
                    },
                ],
                "robot": "web",
            },
        )
        code, data = _json("POST", base + "/api/board/reset", {})
        assert code == 200 and data["ok"]
        assert data["goals"] == []
        assert data["progress"] == {}
        assert data["traces"] == {}
        assert data.get("history") == {}
        assert data["t0"]["uploaded"] == []
        assert data.get("started") is False
    finally:
        httpd.shutdown()


def test_board_reset_sets_t0_robots():
    from lavira_m3.decision import board_state as mod

    mod.STORE = BoardStore()
    httpd, base = _start()
    try:
        code, data = _json("POST", base + "/api/board/reset", {"t0_robots": "large,small_b"})
        assert code == 200 and data["ok"]
        assert data.get("started") is False
        assert data["goals"] == []
        assert data["t0"]["expected"] == ["large", "small_b"]
        assert "small_a" not in data["t0"]["expected"]
    finally:
        httpd.shutdown()


def test_web_init_keeps_reset_t0_robots():
    from lavira_m3.decision import board_state as mod

    mod.STORE = BoardStore()
    httpd, base = _start()
    try:
        code, data = _json("POST", base + "/api/board/reset", {"t0_robots": "large,small_a"})
        assert code == 200 and data["t0"]["expected"] == ["large", "small_a"]
        code, data = _json(
            "POST",
            base + "/api/board/init",
            {
                "goals": [{"goal_index": 0, "modality": "ON", "object_category": "chair"}],
                "robot": "web",
            },
        )
        assert code == 200 and data["ok"]
        assert data.get("started") is True
        assert data["t0"]["expected"] == ["large", "small_a"]
        assert "small_b" not in data["t0"]["expected"]
    finally:
        httpd.shutdown()


def test_board_default_t0_includes_small_a():
    from lavira_m3.decision import board_state as mod
    from lavira_m3.decision.config import t0_robots

    old = os.environ.get("T0_ROBOTS")
    os.environ.pop("T0_ROBOTS", None)
    try:
        assert t0_robots() == ["large", "small_a", "small_b"]
        mod.STORE = BoardStore()
        httpd, base = _start()
        try:
            code, data = _json("POST", base + "/api/board/reset", {})
            assert code == 200 and data["ok"]
            assert data["t0"]["expected"] == ["large", "small_a", "small_b"]
        finally:
            httpd.shutdown()
    finally:
        if old is None:
            os.environ.pop("T0_ROBOTS", None)
        else:
            os.environ["T0_ROBOTS"] = old


def test_found_events_use_configured_names_without_audio_assets():
    from lavira_m3.decision.board_state import found_announce_text, found_announce_wav

    for name in ("large", "small_a", "small_b"):
        assert found_announce_text(name, "ON") == f"Object goal was found by {name}."
        assert found_announce_wav(name, "ON") == ""


def test_found_announces_once():
    from lavira_m3.decision import board_state as mod
    from lavira_m3.decision.board_state import (
        announce_found,
        play_all_found_wav,
        play_systems_online_wav,
    )

    assert announce_found("small_b", "goal_0", "chair", "ON") is False
    assert play_all_found_wav() is False
    assert play_systems_online_wav() is False
    store = BoardStore()
    with patch.object(mod, "play_start_search_wav", return_value=True):
        store.init_goals(
            [{"goal_index": 0, "modality": "ON", "object_category": "office chair"}],
            "web",
        )
    with patch.object(mod, "announce_found", return_value=True) as speak:
        first = store.post_found("small_b", "goal_0")
        again = store.post_found("large", "goal_0")
        time.sleep(0.05)
        assert first["found_by"][0] == 2
        assert again["found_by"][0] == 2
        assert speak.call_count == 0


def test_board_has_no_bundled_audio_dependency():
    from lavira_m3.decision.board_state import (
        ALL_FOUND_WAV,
        START_SEARCH_WAV,
        SYSTEMS_ONLINE_WAV,
    )

    assert (ALL_FOUND_WAV, START_SEARCH_WAV, SYSTEMS_ONLINE_WAV) == ("", "", "")


def test_systems_online_plays_once():
    from lavira_m3.decision import board_state as mod

    store = BoardStore()
    with patch.object(mod, "play_start_search_wav", return_value=True):
        store.init_goals([{"goal_index": 0, "modality": "ON", "object_category": "chair"}], "web")
    with patch.object(mod, "play_systems_online_wav", return_value=True) as speak:
        store.t0_hello("large")
        store.t0_hello("small_a")
        store.t0_hello("small_b")
        time.sleep(0.05)
        assert speak.call_count == 0
        store.t0_hello("small_b")
        time.sleep(0.05)
        assert speak.call_count == 0


def test_all_found_plays_complete_once():
    from lavira_m3.decision import board_state as mod

    store = BoardStore()
    with patch.object(mod, "play_start_search_wav", return_value=True):
        store.init_goals(
            [
                {"goal_index": 0, "modality": "ON", "object_category": "printer"},
                {"goal_index": 1, "modality": "TN", "object_category": "chairs"},
            ],
            "web",
        )
    with patch.object(mod, "announce_found", return_value=True), patch.object(
        mod, "play_all_found_wav", return_value=True
    ) as done:
        store.post_found("small_a", "goal_0")
        time.sleep(0.05)
        assert done.call_count == 0
        store.post_found("small_b", "goal_1")
        time.sleep(0.05)
        assert done.call_count == 0
        store.post_found("large", "goal_1")
        time.sleep(0.05)
        assert done.call_count == 0


def test_clear_runtime_keeps_goals():
    from lavira_m3.decision import board_state as mod

    mod.STORE = BoardStore()
    mod.list_local_m3_processes = lambda: []
    httpd, base = _start()
    try:
        _json(
            "POST",
            base + "/api/board/init",
            {
                "goals": [
                    {"goal_index": 0, "modality": "ON", "object_category": "office chair"},
                    {
                        "goal_index": 1,
                        "modality": "TN",
                        "object_category": "plant",
                        "tn_text": "by the window",
                    },
                ],
                "robot": "web",
            },
        )
        _json(
            "POST",
            base + "/api/board/trace",
            {
                "robot": "small_b",
                "step": 2,
                "va": {"target": "plant", "bbox_2d": [10, 20, 30, 40]},
                "va_image": "aaaa",
                "va_bbox": [10, 20, 30, 40],
            },
        )
        mod.STORE.t0.lock_robot = "large"
        mod.STORE.t0.lock_until = time.time() + 30
        try:
            _json("POST", base + "/api/board/clear-runtime", {})
            raise AssertionError("expected 409 while T0 lock is held")
        except HTTPError as exc:
            assert exc.code == 409
            data = json.loads(exc.read().decode() or "{}")
            assert data.get("ok") is False
            assert "T0" in str(data.get("error") or "")

        mod.STORE.t0.lock_robot = ""
        mod.STORE.t0.lock_until = 0.0
        code, data = _json("POST", base + "/api/board/clear-runtime", {})
        assert code == 200 and data["ok"]
        assert [g["object_category"] for g in data["goals"]] == ["office chair", "plant"]
        assert data["traces"] == {}
        assert data.get("history") == {}
        assert data["progress"] == {}
        assert data["t0"]["uploaded"] == []
        assert data.get("started") is True
    finally:
        httpd.shutdown()


def _tiny_jpeg_b64():
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (16, 12), (20, 80, 180)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def test_history_appends_and_serves_jpeg():
    from lavira_m3.decision import board_state as mod

    mod.STORE = BoardStore()
    httpd, base = _start()
    try:
        jpeg = _tiny_jpeg_b64()
        code, data = _json(
            "POST",
            base + "/api/board/history",
            {
                "robot": "small_a",
                "waypoint_id": 0,
                "step": 1,
                "front_image": jpeg,
                "pose": [0.1, 0.2, 0.0],
            },
        )
        assert code == 200 and data["ok"]
        assert len(data["history"]["small_a"]) == 1
        assert data["history"]["small_a"][0]["front_image"] == "/api/board/history/small_a/0"
        code, data = _json(
            "POST",
            base + "/api/board/history",
            {
                "robot": "small_a",
                "waypoint_id": 1,
                "step": 2,
                "front_image": jpeg,
                "pose": [1.0, 0.0, 0.0],
            },
        )
        assert len(data["history"]["small_a"]) == 2
        code, data = _json(
            "POST",
            base + "/api/board/history",
            {
                "robot": "small_a",
                "waypoint_id": 0,
                "action": "navigate to left",
                "nav_ok": False,
            },
        )
        assert len(data["history"]["small_a"]) == 2
        assert data["history"]["small_a"][0]["action"] == "navigate to left"
        req = Request(base + "/api/board/history/small_a/0")
        with urlopen(req, timeout=3) as resp:
            raw = resp.read()
        assert raw[:2] == b"\xff\xd8"
        code, data = _json("POST", base + "/api/board/reset", {})
        assert data.get("history") == {}
    finally:
        httpd.shutdown()


def test_ra_timing_clear_and_preserve():
    store = BoardStore()
    store.post_trace("large", {"step": 1, "ra_time_ms": 123.4})
    store.post_trace("large", {"step": 1, "nav": {"state": "succeeded"}})
    assert store.traces["large"]["ra_time_ms"] == 123.4
    store.post_trace("large", {"step": 2, "ra_time_ms": None})
    assert store.traces["large"]["ra_time_ms"] is None


def test_clear_va_wipes_and_skips_merge():
    store = BoardStore()
    store.post_trace(
        "small_b",
        {
            "step": 3,
            "va": {"bbox_2d": [1, 2, 3, 4], "target": "chair"},
            "va_bbox": [1, 2, 3, 4],
            "va_image": "oldjpeg",
            "va_map": [1.0, 2.0],
            "found_goal_id": "goal_0",
            "image_size": [640, 480],
        },
    )
    store.post_trace(
        "small_b",
        {
            "step": 4,
            "clear_va": True,
            "direction": "backtrack",
            "la": {"action": "navigate to left"},
        },
    )
    rec = store.traces["small_b"]
    assert rec["va"] == {}
    assert rec["va_bbox"] is None
    assert rec["va_image"] == ""
    assert rec["va_map"] is None
    assert rec["found_goal_id"] is None
    assert rec["image_size"] is None
    assert rec["direction"] == "backtrack"
    assert rec["la"]["action"] == "navigate to left"


if __name__ == "__main__":
    test_ra_timing_clear_and_preserve()
    test_empty_board_not_complete()
    test_web_init_plays_start_wav_once()
    test_board_flow()
    test_board_reset_clears_leftover()
    test_board_reset_sets_t0_robots()
    test_web_init_keeps_reset_t0_robots()
    test_board_default_t0_includes_small_a()
    test_found_events_use_configured_names_without_audio_assets()
    test_found_announces_once()
    test_board_has_no_bundled_audio_dependency()
    test_systems_online_plays_once()
    test_all_found_plays_complete_once()
    test_clear_runtime_keeps_goals()
    test_history_appends_and_serves_jpeg()
    test_clear_va_wipes_and_skips_merge()
    print("board hub ok")
