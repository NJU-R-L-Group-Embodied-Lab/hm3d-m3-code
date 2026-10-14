#!/usr/bin/env python3
"""Offline map2d navigation client tests. Never talks to the real robot."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from lavira_m3.decision.nav_http import Map2dNavClient


class FakeNav:
    def __init__(self):
        self.state = "idle"
        self.message = ""
        self.goal = None
        self.nav_id = 0
        self.preview_id = None
        self.preview = None
        self.accepts = 0
        self.delay_accept = False
        self.executable = True
        self.estop = False
        self.pose_live = True
        self.motion_live = True
        self.expire_preview = False
        self.change_id_after = False
        self.submitted = []
        self.position = [0.0, 0.0, 0.0]
        self.auto_succeed = True
        self.snap_to = None


FAKE = FakeNav()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        return

    def _json(self, obj, code=200):
        raw = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _nav(self):
        return {
            "state": FAKE.state,
            "path": [[0, 0], FAKE.goal[:2]] if FAKE.goal else [],
            "message": FAKE.message,
            "id": FAKE.nav_id,
            "goal": FAKE.goal,
            "estop_locked": FAKE.estop,
            "frame": "odom_lio",
            "pose": {"live": FAKE.pose_live, "yaw": 0.0, "position": list(FAKE.position)},
            "motion": {"live": FAKE.motion_live, "v": 0.0, "w": 0.0},
        }

    def do_GET(self):
        if urlparse(self.path).path == "/v1/scene":
            return self._json({"navigation": self._nav()})
        return self._json({"error": "no"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n).decode() or "{}") if n else {}
        path = urlparse(self.path).path
        if path == "/v1/preview":
            if FAKE.expire_preview:
                FAKE.expire_preview = False
                # Verbatim robot-side error ("preview expired or robot moved"); nav_http matches "过期".
                return self._json({"error": "预览已过期或机器人已移动，请重新预览"}, 409)
            FAKE.preview_id = "prev-1"
            FAKE.preview = dict(body)
            goal = list(FAKE.snap_to) if FAKE.snap_to else [body["x"], body["y"]]
            return self._json(
                {
                    "id": FAKE.preview_id,
                    "goal": goal,
                    "yaw": body["yaw"],
                    "path": [[0, 0], goal],
                    "frame": "odom_lio",
                    "executable": FAKE.executable,
                    "warning": None if FAKE.executable else "blocked",
                }
            )
        if path == "/v1/goal":
            if FAKE.delay_accept:
                time.sleep(0.35)
            if body.get("preview_id") != FAKE.preview_id:
                return self._json({"error": "preview the planned path for this goal first"}, 409)
            FAKE.accepts += 1
            FAKE.submitted.append(body)
            FAKE.nav_id += 1
            FAKE.goal = [body["x"], body["y"], body["yaw"]]
            FAKE.state = "navigating"
            if FAKE.change_id_after:
                threading.Timer(0.15, self._flip_id).start()
            elif FAKE.auto_succeed:
                threading.Timer(0.15, self._succeed).start()
            return self._json(
                {"ok": True, "id": FAKE.nav_id, "goal": FAKE.goal[:2], "yaw": body["yaw"]}
            )
        if path == "/v1/cancel":
            FAKE.state = "cancelled"
            FAKE.message = "cancelled"
            return self._json({"ok": True})
        if path == "/v1/estop":
            FAKE.estop = bool(body.get("locked"))
            FAKE.state = "stopped" if FAKE.estop else "idle"
            FAKE.message = "software e-stop locked" if FAKE.estop else "released"
            return self._json({"ok": True, "locked": FAKE.estop})
        return self._json({"error": "no"}, 404)

    def _succeed(self):
        FAKE.state = "succeeded"
        FAKE.message = "arrived"

    def _flip_id(self):
        FAKE.nav_id += 1
        FAKE.state = "navigating"


def _server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    time.sleep(0.05)
    return httpd, f"http://127.0.0.1:{httpd.server_address[1]}"


def _client(base):
    return Map2dNavClient(base, poll_s=0.05, total_s=2.0, http_timeout_s=0.25)


def test_preview_not_executable():
    global FAKE
    FAKE = FakeNav()
    FAKE.executable = False
    httpd, base = _server()
    try:
        result = _client(base).navigate_to(1.0, 2.0, 0.3)
        assert result.ok is False
        assert result.state == "not_executable"
        assert FAKE.accepts == 0
    finally:
        httpd.shutdown()


def test_success_and_id_match():
    global FAKE
    FAKE = FakeNav()
    httpd, base = _server()
    try:
        result = _client(base).navigate_to(1.0, 2.0, 0.3)
        assert result.ok is True
        assert result.state == "succeeded"
        assert result.execution_id == 1
        assert FAKE.accepts == 1
    finally:
        httpd.shutdown()


def test_expired_preview_retries_once():
    global FAKE
    FAKE = FakeNav()
    FAKE.expire_preview = True
    httpd, base = _server()
    try:
        result = _client(base).navigate_to(1.0, 2.0, 0.3)
        assert result.ok is True
        assert FAKE.accepts == 1
    finally:
        httpd.shutdown()


def test_busy_does_not_cancel():
    global FAKE
    FAKE = FakeNav()
    FAKE.state = "navigating"
    FAKE.nav_id = 9
    httpd, base = _server()
    try:
        result = _client(base).navigate_to(3.0, 4.0, 0.0)
        assert result.ok is False
        assert result.state == "busy"
        assert FAKE.accepts == 0
        assert FAKE.state == "navigating"
    finally:
        httpd.shutdown()


def test_id_change_is_interrupt():
    global FAKE
    FAKE = FakeNav()
    FAKE.change_id_after = True
    httpd, base = _server()
    try:
        result = _client(base).navigate_to(1.0, 2.0, 0.0)
        assert result.ok is False
        assert result.state == "interrupted"
    finally:
        httpd.shutdown()


def test_submit_timeout_does_not_resend():
    global FAKE
    FAKE = FakeNav()
    FAKE.delay_accept = True
    httpd, base = _server()
    try:
        client = Map2dNavClient(base, poll_s=0.05, total_s=1.0, http_timeout_s=0.05)
        result = client.navigate_to(1.0, 2.0, 0.0)
        assert FAKE.accepts <= 1
        assert result.state in (
            "submit_uncertain",
            "interrupted",
            "cancelled",
            "succeeded",
            "timed_out",
        )
    finally:
        httpd.shutdown()


def test_static_t0_preview_does_not_submit():
    import os

    global FAKE
    FAKE = FakeNav()
    os.environ["T0_STATIC"] = "1"
    httpd, base = _server()
    try:
        result = _client(base).navigate_to(1.0, 2.0, 0.3)
        assert result.ok is True
        assert result.state == "previewed"
        assert FAKE.accepts == 0
        assert result.as_dict().get("path")
    finally:
        os.environ.pop("T0_STATIC", None)
        httpd.shutdown()


def test_static_t0_execute_true_submits_turn():
    import os

    global FAKE
    FAKE = FakeNav()
    os.environ["T0_STATIC"] = "1"
    httpd, base = _server()
    try:
        result = _client(base).navigate_to(1.0, 2.0, 0.3, execute=True)
        assert result.ok is True
        assert result.state == "succeeded"
        assert FAKE.accepts == 1
    finally:
        os.environ.pop("T0_STATIC", None)
        httpd.shutdown()


def test_reach_stops_at_02():
    global FAKE
    FAKE = FakeNav()
    FAKE.auto_succeed = False
    FAKE.position = [0.0, 0.0, 0.0]
    httpd, base = _server()

    def _approach():
        FAKE.position = [1.0, 1.85, 0.0]

    try:
        threading.Timer(0.08, _approach).start()
        result = Map2dNavClient(base, poll_s=0.04, total_s=1.5, http_timeout_s=0.4).navigate_to(
            1.0, 2.0, 0.0
        )
        assert result.ok is True
        assert result.state == "succeeded"
        assert "0.75" in result.message
        assert FAKE.state == "cancelled"
        assert FAKE.accepts == 1
    finally:
        httpd.shutdown()


def test_in_place_yaw_does_not_early_stop():
    global FAKE
    FAKE = FakeNav()
    FAKE.auto_succeed = False
    FAKE.position = [1.0, 2.0, 0.0]
    httpd, base = _server()
    try:
        result = Map2dNavClient(base, poll_s=0.04, total_s=0.25, http_timeout_s=0.4).navigate_to(
            1.0, 2.0, 1.2
        )
        assert result.state == "timed_out"
        assert FAKE.accepts == 1
    finally:
        httpd.shutdown()


def test_already_within_02_skips_submit():
    global FAKE
    FAKE = FakeNav()
    FAKE.position = [1.0, 1.85, 0.0]
    httpd, base = _server()
    try:
        result = _client(base).navigate_to(1.0, 2.0, 0.0)
        assert result.ok is True
        assert result.state == "succeeded"
        assert FAKE.accepts == 0
        assert "0.75" in result.message
    finally:
        httpd.shutdown()


def test_wait_uses_snapped_preview_goal():
    global FAKE
    FAKE = FakeNav()
    FAKE.auto_succeed = False
    FAKE.snap_to = [1.0, 1.5]
    FAKE.position = [0.0, 0.0, 0.0]
    httpd, base = _server()

    def _approach():
        FAKE.position = [1.0, 1.4, 0.0]

    try:
        threading.Timer(0.08, _approach).start()
        result = Map2dNavClient(base, poll_s=0.04, total_s=1.5, http_timeout_s=0.4).navigate_to(
            1.0, 5.0, 0.0
        )
        assert result.ok is True
        assert result.state == "progressed"
        assert result.extra["completion"] == "intermediate"
        assert result.extra["requested_goal"] == [1.0, 5.0]
        assert result.extra["accepted_goal"] == [1.0, 1.5]
        assert result.target[1] == 1.5
        assert FAKE.submitted[0]["y"] == 1.5
    finally:
        httpd.shutdown()


def test_already_within_snapped_goal_skips_submit():
    global FAKE
    FAKE = FakeNav()
    FAKE.snap_to = [1.0, 1.5]
    FAKE.position = [1.0, 1.6, 0.0]
    httpd, base = _server()
    try:
        result = _client(base).navigate_to(1.0, 5.0, 0.0)
        assert result.ok is False
        assert result.state == "blocked"
        assert result.extra["completion"] == "blocked"
        assert result.extra["requested_goal"] == [1.0, 5.0]
        assert result.extra["accepted_goal"] == [1.0, 1.5]
        assert FAKE.accepts == 0
    finally:
        httpd.shutdown()


def test_wait_stops_at_travel_budget():
    global FAKE
    FAKE = FakeNav()
    FAKE.auto_succeed = False
    FAKE.position = [0.0, 0.0, 0.0]
    httpd, base = _server()
    walked = [0.0]

    def _travel():
        walked[0] += 1.1
        return (walked[0], 0.0, 0.0)

    try:
        client = Map2dNavClient(base, poll_s=0.04, total_s=1.5, http_timeout_s=0.4)
        client.travel_pose_fn = _travel
        result = client.navigate_to(20.0, 0.0, 0.0)
        assert result.ok is True
        assert result.state == "progressed"
        assert result.extra["completion"] == "intermediate"
        assert result.extra["travel_m"] >= 5.0
        assert "travel budget" in result.message
        assert FAKE.state == "cancelled"
        assert FAKE.accepts == 1
    finally:
        httpd.shutdown()


def test_no_progress_cancels():
    global FAKE
    FAKE = FakeNav()
    FAKE.auto_succeed = False
    FAKE.position = [0.0, 0.0, 0.0]
    httpd, base = _server()
    try:
        result = Map2dNavClient(
            base,
            poll_s=0.04,
            total_s=4.0,
            http_timeout_s=0.4,
            progress_stall_s=0.2,
        ).navigate_to(1.0, 2.0, 0.0)
        assert result.ok is False
        assert result.state == "stalled"
        assert "no progress" in result.message
        assert FAKE.state == "cancelled"
    finally:
        httpd.shutdown()


def test_post_robot_plan_skips_short_and_posts_polyline():
    from lavira_m3.decision.nav_http import post_robot_plan

    seen = []

    class PlanHandler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            return

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n).decode() or "{}") if n else {}
            seen.append((self.path, body))
            raw = json.dumps({"ok": True}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), PlanHandler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    import os

    os.environ["MAP2D_NAV_URL"] = "http://127.0.0.1:%s" % httpd.server_address[1]
    try:
        post_robot_plan("small_b", 1.0, 2.0, 0.0, [])
        post_robot_plan("small_b", 1.0, 2.0, 0.0, [[0.0, 0.0], [1.0, 2.0]])
        post_robot_plan(
            "small_b", 1.0, 2.0, 0.3, [[0.0, 0.0], [0.4, 0.1], [1.0, 2.0]], "camera_init"
        )
        dense = [[i * 0.05, 0.0] for i in range(61)]
        post_robot_plan("small_b", 3.0, 0.0, 0.0, dense, "camera_init")
        assert [p for p, _ in seen] == ["/v1/plan/small_b", "/v1/plan/small_b"]
        assert seen[0][1]["path"] == [[0.0, 0.0], [0.4, 0.1], [1.0, 2.0]]
        assert seen[0][1]["frame"] == "camera_init"
        assert seen[1][1]["path"] == [[0.0, 0.0], [3.0, 0.0]]
    finally:
        os.environ.pop("MAP2D_NAV_URL", None)
        httpd.shutdown()


def test_halt_navigation_cancels_and_locks_estop():
    import os
    import tempfile

    global FAKE
    FAKE = FakeNav()
    FAKE.state = "navigating"
    latch = tempfile.NamedTemporaryFile(prefix="lavira_estop_test_", delete=False)
    latch.close()
    os.unlink(latch.name)
    os.environ["LAVIRA_ESTOP_FILE"] = latch.name
    httpd, base = _server()
    try:
        result = _client(base).halt_navigation()
        assert result.ok
        assert FAKE.state == "stopped"
        assert FAKE.estop is True
        assert os.path.exists(latch.name)
    finally:
        os.environ.pop("LAVIRA_ESTOP_FILE", None)
        if os.path.exists(latch.name):
            os.unlink(latch.name)
        httpd.shutdown()


def test_plan_callback_before_execution():
    global FAKE
    FAKE = FakeNav()
    httpd, base = _server()
    try:
        client = _client(base)
        calls = []

        def on_plan(path, goal):
            assert len(path) >= 2
            assert FAKE.accepts == 0  # Must publish before motion is submitted.
            calls.append(goal)

        client.on_plan = on_plan
        result = client.navigate_to(1.0, 2.0, 0.3)
        assert result.ok and len(calls) == 1
    finally:
        httpd.shutdown()


if __name__ == "__main__":
    test_plan_callback_before_execution()
    test_preview_not_executable()
    test_success_and_id_match()
    test_expired_preview_retries_once()
    test_busy_does_not_cancel()
    test_id_change_is_interrupt()
    test_submit_timeout_does_not_resend()
    test_static_t0_preview_does_not_submit()
    test_static_t0_execute_true_submits_turn()
    test_reach_stops_at_02()
    test_in_place_yaw_does_not_early_stop()
    test_already_within_02_skips_submit()
    test_wait_uses_snapped_preview_goal()
    test_already_within_snapped_goal_skips_submit()
    test_wait_stops_at_travel_budget()
    test_no_progress_cancels()
    test_post_robot_plan_skips_short_and_posts_polyline()
    test_halt_navigation_cancels_and_locks_estop()
    print("nav http ok")
