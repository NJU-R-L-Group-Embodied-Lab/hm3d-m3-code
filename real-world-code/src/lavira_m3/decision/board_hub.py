"""HTTP routes for the goal board and initial team coordination."""

from __future__ import annotations

import json
import logging
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict
from urllib.parse import parse_qs, urlparse

from . import board_state
from .board_state import log


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        if "/api/" in self.path:
            return
        super().log_message(fmt, *args)

    def _json(self, obj: Dict[str, Any], code: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> Dict[str, Any]:
        n = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(n) if n else b"{}"
        try:
            return json.loads(raw.decode() or "{}")
        except json.JSONDecodeError:
            return {}

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _bytes(self, body: bytes, ctype: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/board":
            return self._json(board_state.STORE.snapshot())
        if parsed.path == "/api/board/prompt":
            robot = (parse_qs(parsed.query).get("robot") or [""])[0]
            return self._json(board_state.STORE.prompt(robot))
        if parsed.path.startswith("/api/board/ref/"):
            gid = parsed.path.rsplit("/", 1)[-1]
            data = board_state.STORE.get_ref(gid)
            if not data:
                return self._json({"ok": False, "error": "reference image not found"}, 404)
            return self._bytes(data, "image/jpeg")
        if parsed.path.startswith("/api/board/history/"):
            parts = [p for p in parsed.path.split("/") if p]
            if len(parts) != 5:
                return self._json(
                    {"ok": False, "error": "history path is /api/board/history/<robot>/<id>"}, 404
                )
            try:
                waypoint_id = int(parts[4])
            except ValueError:
                return self._json({"ok": False, "error": "waypoint id must be int"}, 404)
            data = board_state.STORE.get_history_image(parts[3], waypoint_id)
            if not data:
                return self._json({"ok": False, "error": "history image not found"}, 404)
            return self._bytes(data, "image/jpeg")
        if parsed.path == "/api/t0/status":
            robot = (parse_qs(parsed.query).get("robot") or [""])[0]
            return self._json(board_state.STORE.t0_status(robot))
        if parsed.path == "/api/t0/bundle":
            robot = (parse_qs(parsed.query).get("robot") or [""])[0]
            result = board_state.STORE.t0_bundle(robot)
            return self._json(result, 200 if result.get("ok") else 403)
        if parsed.path in ("/", "/health"):
            return self._json({"ok": True, "service": "lavira_m3-board"})
        return self._json({"ok": False, "error": "not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        body = self._read_body()
        if path == "/api/board/reset":
            return self._json(board_state.STORE.reset_board(body.get("t0_robots")))
        if path == "/api/board/clear-runtime":
            result = board_state.STORE.clear_runtime_cache()
            return self._json(result, 200 if result.get("ok") is not False else 409)
        if path == "/api/board/init":
            result = board_state.STORE.init_goals(
                body.get("goals") or [], str(body.get("robot") or "")
            )
            return self._json(result, 200 if result.get("ok") else 400)
        if path == "/api/board/progress":
            result = board_state.STORE.post_progress(
                str(body.get("robot") or ""),
                str(body.get("text") or ""),
                int(body.get("step") or 0),
            )
            return self._json(result, 200 if result.get("ok") else 400)
        if path == "/api/board/found":
            result = board_state.STORE.post_found(
                str(body.get("robot") or ""),
                str(body.get("goal_id") or ""),
                xy=body.get("xy"),
            )
            return self._json(result, 200 if result.get("ok") else 400)
        if path == "/api/board/trace":
            result = board_state.STORE.post_trace(str(body.get("robot") or ""), body)
            return self._json(result, 200 if result.get("ok") else 400)
        if path == "/api/board/history":
            result = board_state.STORE.post_history(str(body.get("robot") or ""), body)
            return self._json(result, 200 if result.get("ok") else 400)
        if path == "/api/t0/hello":
            result = board_state.STORE.t0_hello(str(body.get("robot") or ""))
            return self._json(result, 200 if result.get("ok") else 400)
        if path == "/api/t0/observe":
            result = board_state.STORE.t0_observe(
                str(body.get("robot") or ""),
                body.get("pose"),
                list(body.get("dirs") or []),
                list(body.get("frames") or []),
            )
            return self._json(result, 200 if result.get("ok") else 400)
        if path == "/api/t0/lock":
            result = board_state.STORE.t0_lock(str(body.get("robot") or ""))
            return self._json(result, 200 if result.get("ok") else 409)
        if path == "/api/t0/assign":
            result = board_state.STORE.t0_assign(
                str(body.get("robot") or ""),
                dict(body.get("assignments") or {}),
                str(body.get("team_reasoning") or ""),
                str(body.get("raw") or ""),
                body.get("panoramic_observation"),
                str(body.get("map_understanding") or ""),
                body.get("la_time_ms"),
                body.get("image_goal_scenes"),
            )
            return self._json(result, 200 if result.get("ok") else 400)
        if path == "/api/t0/fail":
            result = board_state.STORE.t0_fail(
                str(body.get("robot") or ""), str(body.get("error") or "")
            )
            return self._json(result, 200 if result.get("ok") else 400)
        return self._json({"ok": False, "error": "not found"}, 404)


def serve(host: str = "127.0.0.1", port: int = 8769) -> None:
    httpd = ThreadingHTTPServer((host, port), Handler)
    log.info("board hub on %s:%s", host, port)
    httpd.serve_forever()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    host = os.environ.get("BOARD_HOST", "127.0.0.1")
    port = int(os.environ.get("BOARD_PORT", "8769"))
    serve(host, port)


if __name__ == "__main__":
    main()
