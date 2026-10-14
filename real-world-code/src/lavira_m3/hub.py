"""Shared maps, camera snapshots, goal board and local monitoring HTTP service."""

import json
import math
import re
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .dashboard import Dashboard
from .decision import board_hub
from .mapping.fusion import Fusion2D
from .mapping.protocol import decode
from .mapping.remote_planning import layers_for_robot
from .profiles import AGENTS
from .settings import FRAME

MAX_BODY = 27_000_000


class MapStore:
    def __init__(self):
        self.lock = threading.RLock()
        self.fusion = Fusion2D()
        self.images = {}
        self.plans = {}

    def upload(self, agent, data):
        if agent not in AGENTS:
            raise ValueError("unknown agent")
        meta, rows = decode(data)
        if meta.frame != FRAME or not -0.1 <= time.time() - meta.timestamp / 1e9 <= 5:
            raise ValueError("map frame or source timestamp invalid")
        if abs(meta.resolution - 0.1) > 1e-8 or tuple(meta.origin) != (0, 0, 0):
            raise ValueError("planning requires 0.1 m cells and zero grid origin")
        with self.lock:
            for source in self.fusion.sources.values():
                if source.ready and (source.meta.resolution, source.meta.origin) != (
                    meta.resolution,
                    meta.origin,
                ):
                    raise ValueError("map geometry mismatch")
            if not self.fusion.apply(agent, meta, rows):
                raise ValueError(self.fusion.last_reject or "map sequence mismatch")
        return dict(ok=True, seq=meta.seq)

    def scene(self):
        with self.lock:
            return dict(self.fusion.scene(), plans=dict(self.plans), agents=list(AGENTS))


STORE = MapStore()
DASHBOARD = Dashboard(STORE)


class Handler(board_hub.Handler):
    def handle_one_request(self):
        started = time.monotonic()
        self.response_bytes = 0
        self.response_code = 0
        try:
            return super().handle_one_request()
        finally:
            if hasattr(self, "path"):
                try:
                    incoming = int(self.headers.get("Content-Length", "0"))
                except (ValueError, AttributeError):
                    incoming = 0
                DASHBOARD.net.record(
                    self.client_address[0],
                    self.path,
                    max(0, incoming),
                    self.response_bytes,
                    (time.monotonic() - started) * 1000,
                    self.response_code,
                )

    def send_response(self, code, message=None):
        self.response_code = code
        return super().send_response(code, message)

    def send_header(self, keyword, value):
        if keyword.lower() == "content-length":
            self.response_bytes = int(value)
        return super().send_header(keyword, value)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        q = parse_qs(parsed.query)
        if path == "/" or re.fullmatch(r"/[a-z_]+\.(?:js|css)", path):
            asset = Path(__file__).parent / "web" / ("index.html" if path == "/" else path[1:])
            if not asset.is_file():
                return self._json(dict(error="asset not found"), 404)
            content_type = {
                ".html": "text/html; charset=utf-8",
                ".js": "text/javascript",
                ".css": "text/css",
            }[asset.suffix]
            return self._bytes(asset.read_bytes(), content_type)
        try:
            if DASHBOARD.get(self, path, q):
                return
        except (ValueError, TypeError) as error:
            return self._json(dict(error=str(error)), 400)
        with STORE.lock:
            if path == "/v1/scene":
                return self._json(STORE.scene())
            if path == "/v1/status":
                return self._json(STORE.fusion.status())
            if path == "/v1/map-updates":
                try:
                    result = STORE.fusion.updates(
                        q.get("map_id", [None])[0], int(q.get("since", [-1])[0])
                    )
                except ValueError:
                    return self._json(dict(error="invalid revision"), 400)
                return self._json(result)
            if path == "/v1/planning":
                agent = q.get("robot", [""])[0]
                if agent not in AGENTS:
                    return self._json(dict(error="unknown agent"), 400)
                if not STORE.fusion.fused.ready:
                    return self._json(dict(available=False), 503)
                result = layers_for_robot(
                    STORE.fusion.status(), STORE.fusion.fused.cells, STORE.fusion.fused.meta, agent
                )
                # Preserve source freshness: an HTTP refresh never makes old data new.
                result["stamp_ns"] = max(
                    (s.meta.timestamp for s in STORE.fusion.sources.values() if s.ready), default=0
                )
                return self._json(result)
            if path.startswith("/v1/image/"):
                key = tuple(path[len("/v1/image/") :].split("/"))
                entry = STORE.images.get(key)
                if not entry or time.monotonic() - entry[0] > 5:
                    return self._json(dict(error="camera stale"), 404)
                body = entry[1]
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
        return super().do_GET()

    def do_POST(self):
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).netloc != self.headers.get("Host"):
            return self._json(dict(error="cross-origin write rejected"), 403)
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 <= length <= MAX_BODY:
                raise ValueError("request too large")
        except ValueError:
            return self._json(dict(error="invalid content length"), 413)
        path = urlparse(self.path).path
        try:
            if DASHBOARD.post(self, path):
                return
        except (ValueError, TypeError, RuntimeError) as error:
            return self._json(dict(error=str(error)), 400)
        if path.startswith("/v1/map/"):
            try:
                result = STORE.upload(path.split("/")[-1], self.rfile.read(length))
            except (ValueError, TypeError, OverflowError) as e:
                return self._json(dict(error=str(e)), 400)
            return self._json(result)
        if path.startswith("/v1/image/"):
            parts = path.split("/")[3:]
            if (
                len(parts) != 2
                or parts[0] not in AGENTS
                or parts[1] not in AGENTS[parts[0]]["directions"]
                or length > 2_000_000
            ):
                return self._json(dict(error="invalid camera upload"), 400)
            data = self.rfile.read(length)
            if not data.startswith(b"\xff\xd8"):
                return self._json(dict(error="expected JPEG"), 400)
            with STORE.lock:
                STORE.images[tuple(parts)] = (time.monotonic(), data)
            return self._json(dict(ok=True))
        if path == "/v1/robot-plan":
            try:
                body = json.loads(self.rfile.read(length))
                agent = body["robot"]
                points = body.get("path", [])
                if (
                    agent not in AGENTS
                    or len(points) > 10000
                    or any(len(p) < 2 or not all(math.isfinite(v) for v in p[:2]) for p in points)
                ):
                    raise ValueError("invalid plan")
                with STORE.lock:
                    STORE.plans[agent] = dict(path=points, updated_at=time.time())
            except (ValueError, KeyError, TypeError):
                return self._json(dict(error="invalid plan"), 400)
            return self._json(dict(ok=True))
        return super().do_POST()


def serve(host="127.0.0.1", port=8770):
    DASHBOARD.net.bind_ip = host
    with ThreadingHTTPServer((host, port), Handler) as server:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            DASHBOARD.recordings.stop()
