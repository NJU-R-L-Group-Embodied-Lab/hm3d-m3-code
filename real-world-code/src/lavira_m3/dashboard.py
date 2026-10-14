"""Read-only dashboard snapshots and bounded recording HTTP endpoints."""

import base64
import copy
import os
import tempfile
import time
from collections import deque
from pathlib import Path

from .decision import board_state
from .mapping.navigation_telemetry import NavigationTelemetry
from .mapping.netmon import NetMonitor
from .profiles import AGENTS
from .recordings import DashboardRecordings
from .settings import FRAME, SAFETY


class Dashboard:
    def __init__(self, store, recording_root=None):
        self.store = store
        self.trails = {agent: deque(maxlen=3000) for agent in AGENTS}
        self.telemetry = NavigationTelemetry()
        self.planning = {}
        self.net = NetMonitor(dict.fromkeys(AGENTS, ""), "127.0.0.1")
        root = recording_root or os.environ.get("LAVIRA_RECORD_DIR", ".runtime/recordings")
        self.recordings = DashboardRecordings(
            Path(root).resolve(), lambda: self.snapshot(images=True)
        )

    def config(self):
        return dict(
            agents=AGENTS,
            frame=FRAME,
            safety=SAFETY,
            recording=dict(
                interval=self.recordings.interval,
                max_frames=self.recordings.max_frames,
                max_bytes=self.recordings.max_bytes,
            ),
        )

    def snapshot(self, images=False, selected=""):
        with self.store.lock:
            scene = self.store.scene()
            status = self.store.fusion.status()
            cameras = {}
            stored_images = {}
            for agent, profile in AGENTS.items():
                pose = status["sources"].get(agent, {}).get("pose", {})
                position = pose.get("position")
                if pose.get("live") and position:
                    trail = self.trails[agent]
                    if (
                        not trail
                        or sum((a - b) ** 2 for a, b in zip(position[:2], trail[-1])) > 0.0001
                    ):
                        trail.append(position[:2])
                cameras[agent] = {}
                stored_images[agent] = {}
                for camera in profile["directions"]:
                    entry = self.store.images.get((agent, camera))
                    age = time.monotonic() - entry[0] if entry else None
                    live = age is not None and age <= 5
                    cameras[agent][camera] = dict(online=live, age_s=age)
                    if images and live:
                        stored_images[agent][camera] = (
                            "data:image/jpeg;base64," + base64.b64encode(entry[1]).decode()
                        )
            scene["trails"] = {agent: list(trail) for agent, trail in self.trails.items()}
            planning = copy.deepcopy(self.planning)
        with board_state.STORE.lock:
            board = copy.deepcopy(board_state.STORE.snapshot())
            if images:
                for goal in board["goals"]:
                    reference = goal.get("reference_image", "")
                    blob = board_state.STORE.refs.get(reference.rsplit("/", 1)[-1])
                    if blob:
                        goal["reference_image"] = (
                            "data:image/jpeg;base64," + base64.b64encode(blob).decode()
                        )
        return copy.deepcopy(
            dict(
                t=time.time(),
                scene=scene,
                status=status,
                board=board,
                cameras=cameras,
                images=stored_images,
                planning=planning,
                telemetry=self.telemetry.snapshot(selected, include_all=images),
            )
        )

    def get(self, handler, path, query):
        def first(key, default=""):
            return query.get(key, [default])[0]

        if path == "/v1/dashboard/config":
            handler._json(self.config())
        elif path == "/v1/dashboard":
            handler._json(self.snapshot(selected=first("robot")))
        elif path == "/v1/planning-grid":
            from .mapping.remote_planning import layers_for_robot

            agent = first("robot")
            if agent not in AGENTS:
                raise ValueError("unknown agent")
            with self.store.lock:
                if not self.store.fusion.fused.ready:
                    raise ValueError("waiting for a fresh planning map")
                result = layers_for_robot(
                    self.store.fusion.status(),
                    self.store.fusion.fused.cells,
                    self.store.fusion.fused.meta,
                    agent,
                )
                result.pop("planning", None)
                result.pop("environment", None)
                result.pop("peers", None)
                self.planning[agent] = result
            handler._json(result)
        elif path == "/v1/net":
            handler._json(self.net.snapshot())
        elif path == "/v1/record/status":
            handler._json(self.recordings.status())
        elif path == "/v1/recordings":
            handler._json(dict(recordings=self.recordings.list()))
        elif path == "/v1/record/frame":
            handler._json(self.recordings.frame(first("id"), int(first("index", "0"))))
        elif path == "/v1/record/download":
            recording_id = first("id")
            with tempfile.TemporaryFile() as output:
                self.recordings.export(recording_id, output)
                length = output.tell()
                output.seek(0)
                handler.send_response(200)
                handler.send_header("Content-Type", "application/zip")
                handler.send_header("Content-Length", str(length))
                handler.send_header(
                    "Content-Disposition", f'attachment; filename="{recording_id}.zip"'
                )
                handler.end_headers()
                while chunk := output.read(65536):
                    handler.wfile.write(chunk)
        else:
            return False
        return True

    def post(self, handler, path):
        if path == "/v1/record/start":
            handler._json(self.recordings.start())
        elif path == "/v1/record/stop":
            handler._json(self.recordings.stop())
        elif path == "/v1/clear":
            from .mapping.fusion import Fusion2D

            with self.store.lock:
                self.store.fusion = Fusion2D()
                self.store.plans.clear()
                self.planning.clear()
                for trail in self.trails.values():
                    trail.clear()
            handler._json(dict(ok=True))
        elif path.startswith("/v1/telemetry/"):
            result = self.telemetry.ingest(path.rsplit("/", 1)[1], handler._read_body())
            handler._json(result, 200 if result["ok"] else 409)
        else:
            return False
        return True
