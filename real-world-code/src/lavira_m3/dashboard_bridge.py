"""Publish fresh camera and MPC observations independently of map exchange."""

import base64
import json
import logging
import math
import threading
import uuid

from .adapters.ros_io import stamp_ns
from .decision.model_messages import encode_image
from .settings import AGENT, FRAME, PROFILE

log = logging.getLogger(__name__)


class DashboardPublisher:
    def __init__(self, io, sensors, request):
        from visualization_msgs.msg import Marker

        self.io, self.sensors, self.request = io, sensors, request
        self.lock = threading.Lock()
        self.markers = {}
        self.session = uuid.uuid4().hex
        self.sequence = 0
        self.stopped = threading.Event()
        self.worker = None
        for field, topic in (("reference", "reference_path"), ("prediction", "predict_path")):
            io.subscribe(Marker, "/mpc/" + topic, lambda msg, key=field: self.on_marker(key, msg))

    def on_marker(self, field, message):
        stamp = stamp_ns(message.header.stamp)
        if message.header.frame_id != FRAME or not -0.05 <= (self.io.now_ns() - stamp) / 1e9 <= 1:
            return
        points = [[p.x, p.y] for p in message.points]
        if len(points) > 120 or any(not all(math.isfinite(v) for v in p) for p in points):
            return
        if message.action in (2, 3):
            points = []
        elif message.action != 0 or message.type != 4:
            return
        with self.lock:
            previous = self.markers.get(field)
            if previous and stamp <= previous[0]:
                return
            self.markers[field] = (stamp, points)

    def publish(self):
        for camera in PROFILE["directions"]:
            if self.stopped.is_set():
                return
            view = self.sensors.view(camera)
            if view:
                jpeg = base64.b64decode(encode_image(view["rgb"], 640))
                self.request("/v1/image/" + AGENT + "/" + camera, jpeg, "image/jpeg")
        pose = self.sensors.pose()
        with self.lock:
            markers = {
                key: points if -0.05 <= (self.io.now_ns() - stamp) / 1e9 <= 1 else []
                for key, (stamp, points) in self.markers.items()
            }
        self.sequence += 1
        state = dict(phase="observing", ages=dict(odom=0 if pose else 999))
        if pose:
            state["pose"] = list(pose)
        packet = dict(session=self.session, seq=self.sequence, state=state, **markers)
        self.request("/v1/telemetry/" + AGENT, json.dumps(packet, allow_nan=False).encode())

    def start(self):
        self.worker = threading.Thread(target=self.run, daemon=True)
        self.worker.start()

    def run(self):
        while not self.stopped.is_set():
            try:
                self.publish()
            except (OSError, ValueError, KeyError) as error:
                log.warning("dashboard observations unavailable: %s", error)
            self.stopped.wait(0.5)

    def close(self):
        self.stopped.set()
        if self.worker:
            self.worker.join(timeout=5)
            if self.worker.is_alive():
                log.warning("dashboard publisher still stopping after HTTP timeout")
