"""Fresh standard-message observations; camera pixels stay independent of ROS."""

import math
import threading
import time

import numpy as np

from .adapters.ros_io import stamp_ns
from .profiles import CONFIG
from .settings import BODY_FRAME, FRAME, PROFILE, SAFETY, TOPICS


def image_array(message):
    enc = message.encoding
    formats = {
        "rgb8": ("u1", 3),
        "bgr8": ("u1", 3),
        "mono8": ("u1", 1),
        "16UC1": ("u2", 1),
        "32FC1": ("f4", 1),
    }
    if enc not in formats:
        raise ValueError("unsupported image encoding: " + enc)
    dtype, channels = formats[enc]
    dtype = np.dtype((">" if message.is_bigendian else "<") + dtype)
    height, width, step = int(message.height), int(message.width), int(message.step)
    if (
        not (0 < height <= 8192 and 0 < width <= 8192)
        or step < width * channels * dtype.itemsize
        or len(message.data) != height * step
    ):
        raise ValueError("invalid image layout")
    values = np.ndarray(
        (height, width, channels),
        dtype=dtype,
        buffer=message.data,
        strides=(step, dtype.itemsize * channels, dtype.itemsize),
    ).copy()
    if channels == 1:
        values = values[:, :, 0]
    if enc == "rgb8":
        values = values[:, :, ::-1].copy()  # Existing decision encoder accepts BGR.
    if enc == "16UC1":
        values = values.astype(np.float32) * 0.001
    return values


class Sensors:
    def __init__(self, io):
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import CameraInfo, Image

        self.io = io
        self.lock = threading.RLock()
        self.odometry = None
        self.odom_at = 0
        self.images = {}
        self.intrinsics = {}
        io.subscribe(Odometry, TOPICS["odometry"], self.on_odom)
        cameras = CONFIG.get("cameras", {})
        for direction in PROFILE["directions"]:
            topics = cameras.get(direction, {})
            io.subscribe(
                Image,
                topics.get("rgb", "/camera/" + direction + "/color/image_raw"),
                lambda m, d=direction: self.on_image(d, "rgb", m),
            )
            io.subscribe(
                Image,
                topics.get("depth", "/camera/" + direction + "/depth/image_raw"),
                lambda m, d=direction: self.on_image(d, "depth", m),
            )
            io.subscribe(
                CameraInfo,
                topics.get("info", "/camera/" + direction + "/color/camera_info"),
                lambda m, d=direction: self.on_info(d, m),
            )

    def fresh(self, stamp, limit):
        return -0.05 <= (self.io.now_ns() - stamp_ns(stamp)) / 1e9 <= limit

    def on_odom(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        values = (p.x, p.y, p.z, q.x, q.y, q.z, q.w)
        with self.lock:
            if (
                m.header.frame_id != FRAME
                or m.child_frame_id != BODY_FRAME
                or not all(math.isfinite(v) for v in values)
                or abs(sum(v * v for v in values[3:]) - 1) > 0.02
                or not self.fresh(m.header.stamp, SAFETY["odom_timeout"])
            ):
                self.odometry = None
                return
            if self.odometry and stamp_ns(m.header.stamp) <= stamp_ns(self.odometry.header.stamp):
                return
            self.odometry = m
            self.odom_at = time.monotonic()

    def pose(self):
        with self.lock:
            m = self.odometry
            if (
                m is None
                or time.monotonic() - self.odom_at > SAFETY["odom_timeout"]
                or not self.fresh(m.header.stamp, SAFETY["odom_timeout"])
            ):
                return None
            p, q = m.pose.pose.position, m.pose.pose.orientation
            return (
                p.x,
                p.y,
                math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)),
            )

    def on_info(self, d, m):
        k = list(m.K if hasattr(m, "K") else m.k)
        if len(k) == 9 and all(math.isfinite(v) for v in k) and k[0] > 0 and k[4] > 0:
            with self.lock:
                self.intrinsics[d] = dict(fx=k[0], fy=k[4], cx=k[2], cy=k[5])

    def on_image(self, d, kind, m):
        if not self.fresh(m.header.stamp, 2.0):
            return
        try:
            array = image_array(m)
        except (ValueError, TypeError):
            return
        if kind == "rgb" and array.ndim != 3:
            return
        if kind == "depth" and array.ndim != 2:
            return
        with self.lock:
            key = (d, kind)
            ns = stamp_ns(m.header.stamp)
            if key in self.images and ns <= self.images[key][1]:
                return
            self.images[key] = (array, ns, time.monotonic())

    def view(self, d):
        with self.lock:
            rgb = self.images.get((d, "rgb"))
            depth = self.images.get((d, "depth"))
            if (
                not rgb
                or time.monotonic() - rgb[2] > 2
                or not -0.05 <= (self.io.now_ns() - rgb[1]) / 1e9 <= 2
            ):
                return None
            result = dict(name=d, rgb=rgb[0].copy(), **self.intrinsics.get(d, {}))
            if depth and abs(depth[1] - rgb[1]) <= 200_000_000:
                result["depth"] = depth[0].copy()
            return result
