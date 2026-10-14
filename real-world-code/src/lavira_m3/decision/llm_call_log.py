"""Persist each LA/VA/T0 JSON return and publish it once on this robot's ROS."""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime
from typing import Any, Optional

log = logging.getLogger("lavira_m3.llm_log")

TOPIC_FMT = "/lavira_m3/%s/llm_call"


def usage_tokens(usage: Any) -> tuple:
    if usage is None:
        return None, None, None
    if isinstance(usage, dict):
        prompt = usage.get("prompt_tokens")
        completion = usage.get("completion_tokens")
        total = usage.get("total_tokens")
    else:
        prompt = getattr(usage, "prompt_tokens", None)
        completion = getattr(usage, "completion_tokens", None)
        total = getattr(usage, "total_tokens", None)
    try:
        prompt = None if prompt is None else int(prompt)
    except (TypeError, ValueError):
        prompt = None
    try:
        completion = None if completion is None else int(completion)
    except (TypeError, ValueError):
        completion = None
    try:
        total = None if total is None else int(total)
    except (TypeError, ValueError):
        total = None
    return prompt, completion, total


def tokens_per_second(
    completion_tokens: Optional[int], duration_ms: Optional[float]
) -> Optional[float]:
    if completion_tokens is None or duration_ms is None:
        return None
    try:
        seconds = float(duration_ms) / 1000.0
    except (TypeError, ValueError):
        return None
    if seconds <= 0:
        return None
    return round(float(completion_tokens) / seconds, 3)


def iso_local(ts: float) -> str:
    return datetime.fromtimestamp(float(ts)).astimezone().isoformat(timespec="milliseconds")


class LlmCallLogger:
    def __init__(self, robot: str, folder: str, session: str):
        self.robot = str(robot)
        self.session = str(session)
        self.folder = folder
        self.path = os.path.join(folder, "llm_calls.jsonl")
        self.topic = TOPIC_FMT % self.robot
        self._lock = threading.Lock()
        self._ros1 = None
        self._ros2_node = None
        self._ros2_pub = None
        os.makedirs(folder, exist_ok=True)
        self.advertise()

    def write(
        self,
        kind: str,
        raw: str,
        returned_at: float,
        duration_ms: float,
        usage: Any = None,
        model: str = "",
        backend: str = "",
        step: Optional[int] = None,
    ) -> dict:
        prompt_tokens, completion_tokens, total_tokens = usage_tokens(usage)
        rec = {
            "kind": str(kind),
            "robot": self.robot,
            "session": self.session,
            "step": step,
            "model": model,
            "backend": backend,
            "raw": raw if raw is not None else "",
            "returned_at": round(float(returned_at), 3),
            "returned_at_iso": iso_local(returned_at),
            "duration_ms": round(float(duration_ms), 1),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "tokens_per_s": tokens_per_second(completion_tokens, duration_ms),
            "topic": self.topic,
        }
        line = json.dumps(rec, ensure_ascii=False)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        self._publish(line)
        return rec

    def advertise(self) -> bool:
        """Hang the topic as soon as ROS is up so rosbag / ros2 bag can subscribe."""
        if os.environ.get("NAV_ROS", "ros2") == "ros2":
            return self._ensure_ros2(announce=True)
        return self._ensure_ros1(announce=True)

    def _ready_payload(self) -> str:
        return json.dumps(
            {
                "kind": "ready",
                "robot": self.robot,
                "session": self.session,
                "topic": self.topic,
                "raw": "",
            },
            ensure_ascii=False,
        )

    def _publish(self, payload: str) -> None:
        if os.environ.get("NAV_ROS", "ros2") == "ros2":
            self._publish_ros2(payload)
        else:
            self._publish_ros1(payload)

    def _ensure_ros1(self, announce: bool = False) -> bool:
        try:
            import rospy
            from std_msgs.msg import String
        except Exception:
            return False
        try:
            if not rospy.core.is_initialized():
                return False
            created = False
            if self._ros1 is None:
                self._ros1 = rospy.Publisher(self.topic, String, queue_size=10, latch=True)
                created = True
                log.info("advertised ROS1 %s", self.topic)
            if announce and created:
                self._ros1.publish(String(data=self._ready_payload()))
            return True
        except Exception:
            log.exception("ROS1 advertise %s failed", self.topic)
            return False

    def _publish_ros1(self, payload: str) -> None:
        if not self._ensure_ros1():
            return
        try:
            from std_msgs.msg import String

            self._ros1.publish(String(data=payload))
        except Exception:
            log.exception("ROS1 publish %s failed", self.topic)

    def _ensure_ros2(self, announce: bool = False) -> bool:
        try:
            import rclpy
            from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
            from std_msgs.msg import String
        except Exception:
            return False
        try:
            if not rclpy.ok():
                return False
            created = False
            if self._ros2_pub is None:
                # Use reliable volatile telemetry QoS.
                qos = QoSProfile(
                    depth=10,
                    reliability=ReliabilityPolicy.RELIABLE,
                    durability=DurabilityPolicy.VOLATILE,
                    history=HistoryPolicy.KEEP_LAST,
                )
                self._ros2_node = rclpy.create_node("lavira_m3_llm_%s" % self.robot)
                self._ros2_pub = self._ros2_node.create_publisher(String, self.topic, qos)
                created = True
                log.info("advertised ROS2 %s", self.topic)
            if announce and created:
                msg = String()
                msg.data = self._ready_payload()
                self._ros2_pub.publish(msg)
            return True
        except Exception:
            log.exception("ROS2 advertise %s failed", self.topic)
            return False

    def _publish_ros2(self, payload: str) -> None:
        if not self._ensure_ros2():
            return
        try:
            from std_msgs.msg import String

            msg = String()
            msg.data = payload
            self._ros2_pub.publish(msg)
        except Exception:
            log.exception("ROS2 publish %s failed", self.topic)
