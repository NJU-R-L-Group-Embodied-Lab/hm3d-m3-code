"""Blocking workflow interface over ROS2 navigation actions with bounded waits."""

import math
import threading
import time

from nav2_msgs.action import ComputePathToPose, NavigateToPose
from rclpy.action import ActionClient
from std_srvs.srv import SetBool

from ...contracts import NavResult
from ...settings import FRAME


class NavClient:
    def __init__(self, io, control=False):
        self.io = io
        self.control = control
        self.handle = None
        self.lock = threading.RLock()
        self.compute = ActionClient(io.node, ComputePathToPose, "compute_path_to_pose")
        self.move = ActionClient(io.node, NavigateToPose, "navigate_to_pose")
        self.estop = io.node.create_client(SetBool, "/navigation/estop")
        self.last_preview = {}

    def wait(self, future, timeout):
        ready = threading.Event()
        future.add_done_callback(lambda _: ready.set())
        if not ready.wait(timeout):
            raise TimeoutError("navigation response deadline")
        return future.result()

    def goal(self, kind, x, y, yaw):
        if not all(math.isfinite(v) for v in (x, y, yaw)):
            raise ValueError("nonfinite goal")
        g = kind.Goal()
        pose = g.goal if kind is ComputePathToPose else g.pose
        pose.header.frame_id = FRAME
        pose.header.stamp = self.io.stamp()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        pose.pose.orientation.z = math.sin(yaw / 2)
        pose.pose.orientation.w = math.cos(yaw / 2)
        return g

    def send(self, client, goal, timeout):
        if not client.wait_for_server(timeout_sec=2.0):
            raise TimeoutError("navigation action unavailable")
        future = client.send_goal_async(goal)
        try:
            handle = self.wait(future, 2.0)
        except TimeoutError:
            # A late acceptance must be cancelled; never leave an orphan goal running.
            future.add_done_callback(
                lambda f: f.result().cancel_goal_async() if f.result().accepted else None
            )
            raise
        if not handle.accepted:
            raise ValueError("navigation goal rejected")
        with self.lock:
            self.handle = handle
        try:
            return self.wait(handle.get_result_async(), timeout)
        except BaseException:
            self.wait(handle.cancel_goal_async(), 2.0)
            raise
        finally:
            with self.lock:
                self.handle = None

    def preview_plan(self, x, y, yaw, start=None, timeout_s=6.0):
        self.last_preview = {}
        try:
            result = self.send(self.compute, self.goal(ComputePathToPose, x, y, yaw), timeout_s)
            if result.status != 4:
                return []
            path = [[p.pose.position.x, p.pose.position.y] for p in result.result.path.poses]
            self.last_preview = dict(path=path, goal=[x, y], frame=FRAME)
            return path
        except (ValueError, TimeoutError):
            return []

    def navigate_to(self, x, y, yaw, **kwargs):
        if not self.control:
            return NavResult(False, "disabled", "motion requires --control")
        try:
            deadline = getattr(self, "round_deadline", time.monotonic() + 30) - time.monotonic()
            result = self.send(self.move, self.goal(NavigateToPose, x, y, yaw), max(0.1, deadline))
            return NavResult(
                result.status == 4,
                "succeeded" if result.status == 4 else "aborted",
                getattr(result.result, "error_msg", ""),
                extra=self.last_preview,
            )
        except (TimeoutError, ValueError) as e:
            return NavResult(False, "failed", str(e))

    def cancel(self):
        with self.lock:
            handle = self.handle
        if handle:
            try:
                self.wait(handle.cancel_goal_async(), 2.0)
            except TimeoutError:
                return NavResult(False, "cancel_failed", "cancel deadline")
        return NavResult(True, "cancelled", "navigation cancelled")

    def lock_estop(self):
        request = SetBool.Request()
        request.data = True
        if not self.estop.wait_for_service(timeout_sec=1.0):
            return NavResult(False, "unavailable", "estop service unavailable")
        result = self.wait(self.estop.call_async(request), 2.0)
        return NavResult(result.success, "estop", result.message)

    def close(self):
        self.cancel()
