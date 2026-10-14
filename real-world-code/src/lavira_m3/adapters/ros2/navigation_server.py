#!/usr/bin/env python3
"""Existing LaViRA action contract, EGO planning, no controller here."""

import argparse
import json
import math
import threading
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import ComputePathToPose, NavigateToPose
from nav_msgs.msg import OccupancyGrid, Odometry
from nav_msgs.msg import Path as RosPath
from quadrotor_msgs.msg import GoalSet
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
from std_msgs.msg import String, UInt64
from tf2_ros import Buffer, TransformListener
from traj_utils.msg import PolyTraj

from .gate import BODY_FRAME, FRAME, PROFILE, TOPICS, finite_pose, stamp_ns, tf_matches


def sample_trajectory(m):
    if (
        m.order != 5
        or not 1 <= len(m.duration) <= 500
        or any(len(c) != 6 * len(m.duration) for c in (m.coef_x, m.coef_y, m.coef_z))
        or any(not math.isfinite(d) or not 0 < d <= 180 for d in m.duration)
        or sum(m.duration) > 300
        or any(not math.isfinite(v) for c in (m.coef_x, m.coef_y, m.coef_z) for v in c)
    ):
        return []
    out = []
    for i, d in enumerate(m.duration):
        if not math.isfinite(d) or d <= 0:
            return []
        for j in range(max(1, math.ceil(d / 0.05)) + 1):
            t = d * j / max(1, math.ceil(d / 0.05))
            xy = []
            for cs in (m.coef_x, m.coef_y):
                v = 0.0
                for c in cs[6 * i : 6 * i + 6]:
                    v = v * t + c
                xy.append(float(v))
            if not all(math.isfinite(x) for x in xy):
                return []
            if not out or xy != out[-1]:
                out.append(xy)
    return out


class Navigation(Node):
    def __init__(self, control=False):
        super().__init__("lavira_ego_navigation")
        self.control = control
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self)
        self.lock = threading.Lock()
        self.busy = False
        self.task = 0
        self.cancelled = set()
        self.traj = None
        self.odom = None
        self.grid = None
        self.map_stamp = 0
        self.map_ok = False
        self.state_pub = self.create_publisher(String, "/navigation/state", 1)
        self.create_timer(
            0.1,
            lambda: self.state_pub.publish(
                String(
                    data=json.dumps(
                        dict(
                            ready=bool(self.ready()),
                            control=self.control,
                            busy=self.busy,
                            tf_ok=tf_matches(self.tf, self.odom, self.now_ns()),
                            map_ok=self.map_ok,
                            map_age=(self.now_ns() - self.map_stamp) / 1e9,
                            odom_age=(self.now_ns() - stamp_ns(self.odom.header)) / 1e9
                            if self.odom
                            else None,
                            goal_subscribers=self.count_subscribers("/navigation/goal"),
                        )
                    )
                )
            ),
        )
        self.goal_pub = self.create_publisher(GoalSet, "/navigation/goal", 10)
        self.cancel_pub = self.create_publisher(UInt64, "/navigation/cancel", 10)
        self.report = self.create_publisher(
            String,
            "/navigation/report",
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.path_pub = self.create_publisher(RosPath, "/navigation/preview_path", 1)
        self.create_subscription(
            Odometry, TOPICS["odometry"], self.on_odom, qos_profile_sensor_data
        )
        self.create_subscription(
            OccupancyGrid,
            "/map2d/planning",
            lambda m: setattr(self, "grid", m),
            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        self.create_subscription(String, "/navigation/map_applied", self.on_map, 1)
        self.create_subscription(PolyTraj, "/ego/trajectory", self.on_traj, 10)
        self.create_subscription(
            UInt64, "/navigation/cancel", lambda m: self.cancelled.add(m.data), 10
        )
        group = ReentrantCallbackGroup()
        self.preview = ActionServer(
            self,
            ComputePathToPose,
            "compute_path_to_pose",
            self.preview_execute,
            goal_callback=self.preview_goal,
            cancel_callback=self.cancel_action,
            callback_group=group,
        )
        self.navigate = ActionServer(
            self,
            NavigateToPose,
            "navigate_to_pose",
            self.navigate_execute,
            goal_callback=self.navigate_goal,
            cancel_callback=self.cancel_action,
            callback_group=group,
        )

    def now_ns(self):
        return self.get_clock().now().nanoseconds

    def on_odom(self, m):
        self.odom = (
            m
            if m.header.frame_id == FRAME and m.child_frame_id == BODY_FRAME and finite_pose(m)
            else None
        )

    def on_map(self, m):
        try:
            v = json.loads(m.data)
            ns = int(v["stamp_ns"])
            accepted = v["accepted"] is True
            if accepted and ns >= self.map_stamp:
                self.map_ok = True
                if ns > self.map_stamp:
                    self.map_stamp = ns
            elif not accepted:
                self.map_ok = False
        except (ValueError, KeyError, TypeError):
            self.map_ok = False

    def on_traj(self, m):
        if (
            m.task_id == self.task
            and m.task_id not in self.cancelled
            and m.header.frame_id == FRAME
            and (self.traj is None or m.traj_id > self.traj.traj_id)
        ):
            self.traj = m

    def ready(self):
        return (
            self.odom is not None
            and -0.05
            <= (self.now_ns() - stamp_ns(self.odom.header)) / 1e9
            <= PROFILE["odom_timeout"]
            and self.map_ok
            and -0.05 <= (self.now_ns() - self.map_stamp) / 1e9 <= PROFILE["map_timeout"]
            and self.count_subscribers("/navigation/goal") >= 3
            and tf_matches(self.tf, self.odom, self.now_ns())
        )

    def reserve(self, pose, execute):
        with self.lock:
            p, q = pose.pose.position, pose.pose.orientation
            valid = (
                pose.header.frame_id == FRAME
                and all(math.isfinite(x) for x in (p.x, p.y, p.z, q.x, q.y, q.z, q.w))
                and abs(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w - 1) < 0.02
            )
            if self.busy or not valid or not self.ready() or (execute and not self.control):
                self.get_logger().warn(
                    "goal reject execute=%s busy=%s valid=%s control=%s %s"
                    % (execute, self.busy, valid, self.control, json.dumps(self._health()))
                )
                return GoalResponse.REJECT
            self.busy = True
            return GoalResponse.ACCEPT

    def preview_goal(self, goal):
        # Explicit starts must match measured localization, never redefine its coordinates.
        if goal.use_start:
            if self.odom is None or goal.start.header.frame_id != FRAME:
                return GoalResponse.REJECT
            a, b = goal.start.pose.position, self.odom.pose.pose.position
            if math.hypot(a.x - b.x, a.y - b.y) > 0.05:
                return GoalResponse.REJECT
        return self.reserve(goal.goal, False)

    def navigate_goal(self, goal):
        return self.reserve(goal.pose, True)

    def cancel_action(self, handle):
        # Bind cancel to this handle's task. Using self.task here used to cancel
        # a newer NavigateToPose when a leftover compute_path was torn down.
        task = getattr(handle, "_lavira_task", None)
        if task:
            self.cancelled.add(task)
            if rclpy.ok():
                self.cancel_pub.publish(UInt64(data=task))
            self.get_logger().warn("cancel_action task=%s current_task=%s" % (task, self.task))
        else:
            self.get_logger().warn(
                "cancel_action ignored unbound handle current_task=%s" % self.task
            )
        return CancelResponse.ACCEPT

    def begin(self, pose, execute):
        self.task = max(time.time_ns(), self.task + 1)
        self.traj = None
        m = GoalSet()
        m.header.frame_id = FRAME
        m.header.stamp = self.get_clock().now().to_msg()
        m.task_id = self.task
        m.execute = execute
        m.goal = [float(pose.pose.position.x), float(pose.pose.position.y), 0.0]
        self.goal_pub.publish(m)
        return [float(v) for v in m.goal[:2]], self.task

    def finish(self, task=None, cancel=True):
        # Only retire this callback's task. A late preview finally:finish() used to
        # cancel whatever self.task is now, which aborts the following execute.
        task = self.task if task is None else task
        publish = False
        with self.lock:
            if cancel and task:
                self.cancelled.add(task)
                publish = True
            if self.task == task:
                self.busy = False
        if publish and rclpy.ok():
            self.cancel_pub.publish(UInt64(data=task))

    def _health(self):
        odom_age = (self.now_ns() - stamp_ns(self.odom.header)) / 1e9 if self.odom else None
        map_age = (self.now_ns() - self.map_stamp) / 1e9 if self.map_stamp else None
        return dict(
            ready=bool(self.ready()),
            map_ok=self.map_ok,
            map_age=map_age,
            odom_ok=self.odom is not None,
            odom_age=odom_age,
            tf_ok=tf_matches(self.tf, self.odom, self.now_ns()),
            goal_subscribers=self.count_subscribers("/navigation/goal"),
        )

    def wait_trajectory(self, handle, task=None, timeout=6):
        task = self.task if task is None else task
        end = time.monotonic() + timeout
        last_ready = True
        while rclpy.ok() and time.monotonic() < end:
            if handle.is_cancel_requested or task in self.cancelled:
                self.get_logger().warn(
                    "wait_trajectory stop task=%s reason=cancel cancel_req=%s in_set=%s health=%s"
                    % (
                        task,
                        bool(handle.is_cancel_requested),
                        task in self.cancelled,
                        json.dumps(self._health()),
                    )
                )
                return None
            ready = self.ready()
            if ready != last_ready:
                self.get_logger().warn(
                    "wait_trajectory ready=%s task=%s health=%s"
                    % (ready, task, json.dumps(self._health()))
                )
                last_ready = ready
            if not ready:
                return None
            if self.traj is not None:
                return self.traj
            time.sleep(0.01)
        self.get_logger().warn(
            "wait_trajectory timeout task=%s health=%s" % (task, json.dumps(self._health()))
        )
        return None

    def cell(self, xy):
        g = self.grid
        if g is None:
            return dict(state="unavailable")
        o = g.info.origin
        q = o.orientation
        dx = xy[0] - o.position.x
        dy = xy[1] - o.position.y
        yaw = 2 * math.atan2(q.z, q.w)
        c, s = math.cos(yaw), math.sin(yaw)
        ix = math.floor((c * dx + s * dy) / g.info.resolution)
        iy = math.floor((-s * dx + c * dy) / g.info.resolution)
        if not (0 <= ix < g.info.width and 0 <= iy < g.info.height):
            return dict(state="outside", index=[ix, iy])
        v = g.data[iy * g.info.width + ix]
        return dict(
            state="unknown"
            if v < 0
            else "blocked"
            if v == 100
            else "soft_traversable"
            if v
            else "free",
            value=v,
            index=[ix, iy],
        )

    def describe(self, request, m=None, phase="no_solution", abort_reason=None):
        path = sample_trajectory(m) if m else []
        p = self.odom.pose.pose.position if self.odom else None
        effective = [float(v) for v in m.effective_goal[:2]] if m else None
        record = dict(
            task_id=self.task,
            phase=phase,
            requested=request,
            target_cell=self.cell(request),
            arrival_tolerance=PROFILE["arrival_tolerance"],
            path_generated=bool(path),
            effective_goal=effective,
            goal_adjusted=bool(effective and math.dist(request, effective) > 0.001),
            preview_endpoint=path[-1] if path else None,
            preview_endpoint_error=math.dist(request, path[-1]) if path else None,
            local_trajectory_partial=bool(path and math.dist(path[-1], effective) > 0.01),
            final_position=[p.x, p.y] if p else None,
            final_localization_error=math.dist(request, (p.x, p.y)) if p else None,
            effective_goal_error=math.dist(effective, (p.x, p.y)) if p and effective else None,
            executed_arrival=phase == "arrived",
            soft_cost_policy="1..99 traversable; no weighted soft cost",
            frame=FRAME,
            abort_reason=abort_reason,
            health=self._health(),
        )
        self.report.publish(String(data=json.dumps(record)))
        return record, path

    def preview_execute(self, h):
        result = ComputePathToPose.Result()
        request, task = self.begin(h.request.goal, False)
        try:
            h._lavira_task = task
            m = self.wait_trajectory(h, task)
            record, path = self.describe(request, m, "path_generated" if m else "no_solution")
            if path:
                result.path.header.frame_id = FRAME
                result.path.header.stamp = self.get_clock().now().to_msg()
                for x, y in path:
                    p = PoseStamped()
                    p.header = result.path.header
                    p.pose.position.x = x
                    p.pose.position.y = y
                    p.pose.orientation.w = 1.0
                    result.path.poses.append(p)
                self.path_pub.publish(result.path)
                h.succeed()
            elif h.is_cancel_requested:
                h.canceled()
            else:
                result.error_code = 208
                result.error_msg = "EGO produced no validated local trajectory"
                h.abort()
            return result
        finally:
            # Preview is display-only for the chassis. Cancelling it used to land
            # on the next NavigateToPose task_id and abort execute in ~0.3 s.
            self.finish(task, cancel=False)

    def navigate_execute(self, h):
        result = NavigateToPose.Result()
        request, task = self.begin(h.request.pose, True)
        try:
            h._lavira_task = task
            m = self.wait_trajectory(h, task)
            if m is None:
                reason = (
                    "cancel"
                    if (h.is_cancel_requested or task in self.cancelled)
                    else "health"
                    if not self.ready()
                    else "timeout"
                )
                result.error_code = 1
                result.error_msg = "No trajectory (%s)" % reason
                self.describe(request, None, "no_solution", abort_reason=reason)
                h.canceled() if h.is_cancel_requested else h.abort()
                return result
            _, path = self.describe(request, m, "executing")
            if path:
                msg = RosPath()
                msg.header.frame_id = FRAME
                msg.header.stamp = self.get_clock().now().to_msg()
                for x, y in path:
                    p = PoseStamped()
                    p.header = msg.header
                    p.pose.position.x = x
                    p.pose.position.y = y
                    p.pose.orientation.w = 1.0
                    msg.poses.append(p)
                self.path_pub.publish(msg)
            deadline = time.monotonic() + 180
            while rclpy.ok() and time.monotonic() < deadline:
                if h.is_cancel_requested or task in self.cancelled:
                    result.error_code = 1
                    result.error_msg = "cancelled after trajectory"
                    h.canceled() if h.is_cancel_requested else h.abort()
                    self.describe(request, m, "cancelled", abort_reason="cancel")
                    return result
                if not self.ready():
                    result.error_code = 2
                    result.error_msg = "map/localization unavailable"
                    h.abort()
                    self.describe(request, m, "health_stop")
                    return result
                m = self.traj or m
                p = self.odom.pose.pose.position
                target = m.effective_goal
                err = math.hypot(p.x - target[0], p.y - target[1])
                if err <= PROFILE["arrival_tolerance"]:
                    h.succeed()
                    self.describe(request, m, "arrived")
                    return result
                f = NavigateToPose.Feedback()
                f.current_pose.header = self.odom.header
                f.current_pose.pose = self.odom.pose.pose
                f.distance_remaining = float(err)
                h.publish_feedback(f)
                time.sleep(0.05)
            result.error_code = 3
            result.error_msg = "execution timeout"
            h.abort()
            self.describe(request, m, "timeout")
            return result
        finally:
            self.finish(task, cancel=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--control", action="store_true")
    a = p.parse_args()
    rclpy.init()
    n = Navigation(a.control)
    e = MultiThreadedExecutor(num_threads=4)
    e.add_node(n)
    try:
        e.spin()
    except KeyboardInterrupt:
        pass
    finally:
        n.finish()
        e.shutdown(timeout_sec=2)
        n.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
