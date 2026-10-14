#!/usr/bin/env python3
"""Validated MPC commands to a standard Twist interface. Motion is opt-in."""

import argparse
import fcntl
import json
import math
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import GoalSet
from rclpy.clock import Clock
from rclpy.clock_type import ClockType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import String, UInt64
from std_srvs.srv import SetBool
from tf2_ros import Buffer, TransformListener
from traj_utils.msg import MotionCommand

from lavira_m3.mapping.peers import PeerGuard
from lavira_m3.settings import (
    AGENT,
    BODY_FRAME,
    FRAME,
    TOPICS,
    lock_path,
)
from lavira_m3.settings import (
    SAFETY as PROFILE,
)


def stamp_ns(h):
    return h.stamp.sec * 1000000000 + h.stamp.nanosec


def finite_pose(m):
    p, q, v, w = (
        m.pose.pose.position,
        m.pose.pose.orientation,
        m.twist.twist.linear,
        m.twist.twist.angular,
    )
    return (
        all(
            math.isfinite(x)
            for x in (p.x, p.y, p.z, q.x, q.y, q.z, q.w, v.x, v.y, v.z, w.x, w.y, w.z)
        )
        and abs(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w - 1) < 0.02
    )


def tf_matches(buffer, odom, now_ns):
    try:
        t = buffer.lookup_transform(FRAME, BODY_FRAME, rclpy.time.Time())
        if not -0.05 <= (now_ns - stamp_ns(t.header)) / 1e9 <= 0.3:
            return False
        p = odom.pose.pose.position
        q = odom.pose.pose.orientation
        a = t.transform.translation
        b = t.transform.rotation
        # Same public body pose from localization, never an invented map/world identity.
        return math.dist((p.x, p.y, p.z), (a.x, a.y, a.z)) <= 0.25 and abs(
            q.x * b.x + q.y * b.y + q.z * b.z + q.w * b.w
        ) >= math.cos(0.2 / 2)
    except Exception:
        return False


class Gate(Node):
    def __init__(self, control=False, simulate=False):
        super().__init__("velocity_gate")
        self.control, self.simulate = control, simulate
        self.task = self.retired = 0
        self.task_start = 0.0
        self.odom = self.cmd = None
        self.map_stamp = 0
        self.map_ok = False
        self.map_received = self.odom_received = self.cmd_received = 0.0
        self.estop = False
        self.peer = PeerGuard(AGENT)
        self.reason = "preview_only" if not control else "idle"
        self.twist_output = self.create_publisher(Twist, TOPICS["command"], 1)
        self.owner_probe = open(lock_path("owner"), "a")
        self.lock = open(lock_path("gate"), "a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.tf = Buffer()
        self.listener = TransformListener(self.tf, self)
        self.output = self.create_publisher(MotionCommand, "/navigation/control_output", 1)
        self.status = self.create_publisher(String, "/navigation/gate_state", 1)
        self.cancel = self.create_publisher(UInt64, "/navigation/cancel", 10)
        self.create_subscription(GoalSet, "/navigation/goal", self.on_goal, 10)
        self.create_subscription(UInt64, "/navigation/cancel", self.on_cancel, 10)
        self.create_subscription(
            Odometry, TOPICS["odometry"], self.on_odom, qos_profile_sensor_data
        )
        self.create_subscription(MotionCommand, "/mpc/command", self.on_cmd, 1)
        self.create_subscription(String, "/navigation/map_applied", self.on_map, 1)
        self.create_subscription(String, "/map2d/peer_state", self.peer.update, 1)
        self.create_service(SetBool, "/navigation/estop", self.stop_service)
        self.create_service(SetBool, "/navigation/enable_motion", self.enable_service)
        self.steady = Clock(clock_type=ClockType.STEADY_TIME)
        self.create_timer(0.01, self.tick, clock=self.steady)
        self.create_timer(0.1, self.state)

    def now_s(self):
        return self.get_clock().now().nanoseconds / 1e9

    def age(self, m):
        return self.now_s() - stamp_ns(m.header) / 1e9

    def on_goal(self, m):
        if (
            m.task_id <= max(self.task, self.retired)
            or m.header.frame_id != FRAME
            or not -0.05 <= self.age(m) <= 0.5
        ):
            return
        self.retired = max(self.retired, self.task)
        self.task = m.task_id if m.execute and self.control and not self.estop else 0
        if not self.task:
            self.retired = max(self.retired, m.task_id)
        self.task_start = time.monotonic()
        self.cmd = None

    def on_cancel(self, m):
        self.retired = max(self.retired, m.data)
        if self.task <= self.retired:
            self.task = 0
            self.cmd = None

    def on_odom(self, m):
        if m.header.frame_id != FRAME or m.child_frame_id != BODY_FRAME or not finite_pose(m):
            self.odom = None
            return
        if self.odom is None or stamp_ns(m.header) > stamp_ns(self.odom.header):
            self.odom = m
            self.odom_received = time.monotonic()

    def on_cmd(self, m):
        if m.task_id == self.task and m.task_id > self.retired:
            if self.cmd is None or stamp_ns(m.header) > stamp_ns(self.cmd.header):
                self.cmd = m
                self.cmd_received = time.monotonic()

    def on_map(self, m):
        try:
            v = json.loads(m.data)
            stamp = int(v["stamp_ns"])
            self.map_ok = (
                v["accepted"] is True
                and stamp > self.map_stamp
                and -0.05 <= self.now_s() - stamp / 1e9 <= PROFILE["map_timeout"]
            )
            if self.map_ok:
                self.map_stamp = stamp
                self.map_received = time.monotonic()
        except (ValueError, KeyError, TypeError):
            self.map_ok = False

    def stop_service(self, req, res):
        if req.data:
            self.estop = True
            self.abort("estop")
        res.success = req.data or not self.estop
        res.message = "Estop latched: restart required" if self.estop else "Estop clear"
        return res

    def enable_service(self, req, res):
        if not req.data:
            self.abort("disabled")
            self.control = False
        res.success = not req.data or (self.control and not self.estop)
        res.message = "Control requires explicit --control at startup"
        return res

    def abort(self, reason):
        self.reason = reason
        if self.task:
            task = self.task
            self.retired = max(self.retired, task)
            self.task = 0
            self.cmd = None
            self.get_logger().warn("gate abort reason=%s task=%s" % (reason, task))
            if rclpy.ok():
                self.cancel.publish(UInt64(data=task))

    def tf_valid(self):
        return tf_matches(self.tf, self.odom, self.get_clock().now().nanoseconds)

    def owner_alive(self):
        # The actual LaViRA client holds EX for its lifetime; kernel releases it on crash.
        try:
            fcntl.flock(self.owner_probe, fcntl.LOCK_SH | fcntl.LOCK_NB)
            fcntl.flock(self.owner_probe, fcntl.LOCK_UN)
            return False
        except BlockingIOError:
            return True

    def health(self):
        if self.estop:
            return "estop"
        if not self.control:
            return "preview_only"
        if not self.task:
            return "idle"
        if not self.owner_alive():
            return "client_exited"
        if (
            not self.map_ok
            or time.monotonic() - self.map_received > PROFILE["map_timeout"]
            or not -0.05 <= self.now_s() - self.map_stamp / 1e9 <= PROFILE["map_timeout"]
        ):
            return "map_stale"
        if (
            self.odom is None
            or time.monotonic() - self.odom_received > PROFILE["odom_timeout"]
            or not -0.05 <= self.age(self.odom) <= PROFILE["odom_timeout"]
        ):
            return "odom_stale"
        if not self.tf_valid():
            return "tf_mismatch"
        if self.cmd is None:
            return "waiting_mpc" if time.monotonic() - self.task_start < 4 else "command_stale"
        if (
            time.monotonic() - self.cmd_received > PROFILE["command_timeout"]
            or self.cmd.header.frame_id != BODY_FRAME
            or not -0.05 <= self.age(self.cmd) <= PROFILE["command_timeout"]
            or (self.cmd.map_stamp_ns > 0 and self.cmd.map_stamp_ns > self.map_stamp)
        ):
            return "command_stale"
        v, w = self.cmd.twist.linear, self.cmd.twist.angular
        if not all(math.isfinite(x) for x in (v.x, v.y, v.z, w.x, w.y, w.z)):
            return "invalid_command"
        if abs(v.y) > 1e-9 or abs(v.z) > 1e-9 or abs(w.x) > 1e-9 or abs(w.y) > 1e-9:
            return "kinematic_mismatch"
        if not self.peer.permits(
            max(-PROFILE["max_speed"], min(PROFILE["max_speed"], v.x)),
            0.0,
            max(-PROFILE["max_yaw_rate"], min(PROFILE["max_yaw_rate"], w.z)),
        ):
            return "peer_interlock"
        if any(
            self.count_publishers(t) != 1
            for t in (
                "/mpc/command",
                TOPICS["command"],
                "/navigation/control_output",
                TOPICS["odometry"],
                "/map2d/planning",
                "/map2d/planning_cloud",
                "/navigation/map_applied",
                "/navigation/goal",
                "/ego/trajectory",
            )
        ):
            return "duplicate_controller"
        return "moving"

    def tick(self):
        out = MotionCommand()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = BODY_FRAME
        reason = self.health()
        if reason == "moving":
            out.task_id = self.task
            out.traj_id = self.cmd.traj_id
            out.map_stamp_ns = self.map_stamp
            out.twist.linear.x = max(
                -PROFILE["max_speed"], min(PROFILE["max_speed"], self.cmd.twist.linear.x)
            )
            out.twist.angular.z = max(
                -PROFILE["max_yaw_rate"], min(PROFILE["max_yaw_rate"], self.cmd.twist.angular.z)
            )
        elif self.task and reason != "waiting_mpc":
            self.abort(reason)
        self.reason = reason
        self.twist_output.publish(out.twist)
        self.output.publish(out)

    def state(self):
        self.status.publish(
            String(
                data=json.dumps(
                    dict(
                        task_id=self.task,
                        reason=self.reason,
                        control=self.control,
                        simulation=self.simulate,
                        estop=self.estop,
                    )
                )
            )
        )

    def close(self):
        self.abort("shutdown")
        if rclpy.ok():
            self.twist_output.publish(Twist())
        self.owner_probe.close()
        self.lock.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--control", action="store_true")
    p.add_argument("--simulate", action="store_true")
    a = p.parse_args()
    rclpy.init()
    n = Gate(a.control, a.simulate)
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.close()
        n.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
