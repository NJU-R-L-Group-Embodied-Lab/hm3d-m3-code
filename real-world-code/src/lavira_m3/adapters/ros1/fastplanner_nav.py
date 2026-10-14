"""L3: LaViRA navigation interface over the isolated EGO + MPC ROS topics."""

import math
import os
import secrets
import threading
import time

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import GoalSet
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Empty, Header, String
from traj_utils.msg import PlannerCommand, PolyTraj

from lavira_m3.contracts import NavResult
from lavira_m3.decision.map_geometry import in_world_xy
from lavira_m3.decision.path_geometry import thin_planner_path
from lavira_m3.mapping.peers import PeerGuard
from lavira_m3.settings import AGENT, FRAME, SAFETY, TOPICS

from .fastplanner_contracts import CommandGate, sample_trajectory, straight_preview

PREFIX = "/navigation"


class FastPlannerNavClient:
    def __init__(
        self,
        frame=FRAME,
        wait_server_s=8.0,
        total_s=180.0,
        poll_s=0.05,
        pose_fn=None,
        enable_motion=None,
    ):
        self.estop = False
        self.frame, self.pose_fn = frame, pose_fn
        self.wait_server_s, self.total_s, self.poll_s = wait_server_s, total_s, poll_s
        self.enable_motion = (
            os.environ.get("ENABLE_MOTION", "false").lower() in ("1", "true", "yes", "on")
            if enable_motion is None
            else enable_motion
        )
        self.lock = threading.RLock()
        self.changed = threading.Event()
        self.closed = threading.Event()
        self.gate = CommandGate()
        self.peers = PeerGuard(AGENT)
        self.generation = secrets.randbelow(2**31) + 1
        self.last_execution_id = None
        self.last_preview = {}
        self.path = []
        self.odom = None
        self.stamps = {}
        self.last_traj_id = -1
        self.last_traj = None
        self.active_goal = None
        self.stop_epoch = 0
        self.goal_pub = rospy.Publisher(PREFIX + "/goal", GoalSet, queue_size=10)
        self.trajectory_pub = rospy.Publisher(PREFIX + "/mpc_trajectory", PolyTraj, queue_size=1)
        self.stop_pub = rospy.Publisher(PREFIX + "/emergency_stop", Empty, queue_size=1)
        self.command_pub = rospy.Publisher(TOPICS["command"], Twist, queue_size=1)
        self.subscribers = [
            rospy.Subscriber(PREFIX + "/trajectory", PolyTraj, self._trajectory, queue_size=1),
            rospy.Subscriber(PREFIX + "/mpc_command", PlannerCommand, self._command, queue_size=1),
            rospy.Subscriber(TOPICS["odometry"], Odometry, self._odometry, queue_size=1),
            rospy.Subscriber(TOPICS["localization"], Odometry, self._localization, queue_size=1),
            rospy.Subscriber(TOPICS["cloud"], PointCloud2, self._cloud, queue_size=1),
            rospy.Subscriber("/navigation/heartbeat", Empty, self._heartbeat, queue_size=1),
            rospy.Subscriber(PREFIX + "/map_applied", Header, self._shared_map, queue_size=1),
            rospy.Subscriber("/map2d/peer_state", String, self._peer_state, queue_size=1),
        ]
        self.worker = threading.Thread(target=self._watchdog, daemon=True)
        self.worker.start()
        rospy.on_shutdown(self.close)

    def _fresh_stamp(self, source, stamp, limit=0.5):
        seconds = stamp.to_sec()
        age = rospy.Time.now().to_sec() - seconds
        if (
            not math.isfinite(seconds)
            or seconds <= self.stamps.get(source, 0)
            or not -0.1 <= age <= limit
        ):
            return False
        self.stamps[source] = seconds
        return True

    def _observe_odom(self, msg, source):
        p, q, v, w = (
            msg.pose.pose.position,
            msg.pose.pose.orientation,
            msg.twist.twist.linear,
            msg.twist.twist.angular,
        )
        values = (p.x, p.y, p.z, q.x, q.y, q.z, q.w, v.x, v.y, v.z, w.x, w.y, w.z)
        with self.lock:
            valid = (
                msg.header.frame_id == self.frame
                and all(math.isfinite(v) for v in values)
                and 0.9 <= q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w <= 1.1
                and in_world_xy(p.x, p.y)
            )
            if not valid:
                self.gate.received[source] = -math.inf
                if self.gate.active:
                    self._stop("invalid_" + source)
            elif self._fresh_stamp(source, msg.header.stamp):
                self.gate.observe(source, time.monotonic())
                if source == "odometry":
                    self.odom = msg

    def _odometry(self, msg):
        self._observe_odom(msg, "odometry")

    def _localization(self, msg):
        self._observe_odom(msg, "localization")

    def _cloud(self, msg):
        with self.lock:
            if (
                msg.header.frame_id == self.frame
                and msg.data
                and self._fresh_stamp("cloud", msg.header.stamp)
            ):
                self.gate.observe("cloud", time.monotonic())

    def _heartbeat(self, msg):
        with self.lock:
            self.gate.observe("heartbeat", time.monotonic())

    def _shared_map(self, msg):
        with self.lock:
            if msg.frame_id != self.frame or msg.stamp.is_zero():
                self.gate.received["shared_map"] = -math.inf
                if self.gate.active:
                    self._stop("invalid_shared_map")
            elif self._fresh_stamp("shared_map", msg.stamp, 5.0):
                self.gate.observe("shared_map", time.monotonic())

    def _peer_state(self, msg):
        with self.lock:
            self.peers.update(msg)
            if self.peers.permits(0, 0, 0):
                self.gate.observe("peer_state", time.monotonic())
            else:
                self.gate.received["peer_state"] = -math.inf
                if self.gate.active:
                    self._stop("invalid_peer_state")

    def _trajectory(self, msg):
        with self.lock:
            if msg.request_id != self.gate.active or msg.traj_id <= self.last_traj_id:
                return
            age = rospy.Time.now().to_sec() - msg.start_time.to_sec()
            if not -0.2 <= age <= 0.5:
                return
            try:
                path = sample_trajectory(msg)
            except ValueError:
                self._stop("invalid_trajectory")
                return
            self.path = path
            self.last_traj = msg
            self.last_traj_id = msg.traj_id
            self.gate.trajectory_until = msg.start_time.to_sec() + sum(msg.duration)
            if self.gate.execute:
                if self.gate.last_command is None:
                    self.gate.last_command = time.monotonic()
                self.trajectory_pub.publish(msg)
            self.changed.set()

    def _command(self, msg):
        with self.lock:
            if msg.request_id != self.gate.active or not self.gate.execute:
                return
            if msg.header.frame_id != "base_link" or not self._fresh_stamp(
                "command", msg.header.stamp, 0.2
            ):
                self._stop("invalid_command_stamp")
                return
            c = msg.twist
            if not self.peers.permits(c.linear.x, c.linear.y, c.angular.z):
                self._stop("peer_obstacle")
                return
            values = (c.linear.x, c.linear.y, c.linear.z, c.angular.x, c.angular.y, c.angular.z)
            request_id = self.gate.active
            if self.gate.accept(
                msg.request_id, values, time.monotonic(), rospy.Time.now().to_sec()
            ):
                self.command_pub.publish(c)
            else:
                self._stop(self.gate.reason or "command_without_trajectory", request_id)

    def _stop(self, reason, request_id=None):
        # Caller holds lock: the stop and command acceptance cannot interleave.
        request_id = request_id or self.gate.active
        self.stop_epoch += 1
        if request_id:
            rospy.loginfo("EGO/MPC request %s stopped: %s", request_id, reason)
        self.gate.stop(reason)
        self.last_traj = None
        self.active_goal = None
        self.command_pub.publish(Twist())
        self.stop_pub.publish(Empty())
        if request_id:
            self.goal_pub.publish(GoalSet(drone_id=0, request_id=request_id, cancel=True))
        self.changed.set()

    def _watchdog(self):
        while not self.closed.wait(0.05):
            with self.lock:
                if self.gate.active:
                    reason = self.gate.expired(time.monotonic(), rospy.Time.now().to_sec())
                    if reason:
                        self._stop(reason)

    def _ensure_client(self):
        deadline = time.monotonic() + self.wait_server_s
        while not rospy.is_shutdown() and not self.closed.is_set():
            with self.lock:
                if (
                    self.goal_pub.get_num_connections()
                    and self.trajectory_pub.get_num_connections()
                    and self.subscribers[1].get_num_connections()
                    and not self.gate.stale(time.monotonic())
                ):
                    return True
            if time.monotonic() >= deadline:
                return False
            self.closed.wait(0.05)
        return False

    def _same_goal(self, x, y):
        goal = self.active_goal
        return bool(goal) and math.hypot(goal[0] - x, goal[1] - y) <= 0.15

    def _forward_last_traj(self):
        if self.last_traj is None:
            return
        if self.gate.last_command is None:
            self.gate.last_command = time.monotonic()
        self.trajectory_pub.publish(self.last_traj)

    def _begin(self, x, y, yaw, execute):
        with self.lock:
            epoch = self.stop_epoch
        if (
            self.frame != FRAME
            or not all(math.isfinite(v) for v in (x, y, yaw))
            or not in_world_xy(x, y)
        ):
            return None
        if not self._ensure_client():
            return None
        with self.lock:
            if self.stop_epoch != epoch or self.closed.is_set():
                return None
            if self.gate.active and execute and self._same_goal(x, y):
                self.gate.promote()
                self._forward_last_traj()
                return self.gate.active
            # Idle start must not publish cancel / emergency_stop; that resets EGO.
            if self.gate.active:
                self._stop("replaced")
            self.generation = self.generation % (2**32 - 1) + 1
            request_id = self.generation
            self.gate.begin(request_id, execute, time.monotonic())
            self.path = []
            self.last_traj = None
            self.last_traj_id = -1
            self.active_goal = [x, y]
            self.changed.clear()
            self.goal_pub.publish(
                GoalSet(
                    header=Header(stamp=rospy.Time.now(), frame_id=self.frame),
                    drone_id=0,
                    request_id=request_id,
                    cancel=False,
                    goal=[x, y, self.odom.pose.pose.position.z],
                )
            )
            return request_id

    def preview_plan(self, x, y, yaw, start=None, timeout_s=4.0):
        """Draw-only path. Must not publish GoalSet; EGO resets EXEC_TRAJ on every goal."""
        del timeout_s
        self.last_preview = {}
        if (
            self.frame != FRAME
            or not all(math.isfinite(v) for v in (x, y, yaw))
            or not in_world_xy(x, y)
        ):
            return []
        origin = start
        if origin is None or len(origin) < 2:
            if self.odom is not None:
                p = self.odom.pose.pose.position
                origin = (p.x, p.y)
            elif self.pose_fn is not None:
                origin = self.pose_fn()
        path = thin_planner_path(straight_preview(origin, (x, y)))
        self.last_preview = dict(path=path, goal=[x, y], yaw=yaw, frame=self.frame)
        return path

    def navigate_to(self, x, y, yaw, goal=None):
        target = (x, y, yaw)
        if self.estop:
            return NavResult(False, "estop", "emergency stop latched; restart required")
        if not self.enable_motion:
            return NavResult(
                False, "disabled", "ENABLE_MOTION is false", target=target, frame=self.frame
            )
        # EGO supplies an XY path, not an in-place orientation trajectory.
        # Do not claim that a rotation-only request was executed.
        with self.lock:
            if self.odom is not None:
                p, q = self.odom.pose.pose.position, self.odom.pose.pose.orientation
                heading = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
                error = math.atan2(math.sin(yaw - heading), math.cos(yaw - heading))
                if math.hypot(p.x - x, p.y - y) <= 0.05 and abs(error) > 0.1:
                    return NavResult(
                        False,
                        "rejected",
                        "EGO/MPC does not support rotation-only goals",
                        target=target,
                        frame=self.frame,
                    )
        request_id = self._begin(x, y, yaw, True)
        if request_id is None:
            return NavResult(
                False,
                "unavailable",
                "invalid goal/frame or planner/sensors unavailable",
                target=target,
                frame=self.frame,
            )
        self.last_execution_id = request_id
        started = time.monotonic()
        result = None
        while not rospy.is_shutdown() and time.monotonic() - started < self.total_s:
            with self.lock:
                if self.gate.active != request_id:
                    result = NavResult(
                        False, self.gate.reason or "interrupted", "navigation stopped"
                    )
                    break
                p = self.odom.pose.pose.position
                if math.hypot(p.x - x, p.y - y) <= SAFETY["arrival_tolerance"]:
                    result = NavResult(
                        True, "succeeded", "reached %.2f m radius" % SAFETY["arrival_tolerance"]
                    )
                    break
                if not self.path and time.monotonic() - started > self.wait_server_s:
                    result = NavResult(
                        False,
                        "unreachable",
                        "EGO returned no valid trajectory before planning timeout",
                    )
                    break
            self.changed.wait(self.poll_s)
            self.changed.clear()
        result = result or NavResult(False, "timed_out", "navigation deadline exceeded")
        with self.lock:
            if self.gate.active == request_id:
                self._stop(result.state)
        result.execution_id, result.target, result.frame = request_id, target, self.frame
        result.extra["path"] = list(self.last_preview.get("path", []))
        return result

    def cancel(self):
        with self.lock:
            self._stop("cancelled")
        return NavResult(
            True, "cancelled", "cancelled own EGO/MPC request", execution_id=self.last_execution_id
        )

    def lock_estop(self):
        with self.lock:
            self.estop = True
            self._stop("estop")
        return NavResult(True, "estop", "emergency stop latched")

    def close(self):
        if self.closed.is_set():
            return
        self.closed.set()
        try:
            self.cancel()
        except rospy.ROSException:
            pass
