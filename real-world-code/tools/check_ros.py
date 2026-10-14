#!/usr/bin/env python3
"""Real local middleware connectivity and command-boundary smoke checks."""

import argparse
import fcntl
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def check_pipeline(ros, io, sensors, directory):
    """Exercise the actual fixture -> HTTP fusion -> ROS planning-map path."""
    from urllib.request import ProxyHandler, build_opener

    from nav_msgs.msg import OccupancyGrid

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base = "http://127.0.0.1:" + str(port)
    config = Path(directory) / "pipeline.json"
    config.write_text(json.dumps(dict(services=dict(map=base, board=base))))
    prefix = [sys.executable, "-m", "lavira_m3.cli", "--ros", ros, "--config", str(config)]
    received = []
    io.subscribe(OccupancyGrid, "/map2d/planning", received.append, durable=True)
    opener = build_opener(ProxyHandler({}))
    children = []
    with open(Path(directory) / "pipeline.log", "w+") as log:
        try:
            for command in (["hub", "--port", str(port)], ["simulate"], ["map-bridge"]):
                children.append(
                    subprocess.Popen(
                        prefix + command, stdout=log, stderr=log, start_new_session=True
                    )
                )
            until = time.monotonic() + 25
            while time.monotonic() < until:
                if any(child.poll() is not None for child in children):
                    raise RuntimeError("pipeline process exited")
                if (
                    received
                    and 0 in received[-1].data
                    and 100 in received[-1].data
                    and sensors.view("front") is not None
                ):
                    with opener.open(base + "/v1/planning?robot=agent_0", timeout=2) as response:
                        planning = json.load(response)
                    with opener.open(base + "/v1/dashboard", timeout=2) as response:
                        dashboard = json.load(response)
                    if not dashboard["cameras"]["agent_0"]["front"]["online"]:
                        time.sleep(0.1)
                        continue
                    with opener.open(base + "/v1/image/agent_0/front", timeout=2) as response:
                        assert response.headers["Content-Type"] == "image/jpeg"
                        assert response.read().startswith(b"\xff\xd8")
                    assert planning["available"]
                    assert 0 in received[-1].data and 100 in received[-1].data
                    assert (
                        len(planning["planning"])
                        == received[-1].info.width * received[-1].info.height
                    )
                    return dict(
                        camera=True, shared_map=True, planning_grid=True, dashboard_camera=True
                    )
                time.sleep(0.1)
            raise TimeoutError("camera/shared planning grid not received")
        except Exception:
            log.flush()
            log.seek(0)
            print(log.read()[-2500:], file=sys.stderr)
            raise
        finally:
            for child in reversed(children):
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGINT)
            for child in reversed(children):
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ros", choices=("ros1", "ros2"), required=True)
    parser.add_argument(
        "--pipeline", action="store_true", help="also test synthetic sensors and HTTP map fusion"
    )
    args = parser.parse_args()
    ros = args.ros
    os.environ["NAV_ROS"] = ros
    os.environ.pop("NAV_CONFIG", None)
    os.environ.pop("NAV_AGENT", None)
    os.environ["ROS_DOMAIN_ID"] = os.environ.get("ROS_DOMAIN_ID", "193")
    os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
    with tempfile.TemporaryDirectory(prefix="unified-ros-") as directory:
        os.environ["NAV_RUNTIME_DIR"] = directory
        master = None
        io = None
        gate = None
        owner = None
        try:
            if ros == "ros1":
                import xmlrpc.client

                with socket.socket() as sock:
                    sock.bind(("127.0.0.1", 0))
                    port = sock.getsockname()[1]
                os.environ["ROS_MASTER_URI"] = "http://127.0.0.1:" + str(port)
                os.environ["ROS_IP"] = "127.0.0.1"
                master = subprocess.Popen(
                    ["roscore", "-p", str(port)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                for _ in range(100):
                    try:
                        xmlrpc.client.ServerProxy(os.environ["ROS_MASTER_URI"]).getPid("probe")
                        break
                    except OSError:
                        time.sleep(0.05)
                else:
                    raise RuntimeError("local ROS master failed to start")
            from geometry_msgs.msg import Twist
            from nav_msgs.msg import OccupancyGrid, Odometry
            from quadrotor_msgs.msg import GoalSet
            from sensor_msgs.msg import CameraInfo, Image, Imu, PointCloud2
            from std_msgs.msg import String
            from traj_utils.msg import PolyTraj

            from lavira_m3.adapters.ros_io import RosIO
            from lavira_m3.sensors import Sensors
            from lavira_m3.settings import AGENT, BODY_FRAME, FRAME, TOPICS, lock_path

            io = RosIO(ros, "unified_probe")
            sensors = Sensors(io)
            received = {}
            types = [
                ("odom", Odometry),
                ("imu", Imu),
                ("cloud", PointCloud2),
                ("image", Image),
                ("camera_info", CameraInfo),
                ("map", OccupancyGrid),
                ("command", Twist),
                ("goal", GoalSet),
                ("trajectory", PolyTraj),
            ]
            pubs = []
            for name, kind in types:
                topic = "/unified_probe/" + name
                io.subscribe(kind, topic, lambda msg, n=name: received.__setitem__(n, msg))
                pubs.append((io.publisher(kind, topic), kind()))
            odom = io.publisher(Odometry, TOPICS["odometry"])
            until = time.monotonic() + 8
            while time.monotonic() < until:
                for pub, msg in pubs:
                    pub.publish(msg)
                o = Odometry()
                o.header.frame_id = FRAME
                o.child_frame_id = BODY_FRAME
                o.header.stamp = io.stamp()
                o.pose.pose.orientation.w = 1.0
                odom.publish(o)
                if len(received) == len(types) and sensors.pose() is not None:
                    break
                time.sleep(0.05)
            assert len(received) == len(types), "missing topics: " + str(
                set(n for n, _ in types) - received.keys()
            )
            assert sensors.pose() == (0.0, 0.0, 0.0)
            time.sleep(0.25)
            assert sensors.pose() is None, "stale odometry accepted"
            checks = dict(standard_and_custom_messages=len(received), sensor_stale_rejection=True)
            if ros == "ros1":
                from lavira_m3.adapters.ros1.fastplanner_nav import FastPlannerNavClient

                nav = FastPlannerNavClient(enable_motion=False)
                assert nav.preview_plan(2, 0, 0, start=(0, 0, 0)) == [[0.0, 0.0], [2.0, 0.0]]
                assert not nav.navigate_to(2, 0, 0).ok
                nav.cancel()
                nav.lock_estop()
                nav.enable_motion = True
                assert nav.navigate_to(2, 0, 0).state == "estop"
                nav.close()
                checks["preview_cannot_move"] = True
                checks["estop_latched"] = True
            else:
                from nav2_msgs.action import ComputePathToPose, NavigateToPose

                from lavira_m3.adapters.ros2.gate import Gate
                from lavira_m3.adapters.ros2.nav_client import NavClient
                from lavira_m3.adapters.ros2.navigation_server import Navigation

                navigation = Navigation(control=False)
                io.executor.add_node(navigation)
                client = NavClient(io)
                try:
                    for action, kind in (
                        (client.compute, ComputePathToPose),
                        (client.move, NavigateToPose),
                    ):
                        try:
                            client.send(action, client.goal(kind, 1, 0, 0), 2)
                        except ValueError as error:
                            assert str(error) == "navigation goal rejected"
                        else:
                            raise AssertionError("action accepted without healthy inputs")
                    checks["action_transport_and_unhealthy_rejection"] = 2
                finally:
                    client.close()
                    client.compute.destroy()
                    client.move.destroy()
                    io.executor.remove_node(navigation)
                    navigation.destroy_node()
                from std_msgs.msg import UInt64
                from std_srvs.srv import SetBool
                from traj_utils.msg import MotionCommand

                owner = open(lock_path("owner"), "a")
                fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
                gate = Gate(control=True, simulate=True)
                # Inject health at the gate boundary; message transport was tested above.
                gate.tf_valid = lambda: True
                gate.count_publishers = lambda _: 1

                def arm(task):
                    stamp = io.stamp()
                    o = Odometry()
                    o.header.stamp = stamp
                    o.header.frame_id = FRAME
                    o.child_frame_id = BODY_FRAME
                    o.pose.pose.orientation.w = 1.0
                    gate.on_odom(o)
                    ns = io.now_ns()
                    gate.on_map(String(data=json.dumps(dict(accepted=True, stamp_ns=ns))))
                    gate.peer.update(
                        String(
                            data=json.dumps(
                                dict(
                                    robot=AGENT,
                                    own=dict(name=AGENT, x=0.0, y=0.0, yaw=0.0, age=0.0),
                                    peers=[],
                                )
                            )
                        )
                    )
                    g = GoalSet()
                    g.header.stamp = stamp
                    g.header.frame_id = FRAME
                    g.task_id = task
                    g.execute = True
                    gate.on_goal(g)
                    c = MotionCommand()
                    c.header.stamp = stamp
                    c.header.frame_id = BODY_FRAME
                    c.task_id = task
                    c.map_stamp_ns = ns
                    c.twist.linear.x = 0.2
                    gate.on_cmd(c)
                    return g, c

                goal, cmd = arm(10)
                assert gate.health() == "moving", gate.health()
                gate.on_cancel(UInt64(data=10))
                gate.on_goal(goal)
                gate.on_cmd(cmd)
                assert gate.task == 0
                arm(11)
                gate.map_received -= 6
                gate.tick()
                assert gate.task == 0
                arm(12)
                gate.odom_received -= 1
                gate.tick()
                assert gate.task == 0
                arm(13)
                gate.cmd_received -= 1
                gate.tick()
                assert gate.task == 0
                arm(14)
                req = SetBool.Request()
                req.data = True
                gate.stop_service(req, SetBool.Response())
                gate.tick()
                assert gate.estop and not gate.task
                req.data = False
                response = gate.stop_service(req, SetBool.Response())
                assert not response.success
                checks.update(
                    cancel_replay_rejected=True,
                    map_odom_command_outage_stop=True,
                    estop_latched=True,
                )
            if args.pipeline:
                checks["pipeline"] = check_pipeline(ros, io, sensors, directory)
            print(json.dumps(dict(ros=ros, passed=checks), indent=2))
        finally:
            if gate:
                gate.close()
                gate.destroy_node()
            if owner:
                owner.close()
            if io:
                io.close()
            if master:
                os.killpg(master.pid, signal.SIGINT)
                try:
                    master.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(master.pid, signal.SIGKILL)
                    master.wait()


if __name__ == "__main__":
    main()
