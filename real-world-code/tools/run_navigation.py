#!/usr/bin/env python3
"""Start the selected native navigation stack; stop all children on any exit."""

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ros", choices=("ros1", "ros2"), required=True)
    p.add_argument("--config")
    p.add_argument("--agent")
    p.add_argument("--control", action="store_true")
    p.add_argument("--check-seconds", type=float, help="startup-only connectivity check, then exit")
    a = p.parse_args()
    if a.config:
        os.environ["NAV_CONFIG"] = str(Path(a.config).resolve())
    if a.agent:
        os.environ["NAV_AGENT"] = a.agent
    os.environ["NAV_ROS"] = a.ros
    sys.path.insert(0, str(ROOT / "src"))
    from lavira_m3.settings import AGENT, SAFETY, TOPICS

    source = ROOT / "src/lavira_m3/adapters/ros2"
    config = ROOT / ".runtime" / a.ros / "runtime-config"
    config.mkdir(parents=True, exist_ok=True)
    planner = yaml.safe_load((source / "planner.yaml").read_text())["/**"]["ros__parameters"]
    planner.update(
        {
            "manager.max_vel": SAFETY["max_speed"],
            "optimization.max_vel": SAFETY["max_speed"],
            "navigation.arrival_tolerance": SAFETY["arrival_tolerance"],
        }
    )
    mpc = yaml.safe_load((source / "mpc.yaml").read_text())["/**"]["ros__parameters"]
    mpc["mpc"].update(max_speed=SAFETY["max_speed"], max_omega=SAFETY["max_yaw_rate"])
    commands = []
    if a.ros == "ros2":
        for name, values in (("planner", planner), ("mpc", mpc)):
            (config / (name + ".yaml")).write_text(
                yaml.safe_dump({"/**": {"ros__parameters": values}})
            )
        commands = [
            [
                "ros2",
                "run",
                "ego_planner",
                "ego_planner_node",
                "--ros-args",
                "--params-file",
                str(config / "planner.yaml"),
                "-r",
                "__node:=ego_planner",
                "-r",
                "/goal:=/navigation/goal",
                "-r",
                "~/odom_world:=" + TOPICS["odometry"],
                "-r",
                "~/planning/trajectory:=/ego/trajectory",
            ],
            [
                "ros2",
                "run",
                "mpc",
                "mpc_controller_node",
                "--ros-args",
                "--params-file",
                str(config / "mpc.yaml"),
                "-r",
                "__node:=mpc",
                "-r",
                "~/odom:=" + TOPICS["odometry"],
                "-r",
                "~/planning/trajectory:=/ego/trajectory",
                "-r",
                "~/cmd:=/mpc/command",
            ],
        ]
        for component in ("gate", "navigation-server"):
            commands.append(
                [
                    sys.executable,
                    "-m",
                    "lavira_m3.cli",
                    "--ros",
                    a.ros,
                    "--agent",
                    AGENT,
                    component,
                ]
                + (["--control"] if a.control else [])
            )
    else:
        (config / "planner.yaml").write_text(
            yaml.safe_dump({k.replace(".", "/"): v for k, v in planner.items()})
        )
        (config / "mpc.yaml").write_text(yaml.safe_dump(mpc))
        from xml.sax.saxutils import escape

        def node(package, executable, name, params, remaps):
            return (
                '<node pkg="'
                + package
                + '" type="'
                + executable
                + '" name="'
                + name
                + '" output="screen" required="true"><rosparam command="load" file="'
                + escape(str(params), {'"': "&quot;"})
                + '"/>'
                + "".join(
                    '<remap from="' + k + '" to="' + escape(v, {'"': "&quot;"}) + '"/>'
                    for k, v in remaps.items()
                )
                + "</node>"
            )

        launch = (
            "<launch>"
            + node(
                "ego_planner",
                "ego_planner_node",
                "ego_planner",
                config / "planner.yaml",
                {
                    "/goal": "/navigation/goal",
                    "odom_world": TOPICS["odometry"],
                    "planning/trajectory": "/navigation/trajectory",
                    "planning/heartbeat": "/navigation/heartbeat",
                    "grid_map/shared": TOPICS["map"],
                    "grid_map/applied": "/navigation/map_applied",
                    "mandatory_stop": "/navigation/emergency_stop",
                },
            )
            + node(
                "mpc",
                "mpc_controller_node",
                "mpc",
                config / "mpc.yaml",
                {
                    "odom": TOPICS["odometry"],
                    "planning/trajectory": "/navigation/mpc_trajectory",
                    "cmd": "/navigation/mpc_command",
                    "/emergency_stop": "/navigation/emergency_stop",
                },
            )
            + "</launch>"
        )
        (config / "navigation.launch").write_text(launch)
        commands = [["roslaunch", str(config / "navigation.launch")]]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    children = []
    started = time.monotonic()
    try:
        for command in commands:
            children.append(subprocess.Popen(command, env=env, start_new_session=True))
        while True:
            exited = [child for child in children if child.poll() is not None]
            if exited:
                raise RuntimeError("navigation component exited: " + str(exited[0].returncode))
            if a.check_seconds and time.monotonic() - started >= a.check_seconds:
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGINT)
        for child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGTERM)
                child.wait(timeout=3)
    if a.check_seconds:
        print("PASS: selected navigation processes remained alive for", a.check_seconds, "seconds")


if __name__ == "__main__":
    main()
