#!/usr/bin/env python3
"""Assemble and build the selected native ROS port from shared source + overlay."""

import argparse
import os
import shlex
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


FETCHED = ("src/localization/FAST_LIO", "src/planner/Ego-Planner")


def require_fetched(ros):
    """GPL components are fetched by native/fetch_native.sh, not stored in the repository."""
    missing = [
        f"native/{layer}/{rel}"
        for layer in ("common", ros)
        for rel in FETCHED
        if not (ROOT / "native" / layer / rel).is_dir()
    ]
    if missing:
        raise SystemExit(
            "missing fetched sources: " + ", ".join(missing) + "\nrun: bash native/fetch_native.sh"
        )


def prepare(ros):
    require_fetched(ros)
    destination = ROOT / ".runtime" / ros
    destination.mkdir(parents=True, exist_ok=True)
    for layer in ("common", ros):
        for source in (ROOT / "native" / layer).rglob("*"):
            if not source.is_file():
                continue
            target = destination / source.relative_to(ROOT / "native" / layer)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or target.read_bytes() != source.read_bytes():
                shutil.copyfile(source, target)
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ros", choices=("ros1", "ros2"), required=True)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--localization", action="store_true", help="Also build FAST-LIO and EKF")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("jobs must be positive")
    workspace = prepare(args.ros)
    if args.prepare_only:
        print(workspace)
        return
    distro = "noetic" if args.ros == "ros1" else "jazzy"
    setup = Path("/opt/ros") / distro / "setup.bash"
    if not setup.is_file():
        parser.error("Missing ROS environment: " + str(setup))
    if args.ros == "ros2":
        command = [
            "colcon",
            "build",
            "--base-paths",
            "src",
            "--executor",
            "sequential",
            "--packages-up-to",
            "ego_planner",
            "mpc",
            "nav2_msgs",
        ]
        if args.localization:
            command += ["fast_lio", "ekf"]
        command += [
            "--cmake-args",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_TESTING=ON",
            "-DPython3_EXECUTABLE=/usr/bin/python3",
        ]
    else:
        packages = "ego_planner;mpc;traj_utils;quadrotor_msgs;plan_env;path_searching;traj_opt"
        if args.localization:
            packages += ";fast_lio;ekf;livox_ros_driver2"
        command = [
            "catkin_make",
            "-j" + str(args.jobs),
            "-l" + str(args.jobs),
            "-DCATKIN_WHITELIST_PACKAGES=" + packages,
            "-DCMAKE_BUILD_TYPE=Release",
            "-DPYTHON_EXECUTABLE=/usr/bin/python3",
            "-DCMAKE_CXX_STANDARD=14",
        ]
    env = dict(os.environ, CMAKE_BUILD_PARALLEL_LEVEL=str(args.jobs))
    env.pop("ROS_DISTRO", None)
    subprocess.run(
        [
            "bash",
            "-c",
            "set -eo pipefail; source " + shlex.quote(str(setup)) + "; exec " + shlex.join(command),
        ],
        cwd=workspace,
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
