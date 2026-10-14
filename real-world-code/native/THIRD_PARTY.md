# Native third-party components

The native ROS workspaces under `native/` are assembled by `tools/build_native.py`
from `native/common` plus the `native/ros1` or `native/ros2` overlay. This table lists
every component in `native/`, where it came from, its license and what we changed.

**GPL components are fetched, not vendored.** FAST-LIO and EGO-Planner v2 are not
stored in this repository. `native/fetch_native.sh` clones them at the commits pinned
below, lays the needed upstream files out as listed in `native/patches/<component>/<layer>.files`,
applies `native/patches/<component>/<layer>.patch` with `git apply`, and installs the
result into the (git-ignored) locations given in the table. The result is byte-identical
to the tree previously vendored here (verified with `diff -r`). The patches are
distributed under the respective upstream GPL license; see
[patches/README.md](patches/README.md). Run `bash native/fetch_native.sh` before
`tools/build_native.py`.

Components in the other rows below are vendored and keep their upstream license. The top-level
`LICENSE` (CC BY-NC-SA 4.0) applies only to the rows marked "HM3D-M3 authors".

| Component | Location | Origin | License | Modifications |
| --- | --- | --- | --- | --- |
| OSQP | `common/deps/osqp` | https://github.com/osqp/osqp, commit `1572ae068e9ce9ca723cf8223548ade1ff7acc29` | Apache-2.0 (`LICENSE`); bundled AMD is BSD-3-Clause (`algebra/_common/lin_sys/qdldl/amd/LICENSE`) | None |
| osqp-eigen | `common/deps/osqp-eigen` | https://github.com/robotology/osqp-eigen, commit `7587e6994dc194cf22511d909bf4cc5d5e0e4eb2` | BSD-3-Clause (`LICENSE`) | None |
| QDLDL | `common/deps/qdldl` | https://github.com/osqp/qdldl, commit `138fdac58b9cd1c4137ff1b99152c8108a6cff5b` | Apache-2.0 (`LICENSE`) | None |
| FAST-LIO (with ikd-Tree and IKFoM toolkit) | **Fetched** by `fetch_native.sh` into `common/src/localization/FAST_LIO`, `ros1/src/localization/FAST_LIO`, `ros2/src/localization/FAST_LIO`; patches in `patches/fast_lio/` | https://github.com/hku-mars/FAST_LIO. ROS 1 and the shared `common/` headers: branch `main`, commit `7cc4175de6f8ba2edf34bab02a42195b141027e9` (2024-07-23; `common/` is unmodified, no patch). ROS 2: branch `ROS2`, commit `87c42cdb8f2c569c27d95920ea6c733cffb3eba8` (2025-01-14; of all `ROS2` commits it gives the smallest patch, tied with the later merge commit `a4743b09` whose files are identical). `include/ikd-Tree` is the upstream ikd-Tree submodule, https://github.com/hku-mars/ikd-Tree commit `e2e3f4e9d3b95a9e66b1ba83dc98d4a05ed8a3c4` (branch `fast_lio`; unmodified in ROS 2, patched in ROS 1) | GPL-2.0 (`LICENSE` in each directory). Some files keep their original BSD notices (LOAM-derived `laserMapping.cpp`, MTK headers from Universität Bremen) | ROS 1: changed `laserMapping.cpp`, `preprocess.cpp/.h`, `CMakeLists.txt` and `package.xml`. ROS 2: changed the same files plus `IMU_Processing.hpp` and the five upstream configs; added `config/unitree_mid360*.yaml`, `include/localization_frames.h` and `tests/localization_frames_test.cpp`. Header files shared by both are in `common/`. `package.xml` license corrected from `BSD` to `GPL-2.0-only` |
| EGO-Planner v2 (`plan_env`, `path_searching`, `traj_opt`, `traj_utils`, `plan_manage`, `quadrotor_msgs`) | **Fetched** by `fetch_native.sh` into `common/src/planner/Ego-Planner`, `ros1/src/planner/Ego-Planner`, `ros2/src/planner/Ego-Planner`; patches in `patches/ego_planner/` | https://github.com/ZJU-FAST-Lab/EGO-Planner-v2, `swarm-playground/main_ws/src/planner` and `main_ws/src/Utils/quadrotor_msgs`, commit `9d85475ea7b9bf5c112cf7c3c3d0d3f9e96d9010` (unmodified files such as `gcopter.hpp`, `lbfgs.hpp`, `poly_traj_utils.hpp`, `root_finder.hpp`, `traj_server.cpp` and the `quadrotor_msgs` sources match it byte for byte or up to whitespace; the patches record the exact bytes, including the older GPL-3.0 `LICENSE` text with `http://` URLs used in the paper tree). Not EGO-Planner v1 (https://github.com/ZJU-FAST-Lab/ego-planner), which uses B-splines (`bspline_opt`) | GPL-3.0 (`LICENSE` in each directory). `quadrotor_msgs` keeps its upstream `BSD` declaration. Upstream's `traj_opt/include/optimizer/root_finder.hpp` is Zhepei Wang's MIT-licensed Root-Finder and its `poly_traj_utils.hpp` contains GCOPTER's MIT-licensed `trajectory.hpp`; upstream ships them without per-file notices and they are vendored here unchanged | Adapted for ground robots: new planar motion cost (`planar_motion*.h`), goal adjustment (`goal_adjustment.*`), tracking-error replanning (`tracking_error_trigger.h`, `ego_replan_fsm.*`), a rewritten occupancy map (`plan_env/grid_map.*`) and a ROS 2 port. `package.xml` license fields set to `GPL-3.0-only` (upstream left them as `TODO`) |
| Livox ROS Driver 2 message definitions | `common/src/localization/livox_ros_driver2/msg`, `ros*/src/localization/livox_ros_driver2` | https://github.com/Livox-SDK/livox_ros_driver2 | MIT for the Livox portions (`LICENSE.txt`, which also lists the licenses of upstream's own bundled dependencies) | Only `CustomMsg.msg` and `CustomPoint.msg` are included, with new minimal build files. No driver or SDK code |
| Nav2 action definitions | `ros2/src/navigation/nav2_msgs` | https://github.com/ros-navigation/navigation2, commit `645abd95f2be02a13ca539b29c2ddc065db34f89` | Apache-2.0 (`LICENSE`) | Only `ComputePathToPose.action` and `NavigateToPose.action` (from `nav2_msgs/action/` at that commit), unmodified, with new minimal build files |
| MPC tracking controller (`mpc`) | `common/src/control/mpc`, `ros*/src/control/mpc` | HM3D-M3 authors, except the three headers in the next row | CC BY-NC-SA 4.0 (top-level `LICENSE`) | n/a |
| MINCO / trajectory / root-finder headers (`mpc_utils/minco.hpp`, `mpc_utils/poly_traj_utils.hpp`, `mpc_utils/root_finder.hpp`) | `common/src/control/mpc/include/mpc_utils` | Zhepei Wang (Root-Finder https://github.com/ZJU-FAST-Lab/Root-Finder, GCOPTER/MINCO https://github.com/ZJU-FAST-Lab/GCOPTER). Copied from https://github.com/ZJU-FAST-Lab/Elastic-Tracker, commit `0a302a2a9cc8b733e74941fdd4af3fb449447bea`, `src/planning/traj_opt/include/traj_opt/` (token match 99-100%) | MIT, `Copyright (c) 2021 Zhepei Wang`; the per-file MIT notice is restored at the top of each header. Note: the Elastic-Tracker repository as a whole is GPL-3.0, but these files carry their author's own MIT notice | Wrapped in namespace `mpc_utils`; `EIGEN_MAKE_ALIGNED_OPERATOR_NEW` added to `Piece`, `Trajectory`, `BandedSystem` and `MinJerkOpt`; `MinJerkOpt` takes the tail state in `generate()` and exposes its gradient |
| EKF pose filter (`ekf_pose`, package `ekf`) | `common/src/localization/ekf_pose`, `ros*/src/localization/ekf_pose` | HKUST Aerial Robotics Group, ELEC5660 course repository: https://github.com/HKUST-Aerial-Robotics/HKUST-ELEC5660-Introduction-to-Aerial-Robotics, branch `2026`, commit `18b04a789aaeb14d78d60c36b71258eaa18ff859`, `lab/lab2/uav_ws/src/RI_Mocap/ekf_pose/`. The upstream `package.xml` names the maintainer "kunyue"; the original author is not otherwise identified. `cmake/FindGLIB2.cmake` and `cmake/FindGTHREAD2.cmake` are KDE CMake modules (Copyright (c) 2008 Laurent Montel, BSD per KDE `COPYING-CMAKE-SCRIPTS`); `cmake/FindEigen3.cmake` is Eigen's (BSD-2-Clause, header kept) | BSD-3-Clause, `Copyright (c) 2026, HKUST Aerial Robotics Group` (`common/src/localization/ekf_pose/LICENSE`, copied from the upstream repository; the upstream `package.xml` itself says `TODO`) | All `common/` files are unchanged except one comment translated in `ekf_node.cpp`. `ros1`: `ekf.h` declares `acc_f_pub()` / `get_filtered_acc_gyro()`, and `ekf_node_vio_timesync_with_acc_pub.cpp` has local edits (parameter loading, gravity, IMU scaling). `ros2`: ROS 2 port (`CMakeLists.txt`, `package.xml`, `ekf.h`, `ekf_node_vio_timesync_with_acc_pub.cpp`), plus new `check_initialization.cpp` and `config/ekf.yaml`. Upstream-only launch/config files are not included |
| FAST-LIO / planner bridge (`fastlio_planner_bridge`) | `ros1/src/planner/fastlio_planner_bridge` | HM3D-M3 authors | CC BY-NC-SA 4.0 (top-level `LICENSE`) | n/a |

## Distribution notes

- The Python package `lavira_m3` (`src/`) does not link against or import any of the
  native code. It talks to it only through ROS topics, actions and HTTP.
- FAST-LIO (GPL-2.0) and EGO-Planner (GPL-3.0) are separate ROS packages that build
  into separate executables. If you distribute binaries built from them, the GPL terms
  of those packages apply to those binaries, including the obligation to provide the
  corresponding source (the pinned upstream commits plus `patches/`, as assembled by
  `fetch_native.sh`).
- To change FAST-LIO or EGO-Planner code, edit the fetched files and run
  `bash native/fetch_native.sh --update-patches`; `--check` verifies that an
  installed tree equals fetch + patch. Never commit the fetched directories.
- Do not mass-format or relicense the vendored or fetched directories. Keep each
  `LICENSE` file next to the code it covers.
