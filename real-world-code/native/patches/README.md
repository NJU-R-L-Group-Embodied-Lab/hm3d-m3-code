# Patches to GPL-licensed native components

`../fetch_native.sh` builds the FAST-LIO and EGO-Planner v2 source directories
from pinned upstream commits plus the files in this directory:

| Component | Upstream | Commit | License |
| --- | --- | --- | --- |
| `fast_lio` (ROS 1 and shared headers) | https://github.com/hku-mars/FAST_LIO, branch `main` | `7cc4175de6f8ba2edf34bab02a42195b141027e9` | GPL-2.0 |
| `fast_lio` (ROS 2) | https://github.com/hku-mars/FAST_LIO, branch `ROS2` | `87c42cdb8f2c569c27d95920ea6c733cffb3eba8` | GPL-2.0 |
| ikd-Tree (FAST-LIO submodule `include/ikd-Tree`) | https://github.com/hku-mars/ikd-Tree, branch `fast_lio` | `e2e3f4e9d3b95a9e66b1ba83dc98d4a05ed8a3c4` | GPL-2.0 (as part of FAST-LIO) |
| `ego_planner` | https://github.com/ZJU-FAST-Lab/EGO-Planner-v2, `swarm-playground/main_ws/src/{planner,Utils/quadrotor_msgs}` | `9d85475ea7b9bf5c112cf7c3c3d0d3f9e96d9010` | GPL-3.0 |

For each component and layer (`common`, `ros1`, `ros2`):

- `<component>/<layer>.files` maps upstream files to their place in
  `native/<layer>/src/localization/FAST_LIO/` or
  `native/<layer>/src/planner/Ego-Planner/`;
- `<component>/<layer>.patch` (git diff format, applied with `git apply`)
  turns them into the HM3D-M3 version, including files that exist only in this
  port (the ROS 2 port, ground-robot planning costs, goal adjustment, tests).
  `fast_lio/common` has no patch: those shared headers are unmodified.

The result is byte-identical to the source tree used for the paper experiments.

## License

The patches modify GPL code and contain GPL-derived code. They are **not**
covered by the CC BY-NC-SA 4.0 license of the rest of this repository:

- `fast_lio/*.patch` are distributed under the **GNU General Public License,
  version 2** (FAST-LIO's license; text in upstream `LICENSE`, installed as
  `FAST_LIO/LICENSE`);
- `ego_planner/*.patch` are distributed under the **GNU General Public License,
  version 3** (EGO-Planner v2's license; text in upstream `LICENSE`, installed
  as `Ego-Planner/LICENSE`).

Files inside the patched trees that carry their own notices (BSD notices in
LOAM-derived `laserMapping.cpp` and the MTK headers of the IKFoM toolkit, Zhepei
Wang's MIT-licensed `root_finder.hpp` / `poly_traj_utils.hpp`) keep them. See
[../THIRD_PARTY.md](../THIRD_PARTY.md) for the full component list.

## Changing the patched code

Edit the fetched files in place, then record the change:

```bash
bash native/fetch_native.sh --update-patches   # rewrites <component>/<layer>.patch
bash native/fetch_native.sh --check            # fetch + patch == installed tree
```

New upstream files must be added to the matching `.files` manifest first;
files listed there that you delete become deletion hunks.
