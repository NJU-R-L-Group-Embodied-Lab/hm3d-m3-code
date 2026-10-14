# Third-party code for the DM3-Nav baseline

`benchmark/dm3_official/runner.py` runs the official DM3-Nav agent stack
(GOAT-era `home_robot`: semantic map, instance memory, Detic/CLIP/SuperGlue
matching, map merger, FMM frontier planner). That code is **not** stored in this
repository. `fetch.sh` clones each upstream at a pinned commit, applies the
small HM3D-M3 patches in `patches/`, and assembles the package tree the runner
imports.

```bash
cd sim-code
SUPERGLUE_DIR=/path/to/SuperGluePretrainedNetwork bash third_party/fetch.sh
```

Output (both git-ignored):

- `third_party/src/<name>/`: pinned clones (`dm3_nav`, `home_robot`, `detic`,
  `centernet2`) and the patched SuperGlue copy (`superglue`)
- `third_party/home_robot/`: the assembled tree; its `home_robot/` package is
  what the runner adds to `sys.path` (override with `DM3_HOME_ROBOT_DIR`)

## Pinned upstreams

| Name | Upstream | Commit | License | Used for |
| --- | --- | --- | --- | --- |
| `dm3_nav` | https://github.com/Multi-Agent-Robotics-and-Autonomy-Lab/DM3_Nav | `5037cf364b3757739d5f7b443b4aa9acb9f2dc9c` | MIT | `home_robot` agent / mapping / planner / perception, and the Detic tree it ships (Detic + CenterNet2 + Deformable-DETR) |
| `home_robot` | https://github.com/facebookresearch/home-robot (branch `goat`) | `b856925e11d888848c514d0c2509a07f298ceb44` | MIT | `home_robot/core/` only |
| `detic` | https://github.com/facebookresearch/Detic | `46f8317a0688b4b94e85816b5839a1dbca26d1e0` | Apache-2.0 | `datasets/` and `detic/data/` only |
| `centernet2` | https://github.com/xingyizhou/CenterNet2 | `8745e012e4dbdf560ac2f27e0b771d4907ad4aaf` | Apache-2.0 | `datasets/` and `centernet/data/` only |
| `superglue` | https://github.com/mukulkhanna/SuperGluePretrainedNetwork | `d2d6dcc21f67690c07a2787acfb357b7ed59b920` | Magic Leap, **non-commercial research only** | SuperPoint + SuperGlue image matching; **user-supplied** |

Why several repos: DM3_Nav's own `.gitignore` (`core`, `data/`, `datasets/`)
kept `home_robot/core/` and Detic's / CenterNet2's `data/` and `datasets/`
directories out of the published repo. `fetch.sh` restores them from the
upstream commits that DM3_Nav is built on (home-robot `goat`, and the Detic /
CenterNet2 commits pinned as Detic submodules). Deformable-DETR (Detic
submodule `11169a60c33333af00a4849f1808023eba96a931`, Apache-2.0) comes from the
copy inside DM3_Nav.

Only the subset that the runner needs is assembled (see the layout section of
`fetch.sh`). detectron2 is not part of this tree: the runner adds a prebuilt
detectron2 @ 38af375 from `data/datasets/dm3_official/detectron2/` if present.

## SuperGlue (manual step)

SuperGluePretrainedNetwork is licensed by Magic Leap for academic or non-profit
non-commercial research use only (see Part 5 of the
top-level `LICENSE`). It is never
downloaded automatically and its code and weights are not redistributed here.
If you accept its license:

```bash
git clone https://github.com/mukulkhanna/SuperGluePretrainedNetwork /path/to/sgpn
git -C /path/to/sgpn checkout d2d6dcc21f67690c07a2787acfb357b7ed59b920
SUPERGLUE_DIR=/path/to/sgpn bash third_party/fetch.sh
```

`fetch.sh` copies the code (without `.git`, `assets/` and `models/weights/`)
and patches the copy, never your checkout. Put `superpoint_v1.pth` and
`superglue_indoor.pth` in `data/datasets/dm3_official/superglue_weights/` or
point `DM3_SUPERGLUE_WEIGHTS_DIR` at them. Without SuperGlue the DM3-Nav
baseline cannot run; the rest of the benchmark is unaffected.

## Patches (`patches/<name>/*.patch`, applied with `patch -p1`)

| Patch | Effect |
| --- | --- |
| `dm3_nav/0001-detic-checkpoint-env-override.patch` | `DM3_DETIC_CHECKPOINT` selects the Detic checkpoint (default path unchanged) |
| `dm3_nav/0002-goat-agent-gate-hsv-debug-dump.patch` | Upstream saves a ~3.5 MB HSV `.npy` every step; only do it at `visualization_level > 2` (debug I/O only) |
| `dm3_nav/0003-planner-empty-instance-map-no-assert.patch` | An empty goal-instance raster (stale Detic instance id) is a plan-to-instance failure instead of an `assert` that kills the worker |
| `superglue/0001-weights-dir-env-override.patch` | `DM3_SUPERGLUE_WEIGHTS_DIR` selects the weights directory (default path unchanged) |

Patched lines are marked `# lavira:` in the code.

## Offline / mirrors

Set `FETCH_MIRROR_DM3_NAV`, `FETCH_MIRROR_HOME_ROBOT`, `FETCH_MIRROR_DETIC` or
`FETCH_MIRROR_CENTERNET2` to a local clone (or another URL) that contains the
pinned commit. `THIRD_PARTY_SRC` and `THIRD_PARTY_OUT` move the clone and
output directories. Re-running `fetch.sh` reuses clones, skips patches that are
already applied, and rebuilds the assembled tree.

## Licenses

The upstream license texts are in Part 5 of the top-level [`LICENSE`](../../LICENSE):
DM3_Nav and home-robot (MIT, Meta Platforms, Inc. and affiliates), Detic, CenterNet2
and Deformable-DETR (Apache-2.0), and SuperGlue (Magic Leap non-commercial; included
for reference only, the code is not redistributed). Each patched file carries a
`# lavira:` change marker; no Apache-2.0 file (Detic, CenterNet2, Deformable-DETR) is
modified. The texts were checked byte-for-byte against the `LICENSE` files of the
pinned upstream commits (2026-10-13).
