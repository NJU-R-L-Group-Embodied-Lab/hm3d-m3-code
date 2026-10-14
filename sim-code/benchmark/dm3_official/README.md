# DM3-Nav official baseline on HM3D-M3

The official DM3-Nav algorithm core, ported into the LaViRA Habitat 0.1.7
stack. Entry point: `python -m benchmark.dm3_official.runner`
(run from `sim-code/`).

Fidelity table: [below](#fidelity-dm3-nav-official-ported-compared-with-the-official-code).

## Setup

The official code is not vendored. Fetch it at the pinned commits and apply
the HM3D-M3 patches:

```bash
cd sim-code
SUPERGLUE_DIR=/path/to/SuperGluePretrainedNetwork bash third_party/fetch.sh
```

This assembles `sim-code/third_party/home_robot/`, which `runner.py` puts on
`sys.path` (override with `DM3_HOME_ROBOT_DIR`). SuperGlue is under a
non-commercial license and must be supplied by the user; see
`sim-code/third_party/README.md`.

Weights and the compiled detectron2 tree are data, not code:

| What | Default (relative to `sim-code/`) | Env override |
| --- | --- | --- |
| Detic SwinB checkpoint | `data/datasets/dm3_official/detic.pth` | `DM3_DETIC_CHECKPOINT` |
| SuperPoint / SuperGlue weights | `data/datasets/dm3_official/superglue_weights/` | `DM3_SUPERGLUE_WEIGHTS_DIR` |
| detectron2 @ 38af375 (built `_C.so`) | `data/datasets/dm3_official/detectron2/` | (added to `sys.path` if present) |

Python packages the `lavira:v4` image lacks (install per run container):
`clip-anytorch pandas bresenham scikit-learn trimesh psutil fvcore iopath yacs
pycocotools portalocker termcolor tabulate cloudpickle`.

## Layout

- `runner.py`: `DM3OfficialRunner`, the official config (`build_dm3_config`) and the detector FOUND gate
- `sim-code/tests/selftests/dm3_runner_selftest.py`: adapter math, no Habitat

## Fidelity: DM3-Nav (official, ported) compared with the official code

Reference: official DM3_Nav @ 5037cf3, plus the directories its `.gitignore`
dropped, restored from the pinned upstream commits (see
`sim-code/third_party/README.md`). Port: `benchmark/dm3_official/runner.py` plus the
official tree `sim-code/third_party/home_robot/`, assembled verbatim by
`sim-code/third_party/fetch.sh` (with the small patches in
`sim-code/third_party/patches/`).

### A. Same as the official code (verbatim upstream code or line-by-line port)

| Component | Official source | Where in the port | Notes |
|---|---|---|---|
| Semantic mapping (log-odds accumulation, 5 cm cells, 48 m global map, min_obs_height 20 cm, raycast explored) | mapping/semantic/categorical_2d_semantic_map_module.py, map_state.py | verbatim upstream | Config values 1:1 (`build_dm3_config` in runner.py) |
| Instance memory (cross-frame 3D association, multi-view InstanceView, score = max over views) | mapping/semantic/instance_tracking_modules.py | verbatim upstream | MIN_PIXELS=50 / MIN_EDGE=10 (official changes kept) |
| Goal matching (category 0.4 / CLIP language 0.24 / SuperGlue image 24.5, memory-then-obs, score agg = max) | agent/goat_agent/goat_matching.py | verbatim upstream | CLIP = ViT-B/32 (clip-anytorch, same OpenAI weights) |
| Detector: Detic SwinB open vocabulary (custom vocab) | perception/detection/detic/ + assets/detic.pth | verbatim upstream + same weights | detectron2@38af375 recompiled in lavira:v4; checkpoint path made an env var (patch dm3_nav/0001) |
| SuperGlue (indoor weights, SuperPoint 1024 keypoints, match_threshold 0.2, confidence_sum) | agent/imagenav_agent/{superglue.py,SuperGluePretrainedNetwork} | verbatim upstream (same weights) | Only the weights path is made an env var (patch superglue/0001) |
| Map merging (per-cell max absolute magnitude, r_comm = 5 m, per-neighbor cooldown 10 steps, exchange of intent / tasks_done / active_task, tie-breaking yield rule) | agent/goat_agent/{map_merger.py,multiagent_goat_agent.py} | verbatim upstream | Identity transform in sim (official sim assumption) |
| Distance-weighted frontier `w=(d_neighbor+1e-4)/(d_self+1e-6)`; at the same position the k-th best frontier is chosen by agent_id priority | navigation_planner/fixed_discrete_planner.py:1163-1356 | verbatim upstream | |
| FMM local planning, collision map, forced STOP after >30 stuck steps, max_steps=500, panorama_start=12x30 deg | fixed_discrete_planner.py, fmm_planner.py, goat_agent.py | verbatim upstream | An empty goal-instance raster is a plan failure instead of an assert (patch dm3_nav/0003) |
| Main loop structure (synchronous two-agent step, idle STOP -> TURN_RIGHT, all-stuck forced STOP, handle_stop timing) | projects/habitat_goat/multiagent_eval_episode.py:135-228 | runner.py, ported line by line | |
| Task allocation (active_task selection, yielding/priority, cooldown) | multiagent_goat_agent.py:146-202, 363-398 | verbatim upstream | |

### B. Intentional protocol differences (HM3D-M3 protocol layer, same as the other main-table rows)

| Item | Official | This port | Reason |
|---|---|---|---|
| Success criterion | Success if within 1 m of a GT viewpoint on the env side (GT leak) | Shared detector FOUND gate on the current RGB (OWL / CLIP crop / SuperGlue), then STOP within 0.9 m; validity is decided by the benchmark (within 1 m of a valid viewpoint) | Same gate as the other adapted baselines; no GT access |
| Behaviour after stop | handle_stop sets tasks_done=True directly | Official handle_stop; the stop is logged in `official_claims`, while found status comes from the detector gate | Keeps the official agent loop unchanged |
| Per-agent budget | Only max_steps=500 | Only max_steps=500 | Same as the official setting |
| Episode cap | env max_episode_steps=2500 (in practice the agent's 500 cap triggers first) | Outer cap 500 (same as the other rows' --max-episode-steps 500) | Consistent protocol |

### C. Environment / implementation differences (not algorithmic)

| Item | Official | This port | Impact |
|---|---|---|---|
| Habitat version | fork 0.2.x + habitat-sim 0.2.x | LaViRA habitat-lab 0.1.7 / sim 0.2.4 | Same HM3D v0.2 scenes and sensor parameters (640x480 hfov79 @ 0.88 m, turn 30 deg, forward 0.25 m); same action semantics |
| Depth near clip | Official `min_depth=0.0` (forced in eval) | LaViRA sensor `MIN_DEPTH=0.1`; the adapter de-normalizes over [_dmin, _dmax] and clips to 0-5 m | Differs only in the 0-10 cm near field |
| GPS/Compass frame | Fork sensors use each agent's own start frame (official runs start both agents at the same pose, so the frame is shared by construction) | Shared episode frame (origin = centroid of the agents' starts, heading = agent0 start heading) | HM3D-M3 starts are spread out (up to 8.7 m apart); in a shared frame the official identity-registration assumption holds exactly. With per-agent frames the identity merge would be wrong, so this is a correctness fix, not a deviation [interpretation] |
| IIN goal images | Rendered in sim from the image_goals camera parameters | Pre-rendered `iin_images/` (same parameters, LaViRA sim render) | Measured RMSE = 0.00 (v1 converter smoke test) |
| Teammate action during a spin | n/a | Teammate repeats its last MOVE_FORWARD, otherwise TURN_RIGHT in place [interpretation] | Keeps the teammate moving during the spin instead of freezing it |
| Spin frames not fed to the agent map | n/a | The 12 spin frames are not fed to semantic_map (mapping_planner=None in collect_and_verify) | Pure in-place rotation without translation; map loss is negligible [interpretation] |
| VISUALIZATION_LEVEL | 1 | 0 | Visualization only; also disables the official unconditional per-step HSV np.save (3.5 MB/step debug I/O, patch dm3_nav/0002) |
| Vocabulary | Official = all scene categories of the loaded split | = the category set of this runner's episode slice | Equivalent on the full 100 episodes; with sharding each shard's vocabulary differs slightly (as with the official per-scene sharding) |
| YOLO secondary verification | USE_YOLO=0, off by default | Not ported | Off in the official default config |
| val_seen official episodes | Official Goat-v1 split | Canonical 100 episodes used directly | Different task definition is part of the benchmark design |

### D. Known risks / to verify

- The 48 m map could overflow under extreme wandering (max centroid-to-goal distance 19.3 m, 4.7 m margin); watch for it in smoke/full runs.
- Detic's actual detection rate with a 33-class custom vocabulary (official GOAT vocabulary has 6 classes; ours has up to ~60); keep in mind during attribution.
- GPU memory and speed of two Detic + SuperGlue + CLIP + two agent maps (per-episode time to be measured in smoke runs).
