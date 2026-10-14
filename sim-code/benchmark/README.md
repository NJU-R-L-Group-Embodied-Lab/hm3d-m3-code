# `benchmark/` — HM3D-M3 episodes, runners and metrics

Run everything from `sim-code/` (`python -m benchmark.<module>`, `bash scripts/...`).
Environment setup and data download are in [`../HOW_TO_RUN.md`](../HOW_TO_RUN.md);
the metrics are defined in [`metrics.py`](metrics.py).

## Scripts

| Run | Script | Episodes / IIN images |
| --- | --- | --- |
| Val, LaViRA-M3 | `scripts/main_table/lavira.sh` | Val1000, `iin_images_official_val1000_mixed3_stretch512` |
| Val, VLFM / EGO-Swarm / Co-NavGPT2 / DM3-Nav | `scripts/main_table/od_baselines.sh` | Val1000, `iin_images_official_val1000_mixed3_stretch_v2` |
| Mini-100 (quick test) | `scripts/mini100/lavira.sh`, `scripts/mini100/od_baselines.sh` | Mini-100, stretch512 |
| Episode generation | `scripts/gen/` | — |
| Offline self-tests | `scripts/run_selftests.sh` | — |

The four adapted baselines use the `stretch_v2` IIN images on Val; the scripts
keep that default. Everything else
uses the `stretch512` images (512×512, HFOV 42°, camera 1.31 m, 1.8 m standoff).

All row scripts call the generic runner `scripts/run_method.sh`:

```bash
# one or more methods on one episode file; args after -- go to every worker
EP=data/datasets/ma_objectnav_ovon/val_unseen/ma_official_mini_100_samefloor_mixed3_kmquota_seed1123.json \
N=100 WORKERS=8 GPUS_CSV=0,1 OUT_ROOT=logs/my_run \
bash scripts/run_method.sh lavira_full -- \
  --iin-images-dir data/datasets/ma_objectnav_ovon/val_unseen/iin_images_official_mini100_mixed3_stretch512 \
  --skip-pano-viz --seed 0
```

| Method | Runner and flags |
| --- | --- |
| `lavira_full` | `ma_search --use-vlm --clean-found-state --disable-path-budget --control-mode k2-async-keep-goals-pa --no-team-topdown-map --la-thinking-budget 1024 --no-la-thinking` |
| `vlfm` | `zeroshot_runner --method vlfm` |
| `ego_swarm` | `ego_runner` |
| `conav` | `conav_runner --vlm qwen` |
| `dm3` | `dm3_official.runner` |

`ma_search --control-mode` also accepts the alias `full`. LaViRA-M3 double-checks
a candidate with the VLM before declaring it; the adapted baselines declare goals
from open-vocabulary detection (`od_found.py`, OWL sidecar).

`run_method.sh` writes `OUT_ROOT/<method>/w<k>/episode_*.json`, merges them into
`metrics_list.json`, `aggregate.json`, `summary.json` and `DONE`, and refreshes
`OUT_ROOT/fair_comparison.{md,json}`. Re-running the same command resumes:
complete methods are skipped and, for LaViRA (`DYNAMIC_QUEUE=1`), finished
episodes are not run again. `MERGE_ONLY=1` only re-merges; `DRY_RUN=1` prints the
worker commands.

### Environment variables

| Variable | Default | Used by |
| --- | --- | --- |
| `LLM_API_KEY` | — (fallbacks `DASHSCOPE_API_KEY`, `LAVIRA_API_KEY`) | LaViRA, Co-NavGPT2, TN generation |
| `LLM_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` (fallback `DASHSCOPE_BASE_URL`) | same |
| `LLM_MODEL` | `qwen3.5-27b` | same |
| `PY` | `python` | all scripts |
| `GPUS_CSV`, `WORKERS`, `N`, `EP`, `OUT_ROOT`, `EXTRA_MODULE_ARGS` | per script | `run_method.sh` and row scripts |
| `OWL_URL` or `OWL_HOST` + `OWL_PORT_BASE` | — / `18210` (sidecar on GPU g listens on `OWL_PORT_BASE+g`) | adapted baselines |
| `OWL_SHM_DIR` | `$PWD/.owl_shm` | adapted baselines + OWL sidecar |
| `VLFM_BLIP2_DIR` | `data/models/blip2-itm-vit-g` | VLFM |
| `DM3_SUPERGLUE_WEIGHTS_DIR`, `DM3_DETIC_CHECKPOINT` | see `dm3_official/README.md` | DM3-Nav |
| `MIN_MEM_GB` | `0` (off) | optional free-memory guard in the row scripts |
| `SKIP_PREFLIGHT`, `DRY_RUN` | `0` | row scripts |

## Episode generation (`scripts/gen/`)

| Step | Command |
| --- | --- |
| Val1000 + Mini-100 geometry (seed 1123, same-floor, mixed3, K∈{2,3,4}, M∈{3,4,5}) | `bash scripts/gen/val1000_mini100.sh` (`generate` → `sample_official`) |
| Train geometry | `SPLIT=t2000 bash scripts/gen/train.sh`, `SPLIT=t14000 bash scripts/gen/train.sh` |
| IIN images + TN text | `EP=... IIN_OUT=... CAMERA=stretch512 bash scripts/gen/fill_iin_tn.sh` (`render_iin` + `gen_tn --sample-mode goat`, sidecars merged by `scripts/gen/merge_tn_sidecars.py`) |
| Val1000 images for the adapted baselines | same with `CAMERA=stretch_v2 DO_TN=0` |

## Module map

| Area | Modules |
| --- | --- |
| Episode generation | `generate` (+ navmesh geometry), `sample_official`, `config` (+ difficulty labels), `routing`, `goal_sources`, `rebake_navmesh` |
| Goal content | `render_iin` (IIN images, GOAT camera sampling, semantic goal instances), `gen_tn` (TN text), `iin_verify` |
| LaViRA-M3 | `ma_search` (+ `async_board_pa`, `pano_views`) and `../vlnce_baselines/` |
| Adapted baselines | `zeroshot_runner` + `vlfm_value` (VLFM), `ego_runner` (EGO-Swarm), `conav_runner` (Co-NavGPT2), `dm3_official/` (DM3-Nav) |
| Detector FOUND | `od_found`, `ovon_od`, `owl_shm`, `dm3_perception` |
| Evaluation | `metrics`, `coord_shell`, `work_queue`, `traj_log`, `frontier` |
| Credentials | `api_auth` (`LLM_*` environment variables; keys are never stored in code) |

## Self-tests

The self-tests live in `../tests/selftests/` (outside this package).
`bash scripts/run_selftests.sh` (a wrapper around `python tests/run_all.py`) runs every test listed in `tests/run_all.py` plus
`tests/golden/check_lavira_prompts.py`, and prints PASS / FAIL / SKIP per test
(SKIP = a missing package of the test's tier or a missing data file). Single
tests run with `python -m tests.selftests.<name>_selftest` from `sim-code/`.
