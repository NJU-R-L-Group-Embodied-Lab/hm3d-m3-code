# How to run HM3D-M³ (simulation)

Run every command in this file from `sim-code/`. The tree has no scene meshes, episode data or
model weights. Sections 3 and 4 explain where to get them.

Shortest path (setup, data, LaViRA-M³ reproduction in a few commands): [`AGENT_SETUP.md`](AGENT_SETUP.md).

## 1. Environment

The paper numbers were produced in this environment. `docker/Dockerfile` rebuilds it from scratch, and `environment.yml` rebuilds it without Docker. Both run the same commands. The native install was verified end to end (self-tests, imports of every runner, and a GPU EGO-Swarm smoke run). The Docker build itself has not been run.

| Component | Version |
| --- | --- |
| Base image | `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04` (CUDA 11.8, cuDNN 8) |
| Python | 3.8.20 (conda env `m3`) |
| PyTorch | 2.4.1+cu118, torchvision 0.19.1+cu118 |
| habitat-sim | 0.1.7, headless (EGL), no Bullet, commit `856d4b0`. The paper image built this commit from source; the recipe uses the conda build of the same commit (`aihabitat::habitat-sim=0.1.7=py3.8_headless_linux_856d4b0…`) |
| habitat-lab | 0.1.7 (`v0.1.7` / `challenge-2021`, commit `d6ed1c0`) + `docker/patches/habitat-lab-vln.patch` |
| GroundingDINO | commit `57535c5`, CUDA ops built with nvcc 11.8. **Required by every LaViRA-M³ row** |
| detectron2 | 0.6, built from source (DM³-Nav only) |
| Python packages | `requirements.txt` (core + LaViRA-M³), `requirements-baselines.txt` (adapted baselines) |
| OWL sidecar | separate image `docker/owl.Dockerfile`: Python 3.10, torch 2.7.1+cu118, transformers, Flask |
| Hosted VLM | Qwen3.5-27B (`qwen3.5-27b`) through an OpenAI-compatible endpoint (Aliyun Bailian / DashScope) |

The habitat-lab patch makes two changes:

- `InstructionSensor` gets an empty `Dict` observation space, so the ObjectNav configs load.
- `tensorflow==1.13.1` is removed from `habitat_baselines/rl/requirements.txt`.

The image also installs an EGL vendor file (`docker/10_nvidia.json`), which headless rendering needs.

Requirements:

- Linux x86_64.
- An NVIDIA driver that supports CUDA 11.8.
- Docker with the NVIDIA Container Toolkit and the `graphics` capability, which EGL needs.
- The Docker Compose v2 plugin (`docker compose ...`; check with `docker compose version`). The legacy standalone `docker-compose` v1 does not read the GPU reservations in `docker/compose.yaml`. If the plugin is missing, install the `docker-compose-plugin` package, or put the `docker-compose-linux-x86_64` binary from the Compose releases page in `~/.docker/cli-plugins/docker-compose`.
- About 30 GB of disk for the image.

## 2. Build and start the containers

```bash
cp .env.example .env              # then edit: LLM_API_KEY, DATA_ROOT, MODEL_ROOT, OWL_GPUS, HOST_UID/HOST_GID
docker compose -f docker/compose.yaml --env-file .env build          # images hm3d-m3:dev, hm3d-m3-owl:dev
docker compose -f docker/compose.yaml --env-file .env up -d m3       # simulation container
docker compose -f docker/compose.yaml --env-file .env exec m3 bash   # shell in /workspace/sim
```

- The `m3` service bind-mounts `sim-code/` at `/workspace/sim`.
- It mounts `DATA_ROOT` at `sim-code/data` and `MODEL_ROOT` at `sim-code/data/models`.
- It uses the host network and `shm_size: 32g`.
- It runs as `HOST_UID:HOST_GID` from `.env` (set them to `id -u` / `id -g`), so `logs/`, `.owl_shm/` and `__pycache__/` in your checkout stay yours. Left empty, the container runs as root and those files become root-owned (see section 10). The `owl` service always runs as root, so `logs/owl/` is root-owned either way.
- Build with `WITH_BASELINES=0` to skip detectron2 and `requirements-baselines.txt`. LaViRA-M³ does not need them. GroundingDINO and `segment_anything` are always installed, because LaViRA-M³ imports them.
- The build downloads from GitHub (micromamba, habitat-lab, GroundingDINO, segment-anything, detectron2), conda-forge / aihabitat, PyPI and download.pytorch.org. Behind a proxy, pass `--network host --build-arg HTTPS_PROXY=... --build-arg HTTP_PROXY=...`. Inside a running container, pip reads the lowercase `https_proxy`/`http_proxy` first, so set both the lowercase and uppercase variables.
- Image sizes (compressed `docker save`): `hm3d-m3:dev` about 9.6 GB, `hm3d-m3-owl:dev` about 5.2 GB. Reserve about 40 GB in the Docker data root for the build. The base images are `nvidia/cuda:11.8.0-cudnn8-{devel,runtime}-ubuntu22.04` from Docker Hub; if Docker Hub is not reachable, pass `--build-arg CUDA_IMAGE=<mirror>/nvidia/cuda:...` with the same tag.

**Without Docker.** The header of `environment.yml` lists every step. It covers the conda env (Python 3.8.20 + habitat-sim + torch + `requirements.txt`), a CUDA 11.8 `nvcc` for the GroundingDINO / detectron2 builds (a CUDA 12.x `nvcc` is rejected by torch 2.4.1+cu118), and the habitat-lab and GroundingDINO steps. Then set these variables yourself:

- `EGL_PLATFORM=surfaceless`
- `MAGNUM_LOG=quiet`
- `HABITAT_SIM_LOG=quiet`
- `PYTHONNOUSERSITE=1`, so that packages in `~/.local` cannot shadow the pins.

**Which packages each method needs:**

| Methods | Needs |
| --- | --- |
| LaViRA-M³ (all rows) | `requirements.txt` + GroundingDINO. `vlnce_baselines/ZS_Evaluator_mp.py` imports `groundingdino` and `segment_anything` at module level, so every LaViRA-M³ row fails at its first VLM step without them. The GroundingDINO weights under `data/grounded_sam/` are not loaded by any shipped method. |
| VLFM, EGO-Swarm, Co-NavGPT2 | additionally `requirements-baselines.txt` (CLIP), plus the OWL sidecar (section 7), which needs OWLv2 |
| DM³-Nav | additionally `requirements-baselines.txt` (fvcore, hydra, `psutil`, ...), detectron2 0.6, and the third-party code from `third_party/fetch.sh` (section 4). Without these, `python -m benchmark.dm3_official.runner` fails at import (`No module named 'home_robot'`, then `'psutil'`). |

## 3. Data

Data is distributed separately from the code. It is CC BY-NC 4.0 and HM3D's terms apply (Part 2 of [`../LICENSE`](../LICENSE)).

| Package | Contents | Where |
| --- | --- | --- |
| Val (1,000 episodes) | episodes + IIN images `stretch512` and `stretch_v2` | GitHub Releases: `https://github.com/NJU-R-L-Group-Embodied-Lab/hm3d-m3-code/releases/tag/v1.0.0` |
| Mini-100 | episodes + IIN images `stretch512` | GitHub Releases: `https://github.com/NJU-R-L-Group-Embodied-Lab/hm3d-m3-code/releases/tag/v1.0.0` |
| Train | train episodes + IIN images `stretch512` | Hugging Face: `https://huggingface.co/datasets/skyhyding/HM3D-M3` |
| Teacher samples (100K) | observation–assignment samples | Hugging Face: `https://huggingface.co/datasets/skyhyding/HM3D-M3` |

Get HM3D v0.2 (val split, with semantics) from Matterport under the HM3D license.

[`download_data.sh`](download_data.sh) (in this directory) downloads, verifies and installs the data:

```bash
bash download_data.sh                         # Val-1000 + Mini-100 (GitHub Release)
bash download_data.sh mini                    # only Mini-100 (quick tests)
bash download_data.sh val --data-root /data/hm3d_m3
bash download_data.sh none --train --teacher  # Hugging Face: train split, 100K teacher samples
```

- Packages: `val` (Val-1000 episodes, IIN `stretch512` and `stretch_v2`), `mini` (Mini-100, IIN `stretch512`), `all` (default), `none`.
- Installs under `--data-root` (default `$DATA_ROOT`, else `sim-code/data`): episodes decompressed to `.json`, each IIN pack extracted to its folder (layout below). HM3D scenes are not downloaded.
- Source: `--base-url` / `$DATA_BASE_URL`, else the GitHub Release link in the top-level `README.md`. curl honors `https_proxy` / `HTTPS_PROXY`.
- Every asset is checked against `SHA256SUMS` before it is installed; a mismatch is kept as `<name>.bad` and the script fails. Re-running is cheap: verified downloads in `<data-root>/downloads/` are reused (`--clean` deletes them, `--force` re-installs).
- `--train` / `--teacher` use `hf download` (or `huggingface-cli download`) on the dataset linked in `README.md` (`--hf-repo` to override). The dataset is gated: accept its terms and log in first. Without the CLI the script prints the manual commands and exits with status 3. Train data goes to `train/`; the teacher samples (`teacher_100k/`) stay in the snapshot directory (`--hf-dir`).

Episode files are shipped as `.json.gz`; the loaders read plain JSON (`download_data.sh` decompresses them; after a manual download run `gunzip`). Each IIN pack contains a merged `manifest.json`; the baseline runners find IIN goal images only through that file.

```text
sim-code/data/                                   (= DATA_ROOT)
  scene_datasets/hm3d-1.0/hm3d_v0.2/        HM3D scenes + hm3d_annotated_basis.scene_dataset_config.json
  datasets/ma_objectnav_ovon/
    val_unseen/
      ma_official_val_1000_samefloor_mixed3_seed1123.json
      ma_official_mini_100_samefloor_mixed3_kmquota_seed1123.json
      iin_images_official_val1000_mixed3_stretch512/
      iin_images_official_val1000_mixed3_stretch_v2/
      iin_images_official_mini100_mixed3_stretch512/
    train/
      ma_train_16000_samefloor_mixed3_seed2026_2027.json
      iin_images_train16000_mixed3_stretch512/
  datasets/dm3_official/                    DM³-Nav weights (section 4)
  models/                                   (= MODEL_ROOT, section 4)
```

## 4. Model weights

| Weights | Needed by | Location (relative to `sim-code/`) | How to get them |
| --- | --- | --- | --- |
| BLIP-2 ITM `Salesforce/blip2-itm-vit-g` | VLFM | `data/models/blip2-itm-vit-g/` (`VLFM_BLIP2_DIR`) | `huggingface-cli download Salesforce/blip2-itm-vit-g --local-dir data/models/blip2-itm-vit-g` |
| OWLv2 `google/owlv2-base-patch16-ensemble` | OWL sidecar (4 baselines) | HF cache `data/models/hf` (`HF_HOME`) | downloaded on first start; afterwards set `HF_HUB_OFFLINE=1` |
| CLIP `openai/clip-vit-base-patch32` (transformers) | OWL sidecar `/score_lang` | HF cache `data/models/hf` | same as OWLv2 |
| CLIP ViT-B/32 (`import clip`) | EGO-Swarm / Co-NavGPT2 TN matching, DM³-Nav | `data/models/clip/ViT-B-32.pt` (mounted at `$HOME/.cache/clip`; `HOME=/opt/m3home` in the image; `clip.load` ignores `XDG_CACHE_HOME`) | downloaded on first use |
| torchvision Mask R-CNN R50-FPN (COCO) | EGO-Swarm / Co-NavGPT2 perception | `data/models/torch` (`TORCH_HOME`) | downloaded on first use |
| Detic `Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.pth` | DM³-Nav | `data/datasets/dm3_official/detic.pth` (`DM3_DETIC_CHECKPOINT`) | Detic model zoo |
| SuperPoint `superpoint_v1.pth`, SuperGlue `superglue_indoor.pth` | DM³-Nav, OWL sidecar `/match_image` | `data/datasets/dm3_official/superglue_weights/` (`DM3_SUPERGLUE_WEIGHTS_DIR`) | **you provide them** (Magic Leap non-commercial license) |

DM³-Nav also needs its third-party code, assembled once:

```bash
SUPERGLUE_DIR=/path/to/SuperGluePretrainedNetwork bash third_party/fetch.sh
```

[`third_party/README.md`](third_party/README.md) covers that step.

## 5. Environment variables (`.env`)

| Variable | Default | Meaning |
| --- | --- | --- |
| `LLM_API_KEY` | — | VLM key. Required for LaViRA-M³, Co-NavGPT2 and TN generation. Fallbacks: `DASHSCOPE_API_KEY`, `LAVIRA_API_KEY` |
| `LLM_BASE_URL` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | OpenAI-compatible endpoint |
| `LLM_MODEL` | `qwen3.5-27b` | VLM name |
| `PY` | `python` | interpreter used by `scripts/` |
| `OWL_GPUS`, `OWL_HOST`, `OWL_PORT_BASE`, `OWL_URL` | `0`, `127.0.0.1`, `18210`, — | OWL sidecar location. The sidecar on GPU g listens on `OWL_PORT_BASE + g` |
| `OWL_SHM_DIR` | `$PWD/.owl_shm` | frame directory shared with the sidecar. Compose uses `/workspace/sim/.owl_shm` |
| `VLFM_BLIP2_DIR` | `data/models/blip2-itm-vit-g` | BLIP-2 ITM weights |
| `DM3_SUPERGLUE_WEIGHTS_DIR`, `DM3_DETIC_CHECKPOINT` | see section 4 | DM³-Nav weights |

Keys are read only from the environment. Never commit `.env`.

## 6. Offline self-tests (no GPU, no data, no API key)

```bash
bash scripts/run_selftests.sh               # = python tests/run_all.py
bash scripts/run_selftests.sh --tier cv     # light tiers only: numpy/pillow + opencv/matplotlib
python tests/run_all.py --list              # every test and its dependency tier
# or a single one:
python -m tests.selftests.benchmark_selftest
```

The tests live in `tests/selftests/`; the list and tiers (`minimal`, `cv`,
`torch`, `habitat`, `home_robot`) are in `tests/run_all.py`. A test whose tier's
packages or data file are missing is reported as SKIP. In the project
environment, expect every test to pass except `dm3_runner_selftest` (needs the
DM³ `home_robot` fork).

## 7. OWL detector sidecar (adapted baselines only)

```bash
docker compose -f docker/compose.yaml --env-file .env up -d owl
docker compose -f docker/compose.yaml --env-file .env ps owl          # waits for "healthy"
curl -fsS http://127.0.0.1:18210/health                                # one port per GPU: 18210 + gpu
```

The `owl` service starts one sidecar per GPU in `OWL_GPUS`. A healthy sidecar returns `"ok": true` and `"superglue": true`. It needs OWLv2 (`google/owlv2-base-patch16-ensemble`) and CLIP (`openai/clip-vit-base-patch32`) in its HF cache, and SuperGlue (section 4).

Without Docker, the sidecar runs in its own Python 3.10 environment (torch 2.7.1+cu118 + `docker/owl-requirements.txt`). Do not run it from the Python 3.8 simulation env: the paper's detector gate used transformers 5.15.1, and that env pins 4.46.3. [`services/README.md`](services/README.md) has the commands and the protocol.

## 8. Running the benchmark

Run these inside the `m3` container.

```bash
# Val (1,000 episodes)
bash scripts/main_table/lavira.sh          # LaViRA-M3
bash scripts/main_table/od_baselines.sh    # VLFM, EGO-Swarm, Co-NavGPT2, DM3-Nav (needs the OWL sidecar)

# Mini-100 (quick test)
bash scripts/mini100/lavira.sh
bash scripts/mini100/od_baselines.sh

# A single method (EP = episode file, IIN_DIR = the matching IIN image folder; both required)
EP=data/datasets/ma_objectnav_ovon/val_unseen/ma_official_mini_100_samefloor_mixed3_kmquota_seed1123.json \
IIN_DIR=data/datasets/ma_objectnav_ovon/val_unseen/iin_images_official_mini100_mixed3_stretch512 \
  bash scripts/run_method.sh <lavira_full|vlfm|ego_swarm|conav|dm3> [extra flags]
```

The header of each script lists its settings, such as GPUs, workers and output directory. Each run writes per-episode metrics and an `aggregate.json`. [`benchmark/metrics.py`](benchmark/metrics.py) defines the metrics.

The launch scripts set each method's FOUND rule and IIN image variant; keep their defaults when comparing with the paper. LaViRA-M³ and Co-NavGPT2 call a hosted VLM, so re-runs and other model snapshots give somewhat different numbers.

## 9. Generating episodes

`scripts/gen/val1000_mini100.sh` and `scripts/gen/train.sh` regenerate the episode files; `scripts/gen/fill_iin_tn.sh` renders the IIN goal images and writes the TN descriptions (TN needs `LLM_API_KEY`). The header of each script lists its settings.

## 10. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `GL::Context: cannot retrieve OpenGL version` (native install) | conda-forge's `libglvnd` is in the env and shadows the system one. Run `micromamba remove --force libegl libgl libglx libglvnd` in the env (see `environment.yml`). |
| `unable to create EGL context` / `no CUDA-capable device` from habitat-sim | Pass the GPU with the `graphics` capability (compose does this) and keep `EGL_PLATFORM=surfaceless`. Check that `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` exists. |
| `OWL_URL is required for ovon_od_detector_stop found` | Start the `owl` service, then set `OWL_URL` or `OWL_HOST`/`OWL_PORT_BASE`. |
| Sidecar is healthy but no IIN goal is ever found | `"superglue": false` in `/health`. Provide SuperGlue (section 4). |
| Two runs interfere: wrong or missing detections, odd `od_<pid>` frames | The sidecar exchanges frames through `OWL_SHM_DIR` in files named by process id. Containers that share one `OWL_SHM_DIR` must share the host PID namespace (`--pid host`; compose sets `pid: host`), or give each container its own `OWL_SHM_DIR` and sidecar. |
| Baseline log shows `IIN imgs=0`, or DM³-Nav fails in SuperGlue with `NoneType.astype` | The IIN folder has no merged `manifest.json`. Use the released IIN packs (they include one). |
| DM³-Nav: `ModuleNotFoundError: home_robot` / `psutil` | Run `third_party/fetch.sh` and install `requirements-baselines.txt`. DM³-Nav also loads CLIP ViT-B/32 from `~/.cache/clip` (section 4). |
| Files under `sim-code/` (`logs/`, `.owl_shm/`, `__pycache__/`) are owned by root after a compose run | The services ran as root: `m3` because `HOST_UID`/`HOST_GID` were empty in `.env`, and `owl` always does (it writes `logs/owl/`). Set `HOST_UID`/`HOST_GID` and recreate `m3` (`up -d --force-recreate m3`). To take back existing files without sudo: `docker run --rm -v "$PWD:/w" hm3d-m3:dev chown -R "$(id -u):$(id -g)" /w`. |
| VLM 401 / empty answers | Check `LLM_API_KEY`, `LLM_BASE_URL` and `LLM_MODEL`. |
| LaViRA worker logs show `Error code: 429 ... limit_burst_rate` followed by `Forcing retry (1/5)` | DashScope limits how fast a key's request rate may ramp up, and all `WORKERS` (default 32) start calling the VLM at once. The client waits 30 s and retries (up to 5 times), so a few 429s at start-up are harmless. If logs show `Max retries (5) reached`, that step got no VLM answer: lower `WORKERS` (e.g. 8–16), especially on a new key, and rerun the affected episodes. |
