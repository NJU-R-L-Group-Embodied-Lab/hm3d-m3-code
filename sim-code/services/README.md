# Services

## `owl_sidecar.py`: OWL detector sidecar

The four adapted baselines (VLFM, EGO-Swarm, Co-NavGPT2, DM³-Nav) declare a goal
FOUND with the shared detector gate `ovon_od_detector_stop`
(`benchmark/od_found.py`, `benchmark/ovon_od.py`). The gate does not load any
detector in the Habitat process: it calls this HTTP sidecar instead.

| Endpoint | Model | Used for |
| --- | --- | --- |
| `GET /health` | — | readiness check |
| `POST /detect`, `POST /detect_batch` | OWLv2 (`google/owlv2-base-patch16-ensemble`) | ON / TN / IIN category detection |
| `POST /match_image` | SuperPoint + SuperGlue (indoor) | IIN goal-image matching |
| `POST /score_lang` | CLIP (`openai/clip-vit-base-patch32`, HF transformers) | TN description scoring |

LaViRA-M³ does not use the sidecar (its FOUND decision is Verifier V2).

### Frames

Habitat workers write raw RGB frames to files in `OWL_SHM_DIR`
(`benchmark/owl_shm.py`) and send only the slot name. The sidecar and the
simulation process must therefore see the same directory **at the same path**.
`docker/compose.yaml` mounts `sim-code/.owl_shm` at `/workspace/sim/.owl_shm` in both
containers. Requests may also carry a base64 JPEG instead.

### Running

With Docker (recommended), from `sim-code/`:

```bash
docker compose -f docker/compose.yaml --env-file .env up -d owl
# one sidecar per GPU in OWL_GPUS, on port OWL_PORT_BASE + gpu (default 18210 + gpu)
curl -fsS http://127.0.0.1:18210/health
```

Without Docker, use a separate Python 3.10 environment (verified with micromamba; the
sidecar must not share the Python 3.8 simulation env, whose transformers pin differs):

```bash
micromamba create -y -p ./envs/owl -c conda-forge --override-channels python=3.10 pip
export PYTHONNOUSERSITE=1          # keep ~/.local packages out of the env
./envs/owl/bin/pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
./envs/owl/bin/pip install -r docker/owl-requirements.txt
# one sidecar per GPU, exactly as in the image (OWL_SIDECAR / OWL_LOG_DIR override the /workspace paths):
PATH=$PWD/envs/owl/bin:$PATH OWL_GPUS=0 OWL_SIDECAR=$PWD/services/owl_sidecar.py \
  OWL_SHM_DIR=$PWD/.owl_shm OWL_LOG_DIR=$PWD/logs/owl bash docker/owl-entrypoint.sh
```

With a pre-populated HF cache set `HF_HUB_OFFLINE=1`.

Then point the simulation side at it: `OWL_URL=http://127.0.0.1:18210` and the
same `OWL_SHM_DIR`. The scripts in `scripts/` derive `OWL_URL` per GPU from
`OWL_HOST` and `OWL_PORT_BASE`.

### Options

| Flag / env | Default | Meaning |
| --- | --- | --- |
| `--model` / `OWL_MODEL` | `google/owlv2-base-patch16-ensemble` | OWLv2 checkpoint (HF id or local dir) |
| `--clip-model` / `CLIP_MODEL` | `openai/clip-vit-base-patch32` | CLIP checkpoint |
| `--device` / `OWL_DEVICE` | `cuda` | torch device |
| `--host` / `OWL_BIND_HOST`, `--port` / `OWL_PORT` | `127.0.0.1`, `18210` | bind address |
| `--shm-dir` / `OWL_SHM_DIR` | — | frame directory |
| `--max-batch` / `OWL_MAX_BATCH`, `--batch-wait-ms` / `OWL_BATCH_WAIT_MS` | `8`, `8` | request batching |
| `--no-match` | off | load OWLv2 only (no SuperGlue, no CLIP) |
| `--skip-clip`, `--skip-superglue` | off | skip one matcher |
| `OWL_SUPERGLUE_DIR` | `third_party/src/superglue` | SuperGluePretrainedNetwork checkout with the HM3D-M3 weights-dir patch |
| `DM3_SUPERGLUE_WEIGHTS_DIR` | `data/datasets/dm3_official/superglue_weights` | `superpoint_v1.pth`, `superglue_indoor.pth` |

The paper numbers were produced with all three models loaded (no `--no-match`,
`--skip-*`) and the default thresholds sent by `benchmark/ovon_od.py`.

SuperGlue is licensed by Magic Leap for non-commercial research only and is not
shipped or downloaded automatically; see `third_party/README.md`. Without it,
start the sidecar with `--skip-superglue`. `/match_image` then returns HTTP 400,
so IIN goals cannot be confirmed and the baseline numbers will not match the paper.
