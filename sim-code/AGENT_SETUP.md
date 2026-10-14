# Agent setup: environment, data and LaViRA-M³ reproduction

Concrete steps for a coding agent (or anyone who wants the shortest path) to set
up the simulation environment, download the data and reproduce the LaViRA-M³
results. Background and options are in [`HOW_TO_RUN.md`](HOW_TO_RUN.md):
sections [1](HOW_TO_RUN.md#1-environment), [2](HOW_TO_RUN.md#2-build-and-start-the-containers),
[3](HOW_TO_RUN.md#3-data) and [8](HOW_TO_RUN.md#8-running-the-benchmark).

What you need from the user: a VLM API key (`LLM_API_KEY`, OpenAI-compatible
endpoint; default DashScope with `qwen3.5-27b`), a data directory (`DATA_ROOT`)
and their own copy of the HM3D v0.2 scenes (Matterport license; not
downloadable by script). Docker with the NVIDIA runtime and a GPU are required.

## 1. Clone and configure

```bash
git clone https://github.com/NJU-R-L-Group-Embodied-Lab/hm3d-m3-code hm3d-m3
cd hm3d-m3/sim-code
cp .env.example .env
# edit .env: LLM_API_KEY, DATA_ROOT, MODEL_ROOT, HOST_UID=$(id -u), HOST_GID=$(id -g)
```

## 2. Build and start the simulation container

```bash
docker compose -f docker/compose.yaml --env-file .env build
docker compose -f docker/compose.yaml --env-file .env up -d m3
```

LaViRA-M³ needs only the `m3` service; the OWL sidecar (section 7 of
`HOW_TO_RUN.md`) is for the adapted baselines.

## 3. Download the data and add HM3D

```bash
bash download_data.sh --data-root <DATA_ROOT>   # Val-1000 + Mini-100 from https://github.com/NJU-R-L-Group-Embodied-Lab/hm3d-m3-code/releases/tag/v1.0.0, checksum-verified
```

Place the HM3D v0.2 scenes (val split, with semantics) so that
`<DATA_ROOT>/scene_datasets/hm3d-1.0/hm3d_v0.2/` holds the scene folders and
`hm3d_annotated_basis.scene_dataset_config.json` (layout in
[`HOW_TO_RUN.md` section 3](HOW_TO_RUN.md#3-data)).

## 4. Check and reproduce

```bash
docker compose -f docker/compose.yaml --env-file .env exec m3 bash   # shell in /workspace/sim
bash scripts/run_selftests.sh        # offline self-tests; only dm3_runner_selftest may SKIP
bash scripts/mini100/lavira.sh       # smoke test: LaViRA-M3 on Mini-100
bash scripts/main_table/lavira.sh    # LaViRA-M3 on Val (1,000 episodes), the paper's main-table row
```

Set `GPUS_CSV` (e.g. `GPUS_CSV=0,1,2,3`) to spread workers over GPUs; the
script header lists all settings. Each run writes per-episode metrics and an
`aggregate.json` under its output directory (`logs/main_table_lavira` by
default). LaViRA-M³ calls a hosted VLM, so numbers vary slightly between runs
and model snapshots.

The adapted baselines run with `bash scripts/main_table/od_baselines.sh` after
the OWL sidecar and the model weights of `HOW_TO_RUN.md` sections 4 and 7 are in
place.
