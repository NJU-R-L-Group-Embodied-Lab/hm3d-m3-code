#!/usr/bin/env bash
# Render IIN goal images and write GOAT-style TN text for an episode file.
#
#   EP=data/datasets/ma_objectnav_ovon/val_unseen/ma_official_val_1000_samefloor_mixed3_seed1123.json \
#   IIN_OUT=data/datasets/ma_objectnav_ovon/val_unseen/iin_images_official_val1000_mixed3_stretch512 \
#   GPUS_CSV=0,1,2,3,4,5,6,7 bash scripts/gen/fill_iin_tn.sh
#
# CAMERA=stretch512 (default; LaViRA rows, Mini-100, train): 512x512, HFOV 42 deg,
#   camera 1.31 m, look-at with pitch, 1.8 m standoff.
# CAMERA=stretch_v2 (Val1000 images used by the four adapted baselines in Table I):
#   Stretch 360x640 view, yaw look-at, object crop, 1.8 m standoff.
# IIN and TN run as N_IIN / N_TN shards, one per GPU in GPUS_CSV (round robin).
# TN shards write JSONL sidecars that are merged into EP at the end
# (scripts/gen/merge_tn_sidecars.py). TN needs LLM_API_KEY.
# Set DO_IIN=0 or DO_TN=0 to skip a stage.
set -euo pipefail
source "$(dirname "$0")/../lib/common.sh"
hm3d_env

EP=${EP:?set EP to the episode JSON}
IIN_OUT=${IIN_OUT:-}
CAMERA=${CAMERA:-stretch512}
SCENES_DIR=${SCENES_DIR:-data/scene_datasets/hm3d-1.0}
GPUS_CSV=${GPUS_CSV:-0}
N_IIN=${N_IIN:-1}
N_TN=${N_TN:-1}
API_WORKERS=${API_WORKERS:-4}
DO_IIN=${DO_IIN:-1}
DO_TN=${DO_TN:-1}
TN_OVERWRITE=${TN_OVERWRITE:-1}
LOG_DIR=${LOG_DIR:-logs/fill_iin_tn_$(basename "$EP" .json)}

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
case "$CAMERA" in
  stretch512) IIN_ARGS=(--camera-profile stretch512 --look-at-object --standoff-m 1.8 --out-size 512) ;;
  stretch_v2) IIN_ARGS=(--camera-profile stretch --look-at-object --crop-object --standoff-m 1.8) ;;
  *) echo "unknown CAMERA=$CAMERA (stretch512 | stretch_v2)" >&2; exit 2 ;;
esac
if [ "$DO_IIN" = "1" ] && [ -z "$IIN_OUT" ]; then
  echo "set IIN_OUT (output dir for the IIN images) or DO_IIN=0" >&2
  exit 2
fi

orch_init "$LOG_DIR" fill-iin-tn
require_file "$EP"
[ "$DO_TN" = "1" ] && preflight_api

PIDS=()
TAGS=()
launch() {
  local tag="$1" gpu="$2"
  shift 2
  log "START $tag gpu=$gpu: $*"
  if [ "$DRY_RUN" = "1" ]; then return 0; fi
  CUDA_VISIBLE_DEVICES="$gpu" "$@" > "$LOG_DIR/$tag.log" 2>&1 &
  PIDS+=("$!")
  TAGS+=("$tag")
}

if [ "$DO_IIN" = "1" ]; then
  [ "$DRY_RUN" = "1" ] || mkdir -p "$IIN_OUT"
  for i in $(seq 0 $((N_IIN - 1))); do
    launch "iin_shard$i" "${GPUS[$((i % ${#GPUS[@]}))]}" \
      "$PY" -u -m benchmark.render_iin --episodes "$EP" --out-dir "$IIN_OUT" \
      --scenes-dir "$SCENES_DIR" "${IIN_ARGS[@]}" --skip-existing \
      --n-shards "$N_IIN" --shard-index "$i" --gpu-device-id 0
  done
fi
if [ "$DO_TN" = "1" ]; then
  TN_FLAGS=(--sample-mode goat)
  [ "$TN_OVERWRITE" = "1" ] && TN_FLAGS+=(--overwrite)
  for i in $(seq 0 $((N_TN - 1))); do
    # TN shards start after the IIN shards so the two do not share a GPU when possible.
    launch "tn_shard$i" "${GPUS[$(((i + N_IIN) % ${#GPUS[@]}))]}" \
      "$PY" -u -m benchmark.gen_tn --episodes "$EP" --scenes-dir "$SCENES_DIR" \
      "${TN_FLAGS[@]}" --n-shards "$N_TN" --shard-index "$i" --gpu-device-id 0 \
      --api-workers "$API_WORKERS" --sidecar "$LOG_DIR/tn_shard$i.jsonl"
  done
fi

if [ "$DRY_RUN" = "1" ]; then
  log "DRY_RUN merge: $PY scripts/gen/merge_tn_sidecars.py --episodes $EP --sidecar-dir $LOG_DIR --replace"
  orch_done
  exit 0
fi

failed=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    log "${TAGS[$i]} exit=0"
  else
    rc=$?
    log "${TAGS[$i]} exit=$rc (see $LOG_DIR/${TAGS[$i]}.log)"
    failed=1
  fi
done
if [ "$DO_TN" = "1" ]; then
  "$PY" -u scripts/gen/merge_tn_sidecars.py --episodes "$EP" --sidecar-dir "$LOG_DIR" --replace \
    | tee -a "$ORCH_LOG"
fi
if [ "$failed" -ne 0 ]; then
  orch_fail 1 "some shards failed"
fi
orch_done
