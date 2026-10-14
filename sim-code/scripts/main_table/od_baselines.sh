#!/usr/bin/env bash
# Table I, the four adapted baselines on Official Val1000 (detector FOUND, STOP 0.9 m):
# VLFM, EGO-Swarm, Co-NavGPT2 and DM3-Nav (detector FOUND gate).
# The paper numbers were produced with the stretch_v2 IIN images
# (iin_images_official_val1000_mixed3_stretch_v2); keep that default to reproduce them.
#
# Needs the OWL detector sidecar (OWL_HOST + OWL_PORT_BASE, one per GPU, or OWL_URL),
# BLIP-2 weights for VLFM (VLFM_BLIP2_DIR), DM3 weights (DM3_SUPERGLUE_WEIGHTS_DIR,
# DM3_DETIC_CHECKPOINT) and the VLM API for Co-NavGPT2 (LLM_API_KEY).
#
# All methods run concurrently. Paper layout (8 GPUs):
#   VLFM_GPUS=0,1,2,3 DM3_GPUS=4,5,6,7 GPUS_CSV=0,1,2,3,4,5,6,7
# with 32 workers each (16 for Co-NavGPT2). Set PARALLEL_METHODS=0 to run them one
# after another. DRY_RUN=1 prints the worker commands only.
set -euo pipefail
source "$(dirname "$0")/../lib/common.sh"
hm3d_env

DATA=${DATA:-data/datasets/ma_objectnav_ovon/val_unseen}
export OUT_ROOT=${OUT_ROOT:-logs/main_table_od_baselines}
export EP=${EP:-$DATA/ma_official_val_1000_samefloor_mixed3_seed1123.json}
IIN=${IIN:-$DATA/iin_images_official_val1000_mixed3_stretch_v2}
export EXTRA_MODULE_ARGS=${EXTRA_MODULE_ARGS:---iin-images-dir $IIN}
export N=${N:-1000}
export MAX_STEPS=${MAX_STEPS:-500}
export OWL_SHM_DIR=${OWL_SHM_DIR:-$PWD/.owl_shm}
GPUS_CSV=${GPUS_CSV:-0}
METHODS=${METHODS:-vlfm dm3 ego_swarm conav}
PARALLEL_METHODS=${PARALLEL_METHODS:-1}

# method -> workers, gpus
plan_for() {
  case "$1" in
    vlfm)      echo "${VLFM_WORKERS:-32} ${VLFM_GPUS:-$GPUS_CSV}" ;;
    dm3)       echo "${DM3_WORKERS:-32} ${DM3_GPUS:-$GPUS_CSV}" ;;
    ego_swarm) echo "${EGO_WORKERS:-32} ${EGO_GPUS:-$GPUS_CSV}" ;;
    conav)     echo "${CONAV_WORKERS:-16} ${CONAV_GPUS:-$GPUS_CSV}" ;;
    *)         echo "unknown method $1" >&2; return 2 ;;
  esac
}

orch_init "$OUT_ROOT" main-table-od
require_file "$EP"
require_dir "$IIN"
ALL_GPUS=$(for m in $METHODS; do read -r _ g <<<"$(plan_for "$m")"; echo "$g"; done \
  | tr ',' '\n' | sort -n -u | paste -sd, -)
preflight_owl "$ALL_GPUS"
preflight_perception
case " $METHODS " in *" conav "*) preflight_api ;; esac
log "CONFIG ep=$EP iin=$IIN n=$N methods=$METHODS parallel=$PARALLEL_METHODS"

failed=0
for m in $METHODS; do
  read -r workers gpus <<<"$(plan_for "$m")"
  run_method_bg "$m" "$workers" "$gpus"
  if [ "$PARALLEL_METHODS" != "1" ]; then
    wait_bg || failed=1
  fi
done
wait_bg || failed=1
[ "$DRY_RUN" = "1" ] && cat "$OUT_ROOT"/_launch_*.log
if [ "$failed" -ne 0 ]; then
  orch_fail 1 "some methods failed"
fi
if [ "$DRY_RUN" != "1" ]; then
  # Final table over all methods (each method is already merged; this only reads).
  METHODS="$METHODS" MERGE_ONLY=0 SKIP_FINAL_TABLE=0 WORKERS=1 GPUS_CSV="$GPUS_CSV" \
    bash "$SIM_ROOT/scripts/run_method.sh" > "$OUT_ROOT/_final_table.log" 2>&1 \
    || orch_fail 1 "final table failed (see $OUT_ROOT/_final_table.log)"
fi
orch_done
