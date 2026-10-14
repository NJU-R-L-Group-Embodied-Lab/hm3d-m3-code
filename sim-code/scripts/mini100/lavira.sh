#!/usr/bin/env bash
# Official Mini-100 (subset of Val1000) x LaViRA-M3 Full, seed 0, stretch512 IIN.
# Quick check of the main row (about 1/10 of the Val1000 cost).
#
#   LLM_API_KEY=... GPUS_CSV=0,1 bash scripts/mini100/lavira.sh
set -euo pipefail
source "$(dirname "$0")/../lib/common.sh"
hm3d_env

DATA=${DATA:-data/datasets/ma_objectnav_ovon/val_unseen}
export OUT_ROOT=${OUT_ROOT:-logs/mini100_lavira}
export EP=${EP:-$DATA/ma_official_mini_100_samefloor_mixed3_kmquota_seed1123.json}
IIN=${IIN:-$DATA/iin_images_official_mini100_mixed3_stretch512}
export EXTRA_MODULE_ARGS=${EXTRA_MODULE_ARGS:---iin-images-dir $IIN --skip-pano-viz --seed 0}
export METHODS=${METHODS:-lavira_full}
export N=${N:-100}
export WORKERS=${WORKERS:-32}
export GPUS_CSV=${GPUS_CSV:-0}
export MAX_STEPS=${MAX_STEPS:-500}
export THINKING_BUDGET=${THINKING_BUDGET:-1024}
export NO_LA_THINKING=${NO_LA_THINKING:-1}
export DYNAMIC_QUEUE=${DYNAMIC_QUEUE:-1}

orch_init "$OUT_ROOT" mini100-lavira
preflight_mem
require_file "$EP"
require_dir "$IIN"
preflight_api
log "CONFIG ep=$EP iin=$IIN n=$N workers=$WORKERS gpus=$GPUS_CSV methods=$METHODS extra=$EXTRA_MODULE_ARGS"

rc=0
bash "$SIM_ROOT/scripts/run_method.sh" > "$OUT_ROOT/_launch.log" 2>&1 || rc=$?
[ "$DRY_RUN" = "1" ] && cat "$OUT_ROOT/_launch.log"
if [ "$rc" -ne 0 ]; then
  orch_fail "$rc" "run_method.sh rc=$rc (see $OUT_ROOT/_launch.log)"
fi
orch_done
