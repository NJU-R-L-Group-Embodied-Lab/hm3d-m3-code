#!/usr/bin/env bash
# Table I, LaViRA-M3 row: Official Val1000, seed 0, Verifier V2.
# Method lavira_full (= ma_search --control-mode k2-async-keep-goals-pa), LA thinking
# off, stretch512 IIN goal images, GOAT TN text already in the episode JSON.
#
#   LLM_API_KEY=... GPUS_CSV=0,1,2,3,4,5,6,7 bash scripts/main_table/lavira.sh
#
# Paper run: 32 workers on 8 GPUs (GPUS_CSV=0,1,2,3,4,5,6,7). Override any variable
# below from the environment; set DRY_RUN=1 to print the worker commands only.
set -euo pipefail
source "$(dirname "$0")/../lib/common.sh"
hm3d_env

DATA=${DATA:-data/datasets/ma_objectnav_ovon/val_unseen}
export OUT_ROOT=${OUT_ROOT:-logs/main_table_lavira}
export EP=${EP:-$DATA/ma_official_val_1000_samefloor_mixed3_seed1123.json}
IIN=${IIN:-$DATA/iin_images_official_val1000_mixed3_stretch512}
export EXTRA_MODULE_ARGS=${EXTRA_MODULE_ARGS:---iin-images-dir $IIN --skip-pano-viz --seed 0}
export METHODS=${METHODS:-lavira_full}
export N=${N:-1000}
export WORKERS=${WORKERS:-32}
export GPUS_CSV=${GPUS_CSV:-0}
export MAX_STEPS=${MAX_STEPS:-500}
export THINKING_BUDGET=${THINKING_BUDGET:-1024}
export NO_LA_THINKING=${NO_LA_THINKING:-1}
export DYNAMIC_QUEUE=${DYNAMIC_QUEUE:-1}

orch_init "$OUT_ROOT" main-table-lavira
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
