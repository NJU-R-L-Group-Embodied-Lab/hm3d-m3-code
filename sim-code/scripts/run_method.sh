#!/usr/bin/env bash
# Generic HM3D-M3 evaluation runner: one or more methods on one episode file.
#
#   bash scripts/run_method.sh [METHOD ...] [--] [EXTRA RUNNER ARGS]
#
# Methods (paper names):
#   lavira_full      LaViRA-M3, main row        (--control-mode k2-async-keep-goals-pa)
#   vlfm             VLFM (BLIP-2 value map)    benchmark.zeroshot_runner --method vlfm
#   ego_swarm        EGO-Swarm                  benchmark.ego_runner
#   conav            Co-NavGPT2                 benchmark.conav_runner --vlm qwen
#   dm3              DM3-Nav (official port)    benchmark.dm3_official.runner
#
# Each method writes OUT_ROOT/<method>/w<k>/episode_*.json; after all workers exit
# the episodes are merged into metrics_list.json / aggregate.json / summary.json
# and a DONE marker, and OUT_ROOT/fair_comparison.{md,json} is refreshed.
# Re-running resumes: complete methods are skipped, and with DYNAMIC_QUEUE=1
# (LaViRA only) finished episodes of a partial run are skipped.
#
# Environment (defaults in brackets):
#   EP [Val1000 JSON]  SCENES_DIR [data/scene_datasets/hm3d-1.0]  OUT_ROOT [logs/run_method]
#   N [1000]  WORKERS [32]  GPUS_CSV [0]  MAX_STEPS [500]  METHODS [lavira_full]
#   THINKING_BUDGET [1024]  NO_LA_THINKING [1 = paper setting: LA thinking off]
#   DYNAMIC_QUEUE [1]  EXTRA_MODULE_ARGS [""]  MERGE_ONLY [0]  SKIP_FINAL_TABLE [0]
#   PY [python]  OWL_URL | OWL_HOST + OWL_PORT_BASE [18210]  OWL_SHM_DIR [$PWD/.owl_shm]
#   IIN_DIR [required]: IIN goal-image directory for EP, passed as --iin-images-dir
#     (e.g. data/datasets/ma_objectnav_ovon/val_unseen/iin_images_official_val1000_mixed3_stretch512).
#     Not needed when --iin-images-dir is already in EXTRA_MODULE_ARGS or the CLI extras
#     (as the scripts/<table>/*.sh row launchers do).
#   DRY_RUN [0]: print the worker commands instead of running them.
set -euo pipefail

cd "$(dirname "$0")/.."

export EGL_PLATFORM=surfaceless
export GLOG_minloglevel=0
export TOKENIZERS_PARALLELISM=false

EP=${EP:-data/datasets/ma_objectnav_ovon/val_unseen/ma_official_val_1000_samefloor_mixed3_seed1123.json}
SCENES_DIR=${SCENES_DIR:-data/scene_datasets/hm3d-1.0}
OUT_ROOT=${OUT_ROOT:-logs/run_method}
N=${N:-1000}
WORKERS=${WORKERS:-32}
GPUS_CSV=${GPUS_CSV:-0}
MAX_STEPS=${MAX_STEPS:-500}
THINKING_BUDGET=${THINKING_BUDGET:-1024}
NO_LA_THINKING=${NO_LA_THINKING:-1}
DYNAMIC_QUEUE=${DYNAMIC_QUEUE:-1}
METHODS=${METHODS:-lavira_full}
PY=${PY:-python}
MERGE_ONLY=${MERGE_ONLY:-0}
DRY_RUN=${DRY_RUN:-0}
OWL_SHM_DIR=${OWL_SHM_DIR:-$PWD/.owl_shm}
IIN_DIR=${IIN_DIR:-}

# Positional arguments: method names, then "--" and extra runner arguments.
CLI_METHODS=()
CLI_EXTRA=()
while [ "$#" -gt 0 ]; do
  if [ "$1" = "--" ]; then
    shift
    CLI_EXTRA=("$@")
    break
  fi
  # The first flag also starts the extra arguments ("--" is optional).
  if [[ "$1" == -* ]]; then
    CLI_EXTRA=("$@")
    break
  fi
  CLI_METHODS+=("$1")
  shift
done
if [ "${#CLI_METHODS[@]}" -gt 0 ]; then
  METHODS="${CLI_METHODS[*]}"
fi

IFS=',' read -r -a GPUS <<< "$GPUS_CSV"

# IIN goal images. Every method evaluates ON + IIN + TN goals, so each one needs
# --iin-images-dir; without it the runners would silently fall back to an old
# default directory that is not part of the released data layout.
method_needs_iin() {
  case "$1" in
    lavira_full|vlfm|ego_swarm|conav|dm3) return 0 ;;
    *) return 1 ;;
  esac
}
iin_arg_given() {
  local a
  # shellcheck disable=SC2086
  for a in ${EXTRA_MODULE_ARGS:-} ${CLI_EXTRA[@]+"${CLI_EXTRA[@]}"}; do
    case "$a" in
      --iin-images-dir|--iin-images-dir=*) return 0 ;;
    esac
  done
  return 1
}
IIN_ARGS=()
if [ "$MERGE_ONLY" != "1" ] && ! iin_arg_given; then
  for _m in $METHODS; do
    if method_needs_iin "$_m"; then
      if [ -z "$IIN_DIR" ]; then
        printf "[run-method] ABORT: method '%s' needs IIN goal images: set IIN_DIR=<iin_images dir for EP> (or pass --iin-images-dir)\n" "$_m" >&2
        exit 2
      fi
      IIN_ARGS=(--iin-images-dir "$IIN_DIR")
      break
    fi
  done
fi
if [ "${#IIN_ARGS[@]}" -gt 0 ] && [ ! -d "$IIN_DIR" ]; then
  if [ "$DRY_RUN" = "1" ]; then
    printf "[run-method] WARN (dry run) missing IIN_DIR %s\n" "$IIN_DIR" >&2
  else
    printf "[run-method] ABORT: IIN_DIR %s is not a directory\n" "$IIN_DIR" >&2
    exit 2
  fi
fi

log() {
  printf "[run-method] %s %s\n" "$(date +"%F %T %Z")" "$*"
}

episode_count() {
  local dir="$1"
  if [ ! -d "$dir" ]; then
    echo 0
    return 0
  fi
  find "$dir" -mindepth 2 -maxdepth 2 -type f \
    -path "$dir/w*/episode_*.json" | wc -l
}

# Keep in sync with EXPECTED_METHODS in tests/selftests/clean_baseline_selftest.py.
method_module_and_args() {
  local method="$1"
  case "$method" in
    # LaViRA-M3 Full: all goals with own FOUND status + teammate progress analysis.
    lavira_full)
      MODULE_ARGS=(benchmark.ma_search --use-vlm --clean-found-state \
                   --disable-path-budget \
                   --control-mode k2-async-keep-goals-pa --no-team-topdown-map \
                   --la-thinking-budget "$THINKING_BUDGET")
      ;;
    vlfm)
      MODULE_ARGS=(benchmark.zeroshot_runner --method vlfm)
      ;;
    ego_swarm)
      MODULE_ARGS=(benchmark.ego_runner)
      ;;
    # Co-NavGPT2: detector FOUND.
    conav)
      MODULE_ARGS=(benchmark.conav_runner --vlm qwen)
      ;;
    dm3)
      MODULE_ARGS=(benchmark.dm3_official.runner)
      ;;
    *)
      log "ABORT: unknown method '$method' (expected one of: lavira_full vlfm ego_swarm conav dm3)"
      exit 2
      ;;
  esac
}

check_existing_outputs() {
  mkdir -p "$OUT_ROOT"
  for method in $METHODS; do
    method_module_and_args "$method"
    local dir="$OUT_ROOT/$method"
    local count
    count=$(episode_count "$dir")
    if [ -f "$dir/DONE" ] && [ "$count" -eq "$N" ]; then
      log "RESUME: $method already complete ($count/$N); will skip"
      continue
    fi
    if [ "$count" -ne 0 ] || [ -f "$dir/DONE" ]; then
      if [ "${DYNAMIC_QUEUE:-0}" = "1" ] && [ ! -f "$dir/DONE" ]; then
        log "RESUME: $method partial $count/$N; dynamic queue will skip finished episodes"
        continue
      fi
      log "ABORT: $dir has partial output ($count/$N) or stale DONE; move it aside before rerun"
      exit 2
    fi
  done
}

merge_method() {
  local method="$1" out="$2"
  "$PY" - "$method" "$out" "$N" "$WORKERS" "$GPUS_CSV" "$MAX_STEPS" <<'PY'
import glob
import json
import os
import sys

from benchmark import metrics

method, out, n, workers, gpus, max_steps = (
    sys.argv[1],
    sys.argv[2],
    int(sys.argv[3]),
    int(sys.argv[4]),
    sys.argv[5],
    int(sys.argv[6]),
)

episode_payloads = []
metric_rows = []
for path in sorted(glob.glob(os.path.join(out, "w*", "episode_*.json"))):
    try:
        with open(path) as fh:
            payload = json.load(fh)
    except Exception as exc:
        print(f"[run-method] skip {path}: {exc!r}", flush=True)
        continue
    metrics_row = payload.get("metrics")
    if isinstance(metrics_row, dict):
        metrics_row = dict(metrics_row)
        metrics_row.setdefault("episode_id", str(payload.get("episode_id", "")))
        metrics_row["_source_file"] = path
        metric_rows.append(metrics_row)
        episode_payloads.append(payload)

if len(metric_rows) != n:
    raise SystemExit(f"[run-method] incomplete {method}: expected {n}, got {len(metric_rows)}")

agg = metrics.aggregate(metric_rows)
# LaViRA rows score claims with Verifier V2; the adapted baselines use the OVON
# detector STOP protocol.
if method.startswith("lavira_"):
    protocol = "verifier_v2_claim_pose"
else:
    protocol = "ovon_od_detector_stop"
summary = {
    "protocol": protocol,
    "method": method,
    "n_requested": n,
    "n_episodes": len(metric_rows),
    "workers": workers,
    "gpus": gpus,
    "max_episode_steps": max_steps,
    "aggregate": agg,
}

with open(os.path.join(out, "metrics_list.json"), "w") as fh:
    json.dump(metric_rows, fh)
with open(os.path.join(out, "aggregate.json"), "w") as fh:
    json.dump(agg, fh, indent=2)
with open(os.path.join(out, "summary.json"), "w") as fh:
    json.dump(summary, fh, indent=2)
with open(os.path.join(out, "DONE"), "w") as fh:
    fh.write(f"{len(metric_rows)} episodes\n")
print(json.dumps(agg, indent=2), flush=True)
PY
}

write_final_table() {
  # shellcheck disable=SC2086
  "$PY" - "$OUT_ROOT" $METHODS <<'PY'
import json
import os
import sys

root = sys.argv[1]
methods = sys.argv[2:]
METHOD_META = {
    "lavira_full": ("LaViRA-M3", "Shared goal board (own FOUND) + teammate progress analysis; Verifier V2"),
    "vlfm": ("VLFM", "Detector FOUND (OVON OD STOP)"),
    "ego_swarm": ("EGO-Swarm", "Detector FOUND (OVON OD STOP)"),
    "conav": ("Co-NavGPT2", "Detector FOUND (OVON OD STOP)"),
    "dm3": ("DM3-Nav", "Official GOAT/Detic/SuperGlue/CLIP port; detector FOUND"),
}
rows = []
for method in methods:
    path = os.path.join(root, method, "aggregate.json")
    if not os.path.exists(path):
        continue
    agg = json.load(open(path))
    name, condition = METHOD_META.get(method, (method, ""))
    rows.append({
        "Method": name,
        "Condition": condition,
        "n": agg.get("n"),
        "SR": agg.get("SR"),
        "Progress": agg.get("Progress"),
        "SRAT": agg.get("SRAT"),
        "MSPL": agg.get("MSPL"),
        "CG_realized": agg.get("CG_realized"),
        "Fairness_Gini": agg.get("Fairness_Gini"),
        "Comm_Volume": agg.get("Comm_Volume"),
    })

def fmt(value, digits=3):
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)

headers = ["Method", "Condition", "n", "SR", "Progress", "SRAT", "MSPL", "CG_realized", "Fairness_Gini", "Comm_Volume"]
lines = [
    "# HM3D-M3 results",
    "",
    "| " + " | ".join(headers) + " |",
    "| " + " | ".join(["---"] * len(headers)) + " |",
]
for row in rows:
    lines.append("| " + " | ".join(fmt(row.get(h), 4 if h in {"SR", "Progress", "SRAT", "MSPL", "CG_realized", "Fairness_Gini"} else 3) for h in headers) + " |")

os.makedirs(root, exist_ok=True)
with open(os.path.join(root, "fair_comparison.md"), "w") as fh:
    fh.write("\n".join(lines) + "\n")
with open(os.path.join(root, "fair_comparison.json"), "w") as fh:
    json.dump(rows, fh, indent=2)
print("\n".join(lines), flush=True)
PY
}

run_method() {
  local method="$1" out="$OUT_ROOT/$method"
  method_module_and_args "$method"
  if [ "${NO_LA_THINKING:-0}" = "1" ] && [ "${MODULE_ARGS[0]}" = "benchmark.ma_search" ]; then
    MODULE_ARGS+=(--no-la-thinking)
  fi
  local existing
  existing=$(episode_count "$out")
  if [ -f "$out/DONE" ] && [ "$existing" -eq "$N" ]; then
    log "SKIP method=$method already complete"
    return 0
  fi
  [ "$DRY_RUN" = "1" ] || mkdir -p "$out"
  log "START method=$method out=$out workers=$WORKERS gpus=$GPUS_CSV dynamic_queue=$DYNAMIC_QUEUE"

  local pids=() labels=() start=0
  local use_dyn=0
  if [ "$DYNAMIC_QUEUE" = "1" ] && [ "${MODULE_ARGS[0]}" = "benchmark.ma_search" ]; then
    use_dyn=1
    if [ "$DRY_RUN" = "1" ]; then
      log "DRY_RUN $PY -m benchmark.work_queue init --episodes $EP --out $out/work_queue.json --n $N --skip-root $out"
    else
      "$PY" -m benchmark.work_queue init \
        --episodes "$EP" --out "$out/work_queue.json" --n "$N" --skip-root "$out"
    fi
  fi
  local base=$((N / WORKERS))
  local rem=$((N % WORKERS))
  for w in $(seq 0 $((WORKERS - 1))); do
    local count=$base
    if [ "$w" -lt "$rem" ]; then
      count=$((count + 1))
    fi
    if [ "$use_dyn" != "1" ] && [ "$count" -le 0 ]; then
      continue
    fi
    local gpu="${GPUS[$((w % ${#GPUS[@]}))]}"
    local save_dir="$out/w$w"
    local owl_url="${OWL_URL:-}"
    if [ -n "${OWL_HOST:-}" ]; then
      owl_url="http://${OWL_HOST}:$((${OWL_PORT_BASE:-18210} + gpu))"
    fi
    local shard_args=()
    if [ "$use_dyn" = "1" ]; then
      shard_args=(--dynamic-queue "$out/work_queue.json" --worker-id "w$w" --limit "$N")
      log "$method worker=$w gpu=$gpu owl=${owl_url:-none} dynamic-queue -> $save_dir"
    else
      shard_args=(--start "$start" --count "$count")
      log "$method worker=$w gpu=$gpu owl=${owl_url:-none} episodes[$start:$((start + count))] -> $save_dir"
    fi
    # EXTRA_MODULE_ARGS is intentionally word-split.
    # shellcheck disable=SC2206
    local extra=(${EXTRA_MODULE_ARGS:-})
    local cmd=("$PY" -u -m "${MODULE_ARGS[@]}" \
      --episodes "$EP" --scenes-dir "$SCENES_DIR" --save-dir "$save_dir" \
      --max-episode-steps "$MAX_STEPS" \
      "${shard_args[@]}" \
      ${IIN_ARGS[@]+"${IIN_ARGS[@]}"} \
      ${extra[@]+"${extra[@]}"} \
      ${CLI_EXTRA[@]+"${CLI_EXTRA[@]}"})
    if [ "$DRY_RUN" = "1" ]; then
      log "DRY_RUN CUDA_VISIBLE_DEVICES=$gpu OWL_URL=$owl_url OWL_SHM_DIR=$OWL_SHM_DIR ${cmd[*]}"
      start=$((start + count))
      continue
    fi
    mkdir -p "$save_dir"
    CUDA_VISIBLE_DEVICES="$gpu" OWL_URL="$owl_url" OWL_SHM_DIR="$OWL_SHM_DIR" \
      "${cmd[@]}" > "$out/w$w.log" 2>&1 &
    pids+=("$!")
    labels+=("$w:$gpu:$start:$count")
    start=$((start + count))
  done
  if [ "$DRY_RUN" = "1" ]; then
    return 0
  fi

  local failed=0
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      log "$method worker ${labels[$i]} exit=0"
    else
      local rc=$?
      log "$method worker ${labels[$i]} exit=$rc"
      failed=1
    fi
  done
  if [ "$failed" -ne 0 ]; then
    log "ABORT method=$method because at least one worker failed"
    exit 1
  fi
  log "MERGE method=$method"
  merge_method "$method" "$out"
  log "DONE method=$method"
}

if [ ! -f "$EP" ]; then
  if [ "$DRY_RUN" = "1" ]; then
    log "WARN (dry run) missing episode file $EP"
  else
    log "ABORT: missing episode file $EP"
    exit 2
  fi
fi
if [ "${#GPUS[@]}" -eq 0 ]; then
  log "ABORT: GPUS_CSV is empty"
  exit 2
fi
if [ "$WORKERS" -le 0 ]; then
  log "ABORT: WORKERS must be > 0"
  exit 2
fi

if [ "$MERGE_ONLY" = "1" ]; then
  log "MERGE_ONLY CONFIG OUT_ROOT=$OUT_ROOT N=$N METHODS=$METHODS"
  for method in $METHODS; do
    out="$OUT_ROOT/$method"
    count=$(episode_count "$out")
    if [ "$count" -ne "$N" ]; then
      log "ABORT merge-only method=$method expected=$N found=$count"
      exit 2
    fi
    merge_method "$method" "$out"
    log "MERGE_ONLY DONE method=$method"
  done
  exit 0
fi

if [ "$DRY_RUN" = "1" ]; then
  for method in $METHODS; do
    method_module_and_args "$method"
  done
else
  check_existing_outputs
fi
log "CONFIG EP=$EP N=$N WORKERS=$WORKERS GPUS=$GPUS_CSV MAX_STEPS=$MAX_STEPS DYNAMIC_QUEUE=$DYNAMIC_QUEUE NO_LA_THINKING=$NO_LA_THINKING METHODS=$METHODS IIN=${IIN_ARGS[1]:-<from extra args>} EXTRA=${EXTRA_MODULE_ARGS:-} ${CLI_EXTRA[*]:-}"
for method in $METHODS; do
  run_method "$method"
  if [ "$DRY_RUN" != "1" ] && [ "${SKIP_FINAL_TABLE:-0}" != "1" ]; then
    write_final_table
  fi
done
log "ALL_DONE out=$OUT_ROOT"
