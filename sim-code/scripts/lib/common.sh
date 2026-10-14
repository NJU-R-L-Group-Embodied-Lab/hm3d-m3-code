# Shared helpers for the HM3D-M3 sim scripts. Source this file; do not run it.
#
# After sourcing, the working directory is sim-code/ (so `python -m benchmark.*` and
# the relative data/ paths resolve) and SIM_ROOT points at it.
#
# Environment (all optional):
#   PY             python interpreter (default: python)
#   MIN_MEM_GB     abort if MemAvailable is below this many GB (default: 0 = no check;
#                  the paper runs with 32 LaViRA workers used a 200 GB guard)
#   SKIP_PREFLIGHT 1 = skip the API / OWL / perception preflights
#   DRY_RUN        1 = print what would run; no preflights, no GPU work

SIM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$SIM_ROOT"

export PY="${PY:-python}"
export DRY_RUN="${DRY_RUN:-0}"
export SKIP_PREFLIGHT="${SKIP_PREFLIGHT:-0}"

# Headless rendering / thread settings shared by every launcher.
hm3d_env() {
  ulimit -n 65535 2>/dev/null || true
  export OMP_NUM_THREADS=1
  export MKL_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export TOKENIZERS_PARALLELISM=false
  export EGL_PLATFORM=surfaceless
  export MAGNUM_LOG=quiet
  export HABITAT_SIM_LOG=quiet
  unset DISPLAY || true
}

# orch_init <out_root> <tag>: create OUT_ROOT and the orchestrator log.
orch_init() {
  ORCH_OUT="$1"
  ORCH_TAG="$2"
  mkdir -p "$ORCH_OUT"
  ORCH_LOG="$ORCH_OUT/_orchestrator.log"
}

log() {
  local line
  line=$(printf '[%s] %s %s' "${ORCH_TAG:-hm3d-m3}" "$(date +'%F %T %Z')" "$*")
  if [ -n "${ORCH_LOG:-}" ]; then
    printf '%s\n' "$line" | tee -a "$ORCH_LOG"
  else
    printf '%s\n' "$line"
  fi
}

# orch_fail <rc> <message>: log, write ORCH_EXIT, exit.
orch_fail() {
  local rc="$1"
  shift
  log "ABORT $*"
  if [ -n "${ORCH_OUT:-}" ] && [ "$DRY_RUN" != "1" ]; then
    echo 1 > "$ORCH_OUT/ORCH_EXIT"
  fi
  exit "$rc"
}

orch_done() {
  if [ "$DRY_RUN" != "1" ]; then
    touch "$ORCH_OUT/ORCH_DONE"
    echo 0 > "$ORCH_OUT/ORCH_EXIT"
  fi
  log "ORCH_DONE out=$ORCH_OUT"
}

# require_file <path> / require_dir <path>: abort unless present (warn in DRY_RUN).
require_file() {
  if [ ! -f "$1" ]; then
    if [ "$DRY_RUN" = "1" ]; then log "WARN (dry run) missing file $1"; return 0; fi
    orch_fail 2 "missing file $1"
  fi
}

require_dir() {
  if [ ! -d "$1" ]; then
    if [ "$DRY_RUN" = "1" ]; then log "WARN (dry run) missing dir $1"; return 0; fi
    orch_fail 2 "missing dir $1"
  fi
}

# Optional free-memory guard (MIN_MEM_GB, default off).
preflight_mem() {
  local min_gb="${MIN_MEM_GB:-0}"
  if [ "$min_gb" = "0" ] || [ ! -r /proc/meminfo ]; then
    return 0
  fi
  local avail_kb
  avail_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  if (( avail_kb < min_gb * 1000000 )); then
    orch_fail 1 "MemAvailable ${avail_kb} kB < ${min_gb} GB (MIN_MEM_GB)"
  fi
}

# One-token request against the configured VLM endpoint (LLM_* env vars).
preflight_api() {
  if [ "$DRY_RUN" = "1" ] || [ "$SKIP_PREFLIGHT" = "1" ]; then
    return 0
  fi
  log "PREFLIGHT VLM API"
  if ! "$PY" - <<'PY'
from openai import OpenAI
from benchmark.api_auth import get_llm_model, require_llm
url, key = require_llm()
OpenAI(api_key=key, base_url=url, timeout=30).chat.completions.create(
    model=get_llm_model(),
    max_tokens=1,
    temperature=0.0,
    messages=[{"role": "user", "content": "Reply OK"}],
)
print("api_ok", get_llm_model())
PY
  then
    orch_fail 3 "VLM API preflight failed (set LLM_API_KEY / LLM_BASE_URL / LLM_MODEL)"
  fi
  log "PREFLIGHT VLM API OK"
}

# preflight_owl <gpu_csv>: the OWL sidecar is required for detector-FOUND baselines.
preflight_owl() {
  local gpus_csv="$1"
  if [ -z "${OWL_HOST:-}" ] && [ -z "${OWL_URL:-}" ]; then
    if [ "$DRY_RUN" = "1" ]; then log "WARN (dry run) OWL_HOST / OWL_URL unset"; return 0; fi
    orch_fail 3 "OWL_HOST or OWL_URL required for detector FOUND (see services/)"
  fi
  export OWL_PORT_BASE="${OWL_PORT_BASE:-18210}"
  export OWL_EVERY="${OWL_EVERY:-2}"
  [ -n "${OWL_HOST:-}" ] && export OWL_HOST
  if [ "$DRY_RUN" = "1" ] || [ "$SKIP_PREFLIGHT" = "1" ] || [ -z "${OWL_HOST:-}" ]; then
    return 0
  fi
  log "PREFLIGHT owl host=$OWL_HOST port_base=$OWL_PORT_BASE gpus=$gpus_csv"
  local g port
  local -a gpus
  IFS=',' read -r -a gpus <<< "$gpus_csv"
  for g in "${gpus[@]}"; do
    port=$((OWL_PORT_BASE + g))
    if ! "$PY" - "$OWL_HOST" "$port" <<'PY'
import sys, urllib.request
host, port = sys.argv[1], sys.argv[2]
urllib.request.urlopen(f"http://{host}:{port}/health", timeout=3).read()
PY
    then
      orch_fail 3 "OWL not healthy at ${OWL_HOST}:${port}"
    fi
    log "OWL ok ${OWL_HOST}:${port}"
  done
}

# CLIP + DM3 text matcher + BLIP-2 (VLFM) smoke check.
preflight_perception() {
  export VLFM_BLIP2_DIR="${VLFM_BLIP2_DIR:-$SIM_ROOT/data/models/blip2-itm-vit-g}"
  if [ ! -f "$VLFM_BLIP2_DIR/config.json" ]; then
    if [ "$DRY_RUN" = "1" ]; then log "WARN (dry run) missing BLIP-2 weights at $VLFM_BLIP2_DIR"; return 0; fi
    orch_fail 3 "missing BLIP-2 weights at $VLFM_BLIP2_DIR (set VLFM_BLIP2_DIR)"
  fi
  if [ "$DRY_RUN" = "1" ] || [ "$SKIP_PREFLIGHT" = "1" ]; then
    return 0
  fi
  log "PREFLIGHT clip + match_tn + blip2"
  if ! "$PY" - <<'PY'
import numpy as np
import clip
from benchmark.dm3_perception import DM3Perception
from benchmark.vlfm_value import shared_blip2, official_prompt
_ = clip.load
perc = DM3Perception(device="cpu")
rgb = np.zeros((224, 224, 3), dtype=np.uint8)
print("match_tn", perc.match_tn(rgb, "a wooden chair next to a window"))
itm = shared_blip2("cuda")
itm.set_targets(["chair"])
print("blip2", official_prompt("chair"), itm.score_image(rgb))
PY
  then
    orch_fail 3 "perception preflight failed"
  fi
  log "PREFLIGHT perception OK"
}

# run_method_bg <method> <workers> <gpus_csv>: start run_method.sh for
# one method in the background (used to run several baselines concurrently).
# Exports from the caller (OUT_ROOT, EP, N, ...) are inherited.
run_method_bg() {
  local method="$1" workers="$2" gpus="$3"
  log "START method=$method workers=$workers gpus=$gpus n=$N"
  METHODS="$method" WORKERS="$workers" GPUS_CSV="$gpus" SKIP_FINAL_TABLE=1 \
    bash "$SIM_ROOT/scripts/run_method.sh" \
    > "$OUT_ROOT/_launch_${method}.log" 2>&1 &
  BG_PIDS+=("$!")
  BG_LABELS+=("$method")
}

# wait_bg: wait for every run_method_bg job; return 1 if any failed.
wait_bg() {
  local i rc failed=0
  for i in "${!BG_PIDS[@]}"; do
    if wait "${BG_PIDS[$i]}"; then
      log "DONE method=${BG_LABELS[$i]}"
    else
      rc=$?
      log "FAILED method=${BG_LABELS[$i]} rc=$rc (see $OUT_ROOT/_launch_${BG_LABELS[$i]}.log)"
      failed=1
    fi
  done
  BG_PIDS=()
  BG_LABELS=()
  return "$failed"
}
BG_PIDS=()
BG_LABELS=()
