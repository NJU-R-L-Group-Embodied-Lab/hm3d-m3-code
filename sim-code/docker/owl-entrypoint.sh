#!/usr/bin/env bash
# Start one OWL sidecar per GPU and wait on them.
#
#   OWL_GPUS        comma-separated GPU ids (default: 0)
#   OWL_PORT_BASE   sidecar on GPU g listens on OWL_PORT_BASE + g (default: 18210)
#   OWL_BIND_HOST   bind address (default: 127.0.0.1; with network_mode: host the
#                   simulation container reaches it as http://127.0.0.1:<port>)
#   OWL_SHM_DIR     shared RGB slot directory; must be the same path the
#                   simulation container uses (default: /workspace/sim/.owl_shm)
#   OWL_LOG_DIR     per-GPU logs (default: $OWL_SHM_DIR/../logs/owl)
#   Extra arguments are passed to every sidecar (e.g. --skip-superglue).
set -euo pipefail

SIDECAR=${OWL_SIDECAR:-/workspace/sim/services/owl_sidecar.py}
GPUS=${OWL_GPUS:-0}
PORT_BASE=${OWL_PORT_BASE:-18210}
BIND=${OWL_BIND_HOST:-127.0.0.1}
export OWL_SHM_DIR=${OWL_SHM_DIR:-/workspace/sim/.owl_shm}
LOG_DIR=${OWL_LOG_DIR:-/workspace/sim/logs/owl}
mkdir -p "$OWL_SHM_DIR" "$LOG_DIR"
chmod a+rwx "$OWL_SHM_DIR" 2>/dev/null || true

pids=()
IFS=',' read -r -a gpu_list <<< "$GPUS"
for g in "${gpu_list[@]}"; do
  port=$((PORT_BASE + g))
  echo "[owl] gpu=$g port=$port shm=$OWL_SHM_DIR log=$LOG_DIR/owl_g${g}.log"
  CUDA_VISIBLE_DEVICES="$g" python "$SIDECAR" \
      --host "$BIND" --port "$port" --device cuda --shm-dir "$OWL_SHM_DIR" "$@" \
      > >(tee -a "$LOG_DIR/owl_g${g}.log") 2>&1 &
  pids+=($!)
done

trap 'kill "${pids[@]}" 2>/dev/null || true' INT TERM
# Exit (and let compose restart us) as soon as any sidecar dies.
wait -n "${pids[@]}"
status=$?
echo "[owl] a sidecar exited with status $status; stopping the others" >&2
kill "${pids[@]}" 2>/dev/null || true
exit "$status"
