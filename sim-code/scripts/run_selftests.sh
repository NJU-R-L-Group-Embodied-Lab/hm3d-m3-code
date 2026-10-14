#!/usr/bin/env bash
# Run the offline self-tests (no simulator episode, GPU or API key needed) and
# report PASS / FAIL / SKIP per test. Thin wrapper around tests/run_all.py, which
# holds the test list and dependency tiers.
#
#   bash scripts/run_selftests.sh               # every test whose deps are installed
#   bash scripts/run_selftests.sh metrics       # only tests whose name contains "metrics"
#   bash scripts/run_selftests.sh --tier cv     # only the light tiers (minimal + cv)
#
# A test is SKIPped when its tier's packages or its data file are missing; set
# STRICT=1 (or pass --strict) to count missing packages as failures. Logs go to
# LOG_DIR (default logs/selftests); per-test timeout TIMEOUT_S (default 600).
# Exit status: 0 if nothing FAILed.
set -uo pipefail

cd "$(dirname "$0")/.."
PY=${PY:-python}

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-}"
export TOKENIZERS_PARALLELISM=false
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

exec "$PY" tests/run_all.py "$@"
