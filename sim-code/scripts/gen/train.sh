#!/usr/bin/env bash
# HM3D-M3 train episodes (geometry only) on the OVON train scenes. Same protocol as
# Official Val: same-floor, mixed3, K in {2,3,4}, M in {3,4,5}.
# The released train split was built from two runs (merged afterwards):
#   SPLIT=t2000   seed 2026, 20 episodes/scene pool, 2000 episodes (rebakes the
#                 TESM v2 navmeshes for Habitat 0.1.7 first)
#   SPLIT=t14000  seed 2027, 120 episodes/scene pool, 14000 episodes, min-cell 400
#                 (reuses the t2000 rebake; set REBAKE=1 if t2000 was not run)
# Each run also writes a scene-stratified 100-episode train-mini file.
# Then fill IIN images and TN text with scripts/gen/fill_iin_tn.sh.
#
#   SPLIT=t2000 bash scripts/gen/train.sh
set -euo pipefail
source "$(dirname "$0")/../lib/common.sh"
hm3d_env

SPLIT=${SPLIT:-t2000}
case "$SPLIT" in
  t2000)  D_SEED=2026; D_EPS=20;  D_N=2000;  D_MIN_CELL=70;  D_REBAKE=1 ;;
  t14000) D_SEED=2027; D_EPS=120; D_N=14000; D_MIN_CELL=400; D_REBAKE=0 ;;
  *) echo "unknown SPLIT=$SPLIT (t2000 | t14000)" >&2; exit 2 ;;
esac
OUT_DIR=${OUT_DIR:-data/datasets/ma_objectnav_ovon/train}
CONTENT_DIR=${CONTENT_DIR:-data/datasets/hm3d_ovon/train/content}
SCENES_DIR=${SCENES_DIR:-data/scene_datasets/hm3d-1.0}
SEED=${SEED:-$D_SEED}
N_TRAIN=${N_TRAIN:-$D_N}
N_MINI=${N_MINI:-100}
EPS_PER_SCENE=${EPS_PER_SCENE:-$D_EPS}
MIN_CELL=${MIN_CELL:-$D_MIN_CELL}
MIN_SCENES=${MIN_SCENES:-120}
REBAKE=${REBAKE:-$D_REBAKE}
DRY_SCENES=${DRY_SCENES:-0}
SKIP_SAMPLE=${SKIP_SAMPLE:-0}
POOL=${POOL:-$OUT_DIR/candidates_train_k234_samefloor_mixed3_seed${SEED}.json}
TRAIN=${TRAIN:-$OUT_DIR/ma_train_${N_TRAIN}_samefloor_mixed3_seed${SEED}.json}
MINI=${MINI:-$OUT_DIR/ma_train_mini_100_samefloor_mixed3_seed${SEED}.json}

run() {
  echo "[gen-train] $*"
  if [ "$DRY_RUN" = "1" ]; then return 0; fi
  "$@"
}

[ "$DRY_RUN" = "1" ] || mkdir -p "$OUT_DIR"
if [ "$REBAKE" = "1" ]; then
  echo "[gen-train] rebake TESM v2 navmeshes for Habitat 0.1.7"
  run "$PY" -u -m benchmark.rebake_navmesh \
    --content-dir "$CONTENT_DIR" \
    --scenes-dir "$SCENES_DIR" \
    --only-tesm2 \
    --radius "${NAVMESH_RADIUS:-0.1}" \
    --height "${NAVMESH_HEIGHT:-0.88}"
fi

GEN_ARGS=(
  --content-dir "$CONTENT_DIR"
  --scenes-dir "$SCENES_DIR"
  --out "$POOL"
  --k-values 2,3,4
  --m-values 3,4,5
  --episodes-per-scene "$EPS_PER_SCENE"
  --seed "$SEED"
  --modality-mode mixed3
  --start-spacing mixed
  --max-tries 80
  --same-floor
)
if [ "$DRY_SCENES" != "0" ]; then
  GEN_ARGS+=(--dry-scenes "$DRY_SCENES")
fi
run "$PY" -u -m benchmark.generate "${GEN_ARGS[@]}"

if [ "$SKIP_SAMPLE" != "0" ]; then
  echo "[gen-train] skip sample (smoke)"
  echo "[gen-train] DONE $POOL"
  exit 0
fi

run "$PY" -u -m benchmark.sample_official \
  --pool "$POOL" \
  --val-out "$TRAIN" \
  --mini-out "$MINI" \
  --n-val "$N_TRAIN" \
  --n-mini "$N_MINI" \
  --seed "$SEED" \
  --min-cell "$MIN_CELL" \
  --min-scenes "$MIN_SCENES" \
  --require-same-floor
echo "[gen-train] DONE $TRAIN $MINI"
