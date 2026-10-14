#!/usr/bin/env bash
# Official Val1000 + Mini-100 episodes (geometry only): same-floor, mixed3
# (every episode has ON + IIN + TN goals), K in {2,3,4}, M in {3,4,5}, seed 1123,
# on the OVON val_unseen scenes. Mini-100 is drawn with the 9-cell K x M quota.
# Then fill IIN images and TN text with scripts/gen/fill_iin_tn.sh.
#
#   bash scripts/gen/val1000_mini100.sh
set -euo pipefail
source "$(dirname "$0")/../lib/common.sh"
hm3d_env

OUT_DIR=${OUT_DIR:-data/datasets/ma_objectnav_ovon/val_unseen}
CONTENT_DIR=${CONTENT_DIR:-data/datasets/hm3d_ovon/val_unseen/content}
SCENES_DIR=${SCENES_DIR:-data/scene_datasets/hm3d-1.0}
SEED=${SEED:-1123}
POOL=${POOL:-$OUT_DIR/candidates_official_k234_samefloor_mixed3_seed${SEED}.json}
VAL=${VAL:-$OUT_DIR/ma_official_val_1000_samefloor_mixed3_seed${SEED}.json}
MINI=${MINI:-$OUT_DIR/ma_official_mini_100_samefloor_mixed3_kmquota_seed${SEED}.json}
QUOTA=${QUOTA:-13,12,10,13,10,10,12,10,10}
EPS_PER_SCENE=${EPS_PER_SCENE:-80}

run() {
  echo "[gen-val] $*"
  if [ "$DRY_RUN" = "1" ]; then return 0; fi
  "$@"
}

[ "$DRY_RUN" = "1" ] || mkdir -p "$OUT_DIR"
run "$PY" -u -m benchmark.generate \
  --content-dir "$CONTENT_DIR" \
  --scenes-dir "$SCENES_DIR" \
  --out "$POOL" \
  --k-values 2,3,4 \
  --m-values 3,4,5 \
  --episodes-per-scene "$EPS_PER_SCENE" \
  --seed "$SEED" \
  --modality-mode mixed3 \
  --start-spacing mixed \
  --max-tries 80 \
  --same-floor
run "$PY" -u -m benchmark.sample_official \
  --pool "$POOL" \
  --val-out "$VAL" \
  --mini-out "$MINI" \
  --n-val 1000 \
  --n-mini 100 \
  --seed "$SEED" \
  --min-cell 70 \
  --min-scenes 36 \
  --require-same-floor \
  --mini-km-quota "$QUOTA" \
  --min-per-scene 2 \
  --max-per-scene 3
echo "[gen-val] DONE $VAL $MINI"
