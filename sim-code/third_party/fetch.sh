#!/usr/bin/env bash
# Fetch the third-party code used by the official DM3-Nav baseline
# (benchmark/dm3_official) at pinned commits, apply the HM3D-M3 patches, and
# assemble the `home_robot` package tree that benchmark/dm3_official/runner.py
# puts on sys.path.
#
#   bash sim-code/third_party/fetch.sh            # from anywhere
#
# Result:
#   sim-code/third_party/src/<name>/      pinned upstream clones (patched in place)
#   sim-code/third_party/home_robot/      assembled tree; contains the `home_robot`
#                                    package (DM3_HOME_ROBOT_DIR overrides it
#                                    at run time)
#
# Environment:
#   FETCH_MIRROR_DM3_NAV, FETCH_MIRROR_HOME_ROBOT, FETCH_MIRROR_DETIC,
#   FETCH_MIRROR_CENTERNET2   clone from this local path / URL instead of GitHub
#                             (the pinned commit must exist there)
#   SUPERGLUE_DIR             a checkout of mukulkhanna/SuperGluePretrainedNetwork
#                             (non-commercial license; never fetched automatically)
#   THIRD_PARTY_SRC           clone directory   (default: sim-code/third_party/src)
#   THIRD_PARTY_OUT           assembled tree    (default: sim-code/third_party/home_robot)
#
# Re-running is safe: existing clones are reused, applied patches are skipped,
# and the assembled tree is rebuilt from scratch.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${THIRD_PARTY_SRC:-$HERE/src}"
OUT_DIR="${THIRD_PARTY_OUT:-$HERE/home_robot}"
PATCH_DIR="$HERE/patches"

# name | upstream URL | pinned commit
DM3_NAV_URL=https://github.com/Multi-Agent-Robotics-and-Autonomy-Lab/DM3_Nav
DM3_NAV_COMMIT=5037cf364b3757739d5f7b443b4aa9acb9f2dc9c
HOME_ROBOT_URL=https://github.com/facebookresearch/home-robot
HOME_ROBOT_COMMIT=b856925e11d888848c514d0c2509a07f298ceb44   # branch `goat`
DETIC_URL=https://github.com/facebookresearch/Detic
DETIC_COMMIT=46f8317a0688b4b94e85816b5839a1dbca26d1e0
CENTERNET2_URL=https://github.com/xingyizhou/CenterNet2
CENTERNET2_COMMIT=8745e012e4dbdf560ac2f27e0b771d4907ad4aaf
SUPERGLUE_URL=https://github.com/mukulkhanna/SuperGluePretrainedNetwork
SUPERGLUE_COMMIT=d2d6dcc21f67690c07a2787acfb357b7ed59b920

log() { echo "[fetch] $*"; }
die() { echo "[fetch] ERROR: $*" >&2; exit 1; }

# fetch_repo <name> <url> <commit> <mirror>
fetch_repo() {
  local name=$1 url=$2 commit=$3 mirror=$4
  local dst="$SRC_DIR/$name" src="${mirror:-$url}"
  if [ -d "$dst/.git" ]; then
    local head
    head=$(git -C "$dst" rev-parse HEAD)
    [ "$head" = "$commit" ] || die "$dst is at $head, expected $commit. Remove it and re-run."
    log "$name: reusing $dst @ ${commit:0:7}"
    return
  fi
  log "$name: cloning $src @ ${commit:0:7}"
  rm -rf "$dst"
  mkdir -p "$dst"
  git -C "$dst" init -q
  git -C "$dst" remote add origin "$src"
  # Shallow fetch of the pinned commit; fall back to a full fetch.
  if ! git -C "$dst" fetch -q --depth 1 origin "$commit" 2>/dev/null; then
    git -C "$dst" fetch -q origin '+refs/heads/*:refs/remotes/origin/*' \
      || die "cannot fetch $name from $src"
  fi
  git -C "$dst" -c advice.detachedHead=false checkout -q "$commit" \
    || die "commit $commit not found in $src"
}

# apply_patches <dir> <patch subdir>
apply_patches() {
  local dir=$1 sub=$2 p
  [ -d "$PATCH_DIR/$sub" ] || return 0
  for p in "$PATCH_DIR/$sub"/*.patch; do
    [ -e "$p" ] || continue
    if patch -d "$dir" -p1 -R --dry-run -s -f < "$p" >/dev/null 2>&1; then
      log "$sub: $(basename "$p") already applied"
    else
      patch -d "$dir" -p1 -s -f --no-backup-if-mismatch < "$p" \
        || die "$sub: $(basename "$p") does not apply"
      log "$sub: applied $(basename "$p")"
    fi
  done
}

command -v git >/dev/null || die "git is required"
command -v patch >/dev/null || die "patch is required"
command -v rsync >/dev/null || die "rsync is required"
mkdir -p "$SRC_DIR"

# ---------------------------------------------------------------- 1. clones
fetch_repo dm3_nav    "$DM3_NAV_URL"    "$DM3_NAV_COMMIT"    "${FETCH_MIRROR_DM3_NAV:-}"
fetch_repo home_robot "$HOME_ROBOT_URL" "$HOME_ROBOT_COMMIT" "${FETCH_MIRROR_HOME_ROBOT:-}"
fetch_repo detic      "$DETIC_URL"      "$DETIC_COMMIT"      "${FETCH_MIRROR_DETIC:-}"
fetch_repo centernet2 "$CENTERNET2_URL" "$CENTERNET2_COMMIT" "${FETCH_MIRROR_CENTERNET2:-}"

# SuperGlue: Magic Leap academic / non-profit non-commercial license.
# Never fetched automatically; the user supplies a checkout via SUPERGLUE_DIR.
HAVE_SG=0
if [ -n "${SUPERGLUE_DIR:-}" ]; then
  [ -f "$SUPERGLUE_DIR/models/superglue.py" ] || die "SUPERGLUE_DIR=$SUPERGLUE_DIR has no models/superglue.py"
  if [ -d "$SUPERGLUE_DIR/.git" ]; then
    sg_head=$(git -C "$SUPERGLUE_DIR" rev-parse HEAD)
    [ "$sg_head" = "$SUPERGLUE_COMMIT" ] \
      || log "WARNING: SUPERGLUE_DIR is at $sg_head; the tested commit is $SUPERGLUE_COMMIT"
  fi
  log "superglue: copying $SUPERGLUE_DIR (code only)"
  rm -rf "$SRC_DIR/superglue"
  rsync -a --exclude=.git --exclude=assets --exclude=models/weights \
    "$SUPERGLUE_DIR/" "$SRC_DIR/superglue/"
  HAVE_SG=1
elif [ -f "$SRC_DIR/superglue/models/superglue.py" ]; then
  log "superglue: reusing $SRC_DIR/superglue from a previous run"
  HAVE_SG=1
fi

# --------------------------------------------------------------- 2. patches
apply_patches "$SRC_DIR/dm3_nav" dm3_nav
if [ "$HAVE_SG" = 1 ]; then apply_patches "$SRC_DIR/superglue" superglue; fi

# ---------------------------------------------------------------- 3. layout
# Only the algorithm core used by benchmark/dm3_official/runner.py.
DM3="$SRC_DIR/dm3_nav/src/home_robot/home_robot"
HR="$SRC_DIR/home_robot/src/home_robot/home_robot"
DST="$OUT_DIR/home_robot"
log "assembling $DST"
rm -rf "$OUT_DIR"
mkdir -p "$DST"/{core,utils,mapping/semantic,navigation_planner,agent/goat_agent,agent/imagenav_agent,perception/detection/detic}

cp "$DM3/__init__.py" "$DST/"
# core/ is missing from the DM3_Nav repo (its .gitignore drops `core`);
# take it from home-robot@goat, which DM3_Nav is built on.
rsync -a "$HR/core/" "$DST/core/"
for f in __init__.py pose.py depth.py rotation.py logger.py camera_params.py \
         visualization.py constants.py spot.py; do
  cp "$DM3/utils/$f" "$DST/utils/"
done
rsync -a "$DM3/utils/geometry/" "$DST/utils/geometry/"

cp "$DM3/mapping/__init__.py" "$DM3/mapping/map_utils.py" "$DST/mapping/"
for f in __init__.py constants.py categorical_2d_semantic_map_module.py \
         categorical_2d_semantic_map_state.py instance_tracking_modules.py; do
  cp "$DM3/mapping/semantic/$f" "$DST/mapping/semantic/"
done

for f in fixed_discrete_planner.py fmm_planner.py co_location_matrix.npy; do
  cp "$DM3/navigation_planner/$f" "$DST/navigation_planner/"
done
if [ -f "$DM3/navigation_planner/__init__.py" ]; then
  cp "$DM3/navigation_planner/__init__.py" "$DST/navigation_planner/"
else
  touch "$DST/navigation_planner/__init__.py"
fi

cp "$DM3/agent/__init__.py" "$DST/agent/"
for f in goat_agent.py multiagent_goat_agent.py goat_matching.py map_merger.py; do
  cp "$DM3/agent/goat_agent/$f" "$DST/agent/goat_agent/"
done
cp "$DM3/agent/imagenav_agent/__init__.py" "$DM3/agent/imagenav_agent/superglue.py" \
   "$DST/agent/imagenav_agent/"
if [ "$HAVE_SG" = 1 ]; then
  rsync -a "$SRC_DIR/superglue/" "$DST/agent/imagenav_agent/SuperGluePretrainedNetwork/"
fi

cp "$DM3/perception/__init__.py" "$DM3/perception/constants.py" \
   "$DM3/perception/matterport_category_mappings.tsv" "$DST/perception/"
mkdir -p "$DST/perception/detection"
cp "$DM3/perception/detection/utils.py" "$DST/perception/detection/"
for f in __init__.py detic_perception.py detic_mask.py; do
  cp "$DM3/perception/detection/detic/$f" "$DST/perception/detection/detic/"
done
# Detic tree as shipped in DM3_Nav (includes CenterNet2 and Deformable-DETR).
# DM3_Nav's .gitignore dropped every `data/` and `datasets/` directory; restore
# them from the Detic and CenterNet2 commits that DM3_Nav vendors.
DETIC_DST="$DST/perception/detection/detic/Detic"
rsync -a "$DM3/perception/detection/detic/Detic/" "$DETIC_DST/"
rsync -a "$SRC_DIR/detic/datasets/"            "$DETIC_DST/datasets/"
rsync -a "$SRC_DIR/detic/detic/data/"          "$DETIC_DST/detic/data/"
rsync -a "$SRC_DIR/centernet2/datasets/"       "$DETIC_DST/third_party/CenterNet2/datasets/"
rsync -a "$SRC_DIR/centernet2/centernet/data/" "$DETIC_DST/third_party/CenterNet2/centernet/data/"

n=$(find "$DST" -name '*.py' | wc -l)
log "done: $n .py files under $DST"

if [ "$HAVE_SG" != 1 ]; then
  cat <<EOF

[fetch] SuperGlue was NOT installed. The DM3-Nav baseline needs it for image goals.
        SuperGluePretrainedNetwork is licensed by Magic Leap for academic or
        non-profit non-commercial research use only, so it is not fetched
        automatically. If you accept that license:

          git clone $SUPERGLUE_URL /path/to/SuperGluePretrainedNetwork
          git -C /path/to/SuperGluePretrainedNetwork checkout $SUPERGLUE_COMMIT
          SUPERGLUE_DIR=/path/to/SuperGluePretrainedNetwork bash $0

        Weights: set DM3_SUPERGLUE_WEIGHTS_DIR to a directory holding
        superpoint_v1.pth and superglue_indoor.pth (default
        data/datasets/dm3_official/superglue_weights/). See sim-code/third_party/README.md.
EOF
fi
