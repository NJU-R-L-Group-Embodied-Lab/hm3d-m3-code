#!/usr/bin/env bash
# Fetch the GPL-licensed native components (FAST-LIO with ikd-Tree / IKFoM, and
# EGO-Planner v2) at pinned upstream commits, apply the HM3D-M3 patches, and
# place the result in the native source layers that tools/build_native.py
# assembles:
#
#   native/{common,ros1,ros2}/src/localization/FAST_LIO/
#   native/{common,ros1,ros2}/src/planner/Ego-Planner/
#
# These directories are git-ignored. Run this once before building:
#
#   bash native/fetch_native.sh             # from real-world-code/ (works from anywhere)
#   python tools/build_native.py --ros ros2 --localization
#
# For each component and layer, patches/<component>/<layer>.files lists which
# upstream file goes where ("SOURCE upstream/path dest/path", dest relative to
# the component directory), and patches/<component>/<layer>.patch (git diff
# format, applied with `git apply`) turns those pristine upstream files into
# the HM3D-M3 version, including files that exist only in this port.
#
# Modes:
#   (default)          fetch, patch and install. An existing directory that
#                      already matches is left alone; one with local edits is
#                      not overwritten unless FETCH_NATIVE_FORCE=1.
#   --check            fetch and patch, then compare with the installed
#                      directories; exit 1 if any differs or is missing.
#   --update-patches   regenerate patches/*/*.patch from the installed
#                      directories (maintainers: edit the fetched code, then
#                      run this to record the change).
#
# Environment:
#   FETCH_MIRROR_FAST_LIO, FETCH_MIRROR_IKD_TREE, FETCH_MIRROR_EGO_PLANNER_V2
#                        clone from this local path / URL instead of GitHub
#                        (the pinned commits must exist there)
#   FETCH_NATIVE_SRC     clone + staging directory
#                        (default: real-world-code/.runtime/native-upstream)
#   FETCH_NATIVE_FORCE=1 overwrite installed directories that have local edits
#
# Re-running is safe: clones are reused and every directory is rebuilt from the
# pinned upstream files plus the patches.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # real-world-code/native
REAL="$(dirname "$HERE")"
SRC_DIR="${FETCH_NATIVE_SRC:-$REAL/.runtime/native-upstream}"
PATCH_DIR="$HERE/patches"
STAGE_DIR="$SRC_DIR/stage"

FAST_LIO_URL=https://github.com/hku-mars/FAST_LIO
FAST_LIO_MAIN_COMMIT=7cc4175de6f8ba2edf34bab02a42195b141027e9   # branch `main` (ROS 1)
FAST_LIO_ROS2_COMMIT=87c42cdb8f2c569c27d95920ea6c733cffb3eba8   # branch `ROS2`
IKD_TREE_URL=https://github.com/hku-mars/ikd-Tree
IKD_TREE_COMMIT=e2e3f4e9d3b95a9e66b1ba83dc98d4a05ed8a3c4        # FAST_LIO's include/ikd-Tree submodule (branch `fast_lio`)
EGO_PLANNER_V2_URL=https://github.com/ZJU-FAST-Lab/EGO-Planner-v2
EGO_PLANNER_V2_COMMIT=9d85475ea7b9bf5c112cf7c3c3d0d3f9e96d9010

# component | directory inside each layer
COMPONENTS=("fast_lio:src/localization/FAST_LIO" "ego_planner:src/planner/Ego-Planner")
LAYERS=(common ros1 ros2)

log() { echo "[fetch_native] $*"; }
die() { echo "[fetch_native] ERROR: $*" >&2; exit 1; }

MODE=install
case "${1:-}" in
  "") ;;
  --check) MODE=check ;;
  --update-patches) MODE=update ;;
  -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
  *) die "unknown argument: $1 (use --check, --update-patches or --help)" ;;
esac

command -v git >/dev/null || die "git is required"

# source key -> "<clone name> <commit>"
source_of() {
  case "$1" in
    FAST_LIO_MAIN)  echo "fast_lio $FAST_LIO_MAIN_COMMIT" ;;
    FAST_LIO_ROS2)  echo "fast_lio $FAST_LIO_ROS2_COMMIT" ;;
    IKD_TREE)       echo "ikd_tree $IKD_TREE_COMMIT" ;;
    EGO_PLANNER_V2) echo "ego_planner_v2 $EGO_PLANNER_V2_COMMIT" ;;
    *) die "unknown source key '$1' in a .files manifest" ;;
  esac
}

# fetch_commits <name> <url> <mirror> <commit>...
# Object-only clone (no checkout); files are read with `git cat-file`.
fetch_commits() {
  local name=$1 url=$2 mirror=$3; shift 3
  local dst="$SRC_DIR/$name" src="${mirror:-$url}" c try
  if [ ! -d "$dst/.git" ]; then
    rm -rf "$dst"; mkdir -p "$dst"
    git -C "$dst" init -q
    git -C "$dst" remote add origin "$src"
  else
    git -C "$dst" remote set-url origin "$src"
  fi
  for c in "$@"; do
    if git -C "$dst" cat-file -e "$c^{commit}" 2>/dev/null; then
      log "$name: have ${c:0:7}"
      continue
    fi
    log "$name: fetching ${c:0:7} from $src"
    for try in 1 2 3; do
      git -C "$dst" fetch -q --depth 1 origin "$c" 2>/dev/null && break
      sleep "$try"
    done
    if ! git -C "$dst" cat-file -e "$c^{commit}" 2>/dev/null; then
      # Servers that refuse fetching by SHA: fall back to all branches.
      for try in 1 2 3; do
        git -C "$dst" fetch -q origin '+refs/heads/*:refs/remotes/origin/*' && break
        sleep "$try"
      done
    fi
    git -C "$dst" cat-file -e "$c^{commit}" 2>/dev/null \
      || die "commit $c not found in $src"
  done
}

# stage_base <component> <layer>: pristine upstream files laid out per manifest
stage_base() {
  local comp=$1 layer=$2 key up dest name commit
  local stage="$STAGE_DIR/$comp/$layer" manifest="$PATCH_DIR/$comp/$layer.files"
  [ -f "$manifest" ] || die "missing $manifest"
  rm -rf "$stage"; mkdir -p "$stage"
  git -C "$stage" init -q
  while read -r key up dest; do
    case "$key" in ""|\#*) continue ;; esac
    read -r name commit <<<"$(source_of "$key")"
    mkdir -p "$stage/$(dirname "$dest")"
    git -C "$SRC_DIR/$name" cat-file blob "$commit:$up" > "$stage/$dest" \
      || die "$comp/$layer: $up not found at $name@${commit:0:7}"
  done < "$manifest"
  git -C "$stage" add -A
  git -C "$stage" -c user.name=fetch_native -c user.email=fetch@example.invalid \
    -c commit.gpgsign=false commit -q --no-verify --allow-empty -m upstream
}

# stage_patched <component> <layer>
stage_patched() {
  local comp=$1 layer=$2
  local stage="$STAGE_DIR/$comp/$layer" patch="$PATCH_DIR/$comp/$layer.patch"
  stage_base "$comp" "$layer"
  if [ -s "$patch" ]; then
    git -C "$stage" apply --whitespace=nowarn "$patch" \
      || die "$comp/$layer: $(basename "$patch") does not apply"
  fi
}

same_tree() {  # same_tree <stage> <target>
  [ -d "$2" ] && diff -r -q --exclude=.git "$1" "$2" >/dev/null 2>&1
}

patch_header() {
  local comp=$1 layer=$2
  case "$comp" in
    fast_lio) cat <<EOF
HM3D-M3 changes to FAST-LIO (native/$layer/src/localization/FAST_LIO).
Upstream: $FAST_LIO_URL ($( [ "$layer" = ros2 ] && echo "branch ROS2 @ $FAST_LIO_ROS2_COMMIT" || echo "branch main @ $FAST_LIO_MAIN_COMMIT" )),
ikd-Tree: $IKD_TREE_URL @ $IKD_TREE_COMMIT.
This patch is a modification of GPL-2.0 code and is distributed under the
GNU General Public License v2 (see the upstream LICENSE file, installed as
FAST_LIO/LICENSE). Generated by fetch_native.sh --update-patches.

EOF
    ;;
    ego_planner) cat <<EOF
HM3D-M3 changes to EGO-Planner v2 (native/$layer/src/planner/Ego-Planner).
Upstream: $EGO_PLANNER_V2_URL @ $EGO_PLANNER_V2_COMMIT
(swarm-playground/main_ws/src/{planner,Utils/quadrotor_msgs}).
This patch is a modification of GPL-3.0 code and is distributed under the
GNU General Public License v3 (see the upstream LICENSE file, installed as
Ego-Planner/LICENSE). Generated by fetch_native.sh --update-patches.

EOF
    ;;
  esac
}

# ---------------------------------------------------------------- 1. clones
mkdir -p "$SRC_DIR"
fetch_commits fast_lio "$FAST_LIO_URL" "${FETCH_MIRROR_FAST_LIO:-}" \
  "$FAST_LIO_MAIN_COMMIT" "$FAST_LIO_ROS2_COMMIT"
fetch_commits ikd_tree "$IKD_TREE_URL" "${FETCH_MIRROR_IKD_TREE:-}" "$IKD_TREE_COMMIT"
fetch_commits ego_planner_v2 "$EGO_PLANNER_V2_URL" "${FETCH_MIRROR_EGO_PLANNER_V2:-}" \
  "$EGO_PLANNER_V2_COMMIT"

# ------------------------------------------------- 2. stage, patch, install
status=0
for entry in "${COMPONENTS[@]}"; do
  comp=${entry%%:*}; sub=${entry#*:}
  for layer in "${LAYERS[@]}"; do
    target="$HERE/$layer/$sub"
    stage="$STAGE_DIR/$comp/$layer"
    case "$MODE" in
      update)
        [ -d "$target" ] || die "$target missing; run without --update-patches first"
        stage_base "$comp" "$layer"
        find "$stage" -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
        cp -R "$target/." "$stage/"
        git -C "$stage" add -A
        patch="$PATCH_DIR/$comp/$layer.patch"
        if git -C "$stage" diff --cached --quiet; then
          rm -f "$patch"
          log "$comp/$layer: identical to upstream, no patch"
        else
          { patch_header "$comp" "$layer"
            git -C "$stage" -c core.quotepath=false -c diff.noprefix=false \
              -c diff.mnemonicPrefix=false diff --cached --no-renames --no-color \
              --no-ext-diff --binary --src-prefix=a/ --dst-prefix=b/
          } > "$patch"
          log "$comp/$layer: wrote $(basename "$patch") ($(grep -c '^diff --git' "$patch") files)"
        fi
        ;;
      check)
        stage_patched "$comp" "$layer"
        if same_tree "$stage" "$target"; then
          log "$comp/$layer: OK"
        else
          log "$comp/$layer: DIFFERS from fetched + patched upstream (or missing): $target"
          status=1
        fi
        ;;
      install)
        stage_patched "$comp" "$layer"
        if same_tree "$stage" "$target"; then
          log "$comp/$layer: up to date"
          continue
        fi
        if [ -d "$target" ] && [ "${FETCH_NATIVE_FORCE:-0}" != 1 ]; then
          die "$target has local changes. Record them with --update-patches, or set FETCH_NATIVE_FORCE=1 to overwrite."
        fi
        rm -rf "$target"; mkdir -p "$target"
        cp -R "$stage/." "$target/"
        rm -rf "$target/.git"
        log "$comp/$layer: installed $target"
        ;;
    esac
  done
done
exit "$status"
