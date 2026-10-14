#!/usr/bin/env bash
# Download, verify and install the HM3D-M3 data into the layout the sim-code/ launchers expect.
# Run it from sim-code/ (or give its path from anywhere).
#
#   bash download_data.sh                    # val + mini (GitHub Release)
#   bash download_data.sh val                # only Val-1000 (episodes + both IIN variants)
#   bash download_data.sh mini               # only Mini-100
#   bash download_data.sh --train --teacher  # also the Hugging Face train split + teacher samples
#
# Packages (positional, default "all"):
#   val      Val-1000 episodes + IIN images stretch512 and stretch_v2
#   mini     Mini-100 episodes + IIN images stretch512
#   all      val mini
#   none     no GitHub Release package (use with --train / --teacher)
#
# Options:
#   --data-root DIR   install root, mounted as sim-code/data (default: $DATA_ROOT or sim-code/data)
#   --dl-dir DIR      download cache (default: <data-root>/downloads/github-v1.0.0)
#   --base-url URL    release download base (default: $DATA_BASE_URL, else derived from the
#                     GitHub Release link in README.md). https://, http://, file:// or a
#                     local directory.
#   --train           Hugging Face: train episodes + train IIN images (installed under train/)
#   --teacher         Hugging Face: 100K observation-assignment samples (kept in the HF directory)
#   --hf-repo ID      Hugging Face dataset id (default: $HF_DATASET_REPO, else from README.md)
#   --hf-dir DIR      HF snapshot directory (default: <data-root>/downloads/hf). If it already
#                     holds the files, nothing is downloaded.
#   --force           re-extract / re-decompress even if the installed copy is up to date
#   --clean           delete the downloaded archives after a successful install
#   -h, --help        this text
#
# Every file is checked against SHA256SUMS before it is installed. Re-running is cheap:
# downloads whose checksum already matches are kept, and installed files carry a stamp
# (<install dir>/.hm3d_m3_installed/) so that unchanged assets are not re-extracted.
# curl honors https_proxy / HTTPS_PROXY / http_proxy / no_proxy from the environment.
# HM3D scenes are not downloaded: get them from Matterport under the HM3D terms.
set -euo pipefail

TAG=v1.0.0
SIM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # sim-code/
REPO_ROOT="$(cd "$SIM_DIR/.." && pwd)"

usage() { sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; }
log() { printf '[download_data] %s\n' "$*"; }
die() { printf '[download_data] ERROR: %s\n' "$*" >&2; exit "${2:-1}"; }

DATA_ROOT=${DATA_ROOT:-$SIM_DIR/data}
DL_DIR=""
BASE_URL=${DATA_BASE_URL:-}
HF_REPO=${HF_DATASET_REPO:-}
HF_DIR=""
WANT_TRAIN=0
WANT_TEACHER=0
FORCE=0
CLEAN=0
PKGS=()

while [ "$#" -gt 0 ]; do
  case "$1" in
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --dl-dir) DL_DIR="$2"; shift 2 ;;
    --base-url) BASE_URL="$2"; shift 2 ;;
    --hf-repo) HF_REPO="$2"; shift 2 ;;
    --hf-dir) HF_DIR="$2"; shift 2 ;;
    --train) WANT_TRAIN=1; shift ;;
    --teacher) WANT_TEACHER=1; shift ;;
    --force) FORCE=1; shift ;;
    --clean) CLEAN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    val|mini|all|none) PKGS+=("$1"); shift ;;
    *) usage >&2; die "unknown argument: $1" 2 ;;
  esac
done
[ "${#PKGS[@]}" -gt 0 ] || PKGS=(all)

WANT_VAL=0 WANT_MINI=0
for p in "${PKGS[@]}"; do
  case "$p" in
    val) WANT_VAL=1 ;;
    mini) WANT_MINI=1 ;;
    all) WANT_VAL=1; WANT_MINI=1 ;;
    none) ;;
  esac
done

mkdir -p "$DATA_ROOT"
DATA_ROOT="$(cd "$DATA_ROOT" && pwd)"
DL_DIR=${DL_DIR:-$DATA_ROOT/downloads/github-$TAG}
HF_DIR=${HF_DIR:-$DATA_ROOT/downloads/hf}
VAL_DIR="$DATA_ROOT/datasets/ma_objectnav_ovon/val_unseen"
TRAIN_DIR="$DATA_ROOT/datasets/ma_objectnav_ovon/train"

# ---------------------------------------------------------------- helpers
if command -v sha256sum >/dev/null 2>&1; then
  sha256_of() { sha256sum "$1" | awk '{print $1}'; }
elif command -v shasum >/dev/null 2>&1; then
  sha256_of() { shasum -a 256 "$1" | awk '{print $1}'; }
else
  die "need sha256sum or shasum"
fi

# expected_sha <SHA256SUMS> <relative name>
expected_sha() {
  awk -v n="$2" '{f=$2; sub(/^\*/, "", f); if (f == n) {print $1; exit}}' "$1"
}

# A URL from README.md (filled in when the release is exported); maintainers' private tree: tools/profiles/public.json.
doc_url() {
  local pat="$1" key="$2" url=""
  if [ -f "$REPO_ROOT/README.md" ]; then
    url=$(grep -oE "$pat" "$REPO_ROOT/README.md" | head -n 1 || true)
  fi
  if [ -z "$url" ] && [ -f "$REPO_ROOT/tools/profiles/public.json" ]; then
    url=$(sed -n "s/.*\"$key\": *\"\\([^\"]*\\)\".*/\\1/p" "$REPO_ROOT/tools/profiles/public.json" | head -n 1)
  fi
  printf '%s' "$url"
}

# fetch <name> <dest>: download (or copy) one release asset.
fetch() {
  local name="$1" dest="$2" tmp="$2.part"
  case "$BASE_URL" in
    http://*|https://*|file://*)
      curl -fL --retry 5 --retry-delay 5 --retry-connrefused --connect-timeout 30 \
        -sS -o "$tmp" "${BASE_URL%/}/$name" || { rm -f "$tmp"; return 1; } ;;
    *)
      cp "${BASE_URL%/}/$name" "$tmp" || { rm -f "$tmp"; return 1; } ;;
  esac
  mv -f "$tmp" "$dest"
}

# get_asset <name>: make sure DL_DIR/<name> exists and matches SHA256SUMS.
get_asset() {
  local name="$1" want have
  want=$(expected_sha "$DL_DIR/SHA256SUMS" "$name")
  [ -n "$want" ] || die "$name is not listed in SHA256SUMS"
  if [ -f "$DL_DIR/$name" ] && [ "$(sha256_of "$DL_DIR/$name")" = "$want" ]; then
    log "ok (cached)  $name"
    return 0
  fi
  log "download     $name"
  fetch "$name" "$DL_DIR/$name" || die "download failed: ${BASE_URL%/}/$name"
  have=$(sha256_of "$DL_DIR/$name")
  if [ "$have" != "$want" ]; then
    mv -f "$DL_DIR/$name" "$DL_DIR/$name.bad"
    die "checksum mismatch for $name (expected $want, got $have; kept as $name.bad)"
  fi
  log "verified     $name"
}

# stamp helpers: <install dir>/.hm3d_m3_installed/<asset> holds the asset's sha256.
is_installed() {  # is_installed <dir> <asset> <sha> <target>
  [ "$FORCE" = 0 ] && [ -e "$4" ] && [ -f "$1/.hm3d_m3_installed/$2" ] \
    && [ "$(cat "$1/.hm3d_m3_installed/$2")" = "$3" ]
}
mark_installed() { mkdir -p "$1/.hm3d_m3_installed"; printf '%s\n' "$3" > "$1/.hm3d_m3_installed/$2"; }

# install_json <archive> <sha> <dest dir> <dest name>
install_json() {
  local src="$1" sha="$2" dir="$3" out="$4" asset
  asset=$(basename "$src")
  if is_installed "$dir" "$asset" "$sha" "$dir/$out"; then
    log "installed    $out (up to date)"
    return 0
  fi
  mkdir -p "$dir/$(dirname "$out")"
  gunzip -c "$src" > "$dir/$out.part"
  mv -f "$dir/$out.part" "$dir/$out"
  mark_installed "$dir" "$asset" "$sha"
  log "installed    $out"
}

# install_tar <archive> <sha> <dest dir> <top folder> [<more parts> ...]
# The archives hold one top-level folder; it is replaced as a whole.
install_tar() {
  local src="$1" sha="$2" dir="$3" top="$4" asset tmp bad
  shift 4
  asset=$(basename "$src")
  if is_installed "$dir" "$asset" "$sha" "$dir/$top/manifest.json"; then
    log "installed    $top/ (up to date)"
    return 0
  fi
  mkdir -p "$dir"
  tmp="$dir/.extract.$top.$$"
  rm -rf "$tmp"
  mkdir -p "$tmp"
  for a in "$src" "$@"; do
    bad=$(tar -tzf "$a" | grep -vE "^(\\./)?$top(/|$)" | head -n 1 || true)
    [ -z "$bad" ] || { rm -rf "$tmp"; die "$(basename "$a"): unexpected member '$bad' (expected only $top/)"; }
    tar -xzf "$a" -C "$tmp" --no-same-owner
  done
  [ -f "$tmp/$top/manifest.json" ] || { rm -rf "$tmp"; die "$asset: no $top/manifest.json"; }
  rm -rf "$dir/$top"
  mv "$tmp/$top" "$dir/$top"
  rm -rf "$tmp"
  mark_installed "$dir" "$asset" "$sha"
  log "installed    $top/ ($(find "$dir/$top" -type f -name '*.png' | wc -l | tr -d ' ') images)"
}

# ---------------------------------------------------------------- GitHub Release
ASSETS_JSON=()   # "<asset>|<dest name under val_unseen>"
ASSETS_TAR=()    # "<asset>|<top folder>"
if [ "$WANT_VAL" = 1 ]; then
  ASSETS_JSON+=("ma_official_val_1000_samefloor_mixed3_seed1123.json.gz|ma_official_val_1000_samefloor_mixed3_seed1123.json")
  ASSETS_TAR+=("iin_images_official_val1000_mixed3_stretch512.tar.gz|iin_images_official_val1000_mixed3_stretch512")
  ASSETS_TAR+=("iin_images_official_val1000_mixed3_stretch_v2.tar.gz|iin_images_official_val1000_mixed3_stretch_v2")
fi
if [ "$WANT_MINI" = 1 ]; then
  ASSETS_JSON+=("ma_official_mini_100_samefloor_mixed3_kmquota_seed1123.json.gz|ma_official_mini_100_samefloor_mixed3_kmquota_seed1123.json")
  ASSETS_TAR+=("iin_images_official_mini100_mixed3_stretch512.tar.gz|iin_images_official_mini100_mixed3_stretch512")
fi

if [ "${#ASSETS_JSON[@]}" -gt 0 ]; then
  if [ -z "$BASE_URL" ]; then
    rel=$(doc_url 'https://github\.com/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+/releases/tag/[A-Za-z0-9._-]+' DATA_RELEASE_URL)
    case "$rel" in
      https://github.com/*/releases/tag/*) BASE_URL="${rel%/releases/tag/*}/releases/download/${rel##*/releases/tag/}" ;;
      *) die "cannot derive the release URL from README.md; set DATA_BASE_URL (e.g. https://github.com/<owner>/<repo>/releases/download/$TAG)" 2 ;;
    esac
  fi
  case "$BASE_URL" in
    http://*|https://*|file://*) command -v curl >/dev/null 2>&1 || die "curl is required" ;;
    *) [ -d "$BASE_URL" ] || die "DATA_BASE_URL is neither a URL nor a directory: $BASE_URL" 2 ;;
  esac
  log "source  $BASE_URL"
  log "install $VAL_DIR"
  mkdir -p "$DL_DIR" "$VAL_DIR"

  fetch SHA256SUMS "$DL_DIR/SHA256SUMS" || die "cannot download SHA256SUMS from $BASE_URL"
  get_asset RELEASE_NOTES.md

  for e in "${ASSETS_JSON[@]}"; do get_asset "${e%%|*}"; done
  for e in "${ASSETS_TAR[@]}"; do get_asset "${e%%|*}"; done
  for e in "${ASSETS_JSON[@]}"; do
    a=${e%%|*}
    install_json "$DL_DIR/$a" "$(expected_sha "$DL_DIR/SHA256SUMS" "$a")" "$VAL_DIR" "${e#*|}"
  done
  for e in "${ASSETS_TAR[@]}"; do
    a=${e%%|*}
    install_tar "$DL_DIR/$a" "$(expected_sha "$DL_DIR/SHA256SUMS" "$a")" "$VAL_DIR" "${e#*|}"
  done
  if [ "$CLEAN" = 1 ]; then
    for e in "${ASSETS_JSON[@]}" "${ASSETS_TAR[@]}"; do rm -f "$DL_DIR/${e%%|*}"; done
    log "removed downloaded archives from $DL_DIR"
  fi
fi

# ---------------------------------------------------------------- Hugging Face
hf_instructions() {
  cat >&2 <<EOF
[download_data] To get the Hugging Face part manually (accept the HM3D terms on the dataset
[download_data] page and log in with 'hf auth login' / 'huggingface-cli login' first):
  pip install -U huggingface_hub
  huggingface-cli download ${HF_REPO:-<owner>/<dataset>} --repo-type dataset --local-dir $HF_DIR $HF_INCLUDE_ARGS
  bash download_data.sh none $HF_FLAGS--hf-dir $HF_DIR --data-root $DATA_ROOT
EOF
}

# verify_hf <prefix regex>: check the listed files of the HF snapshot that match the prefix.
verify_hf() {
  local re="$1" n=0 bad=0 sum path want
  [ -f "$HF_DIR/SHA256SUMS" ] || die "$HF_DIR/SHA256SUMS is missing"
  while read -r sum path; do
    path=${path#\*}
    [[ "$path" =~ $re ]] || continue
    n=$((n + 1))
    if [ ! -f "$HF_DIR/$path" ]; then
      log "missing      hf/$path"; bad=$((bad + 1)); continue
    fi
    want=$sum
    if [ "$(sha256_of "$HF_DIR/$path")" != "$want" ]; then
      log "BAD checksum hf/$path"; bad=$((bad + 1))
    fi
  done < "$HF_DIR/SHA256SUMS"
  [ "$n" -gt 0 ] || die "no file matching '$re' in $HF_DIR/SHA256SUMS"
  [ "$bad" -eq 0 ] || die "$bad of $n Hugging Face files missing or corrupt in $HF_DIR"
  log "verified     $n Hugging Face files ($re)"
}

HF_STATUS=0
if [ "$WANT_TRAIN" = 1 ] || [ "$WANT_TEACHER" = 1 ]; then
  HF_INCLUDE=(SHA256SUMS README.md SCHEMA.md id_remap.json)
  HF_FLAGS=""
  HF_RE='^('
  if [ "$WANT_TRAIN" = 1 ]; then HF_INCLUDE+=("episodes/*" "iin/*"); HF_FLAGS+="--train "; HF_RE+='episodes/|iin/|'; fi
  if [ "$WANT_TEACHER" = 1 ]; then HF_INCLUDE+=("teacher_100k/*"); HF_FLAGS+="--teacher "; HF_RE+='teacher_100k/|'; fi
  HF_RE="${HF_RE%|})"
  HF_INCLUDE_ARGS="--include $(printf "'%s' " "${HF_INCLUDE[@]}")"
  if [ -z "$HF_REPO" ]; then
    hfu=$(doc_url 'https://huggingface\.co/datasets/[A-Za-z0-9._-]+/[A-Za-z0-9._-]+' HF_DATASET_URL)
    case "$hfu" in https://huggingface.co/datasets/*/*) HF_REPO=${hfu#https://huggingface.co/datasets/} ;; esac
  fi
  mkdir -p "$HF_DIR"
  need_dl=1
  if [ -f "$HF_DIR/SHA256SUMS" ] && ( verify_hf "$HF_RE" ) >/dev/null 2>&1; then
    need_dl=0
    log "ok (cached)  Hugging Face files in $HF_DIR (checksums match)"
  fi
  if [ "$need_dl" = 1 ]; then
    HF_CLI=""
    if command -v hf >/dev/null 2>&1; then HF_CLI=hf
    elif command -v huggingface-cli >/dev/null 2>&1; then HF_CLI=huggingface-cli; fi
    if [ -z "$HF_REPO" ]; then
      log "cannot derive the Hugging Face dataset id from README.md; set HF_DATASET_REPO"
      hf_instructions; HF_STATUS=3
    elif [ -z "$HF_CLI" ]; then
      log "neither 'hf' nor 'huggingface-cli' is installed"
      hf_instructions; HF_STATUS=3
    else
      log "download     Hugging Face $HF_REPO -> $HF_DIR ($HF_CLI)"
      if ! "$HF_CLI" download "$HF_REPO" --repo-type dataset --local-dir "$HF_DIR" \
             --include "${HF_INCLUDE[@]}"; then
        log "Hugging Face download failed (gated dataset: accept the terms and log in first)"
        hf_instructions; HF_STATUS=3
      fi
    fi
  fi
  if [ "$HF_STATUS" = 0 ]; then
    [ "$need_dl" = 0 ] || verify_hf "$HF_RE"
    if [ "$WANT_TRAIN" = 1 ]; then
      mkdir -p "$TRAIN_DIR"
      ep=$(cd "$HF_DIR" && ls episodes/*.json.gz | head -n 1)
      install_json "$HF_DIR/$ep" "$(expected_sha "$HF_DIR/SHA256SUMS" "$ep")" "$TRAIN_DIR" \
        "$(basename "${ep%.gz}")"
      parts=()
      while IFS= read -r p; do parts+=("$HF_DIR/$p"); done < <(cd "$HF_DIR" && ls iin/*.tar.gz | sort)
      [ "${#parts[@]}" -gt 0 ] || die "no IIN parts in $HF_DIR/iin"
      top=$(basename "${parts[0]}"); top=${top%%.part*}; top=${top%.tar.gz}
      # One stamp for all parts: the checksum of their concatenated sha256 lines.
      psha=$(for p in "${parts[@]}"; do expected_sha "$HF_DIR/SHA256SUMS" "iin/$(basename "$p")"; done | cksum | awk '{print $1}')
      install_tar "${parts[0]}" "$psha" "$TRAIN_DIR" "$top" "${parts[@]:1}"
    fi
    if [ "$WANT_TEACHER" = 1 ]; then
      log "teacher samples: $HF_DIR/teacher_100k (schema: $HF_DIR/SCHEMA.md)"
    fi
  fi
fi

[ "$HF_STATUS" = 0 ] || log "WARNING: the Hugging Face part was not installed (see above); exit status 3"
log "done. Data root: $DATA_ROOT (mount it as sim-code/data, e.g. DATA_ROOT in sim-code/.env)."
log "HM3D scenes are not included: place them under $DATA_ROOT/scene_datasets/ (sim-code/HOW_TO_RUN.md, section 3)."
exit "$HF_STATUS"
