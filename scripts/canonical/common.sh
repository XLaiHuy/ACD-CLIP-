#!/usr/bin/env bash
set -euo pipefail

# Shared process contract for the canonical Phase2B + SABRA workflow.
# This file contains orchestration/validation only; scientific logic remains
# in train.py, select_phase2b_checkpoint.py, calibrate_sabra.py, and test.py.

CANONICAL_SHA="4aa9b465ddeb072e9218b74982306d6324c62375"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

PYTHON="${PYTHON:-/home/ai4/ENTER/envs/torchhuy/bin/python}"
CLIP_ASSET="${CLIP_ASSET:-/home/ai4/.cache/clip/ViT-L-14-336px.pt}"
VISA_ROOT="${VISA_ROOT:-/home/ai4/caohuy/data/VisA_20220922}"
RUN_ROOT="${RUN_ROOT:-/home/ai4/caohuy/acdclip_runs/canonical_sabra_v1_seed0}"
CONFIG="${CONFIG:-${REPO_ROOT}/configs/phase2b_canonical_v1.json}"
MVTEC_ROOT="${MVTEC_ROOT:-}"
MEDICAL_ROOT="${MEDICAL_ROOT:-}"
ALLOW_DIRTY_CODE="${ALLOW_DIRTY_CODE:-0}"
DRY_RUN="${DRY_RUN:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
EXPORT_FORCE="${EXPORT_FORCE:-0}"
ACDCLIP_REFERENCE_JSON="${ACDCLIP_REFERENCE_JSON:-}"
COMMON_ARGS=()

export PYTHONUNBUFFERED=1
export PYTHON CLIP_ASSET VISA_ROOT RUN_ROOT CONFIG MVTEC_ROOT MEDICAL_ROOT
export ALLOW_DIRTY_CODE DRY_RUN FORCE_RERUN RESUME_CHECKPOINT EXPORT_FORCE ACDCLIP_REFERENCE_JSON PYTHONUNBUFFERED

die() {
  printf '[canonical] ERROR: %s\n' "$*" >&2
  exit 1
}

warn() {
  printf '[canonical] WARNING: %s\n' "$*" >&2
}

banner() {
  printf '\n============================================================\n'
  printf '%s\n' "$*"
  printf '============================================================\n'
}

parse_common_args() {
  COMMON_ARGS=()
  while (($#)); do
    case "$1" in
      --run-root)
        (($# >= 2)) || die "--run-root requires a path"
        RUN_ROOT="$2"; shift 2 ;;
      --mvtec-root)
        (($# >= 2)) || die "--mvtec-root requires a path"
        MVTEC_ROOT="$2"; shift 2 ;;
      --medical-root)
        (($# >= 2)) || die "--medical-root requires a path"
        MEDICAL_ROOT="$2"; shift 2 ;;
      --visa-root)
        (($# >= 2)) || die "--visa-root requires a path"
        VISA_ROOT="$2"; shift 2 ;;
      --clip-asset)
        (($# >= 2)) || die "--clip-asset requires a path"
        CLIP_ASSET="$2"; shift 2 ;;
      --config)
        (($# >= 2)) || die "--config requires a path"
        CONFIG="$2"; shift 2 ;;
      --python)
        (($# >= 2)) || die "--python requires an executable path"
        PYTHON="$2"; shift 2 ;;
      --allow-dirty-code)
        ALLOW_DIRTY_CODE=1; shift ;;
      --dry-run)
        DRY_RUN=1; shift ;;
      --force-rerun)
        FORCE_RERUN=1; shift ;;
      --resume-checkpoint)
        (($# >= 2)) || die "--resume-checkpoint requires a path"
        RESUME_CHECKPOINT="$2"; shift 2 ;;
      --force)
        EXPORT_FORCE=1; shift ;;
      --acdclip-reference-json)
        (($# >= 2)) || die "--acdclip-reference-json requires a path"
        ACDCLIP_REFERENCE_JSON="$2"; shift 2 ;;
      --)
        shift
        while (($#)); do COMMON_ARGS+=("$1"); shift; done
        break ;;
      *)
        COMMON_ARGS+=("$1"); shift ;;
    esac
  done
  export PYTHON CLIP_ASSET VISA_ROOT RUN_ROOT CONFIG MVTEC_ROOT MEDICAL_ROOT
  export ALLOW_DIRTY_CODE DRY_RUN FORCE_RERUN RESUME_CHECKPOINT EXPORT_FORCE ACDCLIP_REFERENCE_JSON
}

check_code_sha() {
  local actual
  actual="$(git -C "$REPO_ROOT" rev-parse HEAD)" || die "not a Git worktree: $REPO_ROOT"
  if [[ "$actual" != "$CANONICAL_SHA" ]]; then
    if [[ "$ALLOW_DIRTY_CODE" == "1" ]]; then
      warn "ALLOW_DIRTY_CODE=1: HEAD=$actual, expected canonical SHA=$CANONICAL_SHA"
    else
      die "HEAD=$actual; expected canonical SHA=$CANONICAL_SHA (set ALLOW_DIRTY_CODE=1 only for an explicit audit)"
    fi
  fi
  export ACTUAL_CODE_SHA="$actual"
}

init_common() {
  parse_common_args "$@"
  check_code_sha
  cd "$REPO_ROOT"
  printf '[canonical] repo=%s\n' "$REPO_ROOT"
  printf '[canonical] code_sha=%s\n' "$ACTUAL_CODE_SHA"
  printf '[canonical] run_root=%s\n' "$RUN_ROOT"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[canonical] DRY_RUN=1 (commands will be printed, model stages will not execute)\n'
  fi
}

print_command() {
  printf 'COMMAND:'
  printf ' %q' "$@"
  printf '\n'
}

run_cmd() {
  print_command "$@"
  [[ "$DRY_RUN" == "1" ]] && return 0
  "$@"
}

run_logged() {
  local log_path="$1"
  shift
  print_command "$@"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '[canonical] DRY_RUN: would tee output to %s\n' "$log_path"
    return 0
  fi
  mkdir -p "$(dirname -- "$log_path")"
  "$@" 2>&1 | tee "$log_path"
}

require_file() {
  local path="$1"
  [[ -f "$path" ]] || die "required file not found: $path"
}

require_dir() {
  local path="$1"
  [[ -d "$path" ]] || die "required directory not found: $path"
}

require_base_assets() {
  [[ -x "$PYTHON" ]] || die "Python executable is not executable: $PYTHON"
  require_file "$CONFIG"
  require_file "$CLIP_ASSET"
  require_dir "$VISA_ROOT"
  local clip_bytes
  clip_bytes="$(stat -c '%s' "$CLIP_ASSET")"
  if ((clip_bytes <= 1024)) && head -c 128 "$CLIP_ASSET" | grep -q 'version https://git-lfs.github.com/spec/v1'; then
    die "CLIP asset is a Git LFS pointer, not a hydrated model: $CLIP_ASSET"
  fi
}

require_mvtec_root() {
  [[ -n "$MVTEC_ROOT" ]] || die "MVTEC_ROOT is required for this stage (use --mvtec-root or export MVTEC_ROOT)"
  require_dir "$MVTEC_ROOT"
}

require_medical_root() {
  [[ -n "$MEDICAL_ROOT" ]] || die "MEDICAL_ROOT is required for final zero-shot evaluation (use --medical-root or export MEDICAL_ROOT)"
  require_dir "$MEDICAL_ROOT"
}

require_clean_stage_output() {
  local marker="$1"
  local stage_dir="${2:-$(dirname -- "$marker")}"
  if [[ -e "$marker" ]]; then
    if [[ "$FORCE_RERUN" == "1" ]]; then
      warn "FORCE_RERUN=1: existing stage marker will be replaced by the canonical command: $marker"
    else
      die "stage output already exists; refusing silent overwrite: $marker (set FORCE_RERUN=1 for explicit rerun)"
    fi
  elif [[ -d "$stage_dir" ]]; then
    local first_entry
    first_entry="$(find "$stage_dir" -mindepth 1 -maxdepth 1 -print -quit)"
    if [[ -n "$first_entry" ]]; then
      die "stage directory contains incomplete output; refusing reuse: $stage_dir (handle it explicitly before rerun)"
    fi
  fi
}

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}
