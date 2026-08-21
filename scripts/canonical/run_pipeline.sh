#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

stage="${1:-}"
if [[ -z "$stage" || "$stage" == "--help" || "$stage" == "-h" ]]; then
  cat <<'USAGE'
Usage: run_pipeline.sh {preflight|train|select|fit-sabra|lambda|medical|export|all} [common options]

Common options: --run-root PATH --mvtec-root PATH --medical-root PATH
  --allow-dirty-code --dry-run --force-rerun --resume-checkpoint PATH
  --force --acdclip-reference-json PATH
USAGE
  [[ -n "$stage" ]] && exit 0
  exit 2
fi
shift

case "$stage" in
  preflight|train|select|fit-sabra|lambda|medical|export|all) ;;
  *) die "unknown pipeline stage: $stage" ;;
esac

if init_common "$@"; then
  :
else
  init_status=$?
  init_stage="$stage"
  [[ "$init_stage" == "all" ]] && init_stage="preflight"
  printf 'PIPELINE_FAILED_STAGE=%s\n' "$init_stage" >&2
  exit "$init_status"
fi
if ((${#COMMON_ARGS[@]} == 0)); then
  :
else
  printf '[canonical] ERROR: unknown pipeline arguments: %s\n' "${COMMON_ARGS[*]}" >&2
  printf 'PIPELINE_FAILED_STAGE=%s\n' "$stage" >&2
  exit 2
fi

run_stage() {
  local stage_name="$1"
  local script_name="$2"
  shift 2
  if "$SCRIPT_DIR/$script_name" "$@"; then
    return 0
  else
    local stage_status=$?
  fi
  printf 'PIPELINE_FAILED_STAGE=%s\n' "$stage_name" >&2
  return "$stage_status"
}

run_export() {
  local -a export_cmd=("$PYTHON" "$SCRIPT_DIR/60_export_results.py" --run-root "$RUN_ROOT" --code-sha "$SCIENTIFIC_CODE_SHA")
  if [[ -n "$ACDCLIP_REFERENCE_JSON" ]]; then
    export_cmd+=(--acdclip-reference-json "$ACDCLIP_REFERENCE_JSON")
  fi
  if [[ "$EXPORT_FORCE" == "1" ]]; then
    export_cmd+=(--force)
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    export_cmd+=(--dry-run)
    # Export validation is artifact-only and safe to run in DRY_RUN; it writes
    # nothing and confirms every completed input is present.
    print_command "${export_cmd[@]}"
    if "${export_cmd[@]}"; then
      return 0
    else
      return $?
    fi
  else
    if run_cmd "${export_cmd[@]}"; then
      return 0
    else
      return $?
    fi
  fi
}

run_function_stage() {
  local stage_name="$1"
  shift
  if "$@"; then
    return 0
  else
    local stage_status=$?
  fi
  printf 'PIPELINE_FAILED_STAGE=%s\n' "$stage_name" >&2
  return "$stage_status"
}

case "$stage" in
  preflight)
    run_stage preflight 00_preflight.sh ;;
  train)
    run_stage train 10_train_phase2b.sh ;;
  select)
    run_stage select 20_select_phase2b.sh ;;
  fit-sabra)
    run_stage fit-sabra 30_fit_sabra_source.sh ;;
  lambda)
    run_stage lambda 40_select_lambda.sh ;;
  medical)
    run_stage medical 50_eval_medical.sh ;;
  export)
    run_function_stage export run_export ;;
  all)
    run_stage preflight 00_preflight.sh
    run_stage train 10_train_phase2b.sh
    run_stage select 20_select_phase2b.sh
    run_stage fit-sabra 30_fit_sabra_source.sh
    run_stage lambda 40_select_lambda.sh
    printf '[canonical] WARNING: entering final zero-shot Medical stage only after frozen SABRA guards passed\n'
    run_stage medical 50_eval_medical.sh
    run_function_stage export run_export ;;
esac

printf 'PIPELINE_STATUS=PASS STAGE=%s\n' "$stage"
