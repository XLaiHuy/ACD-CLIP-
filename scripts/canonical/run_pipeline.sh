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

init_common "$@"
((${#COMMON_ARGS[@]} == 0)) || die "unknown pipeline arguments: ${COMMON_ARGS[*]}"

run_stage() {
  local script_name="$1"
  shift
  "$SCRIPT_DIR/$script_name" "$@"
}

run_export() {
  export_cmd=("$PYTHON" "$SCRIPT_DIR/60_export_results.py" --run-root "$RUN_ROOT" --code-sha "$CANONICAL_SHA")
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
    "${export_cmd[@]}"
  else
    run_cmd "${export_cmd[@]}"
  fi
}

case "$stage" in
  preflight)
    run_stage 00_preflight.sh ;;
  train)
    run_stage 10_train_phase2b.sh ;;
  select)
    run_stage 20_select_phase2b.sh ;;
  fit-sabra)
    run_stage 30_fit_sabra_source.sh ;;
  lambda)
    run_stage 40_select_lambda.sh ;;
  medical)
    run_stage 50_eval_medical.sh ;;
  export)
    run_export ;;
  all)
    run_stage 00_preflight.sh
    run_stage 10_train_phase2b.sh
    run_stage 20_select_phase2b.sh
    run_stage 30_fit_sabra_source.sh
    run_stage 40_select_lambda.sh
    printf '[canonical] WARNING: entering final zero-shot Medical stage only after frozen SABRA guards passed\n'
    run_stage 50_eval_medical.sh
    run_export ;;
esac

printf 'PIPELINE_STATUS=PASS STAGE=%s\n' "$stage"
