#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "${SCRIPT_DIR}/../.." rev-parse --show-toplevel)"
SOURCE="${1:-V}"
NUM_WORKERS="${NUM_WORKERS:-2}"
case "${SOURCE}" in
  V|v)
    SOURCE=V
    LAUNCHER="${SCRIPT_DIR}/run_fresh_visa_source_v2.sh"
    SESSION="${TMUX_SESSION:-rental_20260911_v}"
    ;;
  M|m)
    SOURCE=M
    LAUNCHER="${SCRIPT_DIR}/run_fresh_mvtec_source_v2.sh"
    SESSION="${TMUX_SESSION:-rental_20260911_m}"
    ;;
  *)
    echo "usage: $0 V|M" >&2
    exit 2
    ;;
esac
command -v tmux >/dev/null
if pgrep -f '[t]rain.py' >/dev/null; then
  echo "refusing to start while another train.py process is active" >&2
  exit 2
fi
if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "refusing to reuse existing tmux session: ${SESSION}" >&2
  exit 2
fi
worker_assignment="NUM_WORKERS=$(printf '%q' "${NUM_WORKERS}")"
launcher_command="$(printf '%q' "${LAUNCHER}")"
tmux new-session -d -s "${SESSION}" -c "${ROOT}" \
  "${worker_assignment} bash ${launcher_command}"
printf 'started_session=%s source=%s launcher=%s\n' "${SESSION}" "${SOURCE}" "${LAUNCHER}"
