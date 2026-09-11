#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(git -C "${SCRIPT_DIR}/../.." rev-parse --show-toplevel)"
RAW_MEDVISA="${RAW_MEDVISA:-/data/raw/medvisa}"
RAW_MVTEC="${RAW_MVTEC:-/data/raw/mvtec}"

find_candidate() {
  local candidate
  for candidate in "$@"; do
    if [[ -d "${candidate}" ]] && find "${candidate}" -type f -print -quit | grep -q .; then
      readlink -f "${candidate}"
      return 0
    fi
  done
  return 1
}
link_if_resolved() {
  local destination="$1" source="$2"
  if [[ -e "${destination}" && ! -L "${destination}" ]]; then
    echo "Refusing to replace existing non-symlink: ${destination}" >&2
    return 2
  fi
  ln -sfn "${source}" "${destination}"
  echo "linked ${destination} -> ${source}"
}
mkdir -p "${ROOT}/data"
visa="$(find_candidate "${RAW_MEDVISA}/VisA_20220922" "${RAW_MEDVISA}/data/VisA_20220922" "${RAW_MEDVISA}/VisA" 2>/dev/null || true)"
mvtec="$(find_candidate "${RAW_MVTEC}/mvtec-ad" "${RAW_MVTEC}/MVTec-AD" "${RAW_MVTEC}" 2>/dev/null || true)"
medad="$(find_candidate "${RAW_MEDVISA}/MedAD" "${RAW_MEDVISA}/data/MedAD" 2>/dev/null || true)"
colon="$(find_candidate "${RAW_MEDVISA}/Colon" "${RAW_MEDVISA}/data/Colon" 2>/dev/null || true)"
for pair in "data/VisA_20220922|${visa}" "data/mvtec_ad|${mvtec}" "data/MedAD|${medad}" "data/Colon|${colon}"; do
  destination="${ROOT}/${pair%%|*}"; source="${pair#*|}"
  if [[ -z "${source}" ]]; then echo "Missing extracted source for ${destination}; no link created" >&2; continue; fi
  link_if_resolved "${destination}" "${source}"
done
if [[ -z "${visa}" || -z "${mvtec}" || -z "${medad}" || -z "${colon}" ]]; then
  echo "DATA_LINK_SETUP_INCOMPLETE" >&2
  exit 1
fi
echo "DATA_LINK_SETUP_COMPLETE"
