#!/usr/bin/env bash
set -euo pipefail

EXPECTED_REPOSITORY='https://github.com/XLaiHuy/ACD-CLIP-.git'
EXPECTED_BRANCH='research/p26-sabra-cure-final-architecture-freeze-v1'
EXPECTED_TAG='sabra-final-p26-v1'
EXPECTED_CONFIG_SHA='a9765b4d7862e1b98e348ed8a9e3362dd179d67954cd723008eaecc136affde8'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

REMOTE_URL="$(git remote get-url origin)"
if [[ "${REMOTE_URL}" != "${EXPECTED_REPOSITORY}" ]]; then
  echo "SABRA_FINAL_RESTORE_STATUS=FAIL repository=${REMOTE_URL}" >&2
  exit 2
fi

git fetch origin "${EXPECTED_BRANCH}" --tags
if [[ "$(git branch --show-current)" != "${EXPECTED_BRANCH}" ]]; then
  if [[ -n "$(git status --short)" ]]; then
    echo 'SABRA_FINAL_RESTORE_STATUS=FAIL dirty worktree prevents checkout' >&2
    exit 3
  fi
  git checkout "${EXPECTED_BRANCH}"
fi

LOCAL_SHA="$(git rev-parse HEAD)"
REMOTE_SHA="$(git rev-parse "origin/${EXPECTED_BRANCH}")"
TAG_SHA="$(git rev-parse "refs/tags/${EXPECTED_TAG}^{commit}")"
if [[ "${LOCAL_SHA}" != "${REMOTE_SHA}" || "${LOCAL_SHA}" != "${TAG_SHA}" ]]; then
  echo "SABRA_FINAL_RESTORE_STATUS=FAIL local=${LOCAL_SHA} remote=${REMOTE_SHA} tag=${TAG_SHA}" >&2
  exit 4
fi
if [[ -n "$(git status --short)" ]]; then
  echo 'SABRA_FINAL_RESTORE_STATUS=FAIL worktree not clean' >&2
  exit 5
fi

CONFIG='research/sabra_cure/final_architecture/SABRA_FINAL_CONFIG.json'
if [[ "$(sha256sum "${CONFIG}" | awk '{print $1}')" != "${EXPECTED_CONFIG_SHA}" ]]; then
  echo 'SABRA_FINAL_RESTORE_STATUS=FAIL final config hash mismatch' >&2
  exit 6
fi

python - <<'PY'
import hashlib
import json
from pathlib import Path

root = Path.cwd()
manifest_path = root / "research/sabra_cure/final_architecture/P26_REQUIRED_ARTIFACTS.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
missing = []
mismatched = []
for record in manifest["artifacts"]:
    if not record["required"]:
        continue
    path = root / record["repository_relative_path"]
    if not path.is_file():
        missing.append(record["repository_relative_path"])
        continue
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != record["sha256"]:
        mismatched.append(record["repository_relative_path"])
if missing or mismatched:
    print("MISSING_REQUIRED_ARTIFACTS=" + json.dumps(missing))
    print("HASH_MISMATCHED_ARTIFACTS=" + json.dumps(mismatched))
    raise SystemExit(7)
print("REQUIRED_ARTIFACTS_VERIFIED=" + str(len(manifest["artifacts"])))
PY

python tools/sabra_cure/run_sabra_final.py --check-only
echo "P26_HEAD=${LOCAL_SHA}"
echo 'EXTERNAL_VALIDATION_AUTHORIZED=FALSE'
echo 'SABRA_FINAL_RESTORE_STATUS=READY'
