#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "$SCRIPT_DIR/common.sh"

init_common "$@"
((${#COMMON_ARGS[@]} == 0)) || die "unknown preflight arguments: ${COMMON_ARGS[*]}"

banner "STAGE 0 — CANONICAL PREFLIGHT"
require_base_assets

if [[ -n "$MVTEC_ROOT" ]]; then
  require_mvtec_root
  mvtec_cmd=("$PYTHON" "$REPO_ROOT/select_phase2b_checkpoint.py" --mvtec-root "$MVTEC_ROOT" --preflight-only)
  # This is the existing read-only MVTec adapter preflight, not evaluation.
  print_command "${mvtec_cmd[@]}"
  "${mvtec_cmd[@]}"
fi

if [[ -n "$MEDICAL_ROOT" ]]; then
  # Deliberately validate only the directory.  No Medical samples or metadata
  # are opened and no Medical identity is written to preflight.json.
  require_medical_root
fi

manifest="$RUN_ROOT/manifests/preflight.json"
require_clean_stage_output "$manifest"
if [[ "$DRY_RUN" == "1" ]]; then
  printf '[canonical] DRY_RUN: would write %s\n' "$manifest"
else
  mkdir -p "$(dirname -- "$manifest")"
fi
"$PYTHON" - "$manifest" "$CANONICAL_SHA" "$ACTUAL_CODE_SHA" "$PYTHON" "$CLIP_ASSET" "$VISA_ROOT" "$MVTEC_ROOT" "$CONFIG" "$DRY_RUN" <<'PY'
import hashlib
import json
import platform
import sys
from pathlib import Path


from model.phase2b_runtime import configure_canonical_fp32

configure_canonical_fp32()
import torch


manifest_path = Path(sys.argv[1])
git_sha = sys.argv[2]
actual_git_sha = sys.argv[3]
dry_run = sys.argv[9] == "1"
python_path = sys.argv[4]
clip_path = Path(sys.argv[5]).expanduser().resolve()
visa_path = Path(sys.argv[6]).expanduser().resolve()
mvtec_raw = sys.argv[7]
config_path = Path(sys.argv[8]).expanduser().resolve()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path_identity(path: Path) -> dict[str, object]:
    resolved = str(path.resolve())
    return {
        "path": resolved,
        "exists": path.exists(),
        "path_sha256": hashlib.sha256(resolved.encode("utf-8")).hexdigest(),
    }


config = json.loads(config_path.read_text(encoding="utf-8"))
required = {
    "model_name": "ViT-L-14-336",
    "img_size": 518,
    "n_groups": 3,
    "precision": "fp32",
    "dfg_mode": "attn",
    "dfg_attn_dim": 256,
    "dfg_attn_tau": 8.0,
    "use_ss2d_dfg": True,
    "dfg_ss2d_fusion": "weight_residual",
    "dfg_beta_target": 0.1,
    "dfg_beta_schedule": "warmup010",
    "hybrid_alpha_max": 0.2,
    "soft_prompt_freeze_epochs": 3,
    "lambda_kg": 0.001,
    "lambda_k": 0,
    "candidate_epochs": [10, 12, 14, 16, 18, 20],
}
for key, expected in required.items():
    if config.get(key) != expected:
        raise SystemExit(f"canonical config mismatch for {key}: {config.get(key)!r} != {expected!r}")
if any("h6" in str(key).lower() for key in config):
    raise SystemExit("canonical config contains an H6 field")

cuda_available = bool(torch.cuda.is_available())
if not cuda_available:
    raise SystemExit("canonical preflight requires CUDA, but torch.cuda.is_available() is false")
if bool(torch.backends.cuda.matmul.allow_tf32) or bool(torch.backends.cudnn.allow_tf32):
    raise SystemExit("canonical preflight requires TF32 disabled")
gpu = torch.cuda.get_device_name(0) if cuda_available else None
vram = (torch.cuda.get_device_properties(0).total_memory / 1024**3) if cuda_available else None
payload = {
    "git_sha": git_sha,
    "actual_git_sha": actual_git_sha,
    "python": python_path,
    "python_version": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "cuda_available": cuda_available,
    "gpu": gpu,
    "vram_gib": vram,
    "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
    "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
    "clip_asset": {
        "path": str(clip_path),
        "bytes": clip_path.stat().st_size,
        "sha256": sha256_file(clip_path),
    },
    "visa": path_identity(visa_path),
    "mvtec": path_identity(Path(mvtec_raw).expanduser()) if mvtec_raw else None,
    "config": {"path": str(config_path), "sha256": sha256_file(config_path)},
}
if not dry_run:
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

printf 'CANONICAL_PREFLIGHT=PASS\n'
