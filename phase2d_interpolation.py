"""CPU-only Phase2D A-prime/B checkpoint interpolation utilities."""
import argparse
import copy
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import torch


STATE_DICT_KEYS = ("text_adapter", "image_adapter", "soft_prompt")
INVALIDATED_TRAINING_KEYS = ("optimizer", "optimizer_state", "scheduler", "scheduler_state", "scaler", "scaler_state")
LOCKED_CANDIDATES = (("AB25", 0.25), ("AB50", 0.50), ("AB75", 0.75))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_payload(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def _require_mapping(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a dict, got {type(value).__name__}")


def validate_compatible_payloads(payload_a, payload_b, state_dict_keys=STATE_DICT_KEYS):
    """Validate the interpolation contract and return its tensor summary."""
    _require_mapping(payload_a, "A-prime payload")
    _require_mapping(payload_b, "B payload")
    if payload_a.keys() != payload_b.keys():
        raise ValueError("checkpoint payload schemas differ")
    floating_count = 0
    non_floating_count = 0
    dtypes = set()
    for group in state_dict_keys:
        if group not in payload_a:
            raise ValueError(f"missing model state group: {group}")
        state_a = payload_a[group]
        state_b = payload_b[group]
        _require_mapping(state_a, f"A-prime {group}")
        _require_mapping(state_b, f"B {group}")
        if state_a.keys() != state_b.keys():
            raise ValueError(f"state keys differ for {group}")
        for key, tensor_a in state_a.items():
            tensor_b = state_b[key]
            name = f"{group}.{key}"
            if not torch.is_tensor(tensor_a) or not torch.is_tensor(tensor_b):
                raise ValueError(f"state value is not a tensor: {name}")
            if tensor_a.shape != tensor_b.shape:
                raise ValueError(f"state shapes differ for {name}")
            if tensor_a.is_floating_point() != tensor_b.is_floating_point():
                raise ValueError(f"floating/non-floating mismatch for {name}")
            if tensor_a.is_floating_point():
                floating_count += 1
                dtypes.add(str(tensor_a.dtype))
            else:
                non_floating_count += 1
                if not torch.equal(tensor_a, tensor_b):
                    raise ValueError(f"non-floating tensors differ for {name}")
    return {
        "state_dict_keys": list(state_dict_keys),
        "floating_tensor_count": floating_count,
        "non_floating_tensor_count": non_floating_count,
        "floating_dtypes": sorted(dtypes),
    }


def interpolate_payload(
    payload_a,
    payload_b,
    lambda_b,
    candidate_name,
    parent_a_path,
    parent_a_sha256,
    parent_b_path,
    parent_b_sha256,
    creation_commit,
    created_at=None,
):
    """Return a non-resumable evaluation payload interpolated on CPU in FP32."""
    if not 0.0 <= float(lambda_b) <= 1.0:
        raise ValueError("lambda_b must be in [0, 1]")
    summary = validate_compatible_payloads(payload_a, payload_b)
    candidate = copy.deepcopy(payload_a)
    for training_key in INVALIDATED_TRAINING_KEYS:
        candidate.pop(training_key, None)
    for group in STATE_DICT_KEYS:
        for key, tensor_a in payload_a[group].items():
            tensor_b = payload_b[group][key]
            if tensor_a.is_floating_point():
                value = (1.0 - float(lambda_b)) * tensor_a.float() + float(lambda_b) * tensor_b.float()
                candidate[group][key] = value.to(dtype=tensor_a.dtype)
            else:
                candidate[group][key] = tensor_a.clone()
    candidate["phase2d_interpolation"] = {
        "experiment": "Phase2D_AB_interpolation",
        "parent_a_path": str(parent_a_path),
        "parent_a_sha256": parent_a_sha256,
        "parent_b_path": str(parent_b_path),
        "parent_b_sha256": parent_b_sha256,
        "lambda_b": float(lambda_b),
        "candidate_name": candidate_name,
        "training_performed": False,
        "optimizer_state_interpolated": False,
        "creation_commit": creation_commit,
        "creation_timestamp": created_at or datetime.now(timezone.utc).isoformat(),
        **summary,
    }
    return candidate


def _creation_commit():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def parse_args():
    parser = argparse.ArgumentParser(description="Create locked Phase2D interpolation candidates on CPU")
    parser.add_argument("--a-checkpoint", required=True)
    parser.add_argument("--b-checkpoint", required=True)
    parser.add_argument("--a-sha256", required=True)
    parser.add_argument("--b-sha256", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--creation-commit", default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    a_path = Path(args.a_checkpoint)
    b_path = Path(args.b_checkpoint)
    if sha256_file(a_path) != args.a_sha256:
        raise RuntimeError("A-prime checkpoint SHA-256 mismatch")
    if sha256_file(b_path) != args.b_sha256:
        raise RuntimeError("B checkpoint SHA-256 mismatch")
    payload_a = load_payload(a_path)
    payload_b = load_payload(b_path)
    summary = validate_compatible_payloads(payload_a, payload_b)
    if args.dry_run:
        print(json.dumps({"status": "preflight_pass", **summary}, sort_keys=True))
        return
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    commit = args.creation_commit or _creation_commit()
    manifest = {"parents": {"A_prime": {"path": str(a_path), "sha256": args.a_sha256}, "B": {"path": str(b_path), "sha256": args.b_sha256}}, "candidates": [], **summary}
    for name, lambda_b in LOCKED_CANDIDATES:
        payload = interpolate_payload(
            payload_a, payload_b, lambda_b, name, a_path, args.a_sha256, b_path,
            args.b_sha256, commit,
        )
        filename = f"{name}_lambdaB{lambda_b:.2f}".replace(".", "p") + ".pth"
        path = output_dir / filename
        torch.save(payload, path)
        manifest["candidates"].append({"name": name, "lambda_b": lambda_b, "path": str(path), "sha256": sha256_file(path)})
    (output_dir.parent / "interpolation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
