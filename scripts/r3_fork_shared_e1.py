#!/usr/bin/env python3
"""Create a metadata-only R3 fork of the authoritative shared E1 checkpoint.

The H2 E1 was produced at the historical full-run implementation commit.
R3 needs to preserve its exact model/optimizer/RNG state while recording the
current research-branch identity and one predeclared source-side branch
configuration.  This tool never changes checkpoint tensors.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from h2_clean.contract import canonical_json_hash, parent_scientific_config, sha256_file, state_dict_sha256


HISTORICAL_IMPLEMENTATION_SHA = "31167af5ee3dfff80b74af1e9ee0da4ecc475d2e"
EXPECTED_SOURCE_SHA = "7f9176b7ef53b572935567c574535075a573175b2aa83505d043a71d45b12b35"


def current_git_sha(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def finite_tensors(value: object) -> bool:
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all().item())
    if isinstance(value, dict):
        return all(finite_tensors(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_tensors(item) for item in value)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hybrid-alpha-max", type=float, default=None)
    parser.add_argument("--anchor-lambda", type=float, default=None)
    parser.add_argument("--anchor-family-budget", type=float, default=None)
    parser.add_argument("--enable-anchor", action="store_true")
    args = parser.parse_args()

    source_sha = sha256_file(args.source)
    if source_sha != EXPECTED_SOURCE_SHA:
        raise SystemExit(f"unexpected shared E1 SHA256: {source_sha}")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing fork: {args.output}")

    source = torch.load(args.source, map_location="cpu", weights_only=False)
    if int(source.get("epoch", -1)) != 1:
        raise SystemExit(f"source is not E1: epoch={source.get('epoch')!r}")
    if source.get("implementation_git_sha") != HISTORICAL_IMPLEMENTATION_SHA:
        raise SystemExit("source implementation fingerprint is not the audited H2 full-run SHA")
    if not finite_tensors(source):
        raise SystemExit("source checkpoint contains non-finite tensors")

    payload = copy.deepcopy(source)
    config = dict(payload["resolved_scientific_config"])
    overrides: dict[str, object] = {}
    if args.hybrid_alpha_max is not None:
        config["hybrid_alpha_max"] = float(args.hybrid_alpha_max)
        overrides["hybrid_alpha_max"] = float(args.hybrid_alpha_max)
    if args.anchor_lambda is not None:
        config["anchor_lambda"] = float(args.anchor_lambda)
        overrides["anchor_lambda"] = float(args.anchor_lambda)
    if args.anchor_family_budget is not None:
        config["anchor_family_budget"] = float(args.anchor_family_budget)
        overrides["anchor_family_budget"] = float(args.anchor_family_budget)
    if args.enable_anchor:
        config["use_safe_anchor"] = True
        config["anchor_gradient_budget"] = True
        config["anchor_reference_sha256"] = source_sha
        overrides.update({
            "use_safe_anchor": True,
            "anchor_gradient_budget": True,
            "anchor_reference_sha256": source_sha,
        })

    repo = Path(__file__).resolve().parents[1]
    branch_sha = current_git_sha(repo)
    config["implementation_git_sha"] = branch_sha
    config["working_tree_diff_sha256"] = None
    payload["resolved_scientific_config"] = config
    payload["parent_scientific_config"] = parent_scientific_config(config)
    payload["config_sha256"] = canonical_json_hash(config)
    payload["git_sha"] = branch_sha
    payload["implementation_git_sha"] = branch_sha
    payload["working_tree_diff_sha256"] = None
    payload["r3_fork"] = {
        "type": "metadata_only_shared_e1_fork",
        "source_path": str(args.source),
        "source_sha256": source_sha,
        "source_implementation_git_sha": HISTORICAL_IMPLEMENTATION_SHA,
        "source_image_state_sha256": state_dict_sha256(source["image_adapter"]),
        "overrides": overrides,
    }
    args.output.parent.mkdir(parents=True, exist_ok=False)
    torch.save(payload, args.output)
    print(json.dumps({
        "output": str(args.output),
        "output_sha256": sha256_file(args.output),
        "source_sha256": source_sha,
        "git_sha": branch_sha,
        "overrides": overrides,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
