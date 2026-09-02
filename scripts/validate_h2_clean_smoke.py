#!/usr/bin/env python3
"""Validate bounded H/A/C/AC smoke and deterministic resume repeat."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from h2_clean.contract import RESUME_BRANCH_KEYS, state_dict_sha256


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def identities(path, epoch):
    values = []
    for line in (path / "train.log").read_text().splitlines():
        if f"batch_identity epoch={epoch} " in line:
            values.append(line.split("batch_identity ", 1)[1])
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--expected-batches", type=int, required=True)
    args = parser.parse_args()
    root = Path(args.root)
    shared = load(root / "shared_e1" / "adapter_1.pth")
    names = ("H", "A", "C", "AC")
    arms = {name: load(root / name / "adapter_2.pth") for name in names}
    configs = {name: arms[name]["resolved_scientific_config"] for name in names}
    parents = {name: arms[name]["parent_scientific_config"] for name in names}
    if any(parents[name] != parents["H"] for name in names):
        raise AssertionError("H/A/C/AC parent scientific configs differ")

    expected = {
        "H": {"use_safe_anchor": False, "anchor_lambda": 0.0, "anchor_reference_sha256": None, "use_cir_training": False, "cir_alpha": 0.0, "cir_peer_count": 8, "cir_spatial_radius": 3},
        "A": {"use_safe_anchor": True, "anchor_lambda": 0.001, "use_cir_training": False, "cir_alpha": 0.0, "cir_peer_count": 8, "cir_spatial_radius": 3},
        "C": {"use_safe_anchor": False, "anchor_lambda": 0.0, "use_cir_training": True, "cir_alpha": 0.5, "cir_peer_count": 8, "cir_spatial_radius": 3},
        "AC": {"use_safe_anchor": True, "anchor_lambda": 0.001, "use_cir_training": True, "cir_alpha": 0.5, "cir_peer_count": 8, "cir_spatial_radius": 3},
    }
    non_branch = sorted(set(configs["H"]) - set(RESUME_BRANCH_KEYS))
    for name in names:
        if any(configs[name][key] != configs["H"][key] for key in non_branch):
            raise AssertionError(f"{name} differs from H outside intervention branch keys")
        for key, value in expected[name].items():
            if key == "anchor_reference_sha256" and name in ("A", "AC"):
                if not configs[name].get(key):
                    raise AssertionError(f"{name} has no anchor reference SHA")
            elif configs[name].get(key) != value:
                raise AssertionError(f"{name} {key}={configs[name].get(key)!r}, expected {value!r}")
    h, a, c, ac = (configs[name] for name in names)
    cir_keys = ("use_cir_training", "cir_alpha", "cir_peer_count", "cir_spatial_radius")
    anchor_keys = ("use_safe_anchor", "anchor_lambda", "anchor_reference_sha256")
    if any(a[key] != h[key] for key in cir_keys):
        raise AssertionError("H versus A is not anchor-only")
    if any(c[key] != h[key] for key in anchor_keys):
        raise AssertionError("H versus C is not CIR-only")
    if any(ac[key] != c[key] for key in cir_keys):
        raise AssertionError("C versus AC CIR branch changed unexpectedly")
    if any(ac[key] != a[key] for key in anchor_keys):
        raise AssertionError("A versus AC anchor branch changed unexpectedly")

    arm_ids = {name: identities(root / name, 2) for name in names}
    if any(len(values) != args.expected_batches for values in arm_ids.values()):
        raise AssertionError(f"unexpected E2 batch identity count: {arm_ids}")
    if any(values != arm_ids["H"] for values in arm_ids.values()):
        raise AssertionError("H/A/C/AC sample paths or augmentation identities differ")

    resume_names = ("H_resume_1", "H_resume_2")
    resumes = {}
    for name in resume_names:
        checkpoint = root / name / "adapter_3.pth"
        if not checkpoint.is_file():
            raise AssertionError(f"missing {checkpoint}")
        resumes[name] = load(checkpoint)
    resume_ids = {name: identities(root / name, 3) for name in resume_names}
    if any(len(values) != args.expected_batches for values in resume_ids.values()):
        raise AssertionError(f"unexpected resumed batch identity count: {resume_ids}")
    if resume_ids["H_resume_1"] != resume_ids["H_resume_2"]:
        raise AssertionError("repeated resume next-batch identities differ")
    state_hashes = {
        name: state_dict_sha256(resumes[name]["model_state"]["image_adapter"])
        for name in resume_names
    }
    if len(set(state_hashes.values())) != 1:
        raise AssertionError(f"repeated resume image-adapter states differ: {state_hashes}")
    if shared["resolved_operational_config"].get("save_path") == arms["H"]["resolved_operational_config"].get("save_path"):
        raise AssertionError("operational save paths were not separated")

    result = {
        "shared_e1": True,
        "arms": names,
        "same_e2_batch_identity": True,
        "resume_repeat_same_e3_batch_identity": True,
        "resume_repeat_same_image_adapter_state": True,
        "expected_batches": args.expected_batches,
        "e2_batch_identity": arm_ids["H"],
        "e3_batch_identity": resume_ids["H_resume_1"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
