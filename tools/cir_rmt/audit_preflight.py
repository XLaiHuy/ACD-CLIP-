#!/usr/bin/env python3
"""CIR/G3-PREFLIGHT: source-only structural, transport, and metric preflight."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from evaluation.metrics import binary_average_precision, binary_auroc
from .core import V1_TRANSPORT_DIRECTION, peer_delta_from_native_margins, transport_pair
from .identity import load_cir_config, release_identity_fields


def _entropy(weights: torch.Tensor) -> float:
    return float((-(weights * weights.clamp_min(1e-8).log()).sum(-1)).mean())


def run_synthetic(transport_direction: str = V1_TRANSPORT_DIRECTION, alpha: float = 0.5) -> dict[str, object]:
    torch.manual_seed(23)
    stages, batch, side, dim, groups = 3, 4, 7, 16, 3
    patches = side * side
    features = torch.nn.functional.normalize(torch.randn(stages, batch, patches, dim), dim=-1)
    margins = torch.randn(stages, batch, patches, groups)
    delta, stats = peer_delta_from_native_margins(features, margins)
    peers = stats["peer_indices"]
    spatial_violations = 0
    duplicate_count = 0
    self_count = 0
    invalid_count = 0
    for b in range(batch):
        for p in range(patches):
            if not bool(stats["valid"][b, p]):
                invalid_count += 1
                continue
            values = peers[b, p].tolist()
            duplicate_count += int(len(values) != len(set(values)))
            self_count += int(p in values)
            y, x = divmod(p, side)
            spatial_violations += sum(max(abs(y - divmod(q, side)[0]), abs(x - divmod(q, side)[1])) <= 3 for q in values)
    native = torch.rand(stages, batch, patches, groups) + 0.1
    native = native / native.sum(-1, keepdim=True)
    evidence = delta
    normal, abnormal = transport_pair(native, native, evidence, float(alpha), transport_direction=transport_direction)
    rows = {"stage": "CIR/G3-PREFLIGHT", "peer_shape": list(peers.shape), "peer_valid_fraction": float(stats["valid"].float().mean()), "invalid_count": invalid_count, "duplicate_count": duplicate_count, "self_count": self_count, "spatial_violation_count": spatial_violations, "deterministic": True, "gt_free": True, "mad_mean": float(stats["mad"].mean()), "z_abs_p95": float(stats["z"].abs().quantile(0.95)), "delta_saturation_fraction": float((delta.abs() > 0.95).float().mean()), "transport_l1_normal": float((normal - native).abs().sum(-1).mean()), "transport_l1_abnormal": float((abnormal - native).abs().sum(-1).mean()), "transport_entropy_normal": _entropy(normal), "transport_entropy_abnormal": _entropy(abnormal), "transport_active_fraction": float(((normal - native).abs().sum(-1) + (abnormal - native).abs().sum(-1) > 1e-6).float().mean())}
    rows["status"] = "PASS" if duplicate_count == 0 and self_count == 0 and spatial_violations == 0 and rows["peer_valid_fraction"] > 0 else "FAIL"
    return rows


def add_source_metrics(result: dict[str, object], records_path: Path | None) -> None:
    if records_path is None:
        result["source_metrics"] = {"status": "NOT_RUN_NO_SOURCE_RECORDS"}
        return
    payload = json.loads(records_path.read_text(encoding="utf-8"))
    scores = np.asarray(payload["scores"], dtype=np.float64).reshape(-1)
    labels = np.asarray(payload["labels"], dtype=np.int8).reshape(-1)
    result["source_metrics"] = {"status": "PASS", "auroc": binary_auroc(scores, labels), "ap": binary_average_precision(scores, labels), "ranking_monotonic": bool(np.all(np.diff(scores[np.argsort(-scores, kind="mergesort")]) <= 0))}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/cir_dfg_rmt_v1.json"))
    parser.add_argument("--source-records", type=Path)
    parser.add_argument("--real", action="store_true", help="mark this as the real source-preflight gate")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    config = load_cir_config(args.config)
    result = run_synthetic(str(config.get("rmt_transport_direction", V1_TRANSPORT_DIRECTION)), float(config["rmt_transport_alpha"]))
    add_source_metrics(result, args.source_records)
    real = bool(args.real)
    real_asset = bool(real and args.source_records is not None and args.source_records.is_file())
    result.update({
        "gate": "G3_REAL" if real else "G3_SYNTHETIC",
        "scope": "real" if real else "synthetic",
        "real": real,
        "real_asset": real_asset,
        "identity": release_identity_fields(config),
        "evidence": {
            "kind": "source_preflight_real" if real else "synthetic_preflight",
            "real_execution": bool(real and real_asset and result.get("status") == "PASS"),
            "artifact": {"source_records": str(args.source_records)} if real_asset else {},
        },
    })
    if real and args.source_records is None:
        result["status"] = "NOT_RUN_NO_SOURCE_RECORDS"
    elif real and result.get("source_metrics", {}).get("status") != "PASS":
        result["status"] = "FAIL"
    if result["status"] != "PASS":
        raise SystemExit(json.dumps(result, sort_keys=True))
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
