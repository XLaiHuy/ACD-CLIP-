"""Static and optional runtime audits for the frozen P27 execution base."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from tools.sabra.data import EXPECTED_VISA_CLASSES, read_visa_metadata
from tools.sabra_v2.data_protocol import loco_inventory


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = ROOT / "research/sabra_v2/region_distill/P27_PROTOCOL.json"


def _expect(value: Any, expected: Any, label: str) -> None:
    if value != expected:
        raise RuntimeError(f"{label} drift: expected {expected!r}, got {value!r}")


def audit_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Reject any frozen P27 V1 scientific semantic drift."""
    _expect(protocol.get("schema_version"), "P27_REGION_DISTILL_V1", "schema_version")
    geometry = protocol.get("geometry", {})
    _expect(geometry.get("patch_grid"), [37, 37], "patch_grid")
    _expect(geometry.get("region_grid"), [9, 9], "region_grid")
    _expect(geometry.get("visual_dim"), 768, "visual_dim")
    _expect(geometry.get("projection_dim"), 64, "projection_dim")
    _expect(geometry.get("stages"), 3, "stages")
    teacher = protocol.get("teacher", {})
    _expect(teacher.get("historical_alpha"), 0.25, "historical_alpha")
    _expect(teacher.get("r0_margin_scale"), 19.840438842773438, "r0_margin_scale")
    _expect(teacher.get("headroom"), "NOT_CHECKED_HISTORICAL_CACHE_CHECKPOINT_INCOMPATIBLE", "teacher_headroom")
    residual = protocol.get("residual", {})
    _expect(residual.get("normal_scale"), -0.5, "normal_scale")
    _expect(residual.get("anomaly_scale"), 0.5, "anomaly_scale")
    _expect(residual.get("application"), "before_unchanged_phase2b_deployment", "residual_application")
    losses = protocol.get("losses", {})
    _expect(losses.get("distillation"), "SmoothL1", "distillation_loss")
    _expect(losses.get("localization"), "canonical_focal_dice", "localization_loss")
    _expect(losses.get("distillation_weight"), 1.0, "distillation_weight")
    _expect(losses.get("localization_weight"), 1.0, "localization_weight")
    training = protocol.get("training", {})
    _expect(training.get("protocol"), "12_class_LOCO", "training_protocol")
    _expect(training.get("trainable"), ["p27_region_adapter"], "trainable_ownership")
    firewall = protocol.get("firewall", {})
    for key, expected in (("mvtec_opened", False), ("mvtec_data_reads", 0), ("medical_reads", 0), ("full_scientific_training_runs", 0)):
        _expect(firewall.get(key), expected, f"firewall.{key}")
    return {
        "status": "PASS",
        "REGION_TEACHER_HEADROOM": "NOT_CHECKED",
        "teacher_source": "P26-native source GT utility only",
        "student_inference": "GT-free",
        "checks": ["frozen_design", "teacher_provenance", "losses", "firewall"],
    }


def write_audit_report(report: Mapping[str, Any], output: Path) -> tuple[Path, Path]:
    """Write paired machine-readable and concise human-readable audit reports."""
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "P27_AUDIT.json"
    markdown_path = output / "P27_AUDIT.md"
    json_path.write_text(json.dumps(dict(report), indent=2, sort_keys=True) + "\n")
    lines = ["# P27 Audit", "", f"Status: `{report['status']}`", "", f"REGION_TEACHER_HEADROOM: `{report['REGION_TEACHER_HEADROOM']}`", ""]
    for key, value in report.items():
        if key not in {"status", "REGION_TEACHER_HEADROOM"}:
            lines.append(f"- {key}: `{value}`")
    markdown_path.write_text("\n".join(lines) + "\n")
    return json_path, markdown_path


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--held-class", choices=EXPECTED_VISA_CLASSES, required=True)
    parser.add_argument("--metadata", type=Path, default=ROOT / "dataset/hub/VisA.jsonl")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    protocol = json.loads(PROTOCOL_PATH.read_text())
    report = audit_protocol(protocol)
    inventory = loco_inventory(read_visa_metadata(args.metadata), args.held_class)
    report.update({"held_class": args.held_class, "fit_records": len(inventory.fit_rows), "held_records": len(inventory.held_rows), "loco_firewall": "PASS"})
    json_path, markdown_path = write_audit_report(report, args.output)
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path), "status": report["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
