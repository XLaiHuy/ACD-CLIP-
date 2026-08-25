from __future__ import annotations

from pathlib import Path

import pytest

from tools.sabra_v2.audit_region_distill import audit_protocol, write_audit_report
from tools.sabra_v2.evaluate_region_distill import make_parser as make_evaluate_parser
from tools.sabra_v2.train_region_distill import make_parser as make_train_parser


def test_audit_protocol_accepts_only_frozen_p27_v1_contract() -> None:
    protocol = {
        "schema_version": "P27_REGION_DISTILL_V1",
        "geometry": {"patch_grid": [37, 37], "region_grid": [9, 9], "visual_dim": 768, "projection_dim": 64, "stages": 3},
        "teacher": {"historical_alpha": 0.25, "r0_margin_scale": 19.840438842773438, "headroom": "NOT_CHECKED_HISTORICAL_CACHE_CHECKPOINT_INCOMPATIBLE"},
        "residual": {"normal_scale": -0.5, "anomaly_scale": 0.5, "application": "before_unchanged_phase2b_deployment"},
        "losses": {"distillation": "SmoothL1", "localization": "canonical_focal_dice", "distillation_weight": 1.0, "localization_weight": 1.0},
        "training": {"protocol": "12_class_LOCO", "trainable": ["p27_region_adapter"]},
        "firewall": {"mvtec_opened": False, "mvtec_data_reads": 0, "medical_reads": 0, "full_scientific_training_runs": 0},
    }

    report = audit_protocol(protocol)

    assert report["status"] == "PASS"
    assert report["REGION_TEACHER_HEADROOM"] == "NOT_CHECKED"


def test_audit_protocol_rejects_scientific_semantic_drift() -> None:
    protocol = {"schema_version": "P27_REGION_DISTILL_V1", "geometry": {"region_grid": [8, 8]}}

    with pytest.raises(RuntimeError, match="drift"):
        audit_protocol(protocol)


def test_audit_writes_json_and_markdown_report(tmp_path: Path) -> None:
    report = {"status": "PASS", "REGION_TEACHER_HEADROOM": "NOT_CHECKED"}

    json_path, markdown_path = write_audit_report(report, tmp_path)

    assert json_path.read_text().startswith("{")
    assert "P27 Audit" in markdown_path.read_text()


def test_train_and_gt_free_evaluation_parsers_require_frozen_asset_arguments() -> None:
    train = make_train_parser().parse_args(
        ["--held-class", "candle", "--visa-root", "/data/visa", "--p26-checkpoint", "/m/p26.pt", "--clip-asset", "/m/clip.pt", "--output", "/runs/p27"]
    )
    evaluate = make_evaluate_parser().parse_args(
        ["--held-class", "candle", "--visa-root", "/data/visa", "--p26-checkpoint", "/m/p26.pt", "--clip-asset", "/m/clip.pt", "--adapter-checkpoint", "/runs/p27/adapter.pt", "--output", "/runs/p27/predictions"]
    )

    assert train.held_class == evaluate.held_class == "candle"
