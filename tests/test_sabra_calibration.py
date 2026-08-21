from __future__ import annotations

import numpy as np

from calibrate_sabra import lambda_grid, refined_lambda_grid, select_lambda, fit_source_payload
from tools.sabra.artifacts import build_freeze_payload, validate_sabra_freeze
from tools.sabra.relational import FEATURE_ORDER, NEED_ORDER


def _record(class_name: str, offset: float):
    width = 2
    record = {"class_name": class_name, "trust_target": np.asarray([0, 1], dtype=np.int8), "need_target": np.asarray([1, 0], dtype=np.int8)}
    for index, name in enumerate(FEATURE_ORDER):
        record[name] = np.asarray([offset + index * 0.1, offset + index * 0.1 + 0.3], dtype=np.float32)
    for index, name in enumerate(NEED_ORDER):
        record[name] = np.asarray([offset + index * 0.2, offset + index * 0.2 + 0.2], dtype=np.float32)
    return record


def test_lambda_grid_refinement_and_tie():
    assert lambda_grid()[0] == 0.0 and lambda_grid()[-1] == 1.0
    refined = refined_lambda_grid(0.5, exclude=[0.5])
    assert 0.5 not in refined
    selected = select_lambda([
        {"lambda": 0.025, "pixel_auroc": 0.5, "pixel_ap": 0.5, "image_auroc": 0.5, "image_ap": 0.5},
        {"lambda": 0.0, "pixel_auroc": 0.5, "pixel_ap": 0.5, "image_auroc": 0.5, "image_ap": 0.5},
    ])
    assert selected["lambda"] == 0.0


def test_source_artifact_then_final_freeze():
    rows = [_record("a", 0.0), _record("b", 0.4), _record("c", 0.8)]
    source = fit_source_payload(rows, selected_epoch=10, checkpoint_sha256="abc", margin_values=np.asarray([-1.0, 0.0, 2.0, 4.0]), git_sha="git")
    freeze = build_freeze_payload(source, selected_lambda=0.0, selected_score=0.5, git_sha="git", coarse_grid=[0.0, 0.025])
    validate_sabra_freeze(freeze, checkpoint_sha256="abc")
    assert freeze["correction"]["formula"] == "delta=lambda*margin_scale*T*N"
    assert freeze["medical_seen"] is False
