from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import tools.sabra_v2.run_p29r1_forensic as runner


def test_real_run_path_reproduces_missing_residual_summary_dependency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise CLI-run -> first held class with only synthetic frozen inputs."""
    output = tmp_path / "output"
    cache = tmp_path / "cache"
    tier_a = cache / "tier_a" / "candle"
    tier_a.mkdir(parents=True)
    (tier_a / "manifest.json").write_text(
        json.dumps({"sample_ids": ["candle:candle/sample.JPG"]}), encoding="utf-8"
    )
    np.save(tier_a / "native_logits.npy", np.zeros((1,), dtype=np.float32))
    np.save(tier_a / "seg_features.npy", np.zeros((1,), dtype=np.float32))
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}", encoding="utf-8")
    preaudit = tmp_path / "preaudit.json"
    preaudit.write_text("{}", encoding="utf-8")
    preflight = tmp_path / "preflight.json"
    preflight.write_text(json.dumps({"status": "PASS", "estimate": {"decision": "PROCEED"}}), encoding="utf-8")

    row = {"class_name": "candle", "image_path": "candle/sample.JPG"}
    fold = SimpleNamespace(held_rows=(row,), fit_rows=())
    frozen_map = torch.zeros((518, 518), dtype=torch.float32)
    records = [{
        "image_path": row["image_path"],
        "native_abnormal_probability": frozen_map,
        "p27_abnormal_probability": frozen_map,
        "p29_abnormal_probability": frozen_map,
    }]

    def fake_git(*args: str) -> str:
        if args == ("branch", "--show-current"):
            return "research/p29r1-fast-objective-forensic-v1"
        if args == ("rev-parse", "HEAD"):
            return "exec"
        if args == ("status", "--porcelain"):
            return ""
        return ""

    inventory = {
        "p29_predictions": {name: {"expected": f"p29-{name}"} for name in runner.CLASS_NAMES},
        "p27_predictions": {name: {"expected": f"p27-{name}"} for name in runner.CLASS_NAMES},
        "p29_checkpoints": {name: {"expected": f"p29-checkpoint-{name}"} for name in runner.CLASS_NAMES},
        "p27_checkpoints": {name: {"expected": f"p27-checkpoint-{name}"} for name in runner.CLASS_NAMES},
    }
    monkeypatch.setattr(runner, "validate_runner_execution_contract", lambda: None)
    monkeypatch.setattr(runner, "_protocol_payload", lambda path: {})
    monkeypatch.setattr(runner, "_git", fake_git)
    monkeypatch.setattr(runner, "_remote_head", lambda branch: "exec")
    monkeypatch.setattr(runner, "_artifact_inventory", lambda cache_root, metadata: inventory)
    monkeypatch.setattr(runner, "enforce_data_firewall", lambda visa_root, paths: None)
    monkeypatch.setattr(runner, "read_visa_metadata", lambda metadata: [row])
    monkeypatch.setattr(runner, "loco_inventory", lambda rows, held_class: fold)
    monkeypatch.setattr(runner, "_prediction_records", lambda *args: records)
    monkeypatch.setattr(runner, "_load_masks", lambda rows, visa_root: (np.zeros((1, 518, 518), dtype=np.uint8), 0))
    monkeypatch.setattr(runner, "_load_adapter", lambda *args: object())
    monkeypatch.setattr(runner, "_native_cache_probability", lambda native, indices, device: np.zeros((1, 518, 518), dtype=np.float32))
    monkeypatch.setattr(runner, "_teacher_regions", lambda native, indices, masks, device: np.zeros((1, 9, 9), dtype=np.float32))
    monkeypatch.setattr(runner, "_student_regions", lambda adapter, seg, indices, device: np.zeros((3, 1, 9, 9), dtype=np.float32))

    def stop_after_held_class(*args: object, **kwargs: object) -> list[dict[str, object]]:
        raise RuntimeError("synthetic stop after held-class")

    monkeypatch.setattr(runner, "_gradient_probe", stop_after_held_class)

    args = SimpleNamespace(
        execution_base_sha="exec",
        prereg_sha="pre",
        forensic_uuid="new-test-uuid",
        utc_started="2026-08-26T00:00:00+00:00",
        protocol=protocol,
        preaudit_output=preaudit,
        preflight_output=preflight,
        visa_root=tmp_path / "visa",
        cache_root=cache,
        metadata=tmp_path / "metadata.jsonl",
        output_dir=output,
        device="cpu",
    )

    with pytest.raises(RuntimeError, match="synthetic stop after held-class"):
        runner.run(args)
