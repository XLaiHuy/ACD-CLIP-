from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "research/sabra_cure/final_architecture/SABRA_FINAL_CONFIG.json"
ENTRY = ROOT / "tools/sabra_cure/run_sabra_final.py"


def _module():
    spec = importlib.util.spec_from_file_location("run_sabra_final", ENTRY)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_final_config_has_no_open_scientific_choices() -> None:
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert cfg["status"] == "P26_FINAL_ARCHITECTURE_FROZEN"
    assert cfg["architecture"]["policy"] == "NATIVE_ONLY"
    assert cfg["architecture"]["reachable_actions"] == ["KEEP"]
    assert cfg["architecture"]["intervention_enabled"] is False
    assert cfg["external_validation"]["authorized"] is False
    assert cfg["firewall"] == {
        "additional_clip_forwards_in_p26": 0,
        "medical_reads_in_p26": 0,
        "mvtec_reads_in_p26": 0,
        "new_scientific_fits_in_p26": 0,
        "phase2b_steps_in_p26": 0,
        "scientific_attempt_markers_in_p26": 0,
    }
    serialized = json.dumps(cfg).lower()
    for forbidden in ("todo", "tune later", "try both", "best threshold"):
        assert forbidden not in serialized


def test_schema_rejects_hidden_or_changed_science() -> None:
    module = _module()
    cfg = module.load_config(CONFIG)
    module.validate_config(cfg)
    changed = json.loads(json.dumps(cfg))
    changed["architecture"]["reachable_actions"] = ["KEEP", "BOOST"]
    with pytest.raises(ValueError, match="reachable action"):
        module.validate_config(changed)


def test_check_only_verifies_required_artifacts_without_science() -> None:
    module = _module()
    result = module.check_only(CONFIG, ROOT)
    assert result["status"] == "PASS"
    assert result["scientific_evaluation"] is False
    assert result["clip_forwards"] == 0
    assert result["required_artifacts_verified"] >= 3


def test_dry_run_is_synthetic_native_postprocess_only() -> None:
    module = _module()
    result = module.dry_run(CONFIG, ROOT)
    assert result["status"] == "PASS"
    assert result["fixture"] == "SYNTHETIC_NATIVE_LOGITS"
    assert result["clip_forwards"] == 0
    assert result["external_dataset_reads"] == 0
    assert result["output_shape"] == [1, 518, 518]


def test_dry_run_cli_bootstraps_repository_imports() -> None:
    completed = subprocess.run(
        [sys.executable, str(ENTRY), "--dry-run", "--config", str(CONFIG)],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["status"] == "PASS"


def test_run_is_locked_until_separate_authorization() -> None:
    module = _module()
    with pytest.raises(SystemExit, match="AUTHORIZATION REQUIRED"):
        module.main(["--run", "--config", str(CONFIG)])


def test_p26_has_no_attempt_marker() -> None:
    assert not list((ROOT / "research/sabra_cure/final_architecture").glob("**/ATTEMPT_STARTED.json"))
    assert not list((ROOT / "results/sabra_cure/final_architecture").glob("**/ATTEMPT_STARTED.json"))
