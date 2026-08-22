from __future__ import annotations

import importlib
import importlib.util
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("canonical_exporter", ROOT / "scripts/canonical/60_export_results.py")
assert SPEC is not None and SPEC.loader is not None
EXPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORTER)

WRAPPER_SPEC = importlib.util.spec_from_file_location(
    "canonical_phase2b_selector_recovery", ROOT / "scripts/canonical/phase2b_selector_recovery.py"
)
assert WRAPPER_SPEC is not None and WRAPPER_SPEC.loader is not None
SELECTOR_RECOVERY = importlib.util.module_from_spec(WRAPPER_SPEC)
WRAPPER_SPEC.loader.exec_module(SELECTOR_RECOVERY)

SELECTOR = importlib.import_module("select_phase2b_checkpoint")
from evaluation.metrics import selection_score as CANONICAL_SELECTION_SCORE

SCIENTIFIC_CODE_SHA = "4aa9b465ddeb072e9218b74982306d6324c62375"


def _synthetic_phase2b_candidates() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    metrics = {
        10: (0.90, 0.80, 0.70, 0.60),
        12: (0.90, 0.80, 0.70, 0.60),
        14: (0.80, 0.70, 0.60, 0.50),
        16: (0.70, 0.60, 0.50, 0.40),
        18: (0.60, 0.50, 0.40, 0.30),
        20: (0.50, 0.40, 0.30, 0.20),
    }
    for epoch, values in metrics.items():
        rows.append(
            {
                "epoch": epoch,
                "path": f"/synthetic/checkpoints/adapter_{epoch}.pth",
                "sha256": f"{epoch:02d}" * 32,
                "pixel_auroc": values[0],
                "pixel_ap": values[1],
                "image_auroc": values[2],
                "image_ap": values[3],
            }
        )
    return rows


def test_phase2b_recovery_serializes_raw_rows_with_canonical_scores(tmp_path: Path) -> None:
    raw_rows = _synthetic_phase2b_candidates()
    assert all("score" not in row for row in raw_rows)
    assert SELECTOR.selection_score is CANONICAL_SELECTION_SCORE

    original_main = SELECTOR.main
    original_select_candidate = SELECTOR.select_candidate
    selected_inputs: list[list[dict[str, object]]] = []

    def tracked_select_candidate(candidates):
        materialized = [dict(candidate) for candidate in candidates]
        selected_inputs.append(materialized)
        return original_select_candidate(materialized)

    def synthetic_main(argv=None):
        assert argv == ["--synthetic"]
        assert "--metrics-json" not in argv
        selected = SELECTOR.select_candidate(raw_rows)
        SELECTOR._write_selection(tmp_path, raw_rows, selected, code_sha="synthetic-code-sha")
        return 0

    SELECTOR.main = synthetic_main
    SELECTOR.select_candidate = tracked_select_candidate
    try:
        assert SELECTOR_RECOVERY.main(["--synthetic"]) == 0
    finally:
        SELECTOR.main = original_main
        SELECTOR.select_candidate = original_select_candidate

    assert selected_inputs == [raw_rows]
    selection = json.loads((tmp_path / "phase2b_selection.json").read_text(encoding="utf-8"))
    assert selection["selected_epoch"] == 10
    assert len(selection["candidates"]) == len(raw_rows)

    output_by_epoch = {int(row["epoch"]): row for row in selection["candidates"]}
    for raw in raw_rows:
        epoch = int(raw["epoch"])
        output = output_by_epoch[epoch]
        expected_score = CANONICAL_SELECTION_SCORE(
            {name: float(raw[name]) for name in SELECTOR.METRIC_NAMES}
        )
        assert output["score"] == expected_score
        assert output["path"] == raw["path"]
        assert output["sha256"] == raw["sha256"]
        assert output["pAUROC"] == raw["pixel_auroc"]
        assert output["pAP"] == raw["pixel_ap"]
        assert output["iAUROC"] == raw["image_auroc"]
        assert output["iAP"] == raw["image_ap"]


def test_canonical_stage2_uses_recovery_wrapper_without_debug_metrics() -> None:
    stage2 = (ROOT / "scripts/canonical/20_select_phase2b.sh").read_text(encoding="utf-8")
    assert '"$PYTHON" "$SCRIPT_DIR/phase2b_selector_recovery.py"' in stage2
    assert '"$REPO_ROOT/select_phase2b_checkpoint.py"' not in stage2
    assert "--metrics-json" not in stage2

def test_delta_preserves_undefined_metrics() -> None:
    phase2b = {"pixel_auroc": 0.4, "pixel_ap": 0.5, "image_auroc": None, "image_ap": None}
    sabra = {"pixel_auroc": 0.6, "pixel_ap": 0.7, "image_auroc": 0.9, "image_ap": 0.8}
    delta = EXPORTER.delta_metrics(phase2b, sabra)
    assert delta == {"pAUROC": 0.19999999999999996, "pAP": 0.19999999999999996, "iAUROC": None, "iAP": None}


def test_macro_aggregation_ignores_only_null_values() -> None:
    rows = [
        {"pAUROC": 0.2, "pAP": 0.4, "iAUROC": None, "iAP": None},
        {"pAUROC": 0.6, "pAP": 0.8, "iAUROC": 0.5, "iAP": 0.7},
    ]
    assert EXPORTER._mean_defined(rows) == {"pAUROC": 0.4, "pAP": 0.6000000000000001, "iAUROC": 0.5, "iAP": 0.7}


def test_canonical_medical_workflow_uses_evaluation_metadata() -> None:
    from dataset.info import MEDICAL_EVAL_PATHS

    expected = (
        "Brain",
        "Liver",
        "Retina",
        "Colon_clinicDB",
        "Colon_colonDB",
        "Colon_Kvasir",
    )
    canonical = tuple(MEDICAL_EVAL_PATHS)
    assert canonical == expected
    assert "Colon_cvc300" not in canonical

    medical_script = (ROOT / "scripts/canonical/50_eval_medical.sh").read_text(encoding="utf-8")
    exporter_script = (ROOT / "scripts/canonical/60_export_results.py").read_text(encoding="utf-8")
    assert "from dataset.info import MEDICAL_EVAL_PATHS" in medical_script
    assert "for name in MEDICAL_EVAL_PATHS:" in medical_script
    assert "medical_datasets = tuple(MEDICAL_EVAL_PATHS)" in exporter_script
    assert "CLASS_NAMES, is_medical_dataset" not in medical_script
    assert "CLASS_NAMES, is_medical_dataset" not in exporter_script

    required_metrics = {ROOT / "medical" / dataset / "metrics.json" for dataset in canonical}
    assert ROOT / "medical" / "Colon_cvc300" / "metrics.json" not in required_metrics


def _workflow_path_allowed(path: str) -> bool:
    common = ROOT / "scripts/canonical/common.sh"
    result = subprocess.run(
        ["bash", "-c", 'source "$1"; workflow_path_allowed "$2"', "bash", str(common), path],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def test_scientific_ancestor_accepts_workflow_only_history() -> None:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    subprocess.run(["git", "merge-base", "--is-ancestor", SCIENTIFIC_CODE_SHA, head], cwd=ROOT, check=True)
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", f"{SCIENTIFIC_CODE_SHA}..{head}"], cwd=ROOT, text=True
    ).splitlines()
    assert changed
    assert all(_workflow_path_allowed(path) for path in changed)

    # A representative scientific path is deliberately outside the workflow
    # allowlist and must be rejected even when the commit graph is valid.
    assert not _workflow_path_allowed("model/phase2b_runtime.py")
    assert not _workflow_path_allowed("train.py")
