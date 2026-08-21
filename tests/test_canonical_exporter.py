from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("canonical_exporter", ROOT / "scripts/canonical/60_export_results.py")
assert SPEC is not None and SPEC.loader is not None
EXPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORTER)

SCIENTIFIC_CODE_SHA = "4aa9b465ddeb072e9218b74982306d6324c62375"


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
