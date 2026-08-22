from __future__ import annotations

import importlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("canonical_exporter", ROOT / "scripts/canonical/60_export_results.py")
assert SPEC is not None and SPEC.loader is not None
EXPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORTER)

MEDICAL_SPEC = importlib.util.spec_from_file_location(
    "canonical_medical_external", ROOT / "scripts/canonical/medical_compare_external.py"
)
assert MEDICAL_SPEC is not None and MEDICAL_SPEC.loader is not None
MEDICAL_EXTERNAL = importlib.util.module_from_spec(MEDICAL_SPEC)
MEDICAL_SPEC.loader.exec_module(MEDICAL_EXTERNAL)

WRAPPER_SPEC = importlib.util.spec_from_file_location(
    "canonical_phase2b_selector_recovery", ROOT / "scripts/canonical/phase2b_selector_recovery.py"
)
assert WRAPPER_SPEC is not None and WRAPPER_SPEC.loader is not None
SELECTOR_RECOVERY = importlib.util.module_from_spec(WRAPPER_SPEC)
WRAPPER_SPEC.loader.exec_module(SELECTOR_RECOVERY)

SELECTOR = importlib.import_module("select_phase2b_checkpoint")
from evaluation.evaluator import evaluate_records as CANONICAL_EVALUATE_RECORDS
from evaluation.metrics import (
    binary_average_precision as CANONICAL_AP,
    binary_auroc as CANONICAL_AUROC,
    selection_score as CANONICAL_SELECTION_SCORE,
)

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


def _metric_parity_cases() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(20260822)
    random_scores = rng.random(4096, dtype=np.float32)
    random_labels = rng.integers(0, 2, size=4096, dtype=np.int8)
    imbalance_scores = rng.random(10_000, dtype=np.float32)
    imbalance_labels = np.zeros(10_000, dtype=np.int8)
    imbalance_labels[[17, 9999]] = 1
    tied_scores = (rng.integers(0, 9, size=4097, dtype=np.int16).astype(np.float32) / np.float32(8.0))
    tied_labels = rng.integers(0, 2, size=4097, dtype=np.int8)
    all_equal_scores = np.full(101, np.float32(0.5), dtype=np.float32)
    all_equal_labels = np.asarray(([0, 1] * 50) + [1], dtype=np.int8)
    near_scores = np.asarray(
        [
            0.0,
            np.nextafter(np.float32(0.0), np.float32(1.0)),
            np.float32(1e-7),
            np.float32(1.0) - np.finfo(np.float32).eps,
            np.nextafter(np.float32(1.0), np.float32(0.0)),
            1.0,
        ]
        * 20,
        dtype=np.float32,
    )
    near_labels = np.tile(np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int8), 20)
    crossing_scores = np.asarray(([0.1] * 19) + ([0.4] * 37) + ([0.9] * 23), dtype=np.float32)
    crossing_labels = np.asarray(([0, 1, 1] * 27)[: crossing_scores.size], dtype=np.int8)
    ascending_order = np.argsort(random_scores, kind="mergesort")
    descending_order = ascending_order[::-1]
    shuffled_order = rng.permutation(random_scores.size)
    return {
        "random": (random_scores, random_labels),
        "heavy_imbalance": (imbalance_scores, imbalance_labels),
        "many_ties": (tied_scores, tied_labels),
        "all_equal_mixed": (all_equal_scores, all_equal_labels),
        "near_zero_one": (near_scores, near_labels),
        "tie_crosses_chunks": (crossing_scores, crossing_labels),
        "ascending": (random_scores[ascending_order], random_labels[ascending_order]),
        "descending": (random_scores[descending_order], random_labels[descending_order]),
        "random_input_order": (random_scores[shuffled_order], random_labels[shuffled_order]),
    }


def test_external_exact_metrics_match_canonical_to_1e12(tmp_path: Path) -> None:
    for name, (scores, labels) in _metric_parity_cases().items():
        expected_auroc = CANONICAL_AUROC(scores, labels)
        expected_ap = CANONICAL_AP(scores, labels)
        actual_auroc, actual_ap, metadata = MEDICAL_EXTERNAL.exact_metrics_from_arrays(
            scores,
            labels,
            tmp_path / name,
            chunk_elements=17,
        )
        assert metadata["initial_runs"] > 1
        assert abs(actual_auroc - expected_auroc) <= 1e-12, name
        assert abs(actual_ap - expected_ap) <= 1e-12, name


def _synthetic_compare_records() -> list[dict[str, object]]:
    native_maps = [
        [0.1, 0.1, 0.2, 0.4, 0.4, 0.8, 0.9, 0.9, 0.0, 0.3, 0.3, 0.7, 0.2, 0.6, 0.6, 1.0],
        [0.0, 0.2, 0.2, 0.5, 0.5, 0.5, 0.8, 0.8, 0.1, 0.1, 0.4, 0.7, 0.7, 0.9, 0.9, 1.0],
        [0.05, 0.05, 0.3, 0.3, 0.6, 0.6, 0.6, 0.95, 0.1, 0.2, 0.2, 0.4, 0.4, 0.8, 0.8, 0.9],
        [0.0, 0.15, 0.15, 0.35, 0.35, 0.55, 0.75, 0.75, 0.05, 0.25, 0.25, 0.45, 0.65, 0.65, 0.85, 1.0],
    ]
    labels = [
        [0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1],
        [0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1],
        [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],
        [1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    ]
    class_names = ["class_a", "class_a", "class_b", "class_b"]
    image_labels = [0, 1, 0, 0]
    records: list[dict[str, object]] = []
    for index, values in enumerate(native_maps):
        native = np.asarray(values, dtype=np.float32)
        correction = np.asarray(([0.0, 0.05, 0.0, 0.05] * 4), dtype=np.float32)
        sabra = np.minimum(native + correction, np.float32(1.0)).astype(np.float32)
        records.append(
            {
                "class_name": class_names[index],
                "image_path": f"/synthetic/{class_names[index]}/{index:03d}.png",
                "pixel_labels": np.asarray(labels[index], dtype=np.int8),
                "image_labels": np.asarray([image_labels[index]], dtype=np.int8),
                "phase2b": {
                    "pixel_scores": native,
                    "image_scores": np.asarray([0.1 + 0.2 * index]),
                },
                "sabra": {
                    "pixel_scores": sabra,
                    "image_scores": np.asarray([0.15 + 0.2 * index]),
                },
            }
        )
    return records


def _write_synthetic_cache(tmp_path: Path, records: list[dict[str, object]]):
    work_dir = tmp_path / "medical_work" / "Brain"
    writer = MEDICAL_EXTERNAL.InferenceCacheWriter(
        work_dir,
        dataset="Brain",
        data_root=tmp_path / "data",
        image_count=len(records),
        pixels_per_image=16,
        image_size=4,
        checkpoint_sha256="a" * 64,
        freeze_sha256="b" * 64,
        workflow_package_sha="c" * 40,
        evaluator_sha256="d" * 64,
    )
    writer.write_batch(
        class_names=[str(row["class_name"]) for row in records],
        image_paths=[str(row["image_path"]) for row in records],
        pixel_labels=np.stack([np.asarray(row["pixel_labels"]) for row in records]),
        phase2b_pixel_scores=np.stack([np.asarray(row["phase2b"]["pixel_scores"]) for row in records]),
        sabra_pixel_scores=np.stack([np.asarray(row["sabra"]["pixel_scores"]) for row in records]),
        image_labels=np.concatenate([np.asarray(row["image_labels"]) for row in records]),
        phase2b_image_scores=np.concatenate([np.asarray(row["phase2b"]["image_scores"]) for row in records]),
        sabra_image_scores=np.concatenate([np.asarray(row["sabra"]["image_scores"]) for row in records]),
    )
    manifest = writer.complete()
    expected = {
        "dataset": "Brain",
        "data_root": tmp_path / "data",
        "selected_checkpoint_sha256": "a" * 64,
        "sabra_freeze_sha256": "b" * 64,
        "workflow_package_sha": "c" * 40,
        "workflow_evaluator_sha256": "d" * 64,
        "image_size": 4,
        "pixels_per_image": 16,
    }
    validated = MEDICAL_EXTERNAL.validate_inference_cache(work_dir, expected)
    return work_dir, validated, expected


def test_prediction_cache_bitwise_parity_and_truncation_rejection(tmp_path: Path) -> None:
    records = _synthetic_compare_records()
    work_dir, _, expected = _write_synthetic_cache(tmp_path, records)
    expected_labels = np.stack([np.asarray(row["pixel_labels"], dtype=np.int8) for row in records])
    expected_native = np.stack([np.asarray(row["phase2b"]["pixel_scores"], dtype=np.float32) for row in records])
    expected_sabra = np.stack([np.asarray(row["sabra"]["pixel_scores"], dtype=np.float32) for row in records])
    cached_labels = np.load(work_dir / "pixel_labels.npy", mmap_mode="r")
    cached_native = np.load(work_dir / "phase2b_pixel_scores.npy", mmap_mode="r")
    cached_sabra = np.load(work_dir / "sabra_pixel_scores.npy", mmap_mode="r")
    assert cached_native.dtype == np.float32
    assert cached_sabra.dtype == np.float32
    assert cached_labels.dtype == np.int8
    assert cached_native.shape == expected_native.shape
    assert np.array_equal(cached_native, expected_native)
    assert np.array_equal(cached_sabra, expected_sabra)
    assert np.array_equal(cached_labels, expected_labels)
    identities = [
        json.loads(line)
        for line in (work_dir / MEDICAL_EXTERNAL.IDENTITIES_NAME).read_text(encoding="utf-8").splitlines()
    ]
    assert [row["image_path"] for row in identities] == [row["image_path"] for row in records]
    assert [row["order"] for row in identities] == list(range(len(records)))

    wrong_workflow = {**expected, "workflow_package_sha": "e" * 40}
    with pytest.raises(ValueError, match="mismatch for workflow_package_sha"):
        MEDICAL_EXTERNAL.validate_inference_cache(work_dir, wrong_workflow)
    wrong_evaluator = {**expected, "workflow_evaluator_sha256": "f" * 64}
    with pytest.raises(ValueError, match="mismatch for workflow_evaluator_sha256"):
        MEDICAL_EXTERNAL.validate_inference_cache(work_dir, wrong_evaluator)

    score_path = work_dir / "phase2b_pixel_scores.npy"
    with score_path.open("r+b") as handle:
        handle.truncate(score_path.stat().st_size - 1)
    with pytest.raises(ValueError, match="missing or truncated"):
        MEDICAL_EXTERNAL.validate_inference_cache(work_dir, expected)


def test_external_compare_end_to_end_and_export_schema_parity(tmp_path: Path) -> None:
    records = _synthetic_compare_records()
    canonical = CANONICAL_EVALUATE_RECORDS(records, method="compare", allow_undefined_image_metrics=True)
    work_dir, manifest, _ = _write_synthetic_cache(tmp_path, records)
    external, runtime = MEDICAL_EXTERNAL.evaluate_inference_cache(
        work_dir,
        manifest,
        memory_budget_bytes=16 * 1024**2,
        chunk_elements=5,
    )

    for section in ("phase2b", "sabra"):
        for class_name, expected_row in canonical[section].items():
            for metric_name, expected_value in expected_row.items():
                actual_value = external[section][class_name][metric_name]
                if expected_value is None:
                    assert actual_value is None
                else:
                    assert abs(actual_value - expected_value) <= 1e-12
        for metric_name, expected_value in canonical[f"{section}_macro"].items():
            actual_value = external[f"{section}_macro"][metric_name]
            if expected_value is None:
                assert actual_value is None
            else:
                assert abs(actual_value - expected_value) <= 1e-12
    for metric_name, expected_value in canonical["delta"].items():
        actual_value = external["delta"][metric_name]
        if expected_value is None:
            assert actual_value is None
        else:
            assert abs(actual_value - expected_value) <= 1e-12
    assert external["phase2b"]["class_b"]["image_auroc"] is None
    assert external["sabra"]["class_b"]["image_ap"] is None

    output = {
        **external,
        "dataset": "Brain",
        "role": "FINAL_ZERO_SHOT",
        "phase2b_checkpoint_sha256": "a" * 64,
        "sabra_freeze_sha256": "b" * 64,
        "runtime": {
            **runtime,
            "external_memory": True,
            "external_backend": "numpy_external",
        },
    }
    output_dir = tmp_path / "medical" / "Brain"
    MEDICAL_EXTERNAL._write_outputs_atomic(output_dir, output)
    persisted = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    phase2b_export, sabra_export = EXPORTER._compare_metrics(persisted)
    assert phase2b_export["pAUROC"] == external["phase2b_macro"]["pixel_auroc"]
    assert sabra_export["pAP"] == external["sabra_macro"]["pixel_ap"]
    assert EXPORTER._class_metrics(persisted, "phase2b")
    assert EXPORTER._class_metrics(persisted, "sabra")


def test_medical_external_guard_reuses_canonical_checkpoint_and_freeze_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "selected.pth"
    checkpoint.write_bytes(b"synthetic-checkpoint")
    checkpoint_sha = MEDICAL_EXTERNAL.sha256_file(checkpoint)
    selection_path = tmp_path / "selection.json"
    selection_path.write_text(
        json.dumps(
            {
                "status": "FROZEN",
                "selected_checkpoint": str(checkpoint),
                "selected_checkpoint_sha256": checkpoint_sha,
            }
        ),
        encoding="utf-8",
    )
    freeze_path = tmp_path / "SABRA_FREEZE.json"
    freeze_path.write_text(
        json.dumps(
            {
                "provenance": {"git_sha": SCIENTIFIC_CODE_SHA},
                "relational": {"backend": "fast"},
                "medical_seen": False,
            }
        ),
        encoding="utf-8",
    )
    validated: list[str] = []

    def fake_validate(payload, checkpoint_sha256=None):
        assert payload["medical_seen"] is False
        validated.append(str(checkpoint_sha256))

    monkeypatch.setattr(MEDICAL_EXTERNAL, "validate_sabra_freeze", fake_validate)
    selection, _, selected_path, selected_sha, freeze_sha = MEDICAL_EXTERNAL.validate_medical_inputs(
        "Brain", selection_path, freeze_path
    )
    assert selection["status"] == "FROZEN"
    assert selected_path == checkpoint.resolve()
    assert selected_sha == checkpoint_sha
    assert freeze_sha == MEDICAL_EXTERNAL.sha256_file(freeze_path)
    assert validated == [checkpoint_sha]


def test_medical_external_inference_enables_only_sabra_sensitivity_grad(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = MEDICAL_EXTERNAL.torch
    checkpoint_sha = "a" * 64
    forward_calls: list[bool] = []
    compare_grad_states: list[bool] = []
    cached_sabra: list[np.ndarray] = []

    class FakeModel:
        def __init__(self) -> None:
            self.frozen_weight = torch.tensor(1.0, requires_grad=False)
            self.eval_called = False

        def eval(self):
            self.eval_called = True
            return self

    class FakeProgress:
        def update(self, _count: int) -> None:
            pass

        def set_postfix(self, _values: dict[str, str]) -> None:
            pass

        def close(self) -> None:
            pass

    class FakeLoader:
        def __init__(self, _dataset, **_kwargs) -> None:
            pass

        def __iter__(self):
            yield {
                "image": torch.zeros((1, 3, 2, 2), dtype=torch.float32),
                "class_name": ["synthetic"],
                "mask": torch.zeros((1, 2, 2), dtype=torch.int8),
                "label": torch.zeros((1,), dtype=torch.int8),
                "image_path": ["synthetic.png"],
            }

    class FakeWriter:
        def __init__(self, _work_dir, **_kwargs) -> None:
            self.written = 0
            self.cache_write_seconds = 0.0

        def write_batch(self, *, class_names, sabra_pixel_scores, **_kwargs) -> None:
            assert isinstance(sabra_pixel_scores, np.ndarray)
            assert sabra_pixel_scores.dtype == np.float32
            cached_sabra.append(sabra_pixel_scores)
            self.written += len(class_names)

        def complete(self) -> dict[str, int]:
            return {"image_count": self.written}

        def abort(self) -> None:
            raise AssertionError("synthetic inference unexpectedly aborted")

    class FakeForward:
        deployed_segmentation_probability = torch.full((1, 4), 0.25, dtype=torch.float32)
        classification_probability = torch.full((1,), 0.5, dtype=torch.float32)

    model = FakeModel()

    def fake_forward_phase2b(*_args, require_grad: bool, **_kwargs):
        forward_calls.append(require_grad)
        assert require_grad is False
        assert model.frozen_weight.requires_grad is False
        return FakeForward()

    def fake_compare_forward(forward, _freeze, *, domain: str):
        compare_grad_states.append(torch.is_grad_enabled())
        assert torch.is_grad_enabled()
        assert domain == "Medical"
        assert forward.deployed_segmentation_probability.requires_grad is False
        shared = torch.ones((1, 4), dtype=torch.float32, requires_grad=True)
        corrected = shared.square()
        assert corrected.requires_grad
        return {"corrected_probability": corrected}

    monkeypatch.setattr(MEDICAL_EXTERNAL, "load_json_config", lambda _path: {})
    monkeypatch.setattr(
        MEDICAL_EXTERNAL, "_verify_selected_checkpoint", lambda _selection: (tmp_path / "selected.pth", checkpoint_sha)
    )
    monkeypatch.setattr(MEDICAL_EXTERNAL, "load_phase2b_checkpoint", lambda *_args: model)
    monkeypatch.setattr(MEDICAL_EXTERNAL, "dataset_domain", lambda _dataset: "Medical")
    monkeypatch.setattr(MEDICAL_EXTERNAL, "_build_inference_dataset", lambda *_args: [object()])
    monkeypatch.setattr(MEDICAL_EXTERNAL, "DataLoader", FakeLoader)
    monkeypatch.setattr(MEDICAL_EXTERNAL, "InferenceCacheWriter", FakeWriter)
    monkeypatch.setattr(MEDICAL_EXTERNAL, "tqdm", lambda **_kwargs: FakeProgress())
    monkeypatch.setattr(MEDICAL_EXTERNAL, "forward_phase2b", fake_forward_phase2b)
    monkeypatch.setattr(MEDICAL_EXTERNAL, "compare_forward", fake_compare_forward)
    monkeypatch.setattr(MEDICAL_EXTERNAL, "image_score", lambda classification, pixel, _domain: classification + pixel)
    monkeypatch.setattr(MEDICAL_EXTERNAL, "_cuda_runtime_stats", lambda *_args: {})

    manifest, _runtime = MEDICAL_EXTERNAL.infer_to_cache(
        dataset="Brain",
        data_root=tmp_path / "data",
        selection={},
        freeze={},
        checkpoint_sha256=checkpoint_sha,
        freeze_sha256="b" * 64,
        config_path=tmp_path / "config.json",
        clip_asset=tmp_path / "clip.pt",
        device=torch.device("cpu"),
        work_dir=tmp_path / "medical_work" / "Brain",
        workflow_package_sha="c" * 40,
        evaluator_sha256="d" * 64,
    )

    assert manifest["image_count"] == 1
    assert model.eval_called
    assert model.frozen_weight.requires_grad is False
    assert forward_calls == [False]
    assert compare_grad_states == [True]
    assert len(cached_sabra) == 1


def test_deployment_sensitivity_can_reenable_grad_inside_no_grad() -> None:
    torch = MEDICAL_EXTERNAL.torch
    from model.phase2b_runtime import PATCH_COUNT, compute_deployment_sensitivity

    native = torch.zeros((3, 1, PATCH_COUNT, 2), dtype=torch.float32)
    with torch.no_grad():
        with torch.enable_grad():
            sensitivity = compute_deployment_sensitivity(native, domain="Medical")

    assert sensitivity.shape == (1, PATCH_COUNT)
    assert sensitivity.requires_grad is False
    assert torch.isfinite(sensitivity).all()
    assert torch.count_nonzero(sensitivity) > 0


def test_external_metric_memory_is_chunk_bounded(tmp_path: Path) -> None:
    script = ROOT / "scripts/canonical/medical_compare_external.py"

    def run(name: str, pixels: int) -> dict[str, object]:
        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--synthetic-stress",
                str(tmp_path / name),
                "--synthetic-pixels",
                str(pixels),
                "--synthetic-chunk-elements",
                "50000",
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout.splitlines()[-1])

    small = run("small", 250_000)
    large = run("large", 2_500_000)
    assert large["pixels"] == 10 * small["pixels"]
    assert large["initial_runs"] == 10 * small["initial_runs"]
    assert large["peak_rss_bytes"] <= small["peak_rss_bytes"] * 1.5 + 64 * 1024**2
    assert large["raw_disk_bytes"] > small["raw_disk_bytes"]
    assert 0.0 <= large["auroc"] <= 1.0
    assert 0.0 <= large["ap"] <= 1.0


def test_medical_workflow_and_resume_use_external_cache_without_real_execution() -> None:
    stage5 = (ROOT / "scripts/canonical/50_eval_medical.sh").read_text(encoding="utf-8")
    resume = (ROOT / "scripts/canonical/RESUME_CANONICAL_MEDICAL.sh").read_text(encoding="utf-8")
    assert '"$PYTHON" "$SCRIPT_DIR/medical_compare_external.py"' in stage5
    assert '"$REPO_ROOT/test.py"' not in stage5
    for contract in (
        "--batch-size 6",
        "--num-workers 4",
        "--prefetch-factor 2",
        "--pin-memory",
        "--metric-mode exact",
        "--pixel-stride 1",
        "--reuse-inference-cache",
    ):
        assert contract in stage5
    assert 'work_dir="$RUN_ROOT/medical_work/$dataset"' in stage5
    assert "medical_Brain.oom_kill_backup.log" in resume
    assert "./scripts/canonical/run_pipeline.sh medical" in resume
    assert "./scripts/canonical/run_pipeline.sh export" in resume
    assert "run_pipeline.sh train" not in resume
    assert "run_pipeline.sh select" not in resume
    assert "run_pipeline.sh fit-sabra" not in resume
    assert "run_pipeline.sh lambda" not in resume
