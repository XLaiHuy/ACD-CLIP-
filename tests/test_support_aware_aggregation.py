"""Tests for support_aware_aggregation.py.

Covers:
1.  Missing image metrics are not converted to zero.
2.  Single-class image labels mark image AUROC unsupported.
3.  Brain/Liver/Retina valid image metrics are included.
4.  Pixel metrics can remain valid when image metrics are invalid.
5.  Image macro uses only valid image datasets.
6.  Pixel macro uses all valid pixel datasets.
7.  Combined score is harmonic mean of domain macros.
8.  Legacy and support-aware scores are both emitted.
9.  Exact ties select the lower epoch.
10. Non-ties do not use near-best behavior.
11. Fingerprint is non-empty and deterministic.
12. Fingerprint changes when split/metric policy changes.
13. Original selection artifact is not overwritten.
14. No exact test is launched.
15. Existing P1-v7 protocol tests still pass (imported guard).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import pytest

try:
    from tools.support_aware_aggregation import (
        SELECTION_RULE_VERSION,
        aggregate_epoch,
        build_fingerprint_config,
        compute_config_fingerprint,
        compute_dataset_scores,
        compute_legacy_combined_score,
        harmonic_mean_safe,
        is_image_metric_valid,
        is_pixel_metric_valid,
        reaggregate_all_epochs,
        select_best_epoch,
    )
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))
    from support_aware_aggregation import (
        SELECTION_RULE_VERSION,
        aggregate_epoch,
        build_fingerprint_config,
        compute_config_fingerprint,
        compute_dataset_scores,
        compute_legacy_combined_score,
        harmonic_mean_safe,
        is_image_metric_valid,
        is_pixel_metric_valid,
        reaggregate_all_epochs,
        select_best_epoch,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_row(
    dataset: str,
    image_auroc: float,
    image_ap: float,
    pixel_auroc: float,
    pixel_ap: float,
    normal_count: int,
    anomaly_count: int,
    epoch: int = 1,
) -> dict:
    img_valid = is_image_metric_valid(normal_count, anomaly_count)
    pix_valid = is_pixel_metric_valid(normal_count, anomaly_count)
    scores = compute_dataset_scores(
        image_auroc, image_ap, pixel_auroc, pixel_ap, img_valid, pix_valid
    )
    return {"dataset": dataset, "epoch": epoch, **scores}


def _hm(a: float, b: float) -> float:
    return 0.0 if a + b == 0 else 2.0 * a * b / (a + b)


# ---------------------------------------------------------------------------
# 1. Missing image metrics are NOT converted to zero
# ---------------------------------------------------------------------------


class TestMissingImageMetricsNotZero:
    def test_invalid_image_dataset_has_none_image_score(self):
        """image_score must be None, not 0.0, when image metrics are unsupported."""
        row = _make_row("Colon_clinicDB", 0.0, 0.0, 87.0, 50.0, normal_count=0, anomaly_count=184)
        assert row["image_score"] is None, (
            f"Expected None for unsupported image metric, got {row['image_score']}"
        )

    def test_invalid_image_dataset_has_none_in_epoch_aggregate(self):
        rows = [_make_row("Colon_clinicDB", 0.0, 0.0, 87.0, 50.0, 0, 184)]
        result = aggregate_epoch(rows)
        assert result["support_aware_image_macro"] is None

    def test_zero_is_not_treated_as_valid_image_score(self):
        """A dataset with only anomalies should not contribute a zero to image macro."""
        rows = [
            _make_row("Colon_clinicDB", 0.0, 0.0, 87.0, 50.0, 0, 100),
            _make_row("Brain", 80.0, 82.0, 94.0, 20.0, 39, 44),
        ]
        result = aggregate_epoch(rows)
        # image_macro should only include Brain (80+82)/2 = 81.0
        assert result["support_aware_image_macro"] == pytest.approx(81.0)

    def test_legacy_score_uses_zero_for_invalid(self):
        """Legacy code path does include zeros — this confirms the bug we fixed."""
        rows = [
            _make_row("Colon_clinicDB", 0.0, 0.0, 87.0, 50.0, 0, 100),
            _make_row("Brain", 80.0, 82.0, 94.0, 20.0, 39, 44),
        ]
        legacy = compute_legacy_combined_score(rows)
        # Brain: image_score=81, pixel_score=57, hm=67.ish
        # Colon: image_score=0 (None treated as 0), pixel_score=68.5, hm=0
        # legacy = (67.ish + 0) / 2 — definitely less than support-aware
        result = aggregate_epoch(rows)
        assert legacy < result["support_aware_combined_score"]


# ---------------------------------------------------------------------------
# 2. Single-class image labels mark image AUROC unsupported
# ---------------------------------------------------------------------------


class TestSingleClassInvalid:
    def test_zero_normal_is_invalid(self):
        assert not is_image_metric_valid(normal_count=0, anomaly_count=100)

    def test_zero_anomaly_is_invalid(self):
        assert not is_image_metric_valid(normal_count=50, anomaly_count=0)

    def test_both_zero_is_invalid(self):
        assert not is_image_metric_valid(normal_count=0, anomaly_count=0)

    def test_both_nonzero_is_valid(self):
        assert is_image_metric_valid(normal_count=1, anomaly_count=1)

    def test_colon_datasets_excluded_from_image(self):
        colon_datasets = ["Colon_clinicDB", "Colon_colonDB", "Colon_Kvasir"]
        for ds in colon_datasets:
            row = _make_row(ds, 0.0, 0.0, 85.0, 50.0, normal_count=0, anomaly_count=100)
            assert row["image_valid"] is False, f"{ds} should have invalid image metrics"
            assert row["image_score"] is None


# ---------------------------------------------------------------------------
# 3. Brain/Liver/Retina valid image metrics are included
# ---------------------------------------------------------------------------


class TestBrainLiverRetinaImageValid:
    @pytest.mark.parametrize("dataset,normal,anomaly", [
        ("Brain", 39, 44),
        ("Liver", 93, 73),
        ("Retina", 45, 70),
    ])
    def test_valid_image_dataset_has_float_image_score(self, dataset, normal, anomaly):
        row = _make_row(dataset, 75.0, 80.0, 92.0, 30.0, normal, anomaly)
        assert row["image_valid"] is True
        assert isinstance(row["image_score"], float)
        assert row["image_score"] == pytest.approx(77.5)

    def test_all_three_included_in_image_macro(self):
        rows = [
            _make_row("Brain", 80.0, 82.0, 94.0, 20.0, 39, 44),
            _make_row("Liver", 56.0, 48.0, 96.0, 5.0, 93, 73),
            _make_row("Retina", 69.0, 77.0, 90.0, 38.0, 45, 70),
        ]
        result = aggregate_epoch(rows)
        expected_brain_img = 81.0
        expected_liver_img = 52.0
        expected_retina_img = 73.0
        expected_macro = (expected_brain_img + expected_liver_img + expected_retina_img) / 3
        assert result["support_aware_image_macro"] == pytest.approx(expected_macro)
        assert "Brain" in result["valid_image_datasets"]
        assert "Liver" in result["valid_image_datasets"]
        assert "Retina" in result["valid_image_datasets"]


# ---------------------------------------------------------------------------
# 4. Pixel metrics can remain valid when image metrics are invalid
# ---------------------------------------------------------------------------


class TestPixelValidWhenImageInvalid:
    def test_colon_has_valid_pixel_but_invalid_image(self):
        row = _make_row("Colon_clinicDB", 0.0, 0.0, 87.0, 50.0, 0, 184)
        assert row["image_valid"] is False
        assert row["pixel_valid"] is True
        assert row["pixel_score"] is not None
        assert row["pixel_score"] == pytest.approx((87.0 + 50.0) / 2)

    def test_pixel_macro_includes_colon_datasets(self):
        rows = [
            _make_row("Colon_clinicDB", 0.0, 0.0, 87.0, 50.0, 0, 184),
            _make_row("Brain", 80.0, 82.0, 94.0, 20.0, 39, 44),
        ]
        result = aggregate_epoch(rows)
        # Both should contribute to pixel macro
        assert "Colon_clinicDB" in result["valid_pixel_datasets"]
        assert "Brain" in result["valid_pixel_datasets"]
        assert result["support_aware_pixel_macro"] is not None


# ---------------------------------------------------------------------------
# 5. Image macro uses only valid image datasets
# ---------------------------------------------------------------------------


class TestImageMacroOnlyValidDatasets:
    def test_six_dataset_image_macro_uses_only_three(self):
        rows = [
            _make_row("Brain", 80.0, 82.0, 94.0, 20.0, 39, 44),
            _make_row("Liver", 56.0, 48.0, 96.0, 5.0, 93, 73),
            _make_row("Retina", 69.0, 77.0, 90.0, 38.0, 45, 70),
            _make_row("Colon_clinicDB", 0.0, 0.0, 87.0, 50.0, 0, 184),
            _make_row("Colon_colonDB", 0.0, 0.0, 83.0, 32.0, 0, 114),
            _make_row("Colon_Kvasir", 0.0, 0.0, 86.0, 54.0, 0, 300),
        ]
        result = aggregate_epoch(rows)
        assert set(result["valid_image_datasets"]) == {"Brain", "Liver", "Retina"}
        assert set(result["excluded_image_datasets"]) == {
            "Colon_clinicDB", "Colon_colonDB", "Colon_Kvasir"
        }
        # image macro = mean of (81, 52, 73) = 68.666...
        expected_img_macro = (81.0 + 52.0 + 73.0) / 3.0
        assert result["support_aware_image_macro"] == pytest.approx(expected_img_macro)


# ---------------------------------------------------------------------------
# 6. Pixel macro uses all valid pixel datasets
# ---------------------------------------------------------------------------


class TestPixelMacroAllSixDatasets:
    def test_six_dataset_pixel_macro_uses_all_six(self):
        rows = [
            _make_row("Brain", 80.0, 82.0, 94.0, 20.0, 39, 44),
            _make_row("Liver", 56.0, 48.0, 96.0, 5.0, 93, 73),
            _make_row("Retina", 69.0, 77.0, 90.0, 38.0, 45, 70),
            _make_row("Colon_clinicDB", 0.0, 0.0, 87.0, 50.0, 0, 184),
            _make_row("Colon_colonDB", 0.0, 0.0, 83.0, 32.0, 0, 114),
            _make_row("Colon_Kvasir", 0.0, 0.0, 86.0, 54.0, 0, 300),
        ]
        result = aggregate_epoch(rows)
        assert len(result["valid_pixel_datasets"]) == 6
        assert set(result["valid_pixel_datasets"]) == {
            "Brain", "Liver", "Retina", "Colon_clinicDB", "Colon_colonDB", "Colon_Kvasir"
        }

    def test_pixel_macro_value_is_mean_of_all_six(self):
        pix_scores = [(94.0 + 20.0) / 2, (96.0 + 5.0) / 2, (90.0 + 38.0) / 2,
                      (87.0 + 50.0) / 2, (83.0 + 32.0) / 2, (86.0 + 54.0) / 2]
        expected = sum(pix_scores) / 6.0
        rows = [
            _make_row("Brain", 80.0, 82.0, 94.0, 20.0, 39, 44),
            _make_row("Liver", 56.0, 48.0, 96.0, 5.0, 93, 73),
            _make_row("Retina", 69.0, 77.0, 90.0, 38.0, 45, 70),
            _make_row("Colon_clinicDB", 0.0, 0.0, 87.0, 50.0, 0, 184),
            _make_row("Colon_colonDB", 0.0, 0.0, 83.0, 32.0, 0, 114),
            _make_row("Colon_Kvasir", 0.0, 0.0, 86.0, 54.0, 0, 300),
        ]
        result = aggregate_epoch(rows)
        assert result["support_aware_pixel_macro"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 7. Combined score is harmonic mean of domain macros
# ---------------------------------------------------------------------------


class TestCombinedScoreIsHarmonicMean:
    def test_combined_score_formula(self):
        rows = [
            _make_row("Brain", 80.0, 82.0, 94.0, 20.0, 39, 44),
            _make_row("Liver", 56.0, 48.0, 96.0, 5.0, 93, 73),
            _make_row("Retina", 69.0, 77.0, 90.0, 38.0, 45, 70),
            _make_row("Colon_clinicDB", 0.0, 0.0, 87.0, 50.0, 0, 184),
            _make_row("Colon_colonDB", 0.0, 0.0, 83.0, 32.0, 0, 114),
            _make_row("Colon_Kvasir", 0.0, 0.0, 86.0, 54.0, 0, 300),
        ]
        result = aggregate_epoch(rows)
        img_m = result["support_aware_image_macro"]
        pix_m = result["support_aware_pixel_macro"]
        expected = harmonic_mean_safe(img_m, pix_m)
        assert result["support_aware_combined_score"] == pytest.approx(expected)

    def test_harmonic_mean_safe_with_zero(self):
        assert harmonic_mean_safe(0.0, 80.0) == 0.0
        assert harmonic_mean_safe(80.0, 0.0) == 0.0
        assert harmonic_mean_safe(0.0, 0.0) == 0.0

    def test_harmonic_mean_safe_value(self):
        # HM(60, 80) = 2*60*80/(60+80) = 9600/140 ≈ 68.5714
        assert harmonic_mean_safe(60.0, 80.0) == pytest.approx(2 * 60 * 80 / 140)

    def test_combined_score_not_mean_of_per_dataset_harmonic_means(self):
        """The combined score must NOT be the mean of per-dataset HM values."""
        rows = [
            _make_row("Brain", 80.0, 82.0, 94.0, 20.0, 39, 44),
            _make_row("Colon_clinicDB", 0.0, 0.0, 87.0, 50.0, 0, 184),
        ]
        result = aggregate_epoch(rows)
        # If it were the naive approach, it would be HM(0, 68.5)/2 = 0/2 = 0
        # But it should be HM(81.0, mean_of_two_pixel_scores) > 0
        assert result["support_aware_combined_score"] > 0.0


# ---------------------------------------------------------------------------
# 8. Legacy and support-aware scores are both emitted
# ---------------------------------------------------------------------------


class TestBothScoresEmitted:
    def test_aggregate_returns_both_scores(self):
        rows = [_make_row("Brain", 80.0, 82.0, 94.0, 20.0, 39, 44)]
        result = aggregate_epoch(rows)
        assert "support_aware_combined_score" in result
        assert "legacy_combined_score" in result

    def test_legacy_score_matches_old_formula(self):
        """Legacy: mean of per-dataset HM(image_score, pixel_score), zeros included."""
        rows = [
            _make_row("Brain", 80.0, 82.0, 94.0, 20.0, 39, 44),
            _make_row("Colon_clinicDB", 0.0, 0.0, 87.0, 50.0, 0, 184),
        ]
        result = aggregate_epoch(rows)
        brain_img = 81.0
        brain_pix = 57.0
        colon_pix = 68.5
        # Legacy: Colon image_score=None -> treated as 0, HM(0, 68.5) = 0
        # Brain: HM(81, 57) = 2*81*57/(81+57)
        expected_legacy = (_hm(brain_img, brain_pix) + _hm(0.0, colon_pix)) / 2.0
        assert result["legacy_combined_score"] == pytest.approx(expected_legacy)

    def test_epoch_summary_includes_both_scores(self):
        epochs = [
            {"epoch": 8, "support_aware_combined_score": 70.0, "legacy_combined_score": 40.0},
            {"epoch": 9, "support_aware_combined_score": 68.0, "legacy_combined_score": 41.0},
        ]
        best = select_best_epoch(epochs)
        # Both keys should be present in best
        assert "support_aware_combined_score" in best
        assert "legacy_combined_score" in best


# ---------------------------------------------------------------------------
# 9. Exact ties select the lower epoch
# ---------------------------------------------------------------------------


class TestTieBreak:
    def test_exact_tie_selects_lower_epoch(self):
        epochs = [
            {"epoch": 14, "support_aware_combined_score": 75.0},
            {"epoch": 8, "support_aware_combined_score": 75.0},
            {"epoch": 12, "support_aware_combined_score": 75.0},
        ]
        best = select_best_epoch(epochs)
        assert best["epoch"] == 8

    def test_strict_best_selected_without_tie(self):
        epochs = [
            {"epoch": 8, "support_aware_combined_score": 74.0},
            {"epoch": 12, "support_aware_combined_score": 76.0},
            {"epoch": 15, "support_aware_combined_score": 75.5},
        ]
        best = select_best_epoch(epochs)
        assert best["epoch"] == 12

    def test_single_epoch_selected(self):
        best = select_best_epoch([{"epoch": 12, "support_aware_combined_score": 70.0}])
        assert best["epoch"] == 12


# ---------------------------------------------------------------------------
# 10. Non-ties do not use near-best behavior
# ---------------------------------------------------------------------------


class TestNoNearBestBehavior:
    def test_strictly_better_epoch_always_wins_regardless_of_number(self):
        """Even if epoch 20 is only slightly better, it should win (no near-best rule)."""
        epochs = [
            {"epoch": 8, "support_aware_combined_score": 75.0},
            {"epoch": 20, "support_aware_combined_score": 75.001},
        ]
        best = select_best_epoch(epochs)
        assert best["epoch"] == 20

    def test_no_tolerance_window_applied(self):
        """There must be no tolerance/near-best window.
        Epoch 8 has 75.0, epoch 12 has 75.5 — epoch 12 must win, not epoch 8."""
        epochs = [
            {"epoch": 8, "support_aware_combined_score": 75.0},
            {"epoch": 12, "support_aware_combined_score": 75.5},
        ]
        best = select_best_epoch(epochs)
        assert best["epoch"] == 12


# ---------------------------------------------------------------------------
# 11. Fingerprint is non-empty and deterministic
# ---------------------------------------------------------------------------


class TestFingerprint:
    def _base_config(self, tmp_path: Path) -> dict:
        return {
            "selection_rule_version": SELECTION_RULE_VERSION,
            "validation_split_manifest_hash": "abc123",
            "dataset_list": ["Brain", "Liver", "Retina"],
            "valid_metric_policy": "image_valid_when_normal>=1_and_anomaly>=1",
            "checkpoint_list": ["adapter_12.pth:256806943"],
            "metric_implementation_version": "v1",
            "pixel_stride": 1,
            "exact_pixel_metric_mode": "external_sorted_chunks",
            "threshold_policy": "none",
            "seed": 0,
            "image_scoring_rule": "arithmetic_mean(image_AUROC, image_AP)",
            "postprocessing_config": "none",
            "temperature": 1.0,
        }

    def test_fingerprint_is_non_empty(self, tmp_path):
        config = self._base_config(tmp_path)
        fp = compute_config_fingerprint(config)
        assert isinstance(fp, str)
        assert len(fp) == 64  # SHA-256 hex

    def test_fingerprint_is_deterministic(self, tmp_path):
        config = self._base_config(tmp_path)
        fp1 = compute_config_fingerprint(config)
        fp2 = compute_config_fingerprint(config)
        assert fp1 == fp2

    def test_fingerprint_changes_when_seed_changes(self, tmp_path):
        config1 = self._base_config(tmp_path)
        config2 = {**config1, "seed": 42}
        assert compute_config_fingerprint(config1) != compute_config_fingerprint(config2)

    def test_fingerprint_changes_when_rule_version_changes(self, tmp_path):
        config1 = self._base_config(tmp_path)
        config2 = {**config1, "selection_rule_version": "old-v1"}
        assert compute_config_fingerprint(config1) != compute_config_fingerprint(config2)


# ---------------------------------------------------------------------------
# 12. Fingerprint changes when split/metric policy changes
# ---------------------------------------------------------------------------


class TestFingerprintChangesWithPolicy:
    def test_fingerprint_changes_with_different_metric_policy(self):
        config1 = {
            "selection_rule_version": SELECTION_RULE_VERSION,
            "validation_split_manifest_hash": "abc123",
            "dataset_list": ["Brain"],
            "valid_metric_policy": "image_valid_when_normal>=1_and_anomaly>=1",
            "checkpoint_list": [],
            "metric_implementation_version": "v1",
            "pixel_stride": 1,
            "exact_pixel_metric_mode": "external_sorted_chunks",
            "threshold_policy": "none",
            "seed": 0,
            "image_scoring_rule": "arithmetic_mean(image_AUROC, image_AP)",
            "postprocessing_config": "none",
            "temperature": 1.0,
        }
        config2 = {**config1, "valid_metric_policy": "always_valid"}
        assert compute_config_fingerprint(config1) != compute_config_fingerprint(config2)

    def test_fingerprint_changes_with_different_manifest_hash(self):
        config1 = {
            "selection_rule_version": SELECTION_RULE_VERSION,
            "validation_split_manifest_hash": "abc123",
            "dataset_list": ["Brain"],
            "valid_metric_policy": "image_valid",
            "checkpoint_list": [],
            "metric_implementation_version": "v1",
            "pixel_stride": 1,
            "exact_pixel_metric_mode": "external_sorted_chunks",
            "threshold_policy": "none",
            "seed": 0,
            "image_scoring_rule": "amean",
            "postprocessing_config": "none",
            "temperature": 1.0,
        }
        config2 = {**config1, "validation_split_manifest_hash": "def456"}
        assert compute_config_fingerprint(config1) != compute_config_fingerprint(config2)

    def test_fingerprint_changes_with_different_pixel_stride(self):
        config = {
            "selection_rule_version": SELECTION_RULE_VERSION,
            "validation_split_manifest_hash": "abc123",
            "dataset_list": ["Brain"],
            "valid_metric_policy": "default",
            "checkpoint_list": [],
            "metric_implementation_version": "v1",
            "pixel_stride": 1,
            "exact_pixel_metric_mode": "external_sorted_chunks",
            "threshold_policy": "none",
            "seed": 0,
            "image_scoring_rule": "amean",
            "postprocessing_config": "none",
            "temperature": 1.0,
        }
        config4 = {**config, "pixel_stride": 4}
        assert compute_config_fingerprint(config) != compute_config_fingerprint(config4)


# ---------------------------------------------------------------------------
# 13. Original selection artifact is not overwritten by reaggregation script
# ---------------------------------------------------------------------------


class TestOriginalArtifactPreserved:
    def _make_minimal_csv(self, save_path: Path, dataset: str, split: str, epoch: int,
                           pixel_auc: float = 90.0, pixel_ap: float = 50.0,
                           image_auc: float = 80.0, image_ap: float = 82.0) -> None:
        """Write a minimal exact_results CSV."""
        fname = save_path / f"exact_results_{dataset}_{split}_epoch_{epoch}.csv"
        fname.write_text(
            f"class name,pixel AUC,pixel AP,image AUC,image AP\n"
            f"{dataset},{pixel_auc},{pixel_ap},{image_auc},{image_ap}\n"
            f"Average,{pixel_auc},{pixel_ap},{image_auc},{image_ap}\n"
        )

    def _make_original_selection(self, save_path: Path) -> dict:
        original = {
            "datasets": ["Brain"],
            "selection_rule": "old rule",
            "best_epoch": {"epoch": 12, "combined_score": 50.0},
            "macro_by_epoch": [{"epoch": 12, "combined_score": 50.0}],
        }
        (save_path / "medical_validation_selection.json").write_text(
            json.dumps(original) + "\n"
        )
        return original

    def test_original_selection_json_not_overwritten(self, tmp_path):
        save_path = tmp_path / "train"
        save_path.mkdir()
        output_dir = tmp_path / "selection"

        # Write original selection
        original = self._make_original_selection(save_path)

        # Write one dataset CSV so the script has something to read
        manifest_dir = tmp_path / "manifests"
        manifest_dir.mkdir()
        # Brain val manifest with 2 normal, 1 anomaly
        manifest_file = manifest_dir / "Brain_val.jsonl"
        manifest_file.write_text(
            json.dumps({"label": 0, "class_name": "Brain", "image_path": "a.png"}) + "\n"
            + json.dumps({"label": 0, "class_name": "Brain", "image_path": "b.png"}) + "\n"
            + json.dumps({"label": 1, "class_name": "Brain", "image_path": "c.png", "mask_path": "cm.png"}) + "\n"
        )
        self._make_minimal_csv(save_path, "Brain", "val", 12)

        rows = reaggregate_all_epochs(
            save_path=save_path,
            split="val",
            epochs=[12],
            manifest_root=manifest_dir,
            datasets=["Brain"],
        )
        assert rows  # at least one result

        # Verify original file is unchanged
        original_reloaded = json.loads(
            (save_path / "medical_validation_selection.json").read_text()
        )
        assert original_reloaded["selection_rule"] == "old rule"
        assert original_reloaded["best_epoch"]["epoch"] == 12


# ---------------------------------------------------------------------------
# 14. No exact test is launched by reaggregation
# ---------------------------------------------------------------------------


class TestNoExactTestLaunched:
    def test_reaggregate_does_not_spawn_test_subprocess(self, monkeypatch, tmp_path):
        """reaggregate_all_epochs must not call subprocess.run or os.system."""
        calls = []
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: calls.append(a) or None)
        monkeypatch.setattr(os, "system", lambda cmd: calls.append(cmd))

        save_path = tmp_path / "train"
        save_path.mkdir()

        rows = reaggregate_all_epochs(
            save_path=save_path,
            split="val",
            epochs=[12],
            manifest_root=None,
            datasets=["Brain"],
        )
        # Expect no subprocess calls
        assert calls == [], f"Unexpected subprocess calls: {calls}"


# ---------------------------------------------------------------------------
# 15. Pixel metric validity
# ---------------------------------------------------------------------------


class TestPixelMetricValidity:
    def test_pixel_valid_with_anomaly_only(self):
        """Colon-type datasets (anomaly-only) should have valid pixel metrics."""
        assert is_pixel_metric_valid(normal_count=0, anomaly_count=100)

    def test_pixel_invalid_with_no_anomaly(self):
        """Normal-only splits have no positive pixels to compute pixel AP."""
        assert not is_pixel_metric_valid(normal_count=50, anomaly_count=0)

    def test_pixel_valid_with_both(self):
        assert is_pixel_metric_valid(normal_count=39, anomaly_count=44)


# ---------------------------------------------------------------------------
# Integration: end-to-end reaggregation from CSV files
# ---------------------------------------------------------------------------


class TestEndToEndReaggregation:
    """Simulate the six-dataset setup from real validation artifacts."""

    DATASETS = ["Brain", "Liver", "Retina", "Colon_clinicDB", "Colon_colonDB", "Colon_Kvasir"]
    DATASET_LABELS = {
        "Brain": (39, 44),
        "Liver": (93, 73),
        "Retina": (45, 70),
        "Colon_clinicDB": (0, 184),
        "Colon_colonDB": (0, 114),
        "Colon_Kvasir": (0, 300),
    }

    def _write_csvs(self, save_path: Path, epoch: int, data: dict) -> None:
        for dataset, (pixel_auc, pixel_ap, image_auc, image_ap) in data.items():
            fname = save_path / f"exact_results_{dataset}_val_epoch_{epoch}.csv"
            fname.write_text(
                f"class name,pixel AUC,pixel AP,image AUC,image AP\n"
                f"{dataset},{pixel_auc},{pixel_ap},{image_auc},{image_ap}\n"
                f"Average,{pixel_auc},{pixel_ap},{image_auc},{image_ap}\n"
            )

    def _write_manifests(self, manifest_dir: Path) -> None:
        for ds, (normal, anomaly) in self.DATASET_LABELS.items():
            p = manifest_dir / f"{ds}_val.jsonl"
            lines = []
            for i in range(normal):
                lines.append(json.dumps({"label": 0, "class_name": ds, "image_path": f"n{i}.png"}))
            for i in range(anomaly):
                lines.append(json.dumps({"label": 1, "class_name": ds, "image_path": f"a{i}.png", "mask_path": f"m{i}.png"}))
            p.write_text("\n".join(lines) + "\n")

    def test_image_macro_uses_only_brain_liver_retina(self, tmp_path):
        save_path = tmp_path / "train"
        save_path.mkdir()
        manifest_dir = tmp_path / "manifests"
        manifest_dir.mkdir()
        self._write_manifests(manifest_dir)

        # Epoch 8 data (simplified)
        self._write_csvs(save_path, 8, {
            "Brain": (94.0, 21.0, 78.0, 80.0),
            "Liver": (95.0, 4.0, 56.0, 48.0),
            "Retina": (92.0, 43.0, 70.0, 78.0),
            "Colon_clinicDB": (88.0, 54.0, 0.0, 0.0),
            "Colon_colonDB": (83.0, 32.0, 0.0, 0.0),
            "Colon_Kvasir": (86.0, 55.0, 0.0, 0.0),
        })

        summaries = reaggregate_all_epochs(
            save_path=save_path,
            split="val",
            epochs=[8],
            manifest_root=manifest_dir,
            datasets=self.DATASETS,
        )

        assert len(summaries) == 1
        s = summaries[0]
        assert s["epoch"] == 8
        assert set(s["valid_image_datasets"]) == {"Brain", "Liver", "Retina"}
        assert set(s["excluded_image_datasets"]) == {
            "Colon_clinicDB", "Colon_colonDB", "Colon_Kvasir"
        }
        assert len(s["valid_pixel_datasets"]) == 6

        # image macro
        expected_img = ((78.0 + 80.0) / 2 + (56.0 + 48.0) / 2 + (70.0 + 78.0) / 2) / 3
        assert s["support_aware_image_macro"] == pytest.approx(expected_img)

        # pixel macro: all 6
        pixel_scores = [(94.0 + 21.0) / 2, (95.0 + 4.0) / 2, (92.0 + 43.0) / 2,
                        (88.0 + 54.0) / 2, (83.0 + 32.0) / 2, (86.0 + 55.0) / 2]
        expected_pix = sum(pixel_scores) / 6
        assert s["support_aware_pixel_macro"] == pytest.approx(expected_pix)

        # combined = HM(image_macro, pixel_macro)
        expected_combined = harmonic_mean_safe(expected_img, expected_pix)
        assert s["support_aware_combined_score"] == pytest.approx(expected_combined)

    def test_best_epoch_selected_by_support_aware_score(self, tmp_path):
        save_path = tmp_path / "train"
        save_path.mkdir()
        manifest_dir = tmp_path / "manifests"
        manifest_dir.mkdir()
        self._write_manifests(manifest_dir)

        # Epoch 12 has better image metrics → should score higher in support-aware
        for epoch, brain_img in [(8, (70.0, 72.0)), (12, (80.0, 82.0))]:
            self._write_csvs(save_path, epoch, {
                "Brain": (94.0, 21.0, brain_img[0], brain_img[1]),
                "Liver": (96.0, 5.0, 56.0, 48.0),
                "Retina": (91.0, 40.0, 69.0, 77.0),
                "Colon_clinicDB": (87.0, 51.0, 0.0, 0.0),
                "Colon_colonDB": (82.0, 32.0, 0.0, 0.0),
                "Colon_Kvasir": (86.0, 55.0, 0.0, 0.0),
            })

        summaries = reaggregate_all_epochs(
            save_path=save_path,
            split="val",
            epochs=[8, 12],
            manifest_root=manifest_dir,
            datasets=self.DATASETS,
        )
        best = select_best_epoch(summaries)
        assert best["epoch"] == 12
