from __future__ import annotations

import inspect

from tools.sabra_v2 import run_p30r1_scientific_stage2 as runner


def test_stage2_runner_is_candle_only_and_has_no_stage3_or_full_launch() -> None:
    source = inspect.getsource(runner)
    assert runner.P30R1_STAGE2_CLASS == "candle"
    assert "stage3_subset" not in source
    assert "full_12_class" not in source
    assert "P30R1_EXECUTION_MARKER" not in source
    assert '"automatic_rerun": False' in source


def test_stage2_runner_reuses_frozen_trainer_and_exact_schedule() -> None:
    source = inspect.getsource(runner)
    assert '"tools.sabra_v2.train_region_distill_p30r1_cached"' in source
    assert "1962 * P30R1_EPOCHS" in source
    assert "--epochs" in source
    assert "--batch-size" in source
    assert "--learning-rate" in source
    assert "--seed" in source
    assert "P30R1_FORMULATION_HASH" in source
    assert "P30R1_SCORING_GATE.json" in source


def test_stage2_parser_has_no_tunable_scientific_parameters() -> None:
    args = runner.make_parser().parse_args([])
    assert args.output_root == runner.DEFAULT_OUTPUT_ROOT
    assert not hasattr(args, "seed")
    assert not hasattr(args, "epochs")
    assert not hasattr(args, "batch_size")
    assert not hasattr(args, "learning_rate")


def test_prediction_and_scoring_boundaries_are_explicit() -> None:
    source = inspect.getsource(runner)
    assert "PREDICTIONS_FROZEN_BEFORE_SCORING" in source
    assert "VisaEvaluationDataset" in source
    assert "held_gt_reads_before_prediction_freeze" in source
    assert "held_mask_file_reads_after_prediction_freeze" in source
