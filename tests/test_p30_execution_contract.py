from __future__ import annotations

import inspect

from tools.sabra_v2 import evaluate_region_distill_p30_cached as evaluate
from tools.sabra_v2 import run_p30_directional_distillation as runner
from tools.sabra_v2 import train_region_distill_p30_cached as train
from tools.sabra_v2.p30_contract import P30_CLASS_ORDER, P30_UUID
from tools.sabra_v2.region_adapter import RegionResidualAdapter


def _common_train_args() -> list[str]:
    return [
        "--held-class",
        "candle",
        "--visa-root",
        "/data/visa",
        "--p26-checkpoint",
        "/m/p26.pt",
        "--clip-asset",
        "/m/clip.pt",
        "--cache-root",
        "/cache",
        "--output",
        "/run",
        "--p30-execution-base-sha",
        "base",
        "--p30-prereg-sha",
        "prereg",
    ]


def test_p30_training_parser_locks_the_preregistered_scientific_schedule() -> None:
    args = train.make_parser().parse_args(_common_train_args())
    assert (args.epochs, args.batch_size, args.learning_rate, args.seed) == (20, 1, 0.001, 0)
    assert args.p30_uuid == P30_UUID


def test_p30_training_has_one_directional_objective_and_cached_source_inputs() -> None:
    source = inspect.getsource(train)
    assert "p30_directional_loss" in source
    assert "p29_sign_guarded_loss" not in source
    assert "calculate_seg_loss" not in source
    assert "VisaEvaluationDataset" not in source
    assert "forward_phase2b" not in source
    assert "torch.optim.AdamW(adapter.parameters()" in source
    assert "CachedSourceDataset" in source
    assert '"teacher_trainable": False' in source


def test_p30_prediction_is_gt_free_and_batch_one() -> None:
    source = inspect.getsource(evaluate)
    assert "P30_IMMUTABLE_HELD_PREDICTIONS_V1" in source
    assert '"gt_used": False' in source
    assert '"mask_reads": 0' in source
    args = evaluate.make_parser().parse_args([
        "--held-class",
        "candle",
        "--visa-root",
        "/data/visa",
        "--p26-checkpoint",
        "/m/p26.pt",
        "--clip-asset",
        "/m/clip.pt",
        "--cache-root",
        "/cache",
        "--adapter-checkpoint",
        "/run/adapter.pt",
        "--output",
        "/run/predictions",
        "--p30-execution-base-sha",
        "base",
        "--p30-prereg-sha",
        "prereg",
    ])
    assert args.batch_size == 1


def test_p30_runner_has_fixed_stages_and_prediction_barrier() -> None:
    source = inspect.getsource(runner)
    assert tuple(runner.SUBSET_CLASSES) == ("candle", "chewinggum", "macaroni2", "pcb3")
    assert tuple(runner.FULL_CLASSES) == P30_CLASS_ORDER
    assert "P30_EXECUTION_MARKER.json" in source
    assert "P30_SCORING_GATE.json" in source
    assert source.index("_prediction_gate") < source.index("score_region_distill_p30")
    assert "build_region_cache" not in source
    assert "run_region_distill_science" not in source


def test_p30_uses_the_unchanged_adapter_architecture() -> None:
    assert set(RegionResidualAdapter().state_dict())
    source = inspect.getsource(train)
    assert "RegionResidualAdapter" in source
    assert "adapter(seg_features)" in source
