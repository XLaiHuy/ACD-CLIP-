from __future__ import annotations

import inspect

from tools.sabra_v2 import evaluate_region_distill_p29_cached as evaluate
from tools.sabra_v2 import run_p29_sign_guarded_science as runner
from tools.sabra_v2 import train_region_distill_p29_cached as train
from tools.sabra_v2.region_adapter import RegionResidualAdapter
from tools.sabra_v2.student_forward import forward_region_student


def test_scientific_training_parser_locks_the_frozen_p29_schedule() -> None:
    args = train.make_parser().parse_args([
        "--held-class", "candle", "--visa-root", "/data/visa", "--p26-checkpoint", "/m/p26.pt",
        "--clip-asset", "/m/clip.pt", "--cache-root", "/cache", "--output", "/run", "--execution-base-sha", "base",
    ])
    assert (args.epochs, args.batch_size, args.learning_rate, args.seed) == (20, 1, 0.001, 0)


def test_p29_training_uses_only_the_preregistered_objective_and_source_mask() -> None:
    source = inspect.getsource(train)
    assert "p29_sign_guarded_loss" in source
    assert "calculate_seg_loss" not in source
    assert "VisaEvaluationDataset" not in source
    assert "build_region_cache" not in source


def test_p29_prediction_is_gt_free_and_fixed_to_cuda_batch_one_replay() -> None:
    source = inspect.getsource(evaluate)
    assert "P29_IMMUTABLE_HELD_PREDICTIONS_V1" in source
    assert '"gt_used":False' in source
    assert '"mask_reads":0' in source
    args = evaluate.make_parser().parse_args([
        "--held-class", "candle", "--visa-root", "/data/visa", "--p26-checkpoint", "/m/p26.pt",
        "--clip-asset", "/m/clip.pt", "--cache-root", "/cache", "--adapter-checkpoint", "/run/adapter.pt", "--output", "/run/predictions", "--execution-base-sha", "base",
    ])
    assert args.batch_size == 1


def test_one_attempt_runner_has_prediction_gate_before_scoring_and_no_cache_build() -> None:
    source = inspect.getsource(runner)
    assert "P29_ATTEMPT.json" in source
    assert "P29_SCORING_GATE.json" in source
    assert source.index("_prediction_gate") < source.index("score_region_distill_p29_frozen")
    assert "build_region_cache" not in source


def test_p29_uses_the_unchanged_p27_adapter_and_symmetric_deployment_path() -> None:
    source = inspect.getsource(train)
    assert "RegionResidualAdapter" in source
    assert "forward_region_student" in source
    assert set(RegionResidualAdapter().state_dict())


def test_cached_native_probability_is_unchanged_by_p29_student_forward() -> None:
    import torch
    adapter = RegionResidualAdapter()
    for parameter in adapter.parameters():
        parameter.data.zero_()
    seg = torch.zeros((3, 1, 1369, 768), dtype=torch.float32)
    native = torch.randn((3, 1, 1369, 2), dtype=torch.float32)
    output = forward_region_student(adapter, seg, native)
    assert torch.equal(output.native_probability, output.deployed_probability)


def test_only_adapter_is_optimized_and_cached_path_has_no_clip_or_phase2b_forward() -> None:
    source = inspect.getsource(train)
    assert "torch.optim.AdamW(adapter.parameters()" in source
    for forbidden in ("forward_phase2b", "open_clip", "load_clip", "Phase2B(", "clip_model"):
        assert forbidden not in source


def test_checkpoint_reload_order_and_cache_firewalls_reuse_existing_fail_closed_contracts() -> None:
    source = inspect.getsource(train) + inspect.getsource(evaluate)
    assert "CachedSourceDataset" in source
    assert "TierADataset" in source
    assert "shuffle=False" in inspect.getsource(evaluate)
    assert "mvtec" not in source.lower()
    assert "medical" not in source.lower()
