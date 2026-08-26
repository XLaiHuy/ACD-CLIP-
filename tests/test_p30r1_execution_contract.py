from __future__ import annotations

import inspect

from tools.sabra_v2 import run_p30r1_engineering as runner
from tools.sabra_v2 import train_region_distill_p30r1_cached as train
from tools.sabra_v2.p30r1_contract import P30R1_UUID


def test_engineering_runner_has_no_scientific_stage_or_marker_mode() -> None:
    source = inspect.getsource(runner)
    assert "PASS_TO_STAGE2_PROTOCOL" in source
    assert "P30R1_EXECUTION_MARKER" not in source
    assert "P30R1 scientific stage" not in source
    assert "score_region_distill" not in source
    assert "build_region_cache" not in source


def test_trainer_parser_locks_schedule_and_exposes_only_engineering_stages() -> None:
    args = train.make_parser().parse_args([
        "--held-class", "candle",
        "--output", "/tmp/p30r1",
        "--execution-base-sha", "base",
        "--preregistration-sha", "prereg",
        "--max-steps", "1",
    ])
    assert args.p30r1_uuid == P30R1_UUID
    assert (args.epochs, args.batch_size, args.learning_rate, args.seed) == (20, 1, 0.001, 0)
    assert args.stage == "engineering_smoke"
    assert args.warmup_steps == 0
    choices = train.make_parser()._actions[[action.dest for action in train.make_parser()._actions].index("stage")].choices
    assert tuple(choices) == ("engineering_smoke", "engineering_microprofile", "engineering_profile")


def test_trainer_uses_cached_source_inputs_and_exact_production_objective() -> None:
    source = inspect.getsource(train)
    assert "p30r1_teacher_relative_components" in source
    assert "p30_directional_loss" not in source
    assert "p29_sign_guarded_loss" not in source
    assert "load_source_mask=False" in source
    assert "load_native_logits=False" in source
    assert "CachedSourceDataset" in source
    assert '"objective_count": 1' in source
    assert '"teacher_trainable": False' in source
    assert "torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate)" in source


def test_runner_invokes_the_new_cached_trainer_and_reload_probe() -> None:
    source = inspect.getsource(runner)
    assert '"tools.sabra_v2.train_region_distill_p30r1_cached"' in source
    assert "_reload_checkpoint_and_probe" in source
    assert "P30R1_ENGINEERING_QUALIFICATION.json" in source


def test_optional_training_cache_fields_can_avoid_unused_native_and_mask_tensors() -> None:
    source = inspect.getsource(train)
    assert "source_mask_loaded" in source
    assert "native_logits_loaded" in source
