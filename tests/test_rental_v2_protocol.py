from pathlib import Path

ROOT = Path(__file__).parents[1]
RENTAL = ROOT / "handoff" / "rental_20260911"


def test_v2_launchers_use_fresh_e1_as_anchor_and_resume_state():
    for name in ("run_fresh_visa_source_v2.sh", "run_fresh_mvtec_source_v2.sh"):
        source = (RENTAL / name).read_text()
        assert "--epoch 1" in source
        assert "--epoch 20" in source
        assert "--protocol_horizon 20" in source
        assert "--resume \"${RUN_ROOT}/adapter_1.pth\"" in source
        assert "--anchor_reference_path \"${RUN_ROOT}/adapter_1.pth\"" in source
        assert "h2_clean_factorial_e20_20260902_ampfix/shared_e1" not in source


def test_protocol_distinguishes_anchor_from_resume_and_freezes_tta_policy():
    protocol = (RENTAL / "PROTOCOL.md").read_text()
    assert "Anchor reference and training resume state are separate" in protocol
    assert "identity,hflip,vflip,hvflip" in protocol
    assert "TTA is never used to select an epoch" in protocol
    assert "DELTA_TTA.json" in protocol
    assert "summarize_delta_tta.py" in (RENTAL / "eval_selected_checkpoint_v2.sh").read_text()
    assert "select_medical_epoch_v2.py" in (RENTAL / "run_medical_epoch_evaluation_v2.sh").read_text()
    assert (RENTAL / "select_medical_epoch_v2.py").is_file()
    assert "tmux" in (RENTAL / "launch_training_tmux_v2.sh").read_text()
    assert "python_assignment" in (RENTAL / "launch_training_tmux_v2.sh").read_text()
    assert "find -L" in (RENTAL / "common_v2.sh").read_text()
    assert "record_run_environment" in (RENTAL / "common_v2.sh").read_text()
    assert "find -L" in (RENTAL / "preflight_v2.sh").read_text()
    assert "peak_cuda_memory_allocated_gb=" in (RENTAL / "smoke_e1_e2_continuation.sh").read_text()


def test_data_linker_does_not_copy_or_replace_non_symlinks():
    source = (RENTAL / "prepare_data_links.sh").read_text()
    assert "ln -sfn" in source
    assert "Refusing to replace existing non-symlink" in source
    assert "rsync" not in source


def test_medical_selector_uses_ap_auroc_then_earlier_epoch(tmp_path):
    from handoff.rental_20260911.select_medical_epoch_v2 import DATASETS, select_epoch

    for index, dataset in enumerate(DATASETS):
        dataset_root = tmp_path / dataset
        dataset_root.mkdir()
        lines = ["args: {'evaluator_mode': 'benchmark_exact', 'pixel_stride': 1}"]
        for epoch in range(1, 21):
            pixel_ap = 99.0 if epoch in (7, 8, 9) else float(epoch)
            pixel_auroc = (20.0 + epoch) if epoch != 8 else 42.0 + index
            lines.extend([
                f"load model from epoch {epoch}",
                f"final results: Average {pixel_auroc} {pixel_ap} 0.0 0.0",
            ])
        (dataset_root / "test.log").write_text("\n".join(lines) + "\n")

    rows, winner = select_epoch(tmp_path)
    assert len(rows) == 20
    assert winner["epoch"] == 8
