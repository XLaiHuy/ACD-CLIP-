import json
import random

import numpy as np
import pytest
import torch

from tools.sabra import lab_train as lab


def _checkpoint_payload() -> dict:
    return {
        "epoch": 4,
        "seed": 0,
        "batch_size": 1,
        "grad_accum_steps": 6,
        "image_adapter": {"weight": torch.ones(1)},
        "text_adapter": {"weight": torch.ones(1)},
        "soft_prompt": {"ctx": torch.ones(1)},
        "h6_state_dict": {"router.weight": torch.ones(1)},
        "optimizer_state": {"state": {0: {}}, "param_groups": [{"lr": 0.00032805}]},
        "scheduler_state": {"last_epoch": 4, "step_size": 1, "gamma": 0.9, "_last_lr": [0.00032805]},
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": [torch.get_rng_state()],
        "dataloader_generator_state": torch.Generator().manual_seed(0).get_state(),
        "package_config_sha256": lab.sha256_file(lab.PACKAGE / "TRAIN20E_FINAL_CONFIG.json"),
        "dataset_role_contract_sha256": lab.sha256_file(lab.PACKAGE / "DATASET_ROLE_CONTRACT.json"),
        "phase2b_config": {"h6_prediction_routing": "dense"},
        "git_sha": "fd4a80203bd3224af6bcb6d5ae8bf73e63bec1c4",
    }


def _valid_existing_run(tmp_path):
    base = tmp_path / "runs"
    root = base / "lab20e_b1_accum6_seed0"
    root.mkdir(parents=True)
    (root / "checkpoints").mkdir()
    (root / "logs").mkdir()
    provenance = {
        "git_sha": "fd4a80203bd3224af6bcb6d5ae8bf73e63bec1c4",
        "package_config_sha256": lab.sha256_file(lab.PACKAGE / "TRAIN20E_FINAL_CONFIG.json"),
        "dataset_role_contract_sha256": lab.sha256_file(lab.PACKAGE / "DATASET_ROLE_CONTRACT.json"),
    }
    for name, payload in (
        ("RESOLVED_CONFIG.json", {"original": True}),
        ("GIT_PROVENANCE.json", provenance),
        ("ENVIRONMENT.json", {"original": True}),
        ("DATASET_ROLE_CONTRACT.json", {"original": True}),
        ("ASSET_HASHES.json", {"original": True}),
    ):
        (root / name).write_text(json.dumps(payload), encoding="utf-8")
    (root / "logs" / "train_wrapper.log").write_text("original train log\n", encoding="utf-8")
    checkpoint = root / "checkpoints" / "adapter_4.pth"
    torch.save(_checkpoint_payload(), checkpoint)
    return base, root, checkpoint


def _config():
    return lab.apply_training_overrides(lab.load_config(lab.PACKAGE / "TRAIN20E_FINAL_CONFIG.json"), 1, 6)


def test_new_train_rejects_existing_run_root(tmp_path):
    base, _, _ = _valid_existing_run(tmp_path)
    with pytest.raises(FileExistsError, match="RUN_ID_COLLISION"):
        lab.make_run_root("lab20e_b1_accum6_seed0", base)


def test_resume_accepts_existing_valid_run_and_resolves_epoch_five(tmp_path):
    base, root, checkpoint = _valid_existing_run(tmp_path)
    info = lab.resolve_resume_root("lab20e_b1_accum6_seed0", base, checkpoint, _config())
    assert info["root"] == root
    assert info["epoch"] == 4
    assert info["next_epoch"] == 5
    assert info["batch_size"] == 1
    assert info["grad_accum_steps"] == 6
    assert info["seed"] == 0
    assert info["prediction_routing"] == "dense"
    assert info["expected_primary_lr"] == pytest.approx(0.00032805)


def test_resume_rejects_missing_run_root(tmp_path):
    with pytest.raises(FileNotFoundError, match="RESUME_RUN_ROOT_MISSING"):
        lab.resolve_resume_root("lab20e_b1_accum6_seed0", tmp_path, tmp_path / "adapter_4.pth", _config())


def test_resume_rejects_checkpoint_outside_requested_run_root(tmp_path):
    base, _, _ = _valid_existing_run(tmp_path)
    outside = tmp_path / "outside_adapter_4.pth"
    torch.save(_checkpoint_payload(), outside)
    with pytest.raises(ValueError, match="RESUME_CHECKPOINT_ROOT_MISMATCH"):
        lab.resolve_resume_root("lab20e_b1_accum6_seed0", base, outside, _config())


def test_resume_preserves_initial_provenance_and_train_log(tmp_path, monkeypatch):
    base, root, checkpoint = _valid_existing_run(tmp_path)
    original_files = {
        name: (root / name).read_bytes()
        for name in ("RESOLVED_CONFIG.json", "GIT_PROVENANCE.json", "ENVIRONMENT.json", "DATASET_ROLE_CONTRACT.json", "ASSET_HASHES.json")
    }
    original_log = (root / "logs" / "train_wrapper.log").read_bytes()
    captured = {}

    def fake_run(command, log_path, env):
        captured["command"] = command
        captured["log_path"] = log_path
        return 0

    monkeypatch.setenv("VISA_ROOT", "/tmp/visa")
    monkeypatch.setattr(lab, "run_process", fake_run)
    assert lab.train(lab.PACKAGE / "TRAIN20E_FINAL_CONFIG.json", "lab20e_b1_accum6_seed0", base, resume=checkpoint, batch_size=1, grad_accum_steps=6) == 0
    assert (root / "logs" / "train_wrapper.log").read_bytes() == original_log
    assert all((root / name).read_bytes() == value for name, value in original_files.items())
    assert captured["log_path"].name.startswith("resume_from_epoch_4_")
    assert "--resume" in captured["command"]
    assert captured["command"][captured["command"].index("--resume") + 1] == str(checkpoint.resolve())
    assert captured["command"][captured["command"].index("--save_path") + 1] == str((root / "checkpoints").resolve())
    records = list((root / "resume").glob("RESUME_EPOCH4_*.json"))
    assert len(records) == 1
    record = json.loads(records[0].read_text(encoding="utf-8"))
    assert record["next_epoch"] == 5
    assert record["scientific_formula_change"] is False
