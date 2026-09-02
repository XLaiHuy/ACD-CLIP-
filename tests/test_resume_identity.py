import copy
from pathlib import Path

import pytest
import torch
from torch import nn

from h2_clean.contract import (
    build_full_checkpoint,
    canonical_json_hash,
    current_git_sha,
    make_dataloader_generator,
    parent_scientific_config,
    restore_full_checkpoint,
    validate_resume_identity,
)


class IdentityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_adapter = nn.Linear(3, 3)
        self.text_adapter = nn.Linear(3, 3)


def make_payload():
    model = IdentityModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    config = {
        "epoch": 2,
        "seed": 91,
        "amp": False,
        "tf32_enabled": False,
        "dfg_mode": "attn",
        "dfg_attn_dim": 256,
        "dfg_attn_tau": 8.0,
        "use_ss2d_dfg": True,
        "dfg_beta_schedule": "warmup010",
        "lambda_k": 0.002,
        "batch_size": 6,
        "anchor_gradient_budget": False,
        "anchor_family_budget": 0.10,
    }
    repo = Path(__file__).resolve().parents[1]
    payload = build_full_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        epoch=1,
        global_step=4,
        config=config,
        parent_config=parent_scientific_config(config),
        operational_config={"save_path": "/source", "resume": None},
        repo=repo,
        clip_sha256="clip",
        dataset_manifest_sha256="manifest",
        dataloader_generator=make_dataloader_generator(91),
        anchor=None,
        anchor_lambda=0.0,
        seed=91,
        precision="fp32",
        tf32_enabled=False,
    )
    return payload, config, repo


def validate(payload, config, repo):
    validate_resume_identity(
        payload,
        expected_scientific_config=config,
        expected_parent_config=parent_scientific_config(config),
        expected_epoch=1,
        expected_total_epoch=2,
        expected_seed=91,
        expected_clip_sha256="clip",
        expected_manifest_sha256="manifest",
        expected_git_sha=current_git_sha(repo),
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", 999),
        ("anchor_family_budget", 0.20),
        ("dfg_attn_tau", 4.0),
        ("lambda_k", 0.0),
    ],
)
def test_stale_scientific_resume_rejected(field, value):
    payload, config, repo = make_payload()
    stale = copy.deepcopy(payload)
    stale["resolved_scientific_config"][field] = value
    stale["config_sha256"] = canonical_json_hash(stale["resolved_scientific_config"])
    with pytest.raises(ValueError):
        validate(stale, config, repo)



class NoRestore:
    def load_state_dict(self, *args, **kwargs):
        raise AssertionError("stale resume reached state restoration")


@pytest.mark.parametrize(
    ("field", "value"),
    [("seed", 999), ("dfg_attn_tau", 4.0), ("lambda_k", 0.0)],
)
def test_stale_resume_rejected_before_state_restore(field, value):
    payload, config, repo = make_payload()
    stale = copy.deepcopy(payload)
    stale["resolved_scientific_config"][field] = value
    stale["config_sha256"] = canonical_json_hash(stale["resolved_scientific_config"])
    with pytest.raises(ValueError):
        restore_full_checkpoint(
            stale,
            model=IdentityModel(),
            optimizer=NoRestore(),
            scheduler=NoRestore(),
            scaler=NoRestore(),
            dataloader_generator=None,
            expected_scientific_config=config,
            expected_parent_config=parent_scientific_config(config),
            expected_epoch=1,
            expected_total_epoch=2,
            expected_seed=91,
            expected_clip_sha256="clip",
            expected_manifest_sha256="manifest",
            expected_git_sha=current_git_sha(repo),
        )

def test_stale_manifest_resume_rejected():
    payload, config, repo = make_payload()
    stale = copy.deepcopy(payload)
    stale["dataset_manifest_sha256"] = "wrong-manifest"
    with pytest.raises(ValueError, match="manifest"):
        validate(stale, config, repo)



def test_stale_manifest_rejected_before_state_restore():
    payload, config, repo = make_payload()
    stale = copy.deepcopy(payload)
    stale["dataset_manifest_sha256"] = "wrong-manifest"
    with pytest.raises(ValueError, match="manifest"):
        restore_full_checkpoint(
            stale,
            model=IdentityModel(),
            optimizer=NoRestore(),
            scheduler=NoRestore(),
            scaler=NoRestore(),
            dataloader_generator=None,
            expected_scientific_config=config,
            expected_parent_config=parent_scientific_config(config),
            expected_epoch=1,
            expected_total_epoch=2,
            expected_seed=91,
            expected_clip_sha256="clip",
            expected_manifest_sha256="manifest",
            expected_git_sha=current_git_sha(repo),
        )

def test_stale_epoch_resume_rejected():
    payload, config, repo = make_payload()
    stale = copy.deepcopy(payload)
    stale["epoch"] = 2
    with pytest.raises(ValueError, match="epoch"):
        validate(stale, config, repo)


def test_operational_paths_are_not_scientific_identity():
    payload, config, repo = make_payload()
    payload["resolved_operational_config"]["save_path"] = "/a/different/path"
    payload["resolved_operational_config"]["resume"] = "/another/checkpoint.pth"
    validate(payload, config, repo)


def test_shared_e1_can_fork_only_intervention_fields():
    payload, config, repo = make_payload()
    arm_config = dict(config)
    arm_config.update({
        "use_safe_anchor": True,
        "anchor_lambda": 0.001,
        "anchor_reference_sha256": "e1",
        "anchor_gradient_budget": True,
        "use_cir_training": True,
        "cir_alpha": 0.5,
        "cir_peer_count": 8,
        "cir_spatial_radius": 3,
    })
    validate_resume_identity(
        payload,
        expected_scientific_config=arm_config,
        expected_parent_config=parent_scientific_config(arm_config),
        expected_total_epoch=2,
        expected_seed=91,
        expected_clip_sha256="clip",
        expected_manifest_sha256="manifest",
        expected_git_sha=current_git_sha(repo),
    )
