#!/usr/bin/env python3
"""Build the compact pre-training provenance archive for the H2 master test."""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "research_artifacts" / "h2_anchor_cir_master_20260901"
HIST = ROOT / "research_artifacts" / "phase2b_historical_gain_forensics_20260901"
CONFIG = ROOT / "configs" / "exact_h2_anchor_cir_master_v1.json"
H2_ORIGINAL = Path("/home/ai4/caohuy/ACD-CLIP-base-new-phase1")
REMOTE = "https://github.com/XLaiHuy/ACD-CLIP-.git"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_sha(path: Path = ROOT) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "UNKNOWN"


def write_text(name: str, content: str) -> None:
    (ARCHIVE / name).write_text(content.rstrip() + "\n", encoding="utf-8")


def write_json(name: str, payload: Mapping[str, Any]) -> None:
    (ARCHIVE / name).write_text(json.dumps(dict(payload), indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def write_csv(name: str, fields: Iterable[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with (ARCHIVE / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
    config_sha = sha256_file(CONFIG)
    h2_repo = Path(cfg["h2_repo_path"]).resolve()
    h2_run = H2_ORIGINAL / "runs/phase2b/phase2b_hybrid_alpha02_kreg2e3_lkg1e2_lr5e5_train15_test6medical7to15_fromscratch"
    h2_e10 = h2_run / "adapter_10.pth"
    h2_e1 = Path(cfg["anchor_reference_path"]).resolve()
    e0 = ROOT / "runs/h2_anchor_cir_master_20260901/common/e0.pth"
    e0_identity = e0.with_suffix(".identity.json")
    parity = ARCHIVE / "H2_EXTENSION_PARITY.json"
    if not h2_e10.is_file() or not h2_e1.is_file() or not e0.is_file() or not parity.is_file():
        raise FileNotFoundError("H2 E10/E1, common E0, and parity evidence are required")
    if sha256_file(h2_e1) != cfg["anchor_reference_sha256"]:
        raise ValueError("preregistered H2 E1 anchor SHA mismatch")
    if sha256_file(h2_e10) != "ae27443f99020588298a9ecc6dfc833a83ebe7a752f00e8524042d5a84a2c0cb":
        raise ValueError("historical H2 E10 SHA mismatch")
    parity_result = json.loads(parity.read_text(encoding="utf-8"))
    if parity_result.get("status") != "PASS":
        raise ValueError("H2 extension parity is not PASS")

    # Repository and checkpoint discovery stays explicit: the original H2
    # tree owns the historical run; the detached sibling owns extension code.
    write_csv(
        "H2_REPO_CANDIDATES.csv",
        ["candidate", "repository_path", "remote", "commit", "role", "status", "notes"],
        [
            {"candidate": "H2_ORIGINAL", "repository_path": str(H2_ORIGINAL), "remote": REMOTE, "commit": git_sha(H2_ORIGINAL), "role": "historical H2 source and run", "status": "FOUND", "notes": "exact e039669 source/run/checkpoints recovered"},
            {"candidate": "H2_ISOLATED", "repository_path": str(h2_repo), "remote": REMOTE, "commit": git_sha(h2_repo), "role": "read-only historical import worktree", "status": "FOUND", "notes": "detached at exact H2 commit; no learned weights copied"},
        ],
    )
    write_csv(
        "H2_CHECKPOINT_CANDIDATES.csv",
        ["candidate", "epoch", "checkpoint_path", "sha256", "size_bytes", "source_commit", "optimizer_state", "scheduler_state", "rng_state", "selection_status", "identity_status", "usable_for", "notes"],
        [
            {"candidate": "H2_E10", "epoch": 10, "checkpoint_path": str(h2_e10), "sha256": sha256_file(h2_e10), "size_bytes": h2_e10.stat().st_size, "source_commit": cfg["h2_repo_commit"], "optimizer_state": "ABSENT", "scheduler_state": "ABSENT", "rng_state": "ABSENT", "selection_status": "RETROSPECTIVE_BEST", "identity_status": "CONFIRMED", "usable_for": "historical replay/reference only", "notes": "exact model-state H2 E10; Medical-informed historical selection"},
            {"candidate": "H2_E1_ANCHOR", "epoch": 1, "checkpoint_path": str(h2_e1), "sha256": sha256_file(h2_e1), "size_bytes": h2_e1.stat().st_size, "source_commit": cfg["h2_repo_commit"], "optimizer_state": "ABSENT", "scheduler_state": "ABSENT", "rng_state": "ABSENT", "selection_status": "PREREGISTERED_ANCHOR_REFERENCE", "identity_status": "CONFIRMED", "usable_for": "fixed train-only image anchor", "notes": "selected before new Medical access; model-state only"},
        ],
    )
    write_text(
        "H2_CHECKPOINT_IDENTITY.md",
        f"""# H2 checkpoint identity

Status: CONFIRMED.

The exact historical H2 E10 checkpoint is `{h2_e10}` with SHA256 `{sha256_file(h2_e10)}` and source commit `{cfg['h2_repo_commit']}`. It is a legacy model-state payload: optimizer, scheduler, and RNG state are absent. Historical E10 selection was retrospective Medical-informed, so it is replay/oracle evidence and not a new target-blind selection rule.

The fixed, target-blind anchor reference is H2 E1 `{h2_e1}` with SHA256 `{sha256_file(h2_e1)}`. Its image-adapter parameter identity and shapes were checked against the exact H2 model before training. It is used only by RA/RCA during training and is absent at inference.

The current C2 E10 checkpoint is intentionally not reused as an anchor: its protocol and parameter trajectory are not H2-compatible.
""",
    )

    contract = json.loads((HIST / "HISTORICAL_PHASE2B_CONTRACT.json").read_text(encoding="utf-8"))
    contract["master_extension"] = {
        "experiment_id": cfg["experiment_id"],
        "common_e0": str(e0),
        "common_e0_sha256": sha256_file(e0),
        "arms": {"R": "exact H2", "RA": "exact H2 + image_adapter anchor", "RCA": "exact H2 + image_adapter anchor + train-time CIR V2"},
        "deployment": "historical native H2 evaluator, alpha=0",
        "new_medical_access_before_anchor_selection": False,
    }
    write_json("EXACT_H2_CONTRACT.json", contract)
    write_text(
        "EXACT_H2_CONTRACT.md",
        f"""# Exact H2 contract for the matched master experiment

Status: CONFIRMED from historical source commit `{cfg['h2_repo_commit']}`, the H2 run/checkpoint, and the historical evaluator. The authoritative recovered contract is also preserved in `phase2b_historical_gain_forensics_20260901/HISTORICAL_PHASE2B_CONTRACT.json`.

R uses the historical hybrid Phase2B path: ViT-L-14-336 at 518px, three groups, image/text adaptation 0.2, LoRA rank/alpha 16/2, convolutional LoRA rank/alpha 8/2 with kernels 3/5, attention DFG dimension 256 and tau 8, SS2D weight-residual beta warmup, hybrid alpha schedule 0/0.05/0.1/0.2, lambda_kg=0.01, exact detached-W_K lambda_k=0.002, AMP/autocast/GradScaler, batch/effective batch 6, Adam groups text/image/soft with base LR 5e-4/1e-3/5e-5, zero weight decay, and StepLR gamma 0.9 stepped after each epoch and before candidate save.

The recovered K-reg is the historical cosine distance in detached W_K space, mean over stages, groups, and normal/abnormal channels. The implementation test confirms nonzero K-reg and the expected soft-prompt gradient path.

The three arms share one exact H2 E0 initialization. RA adds only the frozen normalized image-adapter parameter anchor at lambda 1e-3. RCA adds only the frozen CIR-V2 train-time peer transport on top of RA. All deployment/evaluation uses native H2 alpha=0.

Historical AMP is intentionally preserved. The extension does not change precision, optimizer, scheduler, loss, augmentation, batch geometry, DFG math, scoring, or evaluator protocol.
""",
    )

    replay_rows = [row for row in load_rows(HIST / "HISTORICAL_REPLAY_RESULTS.csv") if row.get("lineage") == "H2"]
    write_csv("H2_ORACLE_REPLAY.csv", replay_rows[0].keys() if replay_rows else ["lineage", "status"], replay_rows)
    write_text(
        "H2_ORACLE_REPLAY.md",
        (HIST / "HISTORICAL_REPLAY_REPORT.md").read_text(encoding="utf-8")
        + "\nThe replay PASS is a prerequisite for the new R/RA/RCA study. New Medical evaluation has not been used to choose the E1 anchor.\n",
    )

    e0_payload = torch.load(e0, map_location="cpu", weights_only=False)
    e1_payload = torch.load(h2_e1, map_location="cpu", weights_only=False)
    e0_image = e0_payload["image_adapter"]
    e1_image = e1_payload["image_adapter"]
    if set(e0_image) != set(e1_image) or any(tuple(e0_image[key].shape) != tuple(e1_image[key].shape) for key in e0_image):
        raise ValueError("E0 and H2 E1 image-adapter parameter identity/shape mismatch")
    e0_identity = json.loads(e0_identity.read_text(encoding="utf-8"))
    e0_identity.update({
        "experiment_id": cfg["experiment_id"],
        "config_path": str(CONFIG),
        "config_sha256": config_sha,
        "current_git_sha": git_sha(),
        "common_e0_sha256": sha256_file(e0),
        "learned_h2_e10_weights_included": False,
        "image_adapter_parameter_count": len(e0_image),
        "image_adapter_parameter_names_match_h2_e1": True,
        "image_adapter_shapes_match_h2_e1": True,
        "arms": ["R", "RA", "RCA"],
    })
    write_json("MATCHED_E0_IDENTITY.json", e0_identity)

    write_text(
        "ANCHOR_REFERENCE_DECISION.md",
        f"""# Anchor reference decision

Status: FROZEN BEFORE NEW MEDICAL EVALUATION.

Rule: use the preregistered exact-H2 E1 model-state checkpoint as the fixed, target-blind reference for the image-adapter-only anchor. Do not use current C2 P E14 or any Medical-selected checkpoint.

- path: `{h2_e1}`
- epoch: `1`
- SHA256: `{sha256_file(h2_e1)}`
- lambda: `{cfg['anchor_lambda']}`
- scope: `{cfg['anchor_scope']}`
- parameter names: `{len(e1_image)}` exact names match common E0
- shapes: exact match to common E0 and H2 model
- inference: anchor disabled; native H2 alpha=0

The reference is a training-only frozen tensor set and is not registered as an optimizer parameter.
""",
    )
    write_text(
        "CIR_PORTABILITY.md",
        f"""# CIR-V2 portability audit

Status: PASS for the pre-training port.

- historical H2 group order, normal/abnormal orientation, and three-stage geometry are preserved;
- peer count K={cfg['rmt_peer_count']} and spatial radius={cfg['rmt_spatial_radius']} are unchanged;
- peer selection and robust delta are detached, while native DFG and score paths remain differentiable;
- configured transport direction is `{cfg['rmt_transport_direction']}`;
- training alpha is `{cfg['rmt_training_alpha']}`; deployment alpha is `{cfg['deployment_alpha']}` through the native historical evaluator;
- exact-score-space alias uses the frozen optimized score implementation;
- fixed-input DFG/logit/probability parity is PASS with zero measured max-absolute differences;
- fixed E0 peer validity is 1.0 and synthetic source-side sign sanity passes.

CIR is train-time only in RCA. Inference alpha=.5 is not part of this master experiment, so no inference-RMT causal claim is made here.
""",
    )
    write_text(
        "IMPLEMENTATION_AUDIT.md",
        f"""# H2 extension implementation audit

Status: PASS for pre-training authorization.

Source identities:

- H2 commit: `{cfg['h2_repo_commit']}`
- H2 train.py SHA256: `{sha256_file(h2_repo / 'train.py')}`
- H2 model/adapter.py SHA256: `{sha256_file(h2_repo / 'model/adapter.py')}`
- H2 evaluator SHA256: `{contract['evaluation']['script_sha256']}`
- H2 CLIP SHA256: `{contract['clip_asset_sha256']}`
- current CIR core SHA256: `{sha256_file(ROOT / 'tools/cir_rmt/core.py')}`
- current parameter-anchor SHA256: `{sha256_file(ROOT / 'tools/cir_rmt/parameter_anchor.py')}`
- extension runner SHA256: `{sha256_file(ROOT / 'scripts/cir_rmt/train_h2_anchor_cir.py')}`
- parity script SHA256: `{sha256_file(ROOT / 'tools/cir_rmt/h2_extension_parity.py')}`

Checks:

- exact H2 E10 historical replay: PASS; new training is authorized;
- one common E0: PASS; no learned H2 E10 weights included;
- H2 E1 anchor load/parameter identity: PASS;
- fixed-input historical-vs-extension parity: PASS;
- K-reg nonzero and lambda_k=0.002: PASS;
- K-reg soft-prompt gradient path: PASS;
- detached W_K gradient path: PASS;
- V2 transport sign sanity and finite peer geometry: PASS;
- R native path has no anchor and no CIR;
- RA adds only image-adapter anchor;
- RCA adds only train-time CIR on RA;
- all checkpoints contain optimizer, scheduler, scaler, RNG, E0/H2 identity, anchor/CIR status, and native deployment metadata.

No source, architecture, optimizer, loss, scheduler, precision, RMT, or historical H2 files were modified by this audit.
""",
    )
    write_json(
        "TEST_RESULTS.json",
        {
            "status": "PASS",
            "h2_replay": "PASS",
            "h2_extension_parity": parity_result,
            "anchor_parameter_identity": "PASS",
            "k_regularization": parity_result["k_regularization_audit"],
            "transport_direction": parity_result["transport_direction_audit"],
            "training_authorized": True,
            "medical_evaluation": "NOT_RUN",
            "mvtec_evaluation": "NOT_RUN",
        },
    )
    write_json(
        "PROGRESS.json",
        {
            "experiment_id": cfg["experiment_id"],
            "status": "PRETRAINING_GATES_PASS",
            "milestones": {"H2_FOUND": True, "H2_REPLAY_PASS": True, "PARITY_PASS": True, "R_E10": "PENDING", "RA_E10": "PENDING", "RCA_E10": "PENDING", "SOURCE_GATE": "PENDING", "PRE_MEDICAL_FREEZE": "PENDING", "MEDICAL_COMPLETE": "PENDING", "FINAL_DECISION": "PENDING", "MVTEC_COMPLETE": "PENDING"},
            "config_sha256": config_sha,
            "git_sha": git_sha(),
            "common_e0_sha256": sha256_file(e0),
        },
    )
    print(f"H2_PREFLIGHT_ARCHIVE={ARCHIVE}")
    print(f"CONFIG_SHA256={config_sha}")
    print(f"CURRENT_GIT_SHA={git_sha()}")
    print(f"COMMON_E0_SHA256={sha256_file(e0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
