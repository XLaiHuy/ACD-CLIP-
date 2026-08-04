#!/usr/bin/env python3
"""Gated P1-v6 train/validation/test protocol controller."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Callable, Mapping, Sequence


DATASETS = ["Brain", "Liver", "Retina", "Colon_clinicDB", "Colon_colonDB", "Colon_Kvasir"]
DEFAULT_VALIDATION_EPOCHS = list(range(8, 21))


class ProtocolError(RuntimeError):
    def __init__(self, message: str, code: int = 2):
        super().__init__(message)
        self.code = int(code)


def env_bool(env: Mapping[str, str], key: str, default: str = "0") -> bool:
    return str(env.get(key, default)) == "1"


def env_list_int(env: Mapping[str, str], key: str, default: Sequence[int]) -> list[int]:
    text = str(env.get(key, "")).strip()
    if not text:
        return list(default)
    return [int(part) for part in text.split()]


def q(command: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


class GatedProtocol:
    def __init__(self, env: Mapping[str, str] | None = None):
        self.env = dict(os.environ if env is None else env)
        self.protocol_root = Path(self.env.get("PROTOCOL_ROOT", "runs/phase4/progress1_v6_gated_full_seed0"))
        self.train_dir = Path(self.env.get("SAVE_PATH", str(self.protocol_root / "train")))
        self.validation_dir = self.protocol_root / "validation"
        self.test_dir = self.protocol_root / "test"
        self.manifest_root = Path(self.env.get("MEDICAL_MANIFEST_ROOT", str(self.protocol_root / "medical_manifests")))
        self.validation_epochs = env_list_int(self.env, "VALIDATION_EPOCHS", DEFAULT_VALIDATION_EPOCHS)
        self.dry_run = env_bool(self.env, "DRY_RUN")
        self.allow_test_rerun = env_bool(self.env, "ALLOW_EXACT_TEST_RERUN")

    def common_env(self) -> dict[str, str]:
        return {
            **os.environ,
            "SAVE_PATH": str(self.train_dir),
            "CUDA_DEVICE": self.env.get("CUDA_DEVICE", "0"),
            "TEST_BATCH_SIZE": self.env.get("TEST_BATCH_SIZE", "1"),
            "NUM_WORKERS": self.env.get("NUM_WORKERS", "2"),
            "MEDICAL_MANIFEST_ROOT": str(self.manifest_root),
            "H6_DENSE_ROUTING_EPOCHS": "8",
            "H6_SPARSE_TRANSITION_EPOCHS": "4",
            "EXTERNAL_EXACT_PIXEL_METRICS": self.env.get("EXTERNAL_EXACT_PIXEL_METRICS", "1"),
            "EXTERNAL_METRIC_CHUNK_PIXELS": self.env.get("EXTERNAL_METRIC_CHUNK_PIXELS", "5000000"),
        }

    def train_command(self) -> list[str]:
        return [
            "conda", "run", "--no-capture-output", "-n", "torchhuy", "python", "train.py",
            "--dataset", "VisA",
            "--img_size", "518",
            "--epoch", self.env.get("EPOCHS", "20"),
            "--batch_size", self.env.get("BATCH_SIZE", "1"),
            "--cuda_device", self.env.get("CUDA_DEVICE", "0"),
            "--grad_accum_steps", self.env.get("GRAD_ACCUM", "6"),
            "--num_workers", self.env.get("NUM_WORKERS", "2"),
            "--seed", self.env.get("SEED", "0"),
            "--precision", self.env.get("PRECISION", "bf16"),
            "--n_groups", "3",
            "--image_adapt_weight", "0.2",
            "--text_adapt_weight", "0.2",
            "--lora_rank", "16",
            "--lora_alpha", "2.0",
            "--conv_lora_rank", "8",
            "--conv_lora_alpha", "2.0",
            "--conv_kernel_size_list", "3", "5",
            "--dfg_mode", "attn",
            "--dfg_attn_dim", "256",
            "--dfg_attn_tau", "8.0",
            "--use_ss2d_dfg",
            "--dfg_gamma_max", "0.2",
            "--dfg_ss2d_fusion", "weight_residual",
            "--dfg_beta", "0.10",
            "--dfg_beta_schedule", "warmup010",
            "--dfg_beta_target", "0.10",
            "--image_lr", "0.001",
            "--text_lr", "0.0005",
            "--soft_prompt_ctx_len", "4",
            "--soft_prompt_lr", "0.00005",
            "--hybrid_alpha_max", "0.20",
            "--soft_prompt_freeze_epochs", "3",
            "--lambda_kg", "0.001",
            "--lambda_k", "0.0",
            "--grad_clip_norm", "1.0",
            "--grad_checkpointing",
            "--h6_progress", "1",
            "--h6_num_factors", "4",
            "--h6_top_k", "2",
            "--h6_bank_dim", "256",
            "--h6_router_dim", "128",
            "--h6_router_temperature", "1.0",
            "--h6_dense_routing_epochs", "8",
            "--h6_sparse_transition_epochs", "4",
            "--lambda_h6_router_teacher", "0.01",
            "--h6_router_teacher_temperature", "0.15",
            "--h6_router_teacher_start_epoch", "3",
            "--h6_router_teacher_warmup_epochs", "3",
            "--h6_router_teacher_mode", "state_centered_cosine",
            "--h6_teacher_confidence_gate",
            "--h6_teacher_entropy_threshold", "0.98",
            "--h6_teacher_prob_std_threshold", "0.001",
            "--h6_router_query_mode", "local_global_bypass",
            "--h6_router_query_global_weight", "0.10",
            "--h6_router_local_bypass_scale", "0.10",
            "--h6_router_local_bypass_max_ratio", "0.20",
            "--h6_router_local_projection_seed_offset", "7200",
            "--h6_router_key_anchor_enabled",
            "--h6_router_key_anchor_seed_offset", "7300",
            "--h6_router_key_adaptation_initial_ratio", "0.10",
            "--h6_router_key_adaptation_max_ratio", "0.25",
            "--h6_factor_context_anchor_enabled",
            "--h6_factor_context_anchor_seed_offset", "7400",
            "--h6_factor_context_adaptation_initial_ratio", "0.10",
            "--h6_factor_context_adaptation_max_ratio", "0.25",
            "--h6_factor_identity_tangent_projection_enabled",
            "--lambda_h6_dynamic_mean_anchor", "0.001",
            "--h6_dynamic_mean_anchor_min_cosine", "0.70",
            "--h6_dynamic_mean_anchor_start_epoch", "4",
            "--h6_dynamic_mean_anchor_warmup_epochs", "3",
            "--h6_structural_gate_enabled",
            "--h6_structural_gate_patience", "2",
            "--h6_structural_gate_dense_start_epoch", "8",
            "--h6_structural_gate_require_all_levels",
            "--h6_gate_sparse_min_ratio", "0.50",
            "--lambda_h6_center", "0.10",
            "--h6_center_factor_aware",
            "--h6_center_detach_assignment",
            "--h6_center_margin", "0.0",
            "--lambda_h6_vae_rec", "0.05",
            "--beta_h6_vae_kl", "0.00001",
            "--h6_kl_zero_epochs", "8",
            "--h6_kl_warmup_epochs", "4",
            "--h6_kl_free_bits", "0.02",
            "--h6_vae_class_ratio", "0.25",
            "--h6_slot_init_enabled",
            "--h6_slot_init_scale", "0.02",
            "--h6_slot_init_seed_offset", "6100",
            "--h6_factor_grad_diagnostics",
            "--h6_late_factor_identity_enabled",
            "--h6_factor_id_scale", "0.02",
            "--h6_factor_id_max_ratio", "0.05",
            "--lambda_h6_orth", "0.001",
            "--lambda_h6_balance", "0.001",
            "--lambda_h6_concept_key_diversity", "0.0001",
            "--h6_concept_key_cosine_margin", "0.5",
            "--h6_concept_key_diversity_start_epoch", "1",
            "--h6_concept_key_diversity_warmup_epochs", "3",
            "--h6_load_bias_enabled",
            "--h6_load_bias_momentum", "0.9",
            "--h6_load_bias_step", "0.001",
            "--h6_load_bias_max", "0.03",
            "--h6_router_failure_patience", "2",
            "--h6_router_max_sparse_dead_factors", "1",
            "--h6_router_min_unique_topk_pairs", "2",
            "--h6_vae_hidden_dim", "512",
            "--h6_vae_latent_dim", "256",
            "--save_path", str(self.train_dir),
        ]

    def validation_command(self) -> list[str]:
        return ["bash", "scripts/phase4/test_6medical_exact.sh", "--split", "val", *map(str, self.validation_epochs)]

    def summarize_command(self, split: str, epochs: Sequence[int]) -> list[str]:
        return [
            "conda", "run", "--no-capture-output", "-n", "torchhuy", "python",
            "tools/summarize_phase4_results.py",
            "--save_path", str(self.train_dir),
            "--split", split,
            "--epochs", *map(str, epochs),
        ]

    def exact_test_command(self, epoch: int) -> list[str]:
        return ["bash", "scripts/phase4/test_6medical_exact.sh", "--split", "test", str(int(epoch))]

    def verify_static_inputs(self) -> None:
        required = [
            Path("train.py"),
            Path("test.py"),
            Path("tools/prepare_phase4_medical_splits.py"),
            Path("tools/summarize_phase4_results.py"),
            Path("scripts/phase4/test_6medical_exact.sh"),
        ]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise ProtocolError(f"missing required protocol files: {missing}")

    def print_dry_run(self) -> None:
        print("[DRY_RUN] protocol_root=", self.protocol_root)
        print("[DRY_RUN] train_dir=", self.train_dir)
        print("[DRY_RUN] validation_dir=", self.validation_dir)
        print("[DRY_RUN] test_dir=", self.test_dir)
        print("[DRY_RUN] manifest_root=", self.manifest_root)
        print("[DRY_RUN] gate_thresholds=patience=2 dense_start=8 sparse_min_ratio=0.50")
        print("[DRY_RUN] train_command=", q(self.train_command()))
        print("[DRY_RUN] validation_command=", q(self.validation_command()))
        print("[DRY_RUN] summarize_val_command=", q(self.summarize_command("val", self.validation_epochs)))
        print("[DRY_RUN] selection_rule=maximize validation six-dataset macro combined_score, tie-break lower epoch")
        print("[DRY_RUN] exact_test_command=derived from validation/selected_common_epoch.json")

    def require_completed_training(self) -> dict:
        marker = self.train_dir / "GATED_TRAIN_COMPLETED.json"
        if not marker.exists():
            raise ProtocolError(f"missing gated training success marker: {marker}", code=3)
        payload = json.loads(marker.read_text())
        final_checkpoint = Path(payload.get("final_checkpoint", ""))
        if not final_checkpoint.exists():
            raise ProtocolError(f"missing final checkpoint from success marker: {final_checkpoint}", code=3)
        for epoch in range(1, int(payload["final_epoch"]) + 1):
            checkpoint = self.train_dir / f"adapter_{epoch}.pth"
            if not checkpoint.exists():
                raise ProtocolError(f"missing expected checkpoint: {checkpoint}", code=3)
        return payload

    def write_selected_epoch(self) -> dict:
        source = self.train_dir / "medical_validation_selection.json"
        if not source.exists():
            raise ProtocolError(f"missing validation selection file: {source}", code=5)
        selection = json.loads(source.read_text())
        best = selection["best_epoch"]
        epoch = int(best["epoch"])
        checkpoint = self.train_dir / f"adapter_{epoch}.pth"
        if not checkpoint.exists():
            raise ProtocolError(f"selected checkpoint is missing: {checkpoint}", code=5)
        payload = {
            "selected_epoch": epoch,
            "selected_checkpoint": str(checkpoint),
            "validation_datasets": selection.get("datasets", DATASETS),
            "validation_metrics": selection.get("macro_by_epoch", []),
            "aggregate_selection_score": best.get("combined_score"),
            "deterministic_tie_break_reason": "lower epoch wins after sorting combined_score desc, epoch asc",
            "command_config_fingerprint": {
                "validation_epochs": self.validation_epochs,
                "selection_rule": selection.get("selection_rule"),
            },
        }
        target = self.validation_dir / "selected_common_epoch.json"
        write_json_atomic(target, payload)
        return payload

    def protect_exact_test(self) -> None:
        started = self.test_dir / "EXACT_TEST_STARTED.json"
        completed = self.test_dir / "EXACT_TEST_COMPLETED.json"
        if completed.exists() and not self.allow_test_rerun:
            raise ProtocolError(f"exact test already completed: {completed}", code=6)
        if started.exists() and not completed.exists() and not self.allow_test_rerun:
            raise ProtocolError(f"exact test started without completion marker: {started}", code=6)

    def _run_impl(self, runner: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> int:
        self.verify_static_inputs()
        if self.dry_run:
            self.print_dry_run()
            return 0

        self.protocol_root.mkdir(parents=True, exist_ok=True)
        self.validation_dir.mkdir(parents=True, exist_ok=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)

        train = runner(self.train_command())
        if train.returncode == 42:
            print("[GATED_PROTOCOL] structural gate aborted training")
            print("[GATED_PROTOCOL] abort_pth=", self.train_dir / "gated_abort_epoch_<EPOCH>.pth")
            print("[GATED_PROTOCOL] abort_json=", self.train_dir / "gated_abort_epoch_<EPOCH>.json")
            return 42
        if train.returncode != 0:
            print(f"[GATED_PROTOCOL] ordinary training failure status={train.returncode}", file=sys.stderr)
            return int(train.returncode)

        self.require_completed_training()
        validation = runner(self.validation_command(), env=self.common_env())
        if validation.returncode != 0:
            print(f"[GATED_PROTOCOL] validation failed status={validation.returncode}", file=sys.stderr)
            return int(validation.returncode)
        summary = runner(self.summarize_command("val", self.validation_epochs))
        if summary.returncode != 0:
            print(f"[GATED_PROTOCOL] validation summary failed status={summary.returncode}", file=sys.stderr)
            return int(summary.returncode)
        selection = self.write_selected_epoch()
        self.protect_exact_test()

        started = self.test_dir / "EXACT_TEST_STARTED.json"
        completed = self.test_dir / "EXACT_TEST_COMPLETED.json"
        write_json_atomic(started, {"selected_epoch": selection["selected_epoch"], "selected_checkpoint": selection["selected_checkpoint"]})
        test = runner(self.exact_test_command(int(selection["selected_epoch"])), env=self.common_env())
        if test.returncode != 0:
            print(f"[GATED_PROTOCOL] exact test failed status={test.returncode}", file=sys.stderr)
            return int(test.returncode)
        test_summary = runner(self.summarize_command("test", [int(selection["selected_epoch"])]))
        if test_summary.returncode != 0:
            print(f"[GATED_PROTOCOL] test summary failed status={test_summary.returncode}", file=sys.stderr)
            return int(test_summary.returncode)
        write_json_atomic(completed, {"selected_epoch": selection["selected_epoch"], "selected_checkpoint": selection["selected_checkpoint"]})
        return 0

    def run(self, runner: Callable[..., subprocess.CompletedProcess] = subprocess.run) -> int:
        try:
            return self._run_impl(runner)
        except ProtocolError as exc:
            print(f"[GATED_PROTOCOL] {exc}", file=sys.stderr)
            return exc.code


def main() -> int:
    return GatedProtocol().run()


if __name__ == "__main__":
    raise SystemExit(main())
