#!/usr/bin/env python3
"""P1-v7-full gated train -> validation selection -> one exact test controller."""
from __future__ import annotations
import json, os, subprocess, sys
from pathlib import Path
from typing import Mapping

try:  # works both as ``python tools/...`` and as an import in focused tests
    from .phase4_gated_protocol import GatedProtocol, ProtocolError, env_bool, q, write_json_atomic
except ImportError:  # pragma: no cover - direct script execution
    from phase4_gated_protocol import GatedProtocol, ProtocolError, env_bool, q, write_json_atomic


class V8MinimalFixProtocol(GatedProtocol):
    def __init__(self, env: Mapping[str, str] | None = None):
        super().__init__(env)
        self.protocol_root = Path(self.env.get("PROTOCOL_ROOT", "runs/phase4/progress1_v8_minimal_fix_seed0_ready"))
        self.train_dir = self.protocol_root / "train"
        self.validation_dir, self.test_dir = self.protocol_root / "validation", self.protocol_root / "test"
        self.manifest_root = Path(self.env.get("MEDICAL_MANIFEST_ROOT", str(self.protocol_root / "medical_manifests")))
        self.stage = self.env.get("STAGE", "all")
        if self.stage not in {"all", "train", "val", "test"}:
            raise ProtocolError("STAGE must be all|train|val|test")
        self.force_retrain = env_bool(self.env, "FORCE_RETRAIN")
        self.structural_gate_mode = self.env.get("H6_STRUCTURAL_GATE_MODE", "monitor")
        if self.structural_gate_mode not in {"abort", "monitor", "off"}:
            raise ProtocolError("H6_STRUCTURAL_GATE_MODE must be abort|monitor|off")

    def train_command(self) -> list[str]:
        command = super().train_command()
        # v6-only factor identity and its competing loss must never enter v7.
        filtered: list[str] = []
        skip_value_after = {
            "--h6_factor_id_scale", "--h6_factor_id_max_ratio", "--lambda_h6_center",
            "--lambda_h6_orth", "--lambda_h6_balance", "--lambda_h6_router_teacher",
        }
        index = 0
        while index < len(command):
            token = command[index]
            if token == "--h6_late_factor_identity_enabled" or token == "--h6_center_factor_aware":
                index += 1; continue
            if token in skip_value_after:
                index += 2; continue
            filtered.append(token); index += 1
        command = filtered
        command += [
            "--h6_progress_version", "P1-v8-minimal", "--no-h6_expert_enabled",
            "--h6_global_text_mode", "phase2b",
            "--h6_prediction_routing", "scheduled_topk",
            "--h6_diagnostics_mode", "light",
            "--h6_diagnostics_interval", "50",
            "--h6_structural_gate_mode", self.structural_gate_mode,
        ]
        smoke_max_batches = int(self.env.get("SMOKE_MAX_BATCHES", "0"))
        if smoke_max_batches > 0:
            command += ["--h6_smoke_max_batches", str(smoke_max_batches)]
        if env_bool(self.env, "DRIFT_DIAGNOSTICS"):
            command += ["--h6_drift_diagnostics"]
        return command

    def print_dry_run(self) -> None:
        print("[DRY_RUN] p1_v7_config_fingerprint=P1-v7-full/FOFS-paired-experts/M4-K2-r64")
        print("[DRY_RUN] structural_gate_mode=", self.structural_gate_mode)
        super().print_dry_run()
        print("[DRY_RUN] validation_epochs=", " ".join(map(str, self.validation_epochs)))

    def _marker_is_v8(self, marker: dict) -> bool:
        checkpoint = Path(marker.get("final_checkpoint", ""))
        if not checkpoint.exists(): return False
        payload = __import__("torch").load(checkpoint, map_location="cpu")
        return payload.get("h6_config", {}).get("progress_version") == "P1-v8-minimal"

    def run(self) -> int:
        self.verify_static_inputs()
        if self.dry_run:
            self.print_dry_run(); return 0
        common = self.common_env()
        marker = self.train_dir / "GATED_TRAIN_COMPLETED.json"
        if self.stage in {"all", "train"} and (self.force_retrain or not marker.exists()):
            result = subprocess.run(self.train_command(), env=common)
            if result.returncode == 42: return 42
            if result.returncode: return result.returncode
        if self.stage == "train": return 0
        training = self.require_completed_training()
        if not self._marker_is_v8(training): raise ProtocolError("training marker does not refer to P1-v8-minimal checkpoint", 3)
        selection = self.validation_dir / "selected_common_epoch.json"
        if self.stage in {"all", "val"} and not selection.exists():
            result = subprocess.run(self.validation_command(), env=common)
            if result.returncode: return result.returncode
            result = subprocess.run(self.summarize_command("val", self.validation_epochs), env=common)
            if result.returncode: return result.returncode
            self.write_selected_epoch()
        if self.stage == "val": return 0
        if not selection.exists(): raise ProtocolError("missing selected_common_epoch.json", 3)
        selected = json.loads(selection.read_text())
        epoch = int(selected["selected_epoch"])
        completed = self.test_dir / "EXACT_TEST_COMPLETED.json"
        if completed.exists() and not self.allow_test_rerun: return 0
        self.protect_exact_test()
        write_json_atomic(self.test_dir / "EXACT_TEST_STARTED.json", {"progress_version":"P1-v8-minimal", "selected_epoch":epoch})
        result = subprocess.run(self.exact_test_command(epoch), env=common)
        if result.returncode: return result.returncode
        write_json_atomic(completed, {"progress_version":"P1-v8-minimal", "selected_epoch":epoch})
        return 0


def main() -> int:
    try: return V8MinimalFixProtocol().run()
    except ProtocolError as error:
        print(f"protocol error: {error}", file=sys.stderr); return error.code

if __name__ == "__main__": raise SystemExit(main())
