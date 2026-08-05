from tools.phase4_v7_full_protocol import V7FullProtocol


def test_v7_dry_run_command_is_explicit_and_protocol_is_separate(tmp_path):
    protocol = V7FullProtocol({"PROTOCOL_ROOT": str(tmp_path), "DRY_RUN": "1"})
    command = protocol.train_command()
    assert "P1-v7-full" in command and "--h6_expert_enabled" in command
    assert "--h6_center_factor_aware" not in command
    orth = [i for i, value in enumerate(command) if value == "--lambda_h6_orth"]
    assert command[orth[-1] + 1] == "0"
    assert command[command.index("--h6_structural_gate_mode") + 1] == "monitor"
    assert protocol.validation_epochs == list(range(8, 21))


def test_v7_drift_diagnostics_are_opt_in():
    protocol = V7FullProtocol({"DRIFT_DIAGNOSTICS": "1", "SMOKE_MAX_BATCHES": "20"})
    assert "--h6_drift_diagnostics" in protocol.train_command()
