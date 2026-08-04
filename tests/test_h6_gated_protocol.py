import json
import subprocess
from pathlib import Path

from tools.phase4_gated_protocol import GatedProtocol


class FakeRunner:
    def __init__(self, root: Path, train_code=0, validation_code=0, summary_code=0, test_code=0):
        self.root = root
        self.train_code = train_code
        self.validation_code = validation_code
        self.summary_code = summary_code
        self.test_code = test_code
        self.calls = []

    def __call__(self, command, env=None):
        self.calls.append((list(command), env))
        joined = " ".join(command)
        train_dir = self.root / "train"
        if "train.py" in command:
            if self.train_code == 0:
                train_dir.mkdir(parents=True, exist_ok=True)
                for epoch in range(1, 21):
                    (train_dir / f"adapter_{epoch}.pth").write_bytes(b"x")
                (train_dir / "GATED_TRAIN_COMPLETED.json").write_text(
                    json.dumps({
                        "final_epoch": 20,
                        "final_checkpoint": str(train_dir / "adapter_20.pth"),
                        "checkpoint_list": [str(train_dir / f"adapter_{epoch}.pth") for epoch in range(1, 21)],
                    })
                )
            return subprocess.CompletedProcess(command, self.train_code)
        if "test_6medical_exact.sh" in joined and "--split val" in joined:
            return subprocess.CompletedProcess(command, self.validation_code)
        if "summarize_phase4_results.py" in joined and " --split val" in joined:
            if self.summary_code == 0:
                train_dir.mkdir(parents=True, exist_ok=True)
                (train_dir / "medical_validation_selection.json").write_text(
                    json.dumps({
                        "datasets": ["Brain", "Liver", "Retina", "Colon_clinicDB", "Colon_colonDB", "Colon_Kvasir"],
                        "selection_rule": "validation only",
                        "best_epoch": {"epoch": 12, "combined_score": 77.0},
                        "macro_by_epoch": [{"epoch": 12, "combined_score": 77.0}],
                    })
                )
            return subprocess.CompletedProcess(command, self.summary_code)
        if "test_6medical_exact.sh" in joined and "--split test" in joined:
            return subprocess.CompletedProcess(command, self.test_code)
        if "summarize_phase4_results.py" in joined and " --split test" in joined:
            return subprocess.CompletedProcess(command, 0)
        return subprocess.CompletedProcess(command, 0)


def _env(root: Path, **extra):
    return {
        "PROTOCOL_ROOT": str(root),
        "DRY_RUN": "0",
        **{key: str(value) for key, value in extra.items()},
    }


def test_exit_code_42_prevents_validation_and_test(tmp_path):
    runner = FakeRunner(tmp_path, train_code=42)
    code = GatedProtocol(_env(tmp_path)).run(runner)
    assert code == 42
    assert len(runner.calls) == 1


def test_ordinary_train_failure_prevents_validation_and_test(tmp_path):
    runner = FakeRunner(tmp_path, train_code=9)
    code = GatedProtocol(_env(tmp_path)).run(runner)
    assert code == 9
    assert len(runner.calls) == 1


def test_exit_zero_without_completed_marker_prevents_validation(tmp_path):
    def runner(command, env=None):
        return subprocess.CompletedProcess(command, 0)

    code = GatedProtocol(_env(tmp_path)).run(runner)
    assert code == 3


def test_successful_training_runs_validation_and_selection_then_test_once(tmp_path):
    runner = FakeRunner(tmp_path)
    code = GatedProtocol(_env(tmp_path)).run(runner)
    assert code == 0
    calls = [" ".join(call[0]) for call in runner.calls]
    assert any("--split val 8 9 10 11 12 13 14 15 16 17 18 19 20" in call for call in calls)
    assert any("--split test 12" in call for call in calls)
    selected = json.loads((tmp_path / "validation" / "selected_common_epoch.json").read_text())
    assert selected["selected_epoch"] == 12
    assert "selected_checkpoint" in selected
    assert (tmp_path / "test" / "EXACT_TEST_COMPLETED.json").exists()


def test_validation_failure_and_missing_selection_prevent_test(tmp_path):
    runner = FakeRunner(tmp_path, validation_code=7)
    assert GatedProtocol(_env(tmp_path)).run(runner) == 7
    assert not any("--split test" in " ".join(call[0]) for call in runner.calls)

    runner = FakeRunner(tmp_path / "missing", summary_code=1)
    assert GatedProtocol(_env(tmp_path / "missing")).run(runner) == 1
    assert not any("--split test" in " ".join(call[0]) for call in runner.calls)


def test_exact_test_rerun_protection(tmp_path):
    test_dir = tmp_path / "test"
    test_dir.mkdir(parents=True)
    (test_dir / "EXACT_TEST_COMPLETED.json").write_text("{}")
    runner = FakeRunner(tmp_path)
    assert GatedProtocol(_env(tmp_path)).run(runner) == 6

    other = tmp_path / "started_only"
    (other / "test").mkdir(parents=True)
    (other / "test" / "EXACT_TEST_STARTED.json").write_text("{}")
    runner = FakeRunner(other)
    assert GatedProtocol(_env(other)).run(runner) == 6


def test_dry_run_launches_no_subprocess_and_creates_no_markers(tmp_path):
    runner = FakeRunner(tmp_path)
    code = GatedProtocol(_env(tmp_path, DRY_RUN=1)).run(runner)
    assert code == 0
    assert runner.calls == []
    assert not (tmp_path / "test" / "EXACT_TEST_STARTED.json").exists()


def test_test_command_uses_selected_checkpoint_and_selection_has_no_test_metrics(tmp_path):
    runner = FakeRunner(tmp_path)
    assert GatedProtocol(_env(tmp_path)).run(runner) == 0
    selected = json.loads((tmp_path / "validation" / "selected_common_epoch.json").read_text())
    assert selected["selected_checkpoint"].endswith("adapter_12.pth")
    assert set(selected) == {
        "selected_epoch",
        "selected_checkpoint",
        "validation_datasets",
        "validation_metrics",
        "aggregate_selection_score",
        "deterministic_tie_break_reason",
        "command_config_fingerprint",
    }
    assert all("test" not in row for row in selected["validation_metrics"])
