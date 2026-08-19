"""Persist the resumable SABRA machine handoff without credentials."""
from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "runs/phase5/sabra/PRETRAIN_LOGIC_AUDIT"
HANDOFF = ROOT / "runs/phase5/sabra/HANDOFF_20260819"


def cmd(*args: str) -> str:
    return subprocess.check_output(list(args), cwd=ROOT, text=True, stderr=subprocess.STDOUT).strip()


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text if text.endswith("\n") else text + "\n")


def main() -> None:
    local = cmd("git", "rev-parse", "HEAD")
    remote = cmd("git", "rev-parse", "refs/remotes/origin/research/p5-sabra-g")
    divergence = cmd("git", "rev-list", "--left-right", "--count", "HEAD...refs/remotes/origin/research/p5-sabra-g")
    decision_path = AUDIT / "DECISION.json"
    decision = json.loads(decision_path.read_text()) if decision_path.exists() else {}
    manifest_path = AUDIT / "GT_FREE_CACHE_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    implementation = cmd("git", "log", "--format=%H %s", "--all", "--grep=phase5: implement SABRA pretrain logic audit", "-1")
    result = cmd("git", "log", "--format=%H %s", "--all", "--grep=phase5: audit SABRA pretrain logic", "-1") if decision else ""
    status = decision.get("terminal", "PRETRAIN_LOGIC_AUDIT_INCOMPLETE")
    snapshot = {
        "branch": cmd("git", "branch", "--show-current"), "local_head": local, "remote_head": remote,
        "divergence": divergence, "implementation_commit": implementation, "result_commit": result,
        "last_valid_stage": "science_complete" if decision else "gt_free_cache_finalized" if manifest.get("GT_FREE_CACHE_FINALIZED") else "implementation",
        "first_failed_stage": None if decision else "remote_push_or_science",
        "GT_exposure_state": "POST_CACHE_GT_ALLOWED" if manifest.get("GT_FREE_CACHE_FINALIZED") else "GT_FORBIDDEN",
        "GT_FREE_CACHE_FINALIZED": bool(manifest.get("GT_FREE_CACHE_FINALIZED", False)),
        "cache_hashes": manifest.get("shards", {}), "CLIP_hash": manifest.get("clip_sha256"), "checkpoint_hash": manifest.get("checkpoint_sha256"), "config_hash": manifest.get("config_sha256"),
        "environment": {"python": platform.python_version(), "torch": __import__("torch").__version__, "torch_cuda": __import__("torch").version.cuda, "gpu": __import__("torch").cuda.get_device_name(0) if __import__("torch").cuda.is_available() else None},
        "MVTEC_science_reads": 0, "medical_reads": 0, "Phase2B_training_steps": 0,
        "scientific_statuses": decision.get("statuses", {}), "terminal": status,
        "next_allowed_action": "push implementation, then run science" if not decision else "commit/push result and handoff",
        "forbidden_actions": ["MVTec reads", "medical reads", "Phase2B training", "post-hoc formula changes", "credential inclusion"],
        "MUST_RERUN": [] if manifest.get("GT_FREE_CACHE_FINALIZED") else ["GT-free cache"],
        "DO_NOT_RERUN": ["valid finalized cache"] if manifest.get("GT_FREE_CACHE_FINALIZED") else [],
    }
    write(HANDOFF / "HANDOFF_STATE.json", json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    write(HANDOFF / "ARTIFACT_MANIFEST.json", json.dumps({"audit_root": str(AUDIT), "required_artifacts": sorted(p.name for p in AUDIT.glob("*")), "cache_manifest": str(manifest_path)}, indent=2, sort_keys=True) + "\n")
    write(HANDOFF / "GIT_PROVENANCE.txt", f"branch={snapshot['branch']}\nlocal_head={local}\nremote_head={remote}\ndivergence={divergence}\nimplementation={implementation}\nresult={result}\n")
    write(HANDOFF / "ENVIRONMENT_SNAPSHOT.txt", f"python={snapshot['environment']['python']}\ntorch={snapshot['environment']['torch']}\ntorch_cuda={snapshot['environment']['torch_cuda']}\ngpu={snapshot['environment']['gpu']}\n")
    write(HANDOFF / "RESUME_STATUS.md", f"# SABRA resume status\n\nTerminal: `{status}`\n\nLocal HEAD: `{local}`\nRemote HEAD: `{remote}`\nDivergence: `{divergence}`\n\nGT_FREE_CACHE_FINALIZED: `{snapshot['GT_FREE_CACHE_FINALIZED']}`\n\nNext allowed action: {snapshot['next_allowed_action']}.\n")
    write(ROOT / "NEXT_MACHINE_START_SABRA.md", f"# Next machine start\n\nBranch: `research/p5-sabra-g`\n\nRead `PROJECT_HANDOFF_SABRA_20260819.md`, then `runs/phase5/sabra/HANDOFF_20260819/RESUME_STATUS.md`.\n\nCurrent terminal: `{status}`\n\nDo not rerun a valid immutable cache.\n")
    write(ROOT / "PROJECT_HANDOFF_SABRA_20260819.md", f"# SABRA handoff\n\nTerminal: `{status}`\n\nLocal HEAD: `{local}`\nRemote HEAD: `{remote}`\nDivergence: `{divergence}`\n\nImplementation commit: `{implementation}`\nResult commit: `{result}`\n\nGT-free cache finalized: `{snapshot['GT_FREE_CACHE_FINALIZED']}`\n\nNext allowed action: {snapshot['next_allowed_action']}.\n")
    sums = []
    for path in sorted([p for p in AUDIT.rglob("*") if p.is_file()]):
        digest = cmd("sha256sum", str(path.relative_to(ROOT))).split()[0]
        sums.append(f"{digest}  {path.relative_to(ROOT)}")
    write(HANDOFF / "SHA256SUMS.txt", "\n".join(sums))
    print(json.dumps(snapshot, sort_keys=True))


if __name__ == "__main__":
    main()
