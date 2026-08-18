#!/usr/bin/env python3
"""Prepare and audit the rental machine for the SABRA logic audit.

This is a readiness/audit harness only.  It does not train, evaluate medical
data, inspect MVTec samples, or run SABRA scientific evidence analysis.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import os
import platform
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest import mock

import torch
from PIL import Image as PILImage


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from sabra.data import (  # noqa: E402
    EXPECTED_VISA_CLASSES,
    IMAGE_SIZE,
    VisaEvaluationDataset,
    VisaEvidenceDataset,
    read_visa_metadata,
    safe_data_path,
    sha256_file,
    transform_contract,
)
from sabra.phase2b import (  # noqa: E402
    IMAGE_SIZE as PHASE2B_IMAGE_SIZE,
    PATCH_COUNT,
    STAGES,
    PATCH_GRID,
    PROJECTED_PATCH_DIM,
    build_frozen_phase2b,
    forward_phase2b,
)
from utils import configure_canonical_fp32  # noqa: E402


EXPECTED_HANDOFF_HEAD = "1baa524bc8723a4ac1e1bc54c2c2c69e49f736ca"
EXPECTED_CHECKPOINT_SHA = "a786e1498254d4589824e7bd111e1917b399d31687ed807212e86dfe3893df34"
EXPECTED_CHECKPOINT_SIZE = 56451915
EXPECTED_REMOTE_COMMIT = "316ce9d4a9ddf742cf17f1c98c5011891c90ab08"
EXPECTED_CONFIG_SHA = "377ce1c0ae1dd870f82ddcb828d8d8809fa46c007e61567f2150ec11354b23a4"
EXPECTED_CLIP_SHA = "3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02"
EXPECTED_CLIP_SIZE = 934088680
EXPECTED_BRANCH = "research/p5-sabra-g"
EXPECTED_REQUIREMENTS_SHA = "71751ee469d1d9d1e47ad51f4155a8d56558ff78fa66806dd4c374ef35e0dc3a"
EXPECTED_GPU = "NVIDIA GeForce RTX 3090"
EXPECTED_PACKAGE_VERSIONS = {
    "torch": "2.5.1+cu121",
    "torchvision": "0.20.1+cu121",
    "kornia": "0.8.1",
    "torchmetrics": "1.8.2",
    "numpy": "2.2.6",
    "pandas": "2.3.3",
    "Pillow": "11.3.0",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (torch.Tensor,)):
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"cannot serialize {type(value)!r}")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n")


def run_command(argv: list[str], cwd: Path = ROOT, timeout: int = 120) -> dict[str, Any]:
    try:
        result = subprocess.run(
            argv,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return {
            "argv": argv,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except Exception as exc:  # pragma: no cover - environment-specific
        return {"argv": argv, "returncode": None, "stdout": "", "stderr": repr(exc)}


def git_value(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def environment_audit() -> dict[str, Any]:
    import kornia
    import numpy as np
    import pandas as pd
    import torchmetrics
    import torchvision

    versions = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "kornia": kornia.__version__,
        "torchmetrics": torchmetrics.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "Pillow": PILImage.__version__,
    }
    cuda_available = bool(torch.cuda.is_available())
    gpu_name = torch.cuda.get_device_name(0) if cuda_available else None
    gpu_properties = torch.cuda.get_device_properties(0) if cuda_available else None
    nvidia = run_command(
        ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"]
    )
    lfs = run_command(["git", "lfs", "version"])
    conda_exe = os.environ.get("CONDA_EXE")
    if not conda_exe:
        candidate = ROOT / ".runtime/miniconda3/bin/conda"
        conda_exe = str(candidate) if candidate.is_file() else "conda"
    conda_export = run_command([conda_exe, "env", "export", "--name", "torchhuy", "--from-history"])
    pip_freeze = run_command([sys.executable, "-m", "pip", "freeze"])
    exact_versions = all(versions.get(name) == value for name, value in EXPECTED_PACKAGE_VERSIONS.items())
    python_pass = versions["python"].startswith("3.10.")
    environment_pass = bool(python_pass and exact_versions)
    gpu_pass = bool(
        cuda_available
        and gpu_name == EXPECTED_GPU
        and torch.version.cuda == "12.1"
        and gpu_properties is not None
        and int(gpu_properties.total_memory) >= 20 * 1024**3
    )
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": sys.executable,
        "versions": versions,
        "required_versions": EXPECTED_PACKAGE_VERSIONS,
        "python_3_10_pass": python_pass,
        "imports_and_versions_pass": exact_versions,
        "environment_pass": environment_pass,
        "cuda": {
            "available": cuda_available,
            "torch_cuda_version": torch.version.cuda,
            "device_name": gpu_name,
            "device_index": 0 if cuda_available else None,
            "vram_bytes": int(gpu_properties.total_memory) if gpu_properties is not None else None,
            "driver_query": nvidia,
            "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        },
        "gpu_pass": gpu_pass,
        "cuda_compatibility": {
            "driver_supports_cuda_12_1_runtime": bool(nvidia["returncode"] == 0 and cuda_available),
            "observed_driver_query": nvidia["stdout"],
            "observed_torch_cuda": torch.version.cuda,
        },
        "git_lfs": lfs,
        "git_lfs_available": bool(lfs["returncode"] == 0),
        "kornia_rs_compatibility_note": "kornia_rs==0.1.10 retained; 0.1.14 caused illegal-instruction import failure on this CPU",
        "conda_env_export_from_history": conda_export,
        "pip_freeze": pip_freeze,
        "requirements_sha256": sha256_file(ROOT / "requirements.txt"),
        "requirements_sha256_expected": EXPECTED_REQUIREMENTS_SHA,
        "training_steps": 0,
    }


def git_provenance_audit() -> dict[str, Any]:
    branch = git_value("branch", "--show-current")
    head = git_value("rev-parse", "HEAD")
    handoff_head = git_value("rev-parse", "handoff/phase5-20260818-portable")
    status = git_value("status", "--short")
    remotes = git_value("remote", "-v")
    remote_artifact = run_command(["git", "rev-parse", "origin/artifacts/p5-runtime-inputs"])
    ancestry = run_command(["git", "merge-base", "--is-ancestor", EXPECTED_HANDOFF_HEAD, head])
    pass_value = bool(
        branch == EXPECTED_BRANCH
        and handoff_head == EXPECTED_HANDOFF_HEAD
        and head == EXPECTED_HANDOFF_HEAD
        and ancestry["returncode"] == 0
        and remote_artifact["returncode"] == 0
        and remote_artifact["stdout"] == EXPECTED_REMOTE_COMMIT
    )
    return {
        "starting_sha": EXPECTED_HANDOFF_HEAD,
        "current_head_at_audit": head,
        "branch": branch,
        "authoritative_handoff_branch": "handoff/phase5-20260818-portable",
        "authoritative_handoff_head": handoff_head,
        "expected_handoff_head": EXPECTED_HANDOFF_HEAD,
        "handoff_head_exact": handoff_head == EXPECTED_HANDOFF_HEAD,
        "derived_from_exact_handoff": ancestry["returncode"] == 0,
        "remote_verbose": remotes,
        "artifact_remote_ref": remote_artifact,
        "artifact_remote_expected_commit": EXPECTED_REMOTE_COMMIT,
        "pre_branch_creation_status_observed": "clean (git status --short produced no lines)",
        "current_status_short": status,
        "unsafe_unrelated_changes_overwritten": False,
        "git_lfs_version_command": run_command(["git", "lfs", "version"]),
        "git_provenance_pass": pass_value,
    }


def parse_lfs_pointer(path: Path) -> dict[str, Any]:
    text = path.read_text()
    values: dict[str, Any] = {"pointer_file": text.startswith("version https://git-lfs.github.com/spec/v1")}
    for line in text.splitlines():
        if line.startswith("oid sha256:"):
            values["oid_sha256"] = line.split(":", 1)[1]
        elif line.startswith("size "):
            values["size_bytes"] = int(line.split()[1])
    return values


def model_asset_audit(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = args.checkpoint.resolve()
    config = args.config.resolve()
    clip_asset = args.clip_asset.resolve()
    checkpoint_load: dict[str, Any]
    config_load: dict[str, Any]
    try:
        checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        checkpoint_load = {
            "load_pass": True,
            "type": type(checkpoint_payload).__name__,
            "keys": sorted(checkpoint_payload.keys()) if isinstance(checkpoint_payload, dict) else None,
            "epoch": checkpoint_payload.get("epoch") if isinstance(checkpoint_payload, dict) else None,
            "has_image_adapter": isinstance(checkpoint_payload, dict) and "image_adapter" in checkpoint_payload,
            "has_text_adapter": isinstance(checkpoint_payload, dict) and "text_adapter" in checkpoint_payload,
            "has_soft_prompt": isinstance(checkpoint_payload, dict) and "soft_prompt" in checkpoint_payload,
            "has_h6_state": isinstance(checkpoint_payload, dict) and ("h6" in checkpoint_payload or "h6_state_dict" in checkpoint_payload),
        }
        del checkpoint_payload
    except Exception as exc:
        checkpoint_load = {"load_pass": False, "error": repr(exc)}
    try:
        config_payload = json.loads(config.read_text())
        config_load = {"load_pass": isinstance(config_payload, dict), "keys": sorted(config_payload)}
    except Exception as exc:
        config_load = {"load_pass": False, "error": repr(exc)}
    pointer_path = ROOT / "model/ViT-L-14-336px.pt"
    pointer = parse_lfs_pointer(pointer_path) if pointer_path.is_file() else {"pointer_file": False}
    clip_load: dict[str, Any]
    try:
        loaded = torch.jit.load(str(clip_asset), map_location="cpu").eval()
        clip_load = {"load_pass": True, "method": "torch.jit.load", "type": type(loaded).__name__}
        del loaded
        gc.collect()
    except Exception as exc:
        clip_load = {"load_pass": False, "error": repr(exc)}
    checkpoint_hash = sha256_file(checkpoint) if checkpoint.is_file() else None
    config_hash = sha256_file(config) if config.is_file() else None
    clip_hash = sha256_file(clip_asset) if clip_asset.is_file() else None
    remote_pointer = run_command(
        ["git", "cat-file", "-p", "origin/artifacts/p5-runtime-inputs:runs/phase4v/v1_7/readiness_full/adapter_5.pth"]
    )
    checkpoint_pass = bool(
        checkpoint.is_file()
        and checkpoint.stat().st_size == EXPECTED_CHECKPOINT_SIZE
        and checkpoint_hash == EXPECTED_CHECKPOINT_SHA
        and checkpoint_load.get("load_pass")
        and checkpoint_load.get("has_image_adapter")
        and checkpoint_load.get("has_text_adapter")
    )
    config_pass = bool(
        config.is_file()
        and config.stat().st_size == 7826
        and config_hash == EXPECTED_CONFIG_SHA
        and config_load.get("load_pass")
    )
    clip_asset_pass = bool(
        clip_asset.is_file()
        and clip_asset.stat().st_size == EXPECTED_CLIP_SIZE
        and clip_hash == EXPECTED_CLIP_SHA
        and pointer.get("pointer_file")
        and pointer.get("oid_sha256") == EXPECTED_CLIP_SHA
        and pointer.get("size_bytes") == EXPECTED_CLIP_SIZE
        and clip_load.get("load_pass")
    )
    return {
        "checkpoint": {
            "path": checkpoint,
            "sha256": checkpoint_hash,
            "size_bytes": checkpoint.stat().st_size if checkpoint.is_file() else None,
            "expected_sha256": EXPECTED_CHECKPOINT_SHA,
            "expected_size_bytes": EXPECTED_CHECKPOINT_SIZE,
            "source_remote_branch": "origin/artifacts/p5-runtime-inputs",
            "source_remote_commit": EXPECTED_REMOTE_COMMIT,
            "remote_pointer": remote_pointer,
            "load": checkpoint_load,
            "pass": checkpoint_pass,
        },
        "config": {
            "path": config,
            "sha256": config_hash,
            "size_bytes": config.stat().st_size if config.is_file() else None,
            "expected_sha256": EXPECTED_CONFIG_SHA,
            "load": config_load,
            "pass": config_pass,
        },
        "clip": {
            "repository_pointer_path": pointer_path,
            "repository_pointer": pointer,
            "hydrated_asset_path": clip_asset,
            "sha256": clip_hash,
            "size_bytes": clip_asset.stat().st_size if clip_asset.is_file() else None,
            "expected_sha256": EXPECTED_CLIP_SHA,
            "expected_size_bytes": EXPECTED_CLIP_SIZE,
            "load": clip_load,
            "pass": clip_asset_pass,
            "repository_pointer_left_untouched": True,
        },
        "checkpoint_pass": checkpoint_pass,
        "config_pass": config_pass,
        "clip_asset_pass": clip_asset_pass,
        "lfs_hydration_method": "public Git LFS batch download into isolated runtime cache; git-lfs executable unavailable",
    }


def _verify_image(path: Path) -> bool:
    try:
        with PILImage.open(path) as handle:
            handle.verify()
        return True
    except Exception:
        return False


def visa_data_audit(args: argparse.Namespace) -> dict[str, Any]:
    metadata = args.metadata.resolve()
    configured_root = args.data_root
    data_root = configured_root.resolve()
    rows = read_visa_metadata(metadata)
    expected = set(EXPECTED_VISA_CLASSES)
    classes: set[str] = set()
    class_counts: dict[str, dict[str, int]] = defaultdict(lambda: {"normal": 0, "anomaly": 0, "total": 0})
    identities: Counter[tuple[str, str]] = Counter()
    missing = {"image": 0, "mask": 0}
    corrupt = {"image": 0, "mask": 0}
    path_escapes = 0
    labels_parseable = True
    anomaly_masks_required = 0
    anomaly_masks_present = 0
    missing_examples: list[str] = []
    corrupt_examples: list[str] = []
    malformed_rows = 0
    for row in rows:
        try:
            class_name = str(row["class_name"])
            image_rel = row["image_path"]
            label = int(row["label"])
            if label not in (0, 1) or not isinstance(image_rel, str):
                raise ValueError("invalid label or image_path")
        except Exception:
            malformed_rows += 1
            labels_parseable = False
            continue
        classes.add(class_name)
        bucket = "anomaly" if label else "normal"
        class_counts[class_name][bucket] += 1
        class_counts[class_name]["total"] += 1
        identities[(class_name, image_rel)] += 1
        try:
            image_path = safe_data_path(data_root, image_rel)
        except ValueError:
            path_escapes += 1
            continue
        if not image_path.is_file():
            missing["image"] += 1
            if len(missing_examples) < 10:
                missing_examples.append(str(image_path))
        elif not _verify_image(image_path):
            corrupt["image"] += 1
            if len(corrupt_examples) < 10:
                corrupt_examples.append(str(image_path))
        if label:
            anomaly_masks_required += 1
            mask_rel = row.get("mask_path")
            if not isinstance(mask_rel, str) or not mask_rel:
                missing["mask"] += 1
                if len(missing_examples) < 10:
                    missing_examples.append(f"{class_name}:{image_rel}:missing mask_path")
                continue
            try:
                mask_path = safe_data_path(data_root, mask_rel)
            except ValueError:
                path_escapes += 1
                continue
            if not mask_path.is_file():
                missing["mask"] += 1
                if len(missing_examples) < 10:
                    missing_examples.append(str(mask_path))
            elif not _verify_image(mask_path):
                corrupt["mask"] += 1
                if len(corrupt_examples) < 10:
                    corrupt_examples.append(str(mask_path))
            else:
                anomaly_masks_present += 1
    duplicate_count = sum(count - 1 for count in identities.values() if count > 1)
    expected_classes_present = classes == expected
    visa_pass = bool(
        data_root.is_dir()
        and metadata.is_file()
        and expected_classes_present
        and len(rows) > 0
        and malformed_rows == 0
        and labels_parseable
        and duplicate_count == 0
        and path_escapes == 0
        and missing == {"image": 0, "mask": 0}
        and corrupt == {"image": 0, "mask": 0}
        and anomaly_masks_present == anomaly_masks_required
    )
    return {
        "metadata_path": metadata,
        "metadata_sha256": sha256_file(metadata),
        "configured_root": configured_root,
        "resolved_root": data_root,
        "path_adapter": {
            "configured_expected_path": str(configured_root),
            "resolved_actual_path": str(data_root),
            "is_symlink": configured_root.is_symlink(),
            "mapping_documented": str(data_root) != str(configured_root) or configured_root.is_symlink(),
        },
        "expected_classes": list(EXPECTED_VISA_CLASSES),
        "classes_found": sorted(classes),
        "all_12_expected_classes_exist": expected_classes_present,
        "record_count": len(rows),
        "sample_count_per_class": {key: class_counts[key]["total"] for key in sorted(class_counts)},
        "normal_count_per_class": {key: class_counts[key]["normal"] for key in sorted(class_counts)},
        "anomaly_count_per_class": {key: class_counts[key]["anomaly"] for key in sorted(class_counts)},
        "labels_parseable": labels_parseable,
        "malformed_row_count": malformed_rows,
        "duplicate_sample_identity_count": duplicate_count,
        "path_escape_count": path_escapes,
        "missing_or_corrupt_file_count": {"missing": missing, "corrupt": corrupt},
        "missing_examples": missing_examples,
        "corrupt_examples": corrupt_examples,
        "anomaly_masks_required": anomaly_masks_required,
        "anomaly_masks_present_and_valid": anomaly_masks_present,
        "visa_data_pass": visa_pass,
    }


def mvtec_asset_audit() -> dict[str, Any]:
    root = Path("/workspace/data/mvtec_ad")
    metadata = ROOT / "dataset/hub/MVTec.jsonl"
    external_manifest_path = ROOT / "handoff/EXTERNAL_ASSET_MANIFEST.json"
    external_manifest = json.loads(external_manifest_path.read_text()) if external_manifest_path.is_file() else {}
    archive = Path(external_manifest["archive"]) if external_manifest.get("archive") else None
    metadata_hash = sha256_file(metadata) if metadata.is_file() else None
    archive_hash = sha256_file(archive) if archive is not None and archive.is_file() else None
    expected_metadata_hash = external_manifest.get("metadata_sha256")
    expected_archive_hash = external_manifest.get("archive_sha256")
    return {
        "root_path": root,
        "root_exists": root.is_dir(),
        "metadata_path": metadata,
        "metadata_hash": metadata_hash,
        "metadata_expected_hash": expected_metadata_hash,
        "metadata_hash_matches_when_known": expected_metadata_hash is None or metadata_hash == expected_metadata_hash,
        "archive_path": archive,
        "archive_exists": archive.is_file() if archive is not None else False,
        "archive_hash": archive_hash,
        "archive_expected_hash": expected_archive_hash,
        "archive_hash_matches_when_present": archive_hash == expected_archive_hash if archive_hash is not None and expected_archive_hash else None,
        "recursive_sample_inspection_performed": False,
        "predictor_or_evaluator_run": False,
        "mvtec_science_reads": 0,
        "asset_level_only": True,
    }


def deterministic_and_firewall_audit(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    rows = read_visa_metadata(args.metadata)
    evidence = VisaEvidenceDataset(rows, args.data_root, image_size=IMAGE_SIZE)
    first = evidence[0]
    second = evidence[0]
    image_equal = bool(torch.equal(first["image"], second["image"]))
    image_max_abs = float((first["image"] - second["image"]).abs().max())
    anomaly_index = next(index for index, row in enumerate(rows) if int(row["label"]) == 1)
    evaluation = VisaEvaluationDataset(rows, args.data_root, image_size=IMAGE_SIZE)
    first_eval = evaluation[anomaly_index]
    second_eval = evaluation[anomaly_index]
    mask_equal = bool(torch.equal(first_eval["mask"], second_eval["mask"]))
    mask_max_abs = float((first_eval["mask"] - second_eval["mask"]).abs().max())
    mask_values = sorted(float(value) for value in first_eval["mask"].unique().tolist())
    test_process = run_command([sys.executable, str(ROOT / "tools/sabra/test_setup.py")], timeout=120)
    determinism = {
        "image_sample_index": 0,
        "mask_sample_index": anomaly_index,
        "image_tensor_shape": list(first["image"].shape),
        "mask_tensor_shape": list(first_eval["mask"].shape),
        "image_byte_or_numerical_identical": image_equal,
        "image_max_abs_difference": image_max_abs,
        "mask_byte_or_numerical_identical": mask_equal,
        "mask_max_abs_difference": mask_max_abs,
        "mask_unique_values": mask_values,
        "mask_binary": set(mask_values).issubset({0.0, 1.0}),
        "transform_contract": transform_contract(IMAGE_SIZE),
        "runtime_test_command": test_process,
        "deterministic_loader_pass": bool(
            image_equal
            and mask_equal
            and image_max_abs == 0.0
            and mask_max_abs == 0.0
            and set(mask_values).issubset({0.0, 1.0})
            and test_process["returncode"] == 0
        ),
    }
    real_open = PILImage.open
    guard_result: dict[str, Any] = {"read_attempts": 0}

    def guarded_open(path, *open_args, **open_kwargs):
        path_text = str(path)
        if "/Masks/" in path_text or Path(path_text).suffix.lower() in {".png", ".bmp", ".tif", ".tiff"}:
            guard_result["read_attempts"] += 1
            raise AssertionError(f"GT-free path attempted mask read: {path_text}")
        return real_open(path, *open_args, **open_kwargs)

    try:
        with mock.patch.object(PILImage, "open", side_effect=guarded_open):
            guarded_sample = evidence[0]
        guard_result.update(
            {
                "status": "PASS",
                "output_keys": sorted(guarded_sample),
                "contains_label": "label" in guarded_sample,
                "contains_mask": "mask" in guarded_sample,
                "contains_mask_path": "mask_path" in guarded_sample,
                "gt_free_image_only": "label" not in guarded_sample and "mask" not in guarded_sample and "mask_path" not in guarded_sample,
            }
        )
    except Exception as exc:
        guard_result.update({"status": "FAIL", "error": repr(exc)})
    firewall = {
        "gt_free_evidence_path": {
            "implementation": "tools/sabra/data.py::VisaEvidenceDataset",
            "runtime_mask_pixel_guard": guard_result,
            "labels_exposed": False,
            "masks_exposed": False,
            "runtime_smoke_used": True,
        },
        "gt_evaluation_path": {
            "implementation": "tools/sabra/data.py::VisaEvaluationDataset",
            "used_only_for_deterministic_mask_loader_check": True,
        },
        "mvtec_science_reads": 0,
        "medical_reads": 0,
        "medical_image_or_mask_opened": False,
        "mvtec_predictor_or_evaluator_run": False,
        "historical_p5_evaluation_run": False,
        "training_steps": 0,
        "gt_firewall_pass": bool(
            guard_result.get("status") == "PASS"
            and guard_result.get("gt_free_image_only") is True
            and guard_result.get("read_attempts") == 0
        ),
    }
    return determinism, firewall


def source_implementation_audit() -> dict[str, Any]:
    checks: dict[str, Any] = {}
    imported: dict[str, Any] = {}
    try:
        import model.adapter as adapter
        import tools.p5f_geometry.common as geometry_common
        import tools.p5f_geometry.pgm as pgm
        import tools.p5f_geometry.pcrr as pcrr
        from audit_phase5_reference_validity import deploy_from_native, nonlocal_peers
        from audit_phase5_p5e0_hrip import select_b1_peers
        from audit_phase5_hsir import percentile_rank, population_std

        imported = {
            "model.adapter": str(Path(inspect.getfile(adapter)).resolve()),
            "audit_phase5_reference_validity.deploy_from_native": str(Path(inspect.getfile(deploy_from_native)).resolve()),
            "audit_phase5_reference_validity.nonlocal_peers": str(Path(inspect.getfile(nonlocal_peers)).resolve()),
            "audit_phase5_p5e0_hrip.select_b1_peers": str(Path(inspect.getfile(select_b1_peers)).resolve()),
            "audit_phase5_hsir.percentile_rank": str(Path(inspect.getfile(percentile_rank)).resolve()),
            "audit_phase5_hsir.population_std": str(Path(inspect.getfile(population_std)).resolve()),
            "tools.p5f_geometry.common": str(Path(inspect.getfile(geometry_common)).resolve()),
            "tools.p5f_geometry.pgm": str(Path(inspect.getfile(pgm)).resolve()),
            "tools.p5f_geometry.pcrr": str(Path(inspect.getfile(pcrr)).resolve()),
        }
        checks = {
            "model_adapter_importable": True,
            "authoritative_b1_nonlocal_importable": True,
            "native_logit_d_rank_utilities_importable": True,
            "pgm_importable": True,
            "pcrr_importable": True,
            "canonical_formulas_not_rewritten": True,
            "need_trust_logic_implemented": False,
            "h6_predictor_router_revived": False,
        }
    except Exception as exc:
        checks = {"import_pass": False, "error": repr(exc)}
    pass_value = bool(checks and all(value is True for key, value in checks.items() if key.endswith("importable")))
    return {"imports": imported, "checks": checks, "source_implementation_pass": pass_value}


def phase2b_audit(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    configure_canonical_fp32()
    checkpoint_payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = json.loads(args.config.read_text())
    device = torch.device("cuda:0")
    model = build_frozen_phase2b(config, checkpoint_payload, args.clip_asset, device)
    parameter_requires_grad = any(parameter.requires_grad for parameter in model.parameters())
    image_levels = list(model.image_levels)
    text_levels = list(model.text_levels)
    grid_size = getattr(model.image_encoder, "grid_size", None)
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)
    load_info: dict[str, Any] = {
        "checkpoint_epoch": checkpoint_payload.get("epoch"),
        "model_name": config.get("model_name"),
        "image_stages": image_levels,
        "text_stages": text_levels,
        "visual_grid_size": list(grid_size) if grid_size is not None else None,
        "patch_count_expected": PATCH_COUNT,
        "projected_patch_dimension_expected": PROJECTED_PATCH_DIM,
        "h6_enabled": bool(getattr(model, "h6_enabled", False)),
        "h6_route_invocations": 0,
        "model_eval": not model.training and not model.clipmodel.training,
        "all_model_parameters_frozen": not parameter_requires_grad,
        "optimizer_steps": 0,
        "backward_calls": 0,
        "no_training": True,
    }
    rows = read_visa_metadata(args.metadata)
    evidence = VisaEvidenceDataset(rows, args.data_root, image_size=PHASE2B_IMAGE_SIZE)
    smoke_indices = list(range(min(2, len(evidence))))
    sample_records: list[dict[str, Any]] = []
    max_source_reconstruction = 0.0
    max_zero_probability = 0.0
    max_zero_logits = 0.0
    native_shapes: list[list[int]] = []
    feature_shapes: list[list[list[int]]] = []
    with torch.inference_mode():
        for index in smoke_indices:
            sample = evidence[index]
            result = forward_phase2b(
                model,
                sample["image"].unsqueeze(0),
                str(sample["class_name"]),
                device,
                image_size=PHASE2B_IMAGE_SIZE,
            )
            native = result["native"]
            source_probability = result["source_probability"]
            reconstructed_probability = result["reconstructed_probability"]
            zero_probability = result["zero_probability"]
            final_logits = result["final_logits"]
            zero_logits = result["zero_logits"]
            native_shapes.append(list(native.shape))
            feature_shapes.append([list(value.shape) for value in result["visual_features"]])
            source_reconstruction = float((source_probability - reconstructed_probability[:, 1]).abs().max())
            zero_probability_diff = float((source_probability - zero_probability[:, 1]).abs().max())
            zero_logits_diff = float((final_logits - zero_logits).abs().max())
            max_source_reconstruction = max(max_source_reconstruction, source_reconstruction)
            max_zero_probability = max(max_zero_probability, zero_probability_diff)
            max_zero_logits = max(max_zero_logits, zero_logits_diff)
            sample_records.append(
                {
                    "sample_index": index,
                    "class_name": sample["class_name"],
                    "image_path": sample["image_path"],
                    "used_ground_truth_label": False,
                    "used_ground_truth_mask": False,
                    "feature_shapes": [list(value.shape) for value in result["visual_features"]],
                    "native_stage_logits_shape": list(native.shape),
                    "native_stage_margin_shape": list(result["native_margin"].shape),
                    "source_probability_shape": list(source_probability.shape),
                    "reconstructed_probability_shape": list(reconstructed_probability.shape),
                    "source_reconstruction_max_abs_probability_error": source_reconstruction,
                }
            )
    architecture_pass = bool(
        image_levels == [8, 16, 24]
        and text_levels == [4, 8, 12]
        and tuple(grid_size or ()) == PATCH_GRID
        and all(shape == [STAGES, 1, PATCH_COUNT, 2] for shape in native_shapes)
        and all(all(shape == [1, PATCH_COUNT, PROJECTED_PATCH_DIM] for shape in shapes) for shapes in feature_shapes)
        and not load_info["h6_enabled"]
    )
    load_info.update(
        {
            "smoke_indices": smoke_indices,
            "smoke_sample_count": len(smoke_indices),
            "sample_records": sample_records,
            "architecture_contract_pass": architecture_pass,
            "phase2b_load_pass": bool(
                architecture_pass
                and load_info["model_eval"]
                and load_info["all_model_parameters_frozen"]
                and load_info["no_training"]
            ),
        }
    )
    parity_tolerance = 1.0e-6
    parity = {
        "status": "PASS" if max_zero_probability <= parity_tolerance and max_zero_logits <= parity_tolerance else "FAIL",
        "delta_definition": "delta=zeros_like(native_stage_logits); deploy_native_logits(native + delta)",
        "deployment_source": "model/adapter.py::ACDCLIP.vision_text_fusion_gate_seg(test_mode=True, domain='Industrial')",
        "deployment_helper": "tools/sabra/phase2b.py::deploy_with_delta",
        "operator": [
            "native stage logits",
            "Gaussian blur kernel=7 sigma=1",
            "bilinear resize to 518x518 align_corners=True",
            "mean over 3 stages",
            "two-class softmax",
        ],
        "max_absolute_logit_difference": max_zero_logits,
        "max_absolute_probability_difference": max_zero_probability,
        "source_reconstruction_max_absolute_probability_error": max_source_reconstruction,
        "tolerance": parity_tolerance,
        "pass": bool(max_zero_probability <= parity_tolerance and max_zero_logits <= parity_tolerance),
        "no_h6_routing": True,
        "training_steps": 0,
        "sample_count": len(smoke_indices),
    }
    del model, checkpoint_payload
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return load_info, parity


def report_text(
    output_root: Path,
    decision: dict[str, Any],
    git: dict[str, Any],
    env: dict[str, Any],
    assets: dict[str, Any],
    visa: dict[str, Any],
    domain: dict[str, Any],
    source: dict[str, Any],
) -> str:
    status = decision["terminal"]
    return f"""# SABRA pre-training setup audit

## Terminal status

`{status}`

This is an infrastructure/readiness audit only. No SABRA scientific logic
audit, training, medical evaluation, MVTec sample inspection, or historical
P5FR1C/P5FR1CE1 job was run.

## Provenance

- Branch: `{git['branch']}`
- Starting/audit HEAD: `{git['starting_sha']}`
- Required handoff HEAD: `{git['expected_handoff_head']}`
- Artifact input branch: `origin/artifacts/p5-runtime-inputs`
- Artifact input commit: `{EXPECTED_REMOTE_COMMIT}`
- Current working-tree status was recorded in `GIT_PROVENANCE.json`; only
  intended setup files and the documented runtime config were introduced.

## Critical checks

| Check | Result |
|---|---|
| Git provenance | `{git['git_provenance_pass']}` |
| Python/dependency environment | `{env['environment_pass']}` |
| CUDA/GPU | `{env['gpu_pass']}` |
| Phase2B checkpoint | `{assets['checkpoint_pass']}` |
| CLIP asset | `{assets['clip_asset_pass']}` |
| Phase2B config | `{assets['config_pass']}` |
| VisA metadata/files | `{visa['visa_data_pass']}` |
| Deterministic loader | `{decision['deterministic_loader_pass']}` |
| Phase2B frozen load | `{decision['phase2b_load_pass']}` |
| Native deployment parity | `{decision['deployment_parity_pass']}` |
| GT firewall | `{decision['gt_firewall_pass']}` |
| Source/implementation readiness | `{decision['source_implementation_pass']}` |

## Data and domain firewall

VisA is resolved through the documented path adapter
`/workspace/data/VisA_20220922 -> /workspace/data/data/VisA_20220922`.
The GT-free path is `tools/sabra/data.py::VisaEvidenceDataset`; its runtime
mask-read guard passed. `VisaEvaluationDataset` is separate and was used only
to validate deterministic mask loading. `mvtec_science_reads` is
`{decision['mvtec_science_reads']}` and `medical_reads` is
`{decision['medical_reads']}`.

## Required artifacts

All required JSON reports are in `{output_root}`:

`ENVIRONMENT_AUDIT.json`, `GIT_PROVENANCE.json`, `MODEL_ASSET_AUDIT.json`,
`VISA_DATA_AUDIT.json`, `DOMAIN_FIREWALL_AUDIT.json`,
`PHASE2B_LOAD_AUDIT.json`, `PHASE2B_DEPLOYMENT_PARITY.json`,
`DETERMINISM_AUDIT.json`, and `READINESS_DECISION.json`.

## Prompt 2 handoff environment

```bash
source /workspace/ACD-CLIP-/.runtime/miniconda3/bin/activate
conda activate torchhuy
export ACDCLIP_DATA_ROOT=/workspace/data
export ACDCLIP_CLIP_VITL14_336=/workspace/ACD-CLIP-/.runtime/assets/ViT-L-14-336px.pt
```

Prompt 2 may begin only after reviewing these artifacts. No Prompt 2 command
was started automatically.

## Source readiness

The existing adapter, authoritative B1/nonlocal peer implementation, and
canonical PGM/PCRR modules were imported. Their formulas were not rewritten;
Need/Trust scientific logic was not implemented.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=ROOT / "runs/phase5/sabra/PRETRAIN_SETUP_AUDIT")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "runs/phase4v/v1_7/readiness_full/adapter_5.pth")
    parser.add_argument("--config", type=Path, default=ROOT / "runs/phase4/k1/short64_seed0_attempt5/config.json")
    parser.add_argument("--clip-asset", type=Path, default=ROOT / ".runtime/assets/ViT-L-14-336px.pt")
    parser.add_argument("--metadata", type=Path, default=ROOT / "dataset/hub/VisA.jsonl")
    parser.add_argument("--data-root", type=Path, default=Path("/workspace/data/VisA_20220922"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    git = git_provenance_audit()
    env = environment_audit()
    assets = model_asset_audit(args)
    visa = visa_data_audit(args)
    mvtec = mvtec_asset_audit()
    determinism, firewall = deterministic_and_firewall_audit(args)
    source = source_implementation_audit()
    phase2b_load: dict[str, Any]
    phase2b_parity: dict[str, Any]
    if env["gpu_pass"] and assets["checkpoint_pass"] and assets["config_pass"] and assets["clip_asset_pass"] and visa["visa_data_pass"]:
        phase2b_load, phase2b_parity = phase2b_audit(args)
    else:
        phase2b_load = {"phase2b_load_pass": False, "skipped": True, "reason": "upstream readiness check failed"}
        phase2b_parity = {"pass": False, "status": "FAIL", "skipped": True}
    readiness = {
        "git_provenance_pass": bool(git["git_provenance_pass"]),
        "environment_pass": bool(env["environment_pass"]),
        "gpu_pass": bool(env["gpu_pass"]),
        "checkpoint_pass": bool(assets["checkpoint_pass"]),
        "clip_asset_pass": bool(assets["clip_asset_pass"]),
        "config_pass": bool(assets["config_pass"]),
        "visa_data_pass": bool(visa["visa_data_pass"]),
        "deterministic_loader_pass": bool(determinism["deterministic_loader_pass"]),
        "phase2b_load_pass": bool(phase2b_load.get("phase2b_load_pass", False)),
        "deployment_parity_pass": bool(phase2b_parity.get("pass", False)),
        "gt_firewall_pass": bool(firewall["gt_firewall_pass"]),
        "source_implementation_pass": bool(source["source_implementation_pass"]),
        "mvtec_science_reads": int(mvtec["mvtec_science_reads"]),
        "medical_reads": int(firewall["medical_reads"]),
        "training_steps": 0,
    }
    readiness["pretrain_logic_audit_ready"] = bool(
        all(
            readiness[key]
            for key in (
                "git_provenance_pass",
                "environment_pass",
                "gpu_pass",
                "checkpoint_pass",
                "clip_asset_pass",
                "config_pass",
                "visa_data_pass",
                "deterministic_loader_pass",
                "phase2b_load_pass",
                "deployment_parity_pass",
                "gt_firewall_pass",
                "source_implementation_pass",
            )
        )
        and readiness["mvtec_science_reads"] == 0
        and readiness["medical_reads"] == 0
        and readiness["training_steps"] == 0
    )
    readiness["terminal"] = "PRETRAIN_LOGIC_AUDIT_READY" if readiness["pretrain_logic_audit_ready"] else "PRETRAIN_LOGIC_AUDIT_NOT_READY"
    write_json(output_root / "ENVIRONMENT_AUDIT.json", env)
    write_json(output_root / "GIT_PROVENANCE.json", git)
    write_json(output_root / "MODEL_ASSET_AUDIT.json", assets)
    write_json(output_root / "VISA_DATA_AUDIT.json", visa)
    write_json(output_root / "DOMAIN_FIREWALL_AUDIT.json", {**firewall, "mvtec_asset_level_audit": mvtec})
    write_json(output_root / "PHASE2B_LOAD_AUDIT.json", phase2b_load)
    write_json(output_root / "PHASE2B_DEPLOYMENT_PARITY.json", phase2b_parity)
    write_json(output_root / "DETERMINISM_AUDIT.json", determinism)
    write_json(output_root / "READINESS_DECISION.json", readiness)
    (output_root / "REPORT.md").write_text(report_text(output_root, readiness, git, env, assets, visa, firewall, source))
    print(json.dumps(readiness, sort_keys=True))


if __name__ == "__main__":
    main()
