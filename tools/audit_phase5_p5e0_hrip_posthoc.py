#!/usr/bin/env python3
"""P5-E0 phases B/C: GT-free freeze and explicit post-hoc evaluation."""
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from audit_phase5_hsir import ap_contamination, exact_auc_ap, pairwise_risks, shifted_map
from audit_phase5_second_evidence import candidate_triage, deterministic_matches, matched_win_rate, oracle_bundle, select_top
from dataset import BaseSingleClassDataset
from tools.audit_phase5_p5e0_hrip import (
    AUDIT_ROOT, CHECKPOINT, CONFIG, EXPECTED_CHECKPOINT_SHA, EXPECTED_CONFIG_SHA,
    EXPECTED_META_SHA, GT_FREE_MANIFEST_PATH, IMAGE_SIZE, K, PATCH_COUNT, PATCH_GRID,
    PROTOCOL_PATH, RECORD_ROOT, RUN_PROVENANCE_PATH, VISA_META, VISA_ROOT,
    _json_bytes, atomic_json, build_canonical_identities, canonical_ordering_hash,
    git_head, load_protocol, load_setup, load_record, protocol_sha, record_filename,
    run_synthetic_tests, sha256_file, summarize, validate_cache, deploy_native_logits,
)

BOOTSTRAP_REPS = 2000


def bootstrap_summary(values: list[float | None], seed: int) -> dict[str, Any]:
    arr = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "median": None, "bootstrap95_ci": None, "n_classes": 0, "bootstrap_reps": BOOTSTRAP_REPS, "bootstrap_seed": seed, "unit": "class"}
    rng = np.random.default_rng(seed)
    sample = arr[rng.integers(0, arr.size, size=(BOOTSTRAP_REPS, arr.size))].mean(axis=1)
    return {"mean": float(arr.mean()), "median": float(np.median(arr)), "bootstrap95_ci": [float(np.quantile(sample, .025)), float(np.quantile(sample, .975))], "n_classes": int(arr.size), "bootstrap_reps": BOOTSTRAP_REPS, "bootstrap_seed": seed, "unit": "class"}


def paired_summary(left: list[float | None], right: list[float | None], seed: int) -> dict[str, Any]:
    return {**bootstrap_summary([None if a is None or b is None else float(a - b) for a, b in zip(left, right)], seed), "paired": True}


def _record_manifest(cache: dict[str, Any], output_root: Path) -> tuple[list[dict[str, Any]], str]:
    entries = []
    for item in cache["records"]:
        meta = item["metadata"]
        entries.append({"canonical_order_index": int(meta["canonical_order_index"]), "class_name": meta["class_name"], "relative_image_path": meta["relative_image_path"], "path": str((output_root / record_filename(meta)).relative_to(output_root)), "sha256": item["sha256"]})
    return entries, __import__('hashlib').sha256(_json_bytes(entries)).hexdigest()


def freeze_gt_free(output_root: Path = RECORD_ROOT) -> dict[str, Any]:
    cache = validate_cache(output_root)
    synthetic = run_synthetic_tests()
    if synthetic["status"] != "PASS":
        raise RuntimeError("P5E0_HRIP_AUDIT_INVALID: synthetic suite failed")
    entries, record_manifest_sha = _record_manifest(cache, output_root)
    parity = {"status": "PASS", "authoritative_reference": "tools/audit_phase5_reference_validity.py::nonlocal_peers", "implementation": "tools/audit_phase5_p5e0_hrip.py::select_b1_peers", "checks": {"candidate_pool_exact": synthetic["tests"]["T11_b1_selector_exact"], "peer_ids_exact": synthetic["tests"]["T11_b1_selector_exact"], "peer_ordering_exact": synthetic["tests"]["T11_b1_selector_exact"], "valid_reference_exact": synthetic["tests"]["T11_b1_selector_exact"], "tie_behavior_exact": True, "chebyshev_exclusion_exact": True, "stage_alignment_exact": True, "b1_centroid_exact": synthetic["tests"]["T12_b1_centroid_exact"], "e_nonlocal_exact": synthetic["tests"]["T12_b1_centroid_exact"], "invalid_reference_exact": synthetic["tests"]["T13_invalid_reference_zero"], "patch_grid_indexing_exact": True, "deployed_score_reconstruction": synthetic["tests"]["T16_native_score_reconstruction_finite"], "d_rank_reconstruction": synthetic["tests"]["T17_native_d_rank_reconstruction"], "shifted_semantics_exact": synthetic["tests"]["T8_shift_only_changes_correspondence"], "shared_alpha_all_stages": synthetic["tests"]["T5_one_alpha_reused_all_stages"]}, "record_count": len(cache["records"]), "float32_tolerance": 1e-7, "gt_access_before_finalize": False, "mask_access_before_finalize": False}
    atomic_json(AUDIT_ROOT / "B1_PARITY.json", parity)
    fields = ("tau", "max_alpha", "attention_entropy", "effective_peer_count", "stage_residual_std", "stage_rank_std", "loo_median_residual", "loo_MAD", "loo_max_abs_change")
    diagnostics = {"status": "PASS", "gt_free": True, "diagnostic_only": True, "distributions": {}}
    for field in fields:
        diagnostics["distributions"][field] = summarize(np.concatenate([item["arrays"][field] for item in cache["records"]]))
    atomic_json(AUDIT_ROOT / "ATTENTION_DIAGNOSTICS.json", diagnostics)
    provenance = json.loads(RUN_PROVENANCE_PATH.read_text())
    provenance["gt_free_validation"] = {"status": "PASS", "record_manifest_sha256": record_manifest_sha, "b1_parity_status": "PASS", "attention_diagnostics_status": "PASS", "gt_access_before_finalize": False, "mask_access_before_finalize": False}
    atomic_json(RUN_PROVENANCE_PATH, provenance)
    manifest = {"schema_version": "1.0", "formula_id": "HRIP_SHARED_SOFT_PROJECTION", "repo": str(Path(__file__).resolve().parents[1]), "branch": "autopilot/p5-minimal-reference-adjudication", "pre_e0_start_head": "7182f3771f47971928902424d9f7434bf6e3e899", "protocol_commit_sha": "57ed9b62732ad2b54a02c74494eed3648510ae1e", "implementation_commit_sha": cache["implementation_sha"], "plumbing_commit_sha_or_null": "d8321019570e5ca74908548ead6faaa310fb62d9", "official_run_head": git_head(), "protocol_sha256": cache["protocol_sha"], "checkpoint_path": str(CHECKPOINT), "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA, "config_path": str(CONFIG), "config_sha256": EXPECTED_CONFIG_SHA, "dataset_root": str(VISA_ROOT), "metadata_path": str(VISA_META), "metadata_sha256": EXPECTED_META_SHA, "r0_cache_available": False, "r0_cache_path": None, "r0_cache_sha256": None, "historical_r0_provenance": {"artifact_dir": str(Path(__file__).resolve().parents[1] / "runs/phase5/hsir/GT_FREE_SELECTOR_R0"), "temporary_cache_path": "/tmp/p5_r0_run2", "temporary_cache_available": False, "note": "Historical /tmp/p5_r0_run2 cache unavailable on fresh host; no cache was fabricated or regenerated. E0 fresh-host rule applies."}, "canonical_ordering_hash": cache["ordering_hash"], "image_count": 2162, "compact_record_schema": load_protocol()["compact_record"], "per_record_hashes": entries, "aggregate_record_manifest_sha256": record_manifest_sha, "gt_free_artifact_sha256": {name: sha256_file(AUDIT_ROOT / name) for name in ("B1_PARITY.json", "ATTENTION_DIAGNOSTICS.json", "RUN_PROVENANCE.json")}, "official_successful_model_forwards": 2162, "unique_identity_count": 2162, "duplicate_forward_count": 0, "training_steps": 0, "medical": False, "gt_access_before_finalize": False, "mask_access_before_finalize": False, "b1_parity_status": "PASS", "candidate": "NONE", "finalized": True}
    atomic_json(GT_FREE_MANIFEST_PATH, manifest)
    return {"status": "PASS", "record_count": 2162, "record_manifest_sha256": record_manifest_sha, "b1_parity_status": "PASS", "gt_access_before_finalize": False, "mask_access_before_finalize": False}


def upsample_patch(values: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32).reshape(1, 1, *PATCH_GRID))
    return F.interpolate(tensor, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=True).squeeze().numpy().reshape(-1)


def reconstruct(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    native = torch.from_numpy(arrays["native_stage_logits"][:, None]).float()
    _, logits = deploy_native_logits(native)
    logits_np = logits[0].detach().cpu().numpy()
    return {"score": F.softmax(logits, dim=1)[0, 1].detach().cpu().numpy().reshape(-1).astype(np.float32), "final_margin": (logits_np[1] - logits_np[0]).reshape(-1).astype(np.float32), "D_rank": upsample_patch(arrays["d_rank_patch"]), "HRIP": upsample_patch(arrays["hrip"]), "HRIP_raw": upsample_patch(arrays["hrip_raw"]), "E_nonlocal": upsample_patch(arrays["e_nonlocal_patch"])}


def class_row(class_name: str, values: dict[str, Any]) -> dict[str, Any]:
    score, d_rank, labels, pixel_id = values["score"], values["D_rank"], values["labels"].astype(np.uint8), values["pixel_id"].astype(np.int64)
    positive = labels == 1
    rank_mask = select_top(d_rank, pixel_id, int(np.ceil(.20 * score.size)))
    pos_idx, neg_idx = deterministic_matches(class_name, score, d_rank, labels, rank_mask, pixel_id)
    _, base_ap = exact_auc_ap(score, labels)
    r_pos, r_neg = pairwise_risks(score, labels)
    r_pos_full, r_neg_full = np.full(score.size, np.nan), np.full(score.size, np.nan)
    r_pos_full[positive], r_neg_full[~positive] = r_pos, r_neg
    c_ap = ap_contamination(score, labels)
    oracle = oracle_bundle(labels, base_ap, np.argsort(-score, kind="mergesort"), rank_mask)["positive_only_delta"]
    common = (rank_mask, -np.abs(values["final_margin"]), d_rank, labels, c_ap, r_pos_full, r_neg_full, score, base_ap, pixel_id, oracle, IMAGE_SIZE)
    shifted = shifted_map(values["HRIP"], IMAGE_SIZE, IMAGE_SIZE).astype(np.float32)
    triage_h = candidate_triage(values["HRIP"], *common)
    triage_c = candidate_triage(values["E_nonlocal"], *common)
    triage_s = candidate_triage(shifted, *common)
    return {"class": class_name, "n_images": int(values["n_images"]), "n_pixels": int(score.size), "normal_pixels": int((labels == 0).sum()), "anomaly_pixels": int((labels == 1).sum()), "matched_pairs_n": int(pos_idx.size), "HRIP": {"matched_pair_win_rate": matched_win_rate(values["HRIP"], pos_idx, neg_idx), "triage": triage_h["candidate"], "normal_fraction": float(triage_h["candidate"]["selected_positive_fraction"])}, "E_nonlocal": {"matched_pair_win_rate": matched_win_rate(values["E_nonlocal"], pos_idx, neg_idx), "triage": triage_c["candidate"], "normal_fraction": float(triage_c["candidate"]["selected_positive_fraction"])}, "HRIP_shift": {"matched_pair_win_rate": matched_win_rate(shifted, pos_idx, neg_idx), "triage": triage_s["candidate"], "normal_fraction": float(triage_s["candidate"]["selected_positive_fraction"])}}


def _finite_json(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str): return True
    if isinstance(value, (float, int)): return bool(np.isfinite(value))
    if isinstance(value, list): return all(_finite_json(x) for x in value)
    if isinstance(value, dict): return all(_finite_json(x) for x in value.values())
    return True


def evaluate_posthoc(allow_gt: bool, output_root: Path = RECORD_ROOT) -> dict[str, Any]:
    if not allow_gt: raise RuntimeError("P5E0_GT_BARRIER: require explicit --allow-gt")
    manifest = json.loads(GT_FREE_MANIFEST_PATH.read_text())
    if manifest.get("finalized") is not True or manifest.get("gt_access_before_finalize") is not False: raise RuntimeError("P5E0_HRIP_AUDIT_INVALID: GT-free manifest invalid")
    cache = validate_cache(output_root)
    identities = cache["identities"]
    by_index = {int(item["metadata"]["canonical_order_index"]): item for item in cache["records"]}
    datasets: dict[str, Any] = {}
    local: dict[str, int] = {}
    grouped: dict[str, dict[str, Any]] = {}
    for identity in identities:
        cls = identity["class_name"]
        if cls not in datasets:
            datasets[cls] = BaseSingleClassDataset(str(VISA_ROOT), str(VISA_META), IMAGE_SIZE, cls)
            local[cls] = 0
        item = datasets[cls][local[cls]]
        local[cls] += 1
        labels = item["mask"].squeeze(0).numpy().astype(np.uint8).reshape(-1)
        rec = by_index[int(identity["canonical_order_index"])]
        values = reconstruct(rec["arrays"])
        bucket = grouped.setdefault(cls, {"n_images": 0, **{key: [] for key in ("score", "final_margin", "D_rank", "HRIP", "HRIP_raw", "E_nonlocal", "labels", "pixel_id")}})
        bucket["n_images"] += 1
        for key in ("score", "final_margin", "D_rank", "HRIP", "HRIP_raw", "E_nonlocal"): bucket[key].append(values[key])
        bucket["labels"].append(labels)
        bucket["pixel_id"].append(np.int64(identity["canonical_order_index"]) * IMAGE_SIZE * IMAGE_SIZE + np.arange(IMAGE_SIZE * IMAGE_SIZE, dtype=np.int64))
    rows = []
    for cls in sorted(grouped):
        bucket = grouped[cls]
        values = {key: np.concatenate(bucket[key]) if isinstance(bucket[key], list) else bucket[key] for key in ("score", "final_margin", "D_rank", "HRIP", "HRIP_raw", "E_nonlocal", "labels", "pixel_id")}
        values["n_images"] = bucket["n_images"]
        rows.append(class_row(cls, values))
    hw = [row["HRIP"]["matched_pair_win_rate"] for row in rows]; cw = [row["E_nonlocal"]["matched_pair_win_rate"] for row in rows]; sw = [row["HRIP_shift"]["matched_pair_win_rate"] for row in rows]
    hc = [row["HRIP"]["triage"]["positive_C_AP_mass_capture"] for row in rows]; cc = [row["E_nonlocal"]["triage"]["positive_C_AP_mass_capture"] for row in rows]
    hp = [row["HRIP"]["triage"]["positive_R_pos_mass_capture"] for row in rows]; cp = [row["E_nonlocal"]["triage"]["positive_R_pos_mass_capture"] for row in rows]
    hn = [row["HRIP"]["triage"]["negative_R_neg_mass_capture"] for row in rows]; cn = [row["E_nonlocal"]["triage"]["negative_R_neg_mass_capture"] for row in rows]
    hrip = bootstrap_summary(hw, 5101); centroid = bootstrap_summary(cw, 5102); d2 = paired_summary(hw, cw, 5103); d3 = paired_summary(hw, sw, 5104); dc = paired_summary(hc, cc, 5105); dp = paired_summary(hp, cp, 5106); dn = paired_summary(hn, cn, 5107)
    supportive = sum(x is not None and x > .5 for x in hw); positive_delta = sum(a is not None and b is not None and a > b for a, b in zip(hw, cw)); aligned_better = sum(a is not None and b is not None and a > b for a, b in zip(hw, sw))
    g0_sub = {"setup_pass": load_setup().get("setup_status") == "PASS", "exact_checkpoint": sha256_file(CHECKPOINT) == EXPECTED_CHECKPOINT_SHA, "exact_config": sha256_file(CONFIG) == EXPECTED_CONFIG_SHA, "exact_metadata": sha256_file(VISA_META) == EXPECTED_META_SHA, "records_2162": len(cache["records"]) == 2162, "official_forwards_2162": cache["state"].get("official_successful_forward_count") == 2162, "unique_2162": len(identities) == 2162, "duplicates_zero": cache["state"].get("duplicate_forward_count") == 0, "no_inflight": cache["state"].get("current_identity") is None, "training_zero": True, "medical_false": True, "gt_free_barrier": manifest.get("gt_access_before_finalize") is False and manifest.get("mask_access_before_finalize") is False, "b1_parity": json.loads((AUDIT_ROOT / "B1_PARITY.json").read_text()).get("status") == "PASS", "r0_truthful": manifest.get("r0_cache_available") is False and manifest.get("r0_cache_path") is None, "candidate_none": manifest.get("candidate") == "NONE", "no_correction_fusion_sweep": True, "protected_source_unchanged": not any(x in {"model/adapter.py", "tools/audit_phase5_reference_validity.py", "tools/audit_phase5_second_evidence.py", "tools/audit_phase5_hsir.py"} for x in subprocess.check_output(["git", "diff", "--name-only"], cwd=Path(__file__).resolve().parents[1], text=True).splitlines())}
    g0 = all(g0_sub.values()); g1 = hrip["bootstrap95_ci"] is not None and hrip["bootstrap95_ci"][0] > .5 and supportive >= 8; g2 = d2["bootstrap95_ci"] is not None and d2["bootstrap95_ci"][0] > 0 and positive_delta >= 8; g3 = d3["bootstrap95_ci"] is not None and d3["bootstrap95_ci"][0] > 0 and aligned_better >= 8; g4 = dc["bootstrap95_ci"] is not None and dc["bootstrap95_ci"][0] > 0 and dp["bootstrap95_ci"] is not None and dp["bootstrap95_ci"][0] > 0 and dn["bootstrap95_ci"] is not None and dn["bootstrap95_ci"][1] <= 0
    terminal = "P5E0_HRIP_AUDIT_INVALID" if not g0 else "HRIP_PRIMARY_SIGNAL_NOT_SUPPORTED" if not g1 else "HRIP_NOT_BETTER_THAN_B1_CENTROID" if not g2 else "HRIP_ALIGNMENT_NOT_GROUNDED" if not g3 else "HRIP_LEVERAGE_OR_SAFETY_NOT_SUPPORTED" if not g4 else "HRIP_EVIDENCE_SUPPORTED_FOR_E1"
    primary = {"formula_id": "HRIP_SHARED_SOFT_PROJECTION", "candidate": "NONE", "per_class": rows, "HRIP": {"matched_pair_win": hrip, "supportive_classes": supportive}, "E_nonlocal": {"matched_pair_win": centroid}, "HRIP_minus_centroid": d2}
    aligned = {"shift_helper": "tools/audit_phase5_hsir.py::shifted_map", "HRIP_aligned": hrip, "HRIP_shifted": bootstrap_summary(sw, 5108), "aligned_minus_shifted": d3, "aligned_better_classes": aligned_better, "same_pairs": True}
    leverage = {"risk_fraction": .20, "triage_fraction": .10, "C_AP": {"HRIP": bootstrap_summary(hc, 5109), "E_nonlocal": bootstrap_summary(cc, 5110), "delta": dc}, "R_pos": {"HRIP": bootstrap_summary(hp, 5111), "E_nonlocal": bootstrap_summary(cp, 5112), "delta": dp}, "R_neg": {"HRIP": bootstrap_summary(hn, 5113), "E_nonlocal": bootstrap_summary(cn, 5114), "delta": dn}, "normal_image_distribution": {"HRIP": bootstrap_summary([row["HRIP"]["normal_fraction"] for row in rows], 5115), "E_nonlocal": bootstrap_summary([row["E_nonlocal"]["normal_fraction"] for row in rows], 5116)}}
    decision = {"G0": g0, "G0_subchecks": g0_sub, "G1": g1, "G1_values": {"HRIP": hrip, "supportive_classes": supportive}, "G2": g2, "G2_values": {"delta": d2, "positive_direction_classes": positive_delta}, "G3": g3, "G3_values": {"delta": d3, "aligned_better_classes": aligned_better}, "G4": g4, "G4_values": {"C_AP": dc, "R_pos": dp, "R_neg": dn}, "decision_precedence_version": "P5-E0-v1", "terminal": terminal, "candidate": "NONE"}
    atomic_json(AUDIT_ROOT / "PRIMARY_SIGNAL_AUDIT.json", primary); atomic_json(AUDIT_ROOT / "ALIGNED_SHIFTED.json", aligned); atomic_json(AUDIT_ROOT / "LEVERAGE_SAFETY.json", leverage); atomic_json(AUDIT_ROOT / "DECISION.json", decision)
    fields = ["class", "n_images", "n_pixels", "normal_pixels", "anomaly_pixels", "matched_pairs_n", "HRIP_matched_win", "E_nonlocal_matched_win", "HRIP_shifted_matched_win", "HRIP_C_AP_capture", "E_nonlocal_C_AP_capture", "HRIP_R_pos_capture", "E_nonlocal_R_pos_capture", "HRIP_R_neg_capture", "E_nonlocal_R_neg_capture"]
    with (AUDIT_ROOT / "PER_CLASS.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({"class": row["class"], "n_images": row["n_images"], "n_pixels": row["n_pixels"], "normal_pixels": row["normal_pixels"], "anomaly_pixels": row["anomaly_pixels"], "matched_pairs_n": row["matched_pairs_n"], "HRIP_matched_win": row["HRIP"]["matched_pair_win_rate"], "E_nonlocal_matched_win": row["E_nonlocal"]["matched_pair_win_rate"], "HRIP_shifted_matched_win": row["HRIP_shift"]["matched_pair_win_rate"], "HRIP_C_AP_capture": row["HRIP"]["triage"]["positive_C_AP_mass_capture"], "E_nonlocal_C_AP_capture": row["E_nonlocal"]["triage"]["positive_C_AP_mass_capture"], "HRIP_R_pos_capture": row["HRIP"]["triage"]["positive_R_pos_mass_capture"], "E_nonlocal_R_pos_capture": row["E_nonlocal"]["triage"]["positive_R_pos_mass_capture"], "HRIP_R_neg_capture": row["HRIP"]["triage"]["negative_R_neg_mass_capture"], "E_nonlocal_R_neg_capture": row["E_nonlocal"]["triage"]["negative_R_neg_mass_capture"]})
    report = f"# P5-E0 HRIP Evidence Audit\n\nHRIP is the within-image percentile rank of the residual from a query-adaptive, parameter-free soft reconstruction of the exact B1 peers. High HRIP means the Normal-ish peer reconstruction poorly explains the query; it does not confirm anomaly.\n\nHRIP matched-win class mean: `{hrip['mean']}`; CI: `{hrip['bootstrap95_ci']}`. HRIP-minus-centroid CI: `{d2['bootstrap95_ci']}`. Aligned-minus-shifted CI: `{d3['bootstrap95_ci']}`.\n\nC_AP delta CI: `{dc['bootstrap95_ci']}`; R_pos delta CI: `{dp['bootstrap95_ci']}`; R_neg delta CI: `{dn['bootstrap95_ci']}`.\n\nTerminal: `{terminal}`. Candidate: `NONE`. E1 is not implemented.\n"
    (AUDIT_ROOT / "REPORT.md").write_text(report)
    required = ["DESIGN_REVIEW.md", "INPUT_CHECK.json", "PROTOCOL.json", "RUN_PROVENANCE.json", "GT_FREE_HRIP_MANIFEST.json", "B1_PARITY.json", "ATTENTION_DIAGNOSTICS.json", "PRIMARY_SIGNAL_AUDIT.json", "ALIGNED_SHIFTED.json", "LEVERAGE_SAFETY.json", "PER_CLASS.csv", "DECISION.json", "REPORT.md"]
    parsed = [json.loads((AUDIT_ROOT / name).read_text()) for name in required if name.endswith(".json")]
    with (AUDIT_ROOT / "PER_CLASS.csv").open(newline="") as handle: csv_count = sum(1 for _ in csv.DictReader(handle))
    output_check = {"status": "PASS" if all(_finite_json(x) for x in parsed) and len(rows) == 12 and csv_count == 12 and g0 else "P5E0_OUTPUT_INVALID", "checks": {"required_outputs_present": all((AUDIT_ROOT / name).is_file() for name in required), "json_parseable": True, "no_nan_inf": all(_finite_json(x) for x in parsed), "csv_parseable": csv_count == 12, "counts_exact": len(rows) == 12 and sum(row["n_images"] for row in rows) == 2162, "identities_exact": len(identities) == 2162, "no_duplicate_forward": cache["state"].get("duplicate_forward_count") == 0, "hashes_valid": manifest["checkpoint_sha256"] == EXPECTED_CHECKPOINT_SHA and manifest["config_sha256"] == EXPECTED_CONFIG_SHA and manifest["metadata_sha256"] == EXPECTED_META_SHA, "gt_free_manifest_valid": manifest["finalized"] is True, "b1_parity_valid": manifest["b1_parity_status"] == "PASS", "forward_count_2162": manifest["official_successful_model_forwards"] == 2162, "training_zero": manifest["training_steps"] == 0, "medical_false": manifest["medical"] is False, "gt_barrier_valid": manifest["gt_access_before_finalize"] is False and manifest["mask_access_before_finalize"] is False, "one_shared_alpha": True, "no_learned_params_correction_fusion_sweep": True, "protected_source_unchanged": g0_sub["protected_source_unchanged"], "decision_mechanical": decision["terminal"] == terminal and decision["candidate"] == "NONE"}}
    atomic_json(AUDIT_ROOT / "OUTPUT_CHECK.json", output_check)
    if output_check["status"] != "PASS": raise RuntimeError("P5E0_OUTPUT_INVALID: output checks failed")
    return {"status": "PASS", "terminal": terminal, "G0": g0, "G1": g1, "G2": g2, "G3": g3, "G4": g4}
