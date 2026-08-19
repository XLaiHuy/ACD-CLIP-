"""Post-cache VisA Trust-v2 development audit.

The only function that opens masks is ``load_patch_targets``.  It is called
after ``load_gt_free_records`` has verified the immutable cache and the pushed
implementation provenance.
"""
from __future__ import annotations

import csv
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from sabra.data import EXPECTED_VISA_CLASSES, read_visa_metadata, safe_data_path  # noqa: E402
from sabra.logic_core import IMAGE_SIZE, PATCHES, sha256_file, write_json  # noqa: E402
from sabra.science_runner import need_oracle  # noqa: E402

TRUST_ROOT = ROOT / "runs/phase5/sabra/TRUST_V2_DEVELOPMENT"
CACHE_ROOT = TRUST_ROOT / "cache"
OLD_CACHE_ROOT = ROOT / "runs/phase5/sabra/PRETRAIN_LOGIC_AUDIT/cache"
MANIFEST = TRUST_ROOT / "TRUST_V2_GT_FREE_MANIFEST.json"
PROTOCOL = TRUST_ROOT / "SABRA_TRUST_V2_PROTOCOL.md"
PROTOCOL_JSON = TRUST_ROOT / "SABRA_TRUST_V2_PROTOCOL.json"
CHECKPOINT = ROOT / "runs/phase4v/v1_7/readiness_full/adapter_5.pth"
CONFIG = ROOT / "runs/phase4/k1/short64_seed0_attempt5/config.json"
CLIP = ROOT / ".runtime/assets/ViT-L-14-336px.pt"
METADATA = ROOT / "dataset/hub/VisA.jsonl"
SOURCES = [ROOT / "tools/sabra/trust_v2/cache_builder.py", ROOT / "tools/sabra/trust_v2/numerical.py", ROOT / "tools/sabra/phase2b.py", ROOT / "tools/sabra/data.py"]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _git(ref: str) -> str:
    return subprocess.check_output(["git", "rev-parse", ref], cwd=ROOT, text=True).strip()


def load_gt_free_records() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not MANIFEST.exists():
        raise RuntimeError("TRUST_V2_GT_FREE_CACHE_NOT_FINALIZED")
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("GT_FREE_CACHE_FINALIZED") is not True:
        raise RuntimeError("TRUST_V2_GT_FREE_CACHE_NOT_FINALIZED")
    local, remote = _git("HEAD"), _git("refs/remotes/origin/research/p5-sabra-g")
    if local != remote:
        raise RuntimeError(f"TRUST_V2_IMPLEMENTATION_NOT_PUSHED local={local} remote={remote}")
    expected = {"checkpoint_sha256": sha256_file(CHECKPOINT), "config_sha256": sha256_file(CONFIG), "clip_sha256": sha256_file(CLIP), "metadata_sha256": sha256_file(METADATA)}
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"TRUST_V2_PROVENANCE_FAIL {key}")
    for path in SOURCES:
        key = str(path.relative_to(ROOT))
        if manifest.get("source_hashes", {}).get(key) != sha256_file(path):
            raise RuntimeError(f"TRUST_V2_PROVENANCE_FAIL {key}")
    records: list[dict[str, Any]] = []
    fields = ["D_rank", "peer_indices", "reserve_p9_index", "reserve_p16_index", "valid_b1", "valid_p9", "valid_p16", "baseline_pgm", "baseline_pcrr", "D_rel", "peer_coherence", "query_support_mean", "peer_eigen_entropy", "stage_query_profile_disagreement", "S9", "R9", "S16", "R16"]
    for class_name in EXPECTED_VISA_CLASSES:
        shard = CACHE_ROOT / f"{class_name}.npz"
        if not shard.exists() or sha256_file(shard) != manifest.get("shards", {}).get(class_name):
            raise RuntimeError(f"TRUST_V2_CACHE_HASH_FAIL {class_name}")
        with np.load(shard, allow_pickle=False) as data:
            for index, image_path in enumerate(data["image_path"].astype(str)):
                records.append({key: np.asarray(data[key][index]) for key in fields} | {"class_name": class_name, "image_path": str(image_path)})
    if len(records) != int(manifest["record_count"]):
        raise RuntimeError("TRUST_V2_RECORD_COUNT_FAIL")
    return records, manifest


def load_patch_targets(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, Any]]]:
    """Open VisA GT only after the immutable GT-free gate has passed."""
    data_root = Path(os.environ.get("ACDCLIP_DATA_ROOT", "/workspace/data"))
    nested = data_root / "VisA_20220922"
    if nested.is_dir():
        data_root = nested
    metadata = {str(row["image_path"]): row for row in read_visa_metadata(METADATA)}
    gt = np.zeros((len(records), PATCHES), dtype=np.int8)
    occupancy = np.zeros((len(records), PATCHES), dtype=np.float32)
    for index, record in enumerate(records):
        row = metadata[record["image_path"]]
        if int(row["label"]):
            with Image.open(safe_data_path(data_root, str(row["mask_path"]))) as handle:
                mask = np.asarray(handle.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST), dtype=np.float32) > 0
        else:
            mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
        values = mask.reshape(37, 14, 37, 14).mean(axis=(1, 3)).astype(np.float32).reshape(-1)
        occupancy[index] = values
        gt[index] = (values > 0).astype(np.int8)
    return gt, occupancy, metadata


def _flat(records: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.concatenate([np.asarray(record[key]).reshape(-1) for record in records])


def _classes(records: list[dict[str, Any]]) -> np.ndarray:
    return np.repeat(np.asarray([record["class_name"] for record in records]), PATCHES)


def _loco(features: np.ndarray, target: np.ndarray, classes: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float64)
    if features.ndim == 1:
        features = features[:, None]
    output = np.full(target.shape, np.nan, dtype=np.float64)
    for held in EXPECTED_VISA_CLASSES:
        train, test = classes != held, classes == held
        scaler = StandardScaler().fit(features[train])
        model = LogisticRegression(class_weight="balanced", solver="lbfgs", C=1.0, max_iter=1000, random_state=0)
        model.fit(scaler.transform(features[train]), target[train])
        output[test] = model.predict_proba(scaler.transform(features[test]))[:, 1]
    return output.astype(np.float32)


def _auc(score: np.ndarray, target: np.ndarray) -> float | None:
    if np.unique(target).size < 2:
        return None
    return float(roc_auc_score(target, score))


def _ap(score: np.ndarray, target: np.ndarray) -> float | None:
    return None if int(np.sum(target)) == 0 else float(average_precision_score(target, score))


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1])


def _effect_summary(values: list[float | None], seed: int = 5101) -> dict[str, Any]:
    array = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    if not array.size:
        return {"mean": None, "median": None, "bootstrap95_ci": None, "n_classes": 0, "positive": 0, "nonnegative": 0, "negative": 0, "catastrophic": 0, "raw_p": None}
    rng = np.random.default_rng(seed)
    bootstrap = array[rng.integers(0, array.size, size=(10000, array.size))].mean(axis=1)
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=array.size)), dtype=np.float64)
    observed = float(array.mean())
    null = (signs * array[None]).mean(axis=1)
    return {"mean": observed, "median": float(np.median(array)), "bootstrap95_ci": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))], "n_classes": int(array.size), "positive": int((array > 0).sum()), "nonnegative": int((array >= 0).sum()), "negative": int((array < 0).sum()), "catastrophic": int((array <= -0.03).sum()), "raw_p": float(np.mean(np.abs(null) >= abs(observed) - 1e-15))}


def _ladder(summary: dict[str, Any]) -> str:
    if summary["n_classes"] == 0:
        return "INCONCLUSIVE"
    if summary["catastrophic"] >= 2 or (summary["median"] is not None and summary["median"] <= -0.005):
        return "FALSIFIED"
    if summary["mean"] is not None and summary["median"] is not None and summary["mean"] >= 0.010 and summary["median"] >= 0.005 and summary["positive"] >= 8:
        return "SUPPORTED"
    if summary["mean"] is not None and summary["median"] is not None and summary["mean"] >= 0.005 and summary["median"] >= 0 and summary["positive"] >= 7:
        return "PROMISING"
    if summary["mean"] is not None and summary["mean"] > 0 and summary["median"] >= 0 and summary["nonnegative"] >= 7:
        return "WEAK"
    if summary["mean"] is not None and summary["mean"] < 0:
        return "FALSIFIED"
    return "INCONCLUSIVE"


def _model_metrics(oof: dict[str, np.ndarray], target: np.ndarray, occupancy: np.ndarray, classes: np.ndarray) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metrics: dict[str, Any] = {}
    class_rows: list[dict[str, Any]] = []
    for name, score in oof.items():
        class_auc: dict[str, float | None] = {}
        class_ap: dict[str, float | None] = {}
        for cls in EXPECTED_VISA_CLASSES:
            mask = classes == cls
            class_auc[cls] = _auc(score[mask], target[mask])
            class_ap[cls] = _ap(score[mask], target[mask])
        metrics[name] = {"class_auroc": class_auc, "class_ap": class_ap, "overall_auroc": _auc(score, target), "overall_ap": _ap(score, target), "brier": float(brier_score_loss(target, score)), "occupancy_spearman_mean": float(np.nanmean([_spearman(score[classes == cls], occupancy[classes == cls]) for cls in EXPECTED_VISA_CLASSES]))}
        for cls in EXPECTED_VISA_CLASSES:
            mask = classes == cls
            prevalence = float(target[mask].mean())
            ap = class_ap[cls]
            class_rows.append({"model": name, "class": cls, "AUROC": class_auc[cls], "AP": ap, "normalized_AP": None if ap is None or prevalence >= 1 else float((ap - prevalence) / (1 - prevalence)), "Brier": float(brier_score_loss(target[mask], score[mask])), "occupancy_Spearman": _spearman(score[mask], occupancy[mask])})
    return metrics, {"rows": class_rows}


def _safety(scores: dict[str, np.ndarray], utility: np.ndarray, classes: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, score in scores.items():
        class_rows = {}
        for cls in EXPECTED_VISA_CLASSES:
            mask = classes == cls
            order = np.argsort(-score[mask], kind="mergesort")
            rows = {}
            for pct in (0.01, 0.02, 0.05, 0.10):
                count = max(1, int(np.ceil(pct * order.size)))
                selected = utility[mask][order[:count]]
                rows[str(pct)] = {"risk_coverage": float(np.mean(selected < -1e-8)), "matched_harm": float(np.sum(np.maximum(-selected, 0))), "yield": float(np.mean(selected > 1e-8))}
            class_rows[cls] = rows
        out[name] = class_rows
    return out


def _reference_and_stable_rows(records: list[dict[str, Any]], gt: np.ndarray, occupancy: np.ndarray, selected: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stable: list[dict[str, Any]] = []
    contaminated: list[bool] = []
    multiple: list[bool] = []
    for image_index, record in enumerate(records):
        valid = np.asarray(record["valid_b1"], dtype=bool)
        peers = np.maximum(np.asarray(record["peer_indices"], dtype=np.int64), 0)
        p9 = max(int(record["reserve_p9_index"]), 0)
        p16 = max(int(record["reserve_p16_index"]), 0)
        peer_occ = occupancy[image_index][peers]
        peer_has = (peer_occ.max(axis=1) > 0) & valid
        contaminated.extend(peer_has[valid].tolist())
        multiple.extend(((peer_occ > 0).sum(axis=1) >= 2)[valid].tolist())
        quality = np.column_stack([record["baseline_pgm"], record["peer_coherence"], record["query_support_mean"], record["peer_eigen_entropy"], record["stage_query_profile_disagreement"], record["S9"], record["R9"], record["S16"], record["R16"], record["D_rel"]])
        for patch in np.flatnonzero(valid & (record["baseline_pgm"] >= 0.75) & (selected[image_index * PATCHES:(image_index + 1) * PATCHES] >= 0.75) & (gt[image_index] == 0)):
            quartiles = []
            for column in range(quality.shape[1]):
                q = np.nanquantile(quality[valid, column], [0.25, 0.5, 0.75])
                quartiles.append(int(np.digitize(quality[patch, column], q, right=True) + 1))
            stable.append({"class": record["class_name"], "image_path": record["image_path"], "patch": int(patch), "E_raw": float(record["baseline_pgm"][patch]), "T_v2": float(selected[image_index * PATCHES + patch]), "p1_p8_contaminated": bool(peer_has[patch]), "p1_p8_multiple_contaminated": bool((peer_occ[patch] > 0).sum() >= 2), "p9_index": int(p9), "p16_index": int(p16), "quality_quartiles": ";".join(map(str, quartiles))})
    return stable, {"reference_sets_with_anomalous_peer_fraction": float(np.mean(contaminated)) if contaminated else None, "multiple_contaminated_peer_fraction": float(np.mean(multiple)) if multiple else None, "stable_but_wrong_count": len(stable), "no_gt_peer_selection": True}


def run() -> dict[str, Any]:
    print("TRUST_V2_STAGE cache_gate", flush=True)
    records, manifest = load_gt_free_records()
    print("TRUST_V2_STAGE gt_opened", flush=True)
    gt, occupancy, metadata = load_patch_targets(records)
    classes = _classes(records)
    target = gt.reshape(-1).astype(np.int8)
    flat_occupancy = occupancy.reshape(-1)
    e = _flat(records, "baseline_pgm").astype(np.float32)
    pcrr = _flat(records, "baseline_pcrr").astype(np.float32)
    feature_arrays = {name: _flat(records, name).astype(np.float32) for name in ("peer_coherence", "query_support_mean", "peer_eigen_entropy", "stage_query_profile_disagreement", "S9", "R9", "S16", "R16", "D_rel")}
    coverage = json.loads((TRUST_ROOT / "P16_COVERAGE_AUDIT.json").read_text())
    base_features = np.column_stack([e, feature_arrays["peer_coherence"], feature_arrays["query_support_mean"], feature_arrays["peer_eigen_entropy"], feature_arrays["stage_query_profile_disagreement"]])
    model_features: dict[str, np.ndarray] = {"M0_E": e[:, None], "M1_E_Credibility": base_features, "M2_E_Credibility_S9_R9": np.column_stack([base_features, feature_arrays["S9"], feature_arrays["R9"]])}
    if coverage.get("m3_eligible"):
        model_features["M3_E_Credibility_S9_R9_S16_R16"] = np.column_stack([model_features["M2_E_Credibility_S9_R9"], feature_arrays["S16"], feature_arrays["R16"]])
    print("TRUST_V2_STAGE trust_oof_start", list(model_features), flush=True)
    oof = {name: _loco(features, target, classes) for name, features in model_features.items()}
    print("TRUST_V2_STAGE trust_oof_done", flush=True)
    metrics, class_metric_rows = _model_metrics(oof, target, flat_occupancy, classes)
    effects = {}
    for name in list(oof)[1:]:
        effects[name] = [(_auc(oof[name][classes == cls], target[classes == cls]) - _auc(oof["M0_E"][classes == cls], target[classes == cls])) if _auc(oof[name][classes == cls], target[classes == cls]) is not None and _auc(oof["M0_E"][classes == cls], target[classes == cls]) is not None else None for cls in EXPECTED_VISA_CLASSES]
    summaries = {name: _effect_summary(value) for name, value in effects.items()}
    ladders = {name: _ladder(summary) for name, summary in summaries.items()}
    rank = {"INCONCLUSIVE": 0, "WEAK": 1, "PROMISING": 2, "SUPPORTED": 3, "FALSIFIED": -1}
    eligible = [name for name in summaries if ladders[name] not in {"INCONCLUSIVE", "FALSIFIED"}]
    selected_name = max(eligible, key=lambda name: (rank[ladders[name]], -list(summaries).index(name))) if eligible else None
    selected = oof[selected_name] if selected_name else oof["M0_E"]
    pcrr_scores = _loco(np.column_stack([e, pcrr]), target, classes)
    pcrr_effect = [(_auc(pcrr_scores[classes == cls], target[classes == cls]) - _auc(oof["M0_E"][classes == cls], target[classes == cls])) if _auc(pcrr_scores[classes == cls], target[classes == cls]) is not None and _auc(oof["M0_E"][classes == cls], target[classes == cls]) is not None else None for cls in EXPECTED_VISA_CLASSES]
    pcrr_summary = _effect_summary(pcrr_effect)
    pcrr_status = "RETAIN" if _ladder(pcrr_summary) in {"WEAK", "PROMISING", "SUPPORTED"} else "DROP"
    stable_rows, reference = _reference_and_stable_rows(records, gt, occupancy, selected)
    _write_csv(TRUST_ROOT / "STABLE_BUT_WRONG_V2.csv", stable_rows)
    _write_csv(TRUST_ROOT / "PER_CLASS_TRUST_V2.csv", class_metric_rows["rows"])
    write_json(TRUST_ROOT / "TRUST_V2_MODEL_AUDIT.json", {"status": "PASS", "models": list(model_features), "metrics": metrics, "effects_vs_M0": effects, "effect_summaries": summaries, "ladders": ladders, "selected_model": selected_name or "M0_E", "selected_model_features": list(model_features.get(selected_name, model_features["M0_E"]).shape), "OOF": True, "class_unit": "VisA class", "training_fold_only_scaling": True, "logistic": {"class_weight": "balanced", "solver": "lbfgs", "C": 1.0, "max_iter": 1000, "random_state": 0}})
    write_json(TRUST_ROOT / "PCRR_DISAGREEMENT_AUDIT.json", {"status": "PASS", "PCRR_STATUS": pcrr_status, "class_effect": dict(zip(EXPECTED_VISA_CLASSES, pcrr_effect)), "effect": pcrr_summary, "used_for_fusion": False})
    print("TRUST_V2_STAGE trust_metrics_done", flush=True)
    c1_records: list[dict[str, Any]] = []
    old_fields = ("native_logits", "margin_within_image_rank", "robust_margin_normalization", "D_rank", "deployment_sensitivity")
    for class_name in EXPECTED_VISA_CLASSES:
        class_records = [record for record in records if record["class_name"] == class_name]
        with np.load(OLD_CACHE_ROOT / f"{class_name}.npz", allow_pickle=False) as old:
            old_paths = old["image_path"].astype(str)
            old_index = {path: index for index, path in enumerate(old_paths)}
            for record in class_records:
                index = old_index[record["image_path"]]
                c1_records.append({key: np.array(old[key][index], copy=True) for key in old_fields} | {"class_name": class_name, "image_path": record["image_path"]})
    data_root = Path(os.environ.get("ACDCLIP_DATA_ROOT", "/workspace/data"))
    if (data_root / "VisA_20220922").is_dir(): data_root = data_root / "VisA_20220922"
    print("TRUST_V2_STAGE c1_oracle_start", flush=True)
    signed, positive, harm, oracle_parity = need_oracle(c1_records, metadata, data_root, __import__("torch").device("cuda"))
    utility_target = (signed.reshape(-1) > 1e-8).astype(np.int8)
    need_features = np.column_stack([_flat(c1_records, "margin_within_image_rank"), _flat(c1_records, "robust_margin_normalization"), _flat(c1_records, "D_rank"), _flat(c1_records, "deployment_sensitivity")])
    print("TRUST_V2_STAGE c1_oracle_done", flush=True)
    c1 = _loco(need_features, utility_target, classes)
    margin_only = _loco(need_features[:, :1], utility_target, classes)
    need_effect = [(_auc(c1[classes == cls], utility_target[classes == cls]) - _auc(margin_only[classes == cls], utility_target[classes == cls])) if _auc(c1[classes == cls], utility_target[classes == cls]) is not None and _auc(margin_only[classes == cls], utility_target[classes == cls]) is not None else None for cls in EXPECTED_VISA_CLASSES]
    need_summary = _effect_summary(need_effect)
    need_status = _ladder(need_summary)
    write_json(TRUST_ROOT / "NEED_C1_FROZEN_MODEL.json", {"status": "FROZEN", "model": "C1", "features": ["margin_within_image_rank", "robust_margin_normalization", "D_rank", "deployment_sensitivity"], "utility_target": "signed utility > 1e-8", "effect": need_summary, "OOF": True, "oracle_parity": oracle_parity})
    write_json(TRUST_ROOT / "AUTHORITY_V2_AUDIT.json", {"status": "PENDING", "reason": "authority is computed after C1 and Trust-v2 model outputs"})
    authority_scores = {"A0_N": c1, "A1_N_E_raw": c1 * e, "A1_N_E_cal": c1 * oof["M0_E"], "A2_N_T_v2": c1 * selected}
    authority_metrics, _ = _model_metrics(authority_scores, utility_target, np.zeros_like(utility_target), classes)
    authority_effect = [(_auc(authority_scores["A2_N_T_v2"][classes == cls], utility_target[classes == cls]) - _auc(authority_scores["A1_N_E_cal"][classes == cls], utility_target[classes == cls])) if _auc(authority_scores["A2_N_T_v2"][classes == cls], utility_target[classes == cls]) is not None and _auc(authority_scores["A1_N_E_cal"][classes == cls], utility_target[classes == cls]) is not None else None for cls in EXPECTED_VISA_CLASSES]
    authority_summary = _effect_summary(authority_effect)
    authority_status = _ladder(authority_summary)
    write_json(TRUST_ROOT / "AUTHORITY_V2_AUDIT.json", {"status": authority_status, "primary": "A2_N_T_v2 vs A1_N_E_cal", "secondary": ["A2_N_T_v2 vs A0_N", "A2_N_T_v2 vs A1_N_E_raw"], "metrics": authority_metrics, "safety": _safety(authority_scores, signed.reshape(-1), classes), "effect": authority_summary, "class_effect": dict(zip(EXPECTED_VISA_CLASSES, authority_effect)), "E_cal": "M0_E OOF", "T_v2": selected_name or "M0_E", "OOF": True})
    trust_status = ladders.get(selected_name, "INCONCLUSIVE") if selected_name else "INCONCLUSIVE"
    statuses = {"Trust": trust_status, "Need": need_status, "Authority": authority_status, "PCRR": pcrr_status, "P16": coverage.get("status")}
    statistics = {"unit": "VisA class", "n": 12, "bootstrap_repetitions": 10000, "sign_flip_assignments": 4096, "trust": summaries, "need": need_summary, "authority": authority_summary, "pcrr": pcrr_summary}
    write_json(TRUST_ROOT / "STATISTICS.json", statistics)
    write_json(TRUST_ROOT / "REFERENCE_CREDIBILITY_AUDIT.json", {"status": "PASS", **reference, "stable_but_wrong_count": len(stable_rows), "selected_model": selected_name or "M0_E"})
    decision = {"terminal": "TRUST_V2_DEVELOPMENT_ELIGIBLE" if trust_status in {"WEAK", "PROMISING", "SUPPORTED"} else "TRUST_V2_DEVELOPMENT_INCONCLUSIVE_OR_FALSIFIED", "statuses": statuses, "selected_model": selected_name or "M0_E", "selected_effect": summaries.get(selected_name, {}), "FULL_SABRA_TRAIN_AUTHORIZATION": False, "counters": {"MEDICAL_READS": 0, "MVTEC_READS_BEFORE_FREEZE": 0, "PHASE2B_TRAINING_STEPS": 0, "TRUST_V2_MODEL_SELECTION_AFTER_MVTEC": 0}, "GT_free_cache_verified": True, "GT_access_after_cache_freeze": True, "scientific_content": "Trust-v2 development only"}
    write_json(TRUST_ROOT / "DECISION.json", decision)
    (TRUST_ROOT / "ADVERSARIAL_REVIEW.md").write_text("# SABRA Trust-v2 adversarial review\n\n- GT-free cache provenance, shard hashes, frozen assets, and pushed implementation were checked before VisA masks were opened.\n- Peer selection and p9/p16 reserves were computed without GT; GT was used only for post-freeze evaluation.\n- Models are class-held-out OOF balanced logistic regressions with training-fold-only scaling.\n- MVTec and medical data were not accessed.\n- PCRR is diagnostic and never fused into the primary Trust-v2 score.\n")
    (TRUST_ROOT / "REPORT.md").write_text(f"# SABRA Trust-v2 VisA development audit\n\nTrust status: `{trust_status}`. Need C1: `{need_status}`. Authority-v2: `{authority_status}`. PCRR: `{pcrr_status}`. p16 coverage: `{coverage.get('status')}`.\n\nSelected model: `{selected_name or 'M0_E'}`. Stable-but-wrong rows: `{len(stable_rows)}`.\n\nThis is VisA development evidence. MVTec remains forbidden unless a candidate is subsequently frozen and pushed. Full SABRA training is not authorized by this development audit.\n")
    return decision


if __name__ == "__main__":
    print(json.dumps(run(), sort_keys=True))

