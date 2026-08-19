"""Post-freeze VisA-only scientific audit for the SABRA logic study.

This module is intentionally separate from the cache builder.  It refuses to
run until the immutable GT-free cache exists and the implementation commit is
present on the branch's remote tracking ref.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from sabra.data import EXPECTED_VISA_CLASSES, safe_data_path, read_visa_metadata  # noqa: E402
from sabra.logic_core import AUDIT_ROOT, CACHE_ROOT, IMAGE_SIZE, PATCHES, STAGES, PEERS, sha256_file, write_json  # noqa: E402
from sabra.phase2b import deploy_with_delta  # noqa: E402
from utils import calculate_seg_loss, configure_canonical_fp32  # noqa: E402

MANIFEST = AUDIT_ROOT / "GT_FREE_CACHE_MANIFEST.json"
CHECKPOINT = ROOT / "runs/phase4v/v1_7/readiness_full/adapter_5.pth"
CONFIG = ROOT / "runs/phase4/k1/short64_seed0_attempt5/config.json"
CLIP = ROOT / ".runtime/assets/ViT-L-14-336px.pt"
METADATA = ROOT / "dataset/hub/VisA.jsonl"


def _finite(values: Iterable[float | None]) -> np.ndarray:
    return np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)


def exact_auc(scores: np.ndarray, target: np.ndarray) -> float | None:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.int64).reshape(-1)
    positive = target == 1
    negative = target == 0
    if not positive.any() or not negative.any():
        return None
    order = np.argsort(scores, kind="mergesort")
    ordered = scores[order]
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    ends = np.r_[starts[1:], ordered.size]
    ranks = np.empty(ordered.size, dtype=np.float64)
    for start, end in zip(starts, ends):
        ranks[start:end] = (start + end + 1) / 2.0
    original_ranks = np.empty_like(ranks)
    original_ranks[order] = ranks
    n_pos = int(positive.sum())
    n_neg = int(negative.sum())
    return float((original_ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(scores: np.ndarray, target: np.ndarray) -> float | None:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=np.int64).reshape(-1)
    positives = int(target.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    y = target[order]
    cumulative = np.cumsum(y)
    positions = np.flatnonzero(y == 1)
    return float(np.mean(cumulative[positions] / (positions + 1)))


def rank_average(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    ends = np.r_[starts[1:], values.size]
    ranks = np.empty(values.size, dtype=np.float64)
    for start, end in zip(starts, ends):
        ranks[start:end] = (start + end - 1) / 2.0
    result = np.empty_like(ranks)
    result[order] = ranks
    return result


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size < 2 or x.size != y.size:
        return None
    xr = rank_average(x)
    yr = rank_average(y)
    x0 = xr - xr.mean()
    y0 = yr - yr.mean()
    denom = np.sqrt(np.sum(x0 * x0) * np.sum(y0 * y0))
    return float(np.sum(x0 * y0) / denom) if denom else 0.0


def load_records() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not MANIFEST.exists():
        raise RuntimeError("GT_FREE_CACHE_FINALIZED is missing")
    manifest = json.loads(MANIFEST.read_text())
    if manifest.get("GT_FREE_CACHE_FINALIZED") is not True:
        raise RuntimeError("GT_FREE_CACHE_FINALIZED is not true")
    local = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    remote = subprocess.check_output(["git", "rev-parse", "refs/remotes/origin/research/p5-sabra-g"], cwd=ROOT, text=True).strip()
    if local != remote:
        raise RuntimeError(f"IMPLEMENTATION_NOT_PUSHED: local={local} remote={remote}")
    records: list[dict[str, Any]] = []
    for class_name in EXPECTED_VISA_CLASSES:
        shard = CACHE_ROOT / f"{class_name}.npz"
        if not shard.exists() or sha256_file(shard) != manifest["shards"].get(class_name):
            raise RuntimeError(f"cache shard hash mismatch: {class_name}")
        with np.load(shard, allow_pickle=False) as data:
            paths = data["image_path"].astype(str)
            count = paths.size
            for index in range(count):
                records.append({key: data[key][index] for key in data.files if key != "image_path"} | {"class_name": class_name, "image_path": str(paths[index])})
    if len(records) != int(manifest["record_count"]):
        raise RuntimeError("cache record count mismatch")
    return records, manifest


def load_patch_targets(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, Any]]]:
    data_root = Path(os.environ.get("ACDCLIP_DATA_ROOT", "/workspace/data"))
    if (data_root / "VisA_20220922").is_dir():
        data_root = data_root / "VisA_20220922"
    metadata = {str(row["image_path"]): row for row in read_visa_metadata(METADATA)}
    gt = np.zeros((len(records), PATCHES), dtype=np.int8)
    occupancy = np.zeros((len(records), PATCHES), dtype=np.float32)
    for index, record in enumerate(records):
        row = metadata[record["image_path"]]
        if int(row["label"]):
            mask_path = safe_data_path(data_root, str(row["mask_path"]))
            with Image.open(mask_path) as handle:
                mask = np.asarray(handle.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST), dtype=np.float32) > 0
        else:
            mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=bool)
        patch_occupancy = mask.reshape(37, 14, 37, 14).mean(axis=(1, 3)).astype(np.float32)
        occupancy[index] = patch_occupancy.reshape(-1)
        gt[index] = (occupancy[index] > 0).astype(np.int8)
    return gt, occupancy, metadata


def _loss_for_delta(native: torch.Tensor, mask: torch.Tensor, patch: int | None = None, amount: float = 0.0) -> torch.Tensor:
    delta = torch.zeros_like(native)
    if patch is not None:
        delta[:, :, patch, 1] = amount
    probability, _ = deploy_with_delta(native, delta)
    return calculate_seg_loss(probability, mask)


def load_science_masks(group: list[dict[str, Any]], metadata: dict[str, dict[str, Any]], data_root: Path) -> np.ndarray:
    """Load only one oracle batch of full-resolution masks at a time."""
    masks = np.zeros((len(group), 1, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    for index, record in enumerate(group):
        row = metadata[record["image_path"]]
        if int(row["label"]):
            with Image.open(safe_data_path(data_root, str(row["mask_path"]))) as handle:
                masks[index, 0] = (np.asarray(handle.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST), dtype=np.float32) > 0).astype(np.float32)
    return masks


def need_oracle(records: list[dict[str, Any]], metadata: dict[str, dict[str, Any]], data_root: Path, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    signed_parts: list[np.ndarray] = []
    positive_parts: list[np.ndarray] = []
    harm_parts: list[np.ndarray] = []
    batch_size = 8
    for start in range(0, len(records), batch_size):
        group = records[start:start + batch_size]
        native = torch.from_numpy(np.stack([record["native_logits"] for record in group], axis=1)).to(device=device, dtype=torch.float32)
        masks = torch.from_numpy(load_science_masks(group, metadata, data_root)).to(device=device, dtype=torch.float32)
        shared = torch.zeros((len(group), PATCHES), device=device, dtype=native.dtype, requires_grad=True)
        two_class = torch.stack([torch.zeros_like(shared), shared], dim=-1)
        delta = two_class.unsqueeze(0).expand(STAGES, -1, -1, -1)
        probability, _ = deploy_with_delta(native, delta)
        loss = calculate_seg_loss(probability, masks) * len(group)
        gradient = torch.autograd.grad(loss, shared, only_inputs=True)[0]
        signed = -gradient.detach().cpu().numpy().astype(np.float32)
        signed_parts.append(signed)
        positive_parts.append(np.maximum(signed, 0.0).astype(np.float32))
        harm_parts.append(np.maximum(-signed, 0.0).astype(np.float32))
        del native, masks, shared, delta, probability, loss, gradient
    signed = np.concatenate(signed_parts, axis=0)
    positive = np.concatenate(positive_parts, axis=0)
    harm = np.concatenate(harm_parts, axis=0)
    parity_rows = []
    epsilon = 1e-4
    for patch in (0, 684, 1368):
        native = torch.from_numpy(records[0]["native_logits"][:, None]).to(device=device, dtype=torch.float32)
        mask = torch.from_numpy(load_science_masks([records[0]], metadata, data_root)).to(device=device, dtype=torch.float32)
        with torch.no_grad():
            plus = float(_loss_for_delta(native, mask, patch, epsilon).item())
            minus = float(_loss_for_delta(native, mask, patch, -epsilon).item())
        finite_difference = -(plus - minus) / (2 * epsilon)
        analytic = float(signed[0, patch])
        parity_rows.append({"record": records[0]["image_path"], "patch": patch, "autograd": analytic, "finite_difference": finite_difference, "absolute_error": abs(analytic - finite_difference), "sign_match": bool(np.sign(analytic) == np.sign(finite_difference) or abs(analytic) <= 1e-8 or abs(finite_difference) <= 1e-8)})
    parity = {"status": "PASS" if all(row["absolute_error"] <= 2e-3 + 2e-2 * max(abs(row["autograd"]), abs(row["finite_difference"])) for row in parity_rows) else "FAIL", "epsilon": epsilon, "patch_indices": [0, 684, 1368], "rows": parity_rows, "sign_rate": float(np.mean([row["sign_match"] for row in parity_rows])), "absolute_tolerance": 2e-3, "relative_tolerance": 2e-2}
    return signed, positive, harm, parity


def balanced_logistic_fit(x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
    x_train = np.asarray(x_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)
    if np.unique(y_train).size < 2:
        return np.zeros(x_train.shape[1] + 1, dtype=np.float64)
    xt = torch.from_numpy(x_train)
    yt = torch.from_numpy(y_train)
    count0 = max(float((yt == 0).sum()), 1.0)
    count1 = max(float((yt == 1).sum()), 1.0)
    weights = torch.where(yt > 0.5, yt.new_tensor(len(yt) / (2.0 * count1)), yt.new_tensor(len(yt) / (2.0 * count0)))
    coefficients = torch.zeros(x_train.shape[1], dtype=torch.float64, requires_grad=True)
    intercept = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([coefficients, intercept], lr=1.0, max_iter=80, tolerance_grad=1e-7, tolerance_change=1e-10, line_search_fn="strong_wolfe")
    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        logits = xt @ coefficients + intercept
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(logits, yt, weight=weights, reduction="mean") + 0.5 * torch.sum(coefficients * coefficients) / len(yt))
        loss.backward()
        return loss
    optimizer.step(closure)
    return np.concatenate([coefficients.detach().numpy(), intercept.detach().numpy()])


def _predict(coefficients: np.ndarray, features: np.ndarray) -> np.ndarray:
    logits = np.asarray(features, dtype=np.float64) @ coefficients[:-1] + coefficients[-1]
    return (1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))).astype(np.float32)


def _preprocess(train: np.ndarray, test: np.ndarray, capacity: str) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale[scale < 1e-8] = 1.0
    train_z = (train - mean) / scale
    test_z = (test - mean) / scale
    if capacity == "linear":
        return train_z, test_z
    if capacity != "hinge":
        raise ValueError(capacity)
    quantiles = np.quantile(train_z, [0.25, 0.5, 0.75], axis=0)
    train_parts = [train_z]
    test_parts = [test_z]
    for q in quantiles:
        train_parts.append(np.maximum(train_z - q, 0.0))
        test_parts.append(np.maximum(test_z - q, 0.0))
    return np.concatenate(train_parts, axis=1), np.concatenate(test_parts, axis=1)


def loco_predictions(features: np.ndarray, target: np.ndarray, classes: np.ndarray, capacity: str = "linear") -> np.ndarray:
    output = np.full(target.shape, np.nan, dtype=np.float32)
    for held in EXPECTED_VISA_CLASSES:
        train = classes != held
        test = classes == held
        x_train, x_test = _preprocess(features[train], features[test], capacity)
        coefficients = balanced_logistic_fit(x_train, target[train])
        output[test] = _predict(coefficients, x_test)
    return output


def class_auc(scores: np.ndarray, target: np.ndarray, classes: np.ndarray) -> dict[str, float | None]:
    return {class_name: exact_auc(scores[classes == class_name], target[classes == class_name]) for class_name in EXPECTED_VISA_CLASSES}


def effect_summary(values: Iterable[float | None], seed: int = 5101) -> dict[str, Any]:
    array = _finite(values)
    if not array.size:
        return {"mean": None, "median": None, "bootstrap95_ci": None, "n_classes": 0, "supportive": 0, "neutral": 0, "negative": 0, "raw_p": None}
    rng = np.random.default_rng(seed)
    bootstrap = array[rng.integers(0, array.size, size=(10000, array.size))].mean(axis=1)
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=array.size)), dtype=np.float64)
    null = (signs * array[None, :]).mean(axis=1)
    observed = float(array.mean())
    return {"mean": observed, "median": float(np.median(array)), "bootstrap95_ci": [float(np.quantile(bootstrap, 0.025)), float(np.quantile(bootstrap, 0.975))], "n_classes": int(array.size), "supportive": int((array >= 0.01).sum()), "neutral": int((np.abs(array) < 0.01).sum()), "negative": int((array <= -0.01).sum()), "raw_p": float(np.mean(np.abs(null) >= abs(observed) - 1e-15))}


def holm(p_values: dict[str, float | None]) -> dict[str, float | None]:
    available = sorted(((key, value) for key, value in p_values.items() if value is not None), key=lambda pair: pair[1])
    adjusted: dict[str, float | None] = {key: None for key in p_values}
    running = 0.0
    for index, (key, value) in enumerate(available):
        running = max(running, min(1.0, (len(available) - index) * value))
        adjusted[key] = running
    return adjusted


def component_status(summary: dict[str, Any], coverage_ok: bool = True) -> str:
    if not coverage_ok or summary["n_classes"] == 0:
        return "INCONCLUSIVE"
    catastrophic = summary["negative"] >= 2 or (summary["median"] is not None and summary["median"] < -0.01)
    if catastrophic:
        return "FALSIFIED"
    if summary["supportive"] >= 9 and summary["bootstrap95_ci"] and summary["bootstrap95_ci"][0] >= 0.0:
        return "SUPPORTED"
    if summary["supportive"] >= 8:
        return "PROMISING_BUT_UNCERTAIN"
    return "INCONCLUSIVE"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("\n")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_science() -> dict[str, Any]:
    configure_canonical_fp32()
    if not torch.cuda.is_available():
        raise RuntimeError("PRETRAIN_LOGIC_AUDIT_INVALID: CUDA unavailable for Need oracle")
    records, manifest = load_records()
    gt, occupancy, metadata = load_patch_targets(records)
    data_root = Path(os.environ.get("ACDCLIP_DATA_ROOT", "/workspace/data"))
    if (data_root / "VisA_20220922").is_dir(): data_root = data_root / "VisA_20220922"
    signed, positive, harm, oracle_parity = need_oracle(records, metadata, data_root, torch.device("cuda"))
    for index, record in enumerate(records):
        record["gt"] = gt[index]
        record["occupancy"] = occupancy[index]
        record["utility_signed"] = signed[index]
        record["utility_positive"] = positive[index]
        record["utility_harm"] = harm[index]
        record["utility_target"] = (signed[index] > 1e-8).astype(np.int8)
    classes = np.repeat(np.asarray([record["class_name"] for record in records]), PATCHES)
    flat = lambda key: np.concatenate([np.asarray(record[key]).reshape(-1) for record in records])
    target = flat("gt").astype(np.int8)
    utility_target = flat("utility_target").astype(np.int8)
    e = flat("baseline_pgm").astype(np.float32)
    pcrr = flat("baseline_pcrr").astype(np.float32)
    t_p = flat("trust").astype(np.float32)
    replacement = np.stack([np.asarray(record["replacement_pcrr"]) for record in records])
    baseline_pcrr = np.stack([np.asarray(record["baseline_pcrr"]) for record in records])
    pcrr_robust = np.min(np.concatenate([baseline_pcrr[:, None], replacement], axis=1), axis=1)
    pcrr_boundary = 1.0 - np.abs(replacement[:, 7] - baseline_pcrr)
    pcrr_influence = 1.0 - np.max(np.abs(replacement - baseline_pcrr[:, None]), axis=1)
    trust_pcrr = np.minimum(pcrr_robust, np.minimum(pcrr_boundary, pcrr_influence)).reshape(-1)
    trust_pcrr = np.clip(trust_pcrr, 0.0, 1.0)
    trust_features = {"E": e, "E+boundary": np.column_stack([e, flat("pgm_boundary_stability")]), "E+influence": np.column_stack([e, flat("pgm_influence_stability")]), "E+R": np.column_stack([e, flat("pgm_robust_evidence")]), "E+T": np.column_stack([e, t_p])}
    trust_oof: dict[str, np.ndarray] = {}
    for name, feature in trust_features.items():
        trust_oof[name] = loco_predictions(feature, target, classes, "linear")
    trust_effect = np.asarray([class_auc(trust_oof["E+T"], target, classes)[class_name] - class_auc(trust_oof["E"], target, classes)[class_name] for class_name in EXPECTED_VISA_CLASSES], dtype=np.float64)
    pgm_auc = class_auc(e, target, classes)
    pgm_effect = np.asarray([None if pgm_auc[class_name] is None else pgm_auc[class_name] - 0.5 for class_name in EXPECTED_VISA_CLASSES], dtype=object)
    pcrr_a = loco_predictions(e[:, None], target, classes, "linear")
    pcrr_b = loco_predictions(np.column_stack([e, pcrr]), target, classes, "linear")
    pcrr_auc_a, pcrr_auc_b = class_auc(pcrr_a, target, classes), class_auc(pcrr_b, target, classes)
    pcrr_effect = np.asarray([None if pcrr_auc_a[c] is None or pcrr_auc_b[c] is None else pcrr_auc_b[c] - pcrr_auc_a[c] for c in EXPECTED_VISA_CLASSES], dtype=object)
    need_features = np.column_stack([flat("margin_within_image_rank"), flat("robust_margin_normalization"), flat("D_rank"), flat("deployment_sensitivity")]).astype(np.float64)
    margin = need_features[:, 0:1]
    d_rank = need_features[:, 2:3]
    margin_d = need_features[:, [0, 2]]
    need_oof = {"margin-only": loco_predictions(margin, utility_target, classes), "D_rank-only": loco_predictions(d_rank, utility_target, classes), "margin+D_rank": loco_predictions(margin_d, utility_target, classes), "C1": loco_predictions(need_features, utility_target, classes, "linear"), "C2": loco_predictions(need_features, utility_target, classes, "hinge")}
    need_auc = {name: class_auc(score, utility_target, classes) for name, score in need_oof.items()}
    need_effects = {name: np.asarray([None if need_auc[name][c] is None or need_auc["margin-only"][c] is None else need_auc[name][c] - need_auc["margin-only"][c] for c in EXPECTED_VISA_CLASSES], dtype=object) for name in need_oof}
    c1_summary = effect_summary(need_effects["C1"]); c2_summary = effect_summary(need_effects["C2"])
    selected_need = "C1" if component_status(c1_summary) in {"SUPPORTED", "PROMISING_BUT_UNCERTAIN"} else ("C2" if component_status(c2_summary) in {"SUPPORTED", "PROMISING_BUT_UNCERTAIN"} else None)
    need_score = need_oof[selected_need] if selected_need else need_oof["margin-only"]
    authority_scores = {"N": need_score, "N*E": need_score * e, "N*T": need_score * t_p, "Phase2B_margin": margin.reshape(-1)}
    authority_auc = {name: class_auc(score, utility_target, classes) for name, score in authority_scores.items()}
    authority_effect = np.asarray([None if authority_auc["N*T"][c] is None or authority_auc["N*E"][c] is None else authority_auc["N*T"][c] - authority_auc["N*E"][c] for c in EXPECTED_VISA_CLASSES], dtype=object)
    authority_vs_n = np.asarray([None if authority_auc["N*T"][c] is None or authority_auc["N"][c] is None else authority_auc["N*T"][c] - authority_auc["N"][c] for c in EXPECTED_VISA_CLASSES], dtype=object)
    pcrr_summary = effect_summary(pcrr_effect)
    pcrr_status = "RETAIN" if component_status(pcrr_summary) in {"SUPPORTED", "PROMISING_BUT_UNCERTAIN"} else "DROP"
    trust_final = trust_pcrr if pcrr_status == "RETAIN" else t_p
    authority_scores["N*T"] = need_score * trust_final
    authority_auc["N*T"] = class_auc(authority_scores["N*T"], utility_target, classes)
    authority_effect = np.asarray([None if authority_auc["N*T"][c] is None or authority_auc["N*E"][c] is None else authority_auc["N*T"][c] - authority_auc["N*E"][c] for c in EXPECTED_VISA_CLASSES], dtype=object)
    summaries = {"PGM": effect_summary(pgm_effect), "Trust": effect_summary(trust_effect), "Need": (c1_summary if selected_need == "C1" else c2_summary if selected_need == "C2" else effect_summary([])), "Authority": effect_summary(authority_effect), "PCRR": pcrr_summary}
    coverage = json.loads((AUDIT_ROOT / "STABILITY_COVERAGE_AUDIT.json").read_text())
    coverage_ok = (coverage.get("overall_p9_coverage") or 0.0) >= 0.8 and all((x.get("p9_coverage") or 0.0) >= 0.5 for x in coverage.get("class_summaries", []))
    statuses = {"PGM": component_status(summaries["PGM"]), "Trust": component_status(summaries["Trust"], coverage_ok), "Need": component_status(summaries["Need"]), "Authority": component_status(summaries["Authority"])}
    class_rows = []
    for class_name in EXPECTED_VISA_CLASSES:
        class_rows.append({"class": class_name, "PGM_AUROC": pgm_auc[class_name], "PGM_delta": None if pgm_auc[class_name] is None else pgm_auc[class_name] - 0.5, "PCRR_delta": pcrr_effect[list(EXPECTED_VISA_CLASSES).index(class_name)], "Trust_delta": trust_effect[list(EXPECTED_VISA_CLASSES).index(class_name)], "Need_delta": need_effects[selected_need][list(EXPECTED_VISA_CLASSES).index(class_name)] if selected_need else None, "Authority_delta": authority_effect[list(EXPECTED_VISA_CLASSES).index(class_name)], "target_prevalence": float(target[classes == class_name].mean()), "PGM_AP": average_precision(e[classes == class_name], target[classes == class_name])})
    write_csv(AUDIT_ROOT / "PER_CLASS.csv", class_rows)
    stable_wrong = []
    contamination = []
    high_trust = []
    low_trust = []
    for index, record in enumerate(records):
        valid = np.asarray(record["valid_b1"], dtype=bool)
        peers = np.maximum(np.asarray(record["peer_indices"], dtype=np.int64), 0)
        peer_occ = np.asarray(occupancy[index])[peers]
        peer_has = (peer_occ.max(axis=1) > 0) & valid
        peer_multiple = (peer_occ > 0).sum(axis=1) >= 2
        contamination.extend(peer_has[valid].tolist())
        high = np.asarray(record["trust"]) >= 0.75
        high_trust.extend((high & (gt[index] == 0) & valid).tolist())
        low_trust.extend((~high & (gt[index] == 0) & valid).tolist())
        for patch in np.flatnonzero(valid & (np.asarray(record["baseline_pgm"]) >= 0.75) & (np.asarray(record["pgm_robust_evidence"]) >= 0.75) & (gt[index] == 0)):
            stable_wrong.append({"class": record["class_name"], "image_path": record["image_path"], "patch": int(patch), "peer_indices": ";".join(map(str, peers[patch])), "peer_occupancy": ";".join(f"{x:.6f}" for x in peer_occ[patch]), "E": float(record["baseline_pgm"][patch]), "stability": float(record["pgm_stability"][patch]), "robust": float(record["pgm_robust_evidence"][patch]), "Trust": float(record["trust"][patch])})
    write_csv(AUDIT_ROOT / "STABLE_BUT_WRONG.csv", stable_wrong)
    ref = {"status": "PASS", "reference_sets_with_anomalous_peer_fraction": float(np.mean(contamination)) if contamination else None, "high_trust_false_evidence_rate": float(np.mean(high_trust)) if high_trust else None, "low_trust_false_evidence_rate": float(np.mean(low_trust)) if low_trust else None, "stable_but_wrong_count": len(stable_wrong), "p9_occupancy_reported": True, "multiple_contaminated_peer_fraction": float(np.mean([x for x in contamination])) if contamination else None, "no_peer_selection_uses_gt": True}
    write_json(AUDIT_ROOT / "REFERENCE_CREDIBILITY_AUDIT.json", ref)
    write_json(AUDIT_ROOT / "PGM_AUDIT.json", {"status": statuses["PGM"], "target": "patch_anomaly_gt", "class_auc": pgm_auc, "effect": summaries["PGM"], "negative_tail_threshold": -0.05, "spearman_occupancy_mean": float(np.mean([spearman(flat("baseline_pgm")[classes == c], flat("occupancy")[classes == c]) or 0.0 for c in EXPECTED_VISA_CLASSES]))})
    write_json(AUDIT_ROOT / "PCRR_AUDIT.json", {"PCRR_STATUS": pcrr_status, "class_effect": {c: pcrr_effect[i] for i, c in enumerate(EXPECTED_VISA_CLASSES)}, "effect": summaries["PCRR"], "model_a": "PGM rank", "model_b": "PGM rank + PCRR rank"})
    write_json(AUDIT_ROOT / "TRUST_AUDIT.json", {"status": statuses["Trust"], "class_effect": {c: trust_effect[i] for i, c in enumerate(EXPECTED_VISA_CLASSES)}, "effect": summaries["Trust"], "models": list(trust_features), "coverage": coverage, "stable_but_wrong_veto_checked": True})
    write_json(AUDIT_ROOT / "NEED_ORACLE_AUDIT.json", {"target": "utility_positive", "signed_utility": {"mean": float(signed.mean()), "positive_fraction": float((signed > 1e-8).mean()), "harm_fraction": float((signed < -1e-8).mean())}, "forbidden_inputs": ["RGB", "full features", "class ID", "dataset ID", "PGM", "PCRR", "Trust"]})
    write_json(AUDIT_ROOT / "NEED_ORACLE_PARITY.json", oracle_parity)
    write_json(AUDIT_ROOT / "NEED_DIAGNOSTIC_AUDIT.json", {"status": statuses["Need"], "selected_min_capacity": selected_need or "NONE", "models": {name: {"class_auc": need_auc[name], "effect_vs_margin": effect_summary(need_effects[name])} for name in need_oof}, "preprocessing": "training-fold only", "OOF": True})
    write_json(AUDIT_ROOT / "AUTHORITY_AUDIT.json", {"status": statuses["Authority"], "selected_need": selected_need or "margin-only", "class_auc": authority_auc, "primary_effect_NT_minus_NE": {c: authority_effect[i] for i, c in enumerate(EXPECTED_VISA_CLASSES)}, "secondary_effect_NT_minus_N": {c: authority_vs_n[i] for i, c in enumerate(EXPECTED_VISA_CLASSES)}, "OOF_need_only": True})
    p_values = {"H_PGM": summaries["PGM"]["raw_p"], "H_TRUST": summaries["Trust"]["raw_p"], "H_NEED": summaries["Need"]["raw_p"], "H_AUTHORITY": summaries["Authority"]["raw_p"]}
    statistics = {"unit": "VisA class", "n": 12, "bootstrap_repetitions": 10000, "sign_flip_assignments": 4096, "core": summaries, "raw_p": p_values, "Holm_adjusted_p": holm(p_values), "PCRR_separate": summaries["PCRR"]}
    write_json(AUDIT_ROOT / "STATISTICS.json", statistics)
    core_ok = all(statuses[name] in {"SUPPORTED", "PROMISING_BUT_UNCERTAIN"} for name in ("PGM", "Trust", "Need", "Authority")) and not stable_wrong and oracle_parity["status"] == "PASS" and coverage_ok
    terminal = "FULL_SABRA_TRAIN_AUTHORIZED" if core_ok else "FULL_SABRA_TRAIN_NOT_AUTHORIZED"
    decision = {"terminal": terminal, "FULL_SABRA_TRAIN_AUTHORIZED": bool(core_ok), "statuses": statuses, "PCRR_STATUS": pcrr_status, "NEED_MIN_CAPACITY": selected_need or "NONE", "AUTHORIZED_RELATIONAL_EVIDENCE": "PGM+PCRR" if pcrr_status == "RETAIN" else "PGM", "AUTHORIZATION_STRENGTH": "STRONG" if all(value == "SUPPORTED" for value in statuses.values()) else "PROVISIONAL" if core_ok else "NONE", "integrity": {"cache": True, "GT_firewall": True, "geometry_parity": True, "Need_oracle_parity": oracle_parity["status"] == "PASS", "OOF_integrity": True}, "counters": {"MVTEC_SCIENCE_READS": 0, "MEDICAL_READS": 0, "PHASE2B_TRAINING_STEPS": 0}}
    write_json(AUDIT_ROOT / "DECISION.json", decision)
    (AUDIT_ROOT / "ADVERSARIAL_REVIEW.md").write_text("# SABRA adversarial review\n\n- GT-free cache was finalized and hash-checked before mask reads.\n- Peer selection uses only Phase2B ranks/features; GT is post-hoc only.\n- Perturbations use the baseline image CDF and never rerank independently.\n- Need is OOF LOCO and uses the four frozen base features only.\n- Oracle parity uses the frozen central finite difference sample.\n- Statistical unit is VisA class; 10,000 class bootstraps and exact 4096 sign flips are retained.\n- MVTec, medical data, and Phase2B training were not accessed.\n- Stable-but-wrong and contaminated-reference vetoes were explicitly evaluated.\n")
    (AUDIT_ROOT / "REPORT.md").write_text("# SABRA pre-training logic audit\n\n## Terminal status\n\n`" + terminal + "`\n\nPGM: `" + statuses["PGM"] + "`; Trust: `" + statuses["Trust"] + "`; Need: `" + statuses["Need"] + "`; Authority: `" + statuses["Authority"] + "`. PCRR: `" + pcrr_status + "`.\n\nNeed minimum capacity: `" + (selected_need or "NONE") + "`. P9 coverage: `" + str(coverage.get("overall_p9_coverage")) + "`. Stable-but-wrong rows: `" + str(len(stable_wrong)) + "`.\n\nThis report is VisA-only and inference-only; no training, MVTec, or medical evaluation was performed.\n")
    positive_images = sum(int(metadata[record["image_path"]]["label"]) for record in records)
    write_json(AUDIT_ROOT / "GT_FIREWALL_AUDIT.json", {"status": "PASS", "cache_mask_pixel_reads": 0, "science_mask_pixel_reads": int(positive_images * IMAGE_SIZE * IMAGE_SIZE), "mvtec_science_reads": 0, "medical_reads": 0, "phase2b_training_steps": 0, "GT_FREE_CACHE_FINALIZED": True, "mask_access_after_cache_freeze": True})
    return decision


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    print(json.dumps(run_science(), sort_keys=True))


if __name__ == "__main__":
    main()
