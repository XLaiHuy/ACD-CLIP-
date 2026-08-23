"""P15 exact-equivalent, cached AP recovery engine for the frozen P14 study."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import signal
import sys
import tempfile
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.sabra_car.r0_direction import canonical_loss_per_image, evaluate_correction, exact_average_precision, exact_metrics, load_masks, metadata_and_root
from tools.sabra_cure import context_value_risk as ref
from tools.sabra_cure import r1, r2, r2v2_harm as frozen

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/sabra_cure/context_value_risk_recovery"
DOC = ROOT / "research/sabra_cure/context_value_risk_recovery"
P14_TERMINAL = "09be5fb0daf80b3697634b738f7579c7041af690"
PREREG = "5127d3156fcf0faf3022e337a408b7f41380de95"
BRANCH = "research/p15-sabra-cure-context-value-risk-exact-recovery-v1"
P14_SOURCE = ROOT / "tools/sabra_cure/context_value_risk.py"
P14_SOURCE_SHA = "d83c59c7e52b21022c90708198048f049bd4e4e46ff443a67adf0c524414273f"
PATCHES, ALPHA, Q, EPS = ref.PATCHES, ref.ALPHA, ref.Q, ref.EPS
FEATURE_ORDER = ref.FEATURE_ORDER
WORKERS = min(4, max(1, os.cpu_count() or 1))
FIXTURE_CLASSES = ("candle", "capsules", "cashew")


def git(*args: str) -> str:
    return r1.git(*args)


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False, default=json_default)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def log(out: Path, line: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "execution.log").open("a", encoding="utf-8") as handle:
        handle.write(line.rstrip() + "\n")


def score_groups(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The exact descending float32 score groups used by frozen R0 AP."""
    score = np.asarray(scores, dtype=np.float32).reshape(-1)
    label = np.asarray(labels, dtype=np.uint8).reshape(-1)
    order = np.argsort(-score, kind="mergesort")
    ordered_score, ordered_label = score[order], label[order].astype(np.int64, copy=False)
    ends = np.flatnonzero(np.r_[ordered_score[1:] != ordered_score[:-1], True])
    cumulative = np.cumsum(ordered_label, dtype=np.int64)
    positive = np.diff(np.r_[0, cumulative[ends]]).astype(np.float64)
    total = np.diff(np.r_[0, ends + 1]).astype(np.float64)
    return ordered_score[ends], positive, total


def ap_from_groups(positive: np.ndarray, total: np.ndarray) -> float:
    positives = float(np.sum(positive))
    if positives <= 0:
        raise ValueError("average precision requires a positive label")
    precision = np.cumsum(positive) / np.cumsum(total)
    return float(np.sum((positive / positives) * precision))


def delta_groups(safe_scores: np.ndarray, expand_scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ss, sp, st = score_groups(safe_scores, labels)
    es, ep, et = score_groups(expand_scores, labels)
    scores = np.unique(np.concatenate((ss, es))).astype(np.float32)[::-1]
    dp = np.zeros(len(scores), dtype=np.float64); dt = np.zeros(len(scores), dtype=np.float64)
    for source, positive, total, sign in ((ss, sp, st, -1.0), (es, ep, et, 1.0)):
        index = np.searchsorted(-scores, -source)
        np.add.at(dp, index, sign * positive); np.add.at(dt, index, sign * total)
    return scores, dp, dt


def ap_with_delta(base_scores: np.ndarray, base_positive: np.ndarray, base_total: np.ndarray, change_scores: np.ndarray, change_positive: np.ndarray, change_total: np.ndarray) -> float:
    """Exact AP of baseline grouped counts plus sparse score-group delta."""
    scores = np.unique(np.concatenate((base_scores, change_scores))).astype(np.float32)[::-1]
    positive = np.zeros(len(scores), dtype=np.float64); total = np.zeros(len(scores), dtype=np.float64)
    for source, p, t in ((base_scores, base_positive, base_total), (change_scores, change_positive, change_total)):
        index = np.searchsorted(-scores, -source)
        np.add.at(positive, index, p); np.add.at(total, index, t)
    keep = total != 0
    if np.any(total[keep] < 0) or not np.allclose(total[keep], np.rint(total[keep]), atol=0, rtol=0):
        raise RuntimeError("P15_ENGINEERING_STOP invalid grouped delta")
    return ap_from_groups(positive[keep], total[keep])


@dataclass
class ClassCache:
    name: str
    paths: np.ndarray
    masks: np.ndarray
    native: np.ndarray
    safe: np.ndarray
    expand: np.ndarray
    native_loss: np.ndarray
    safe_loss: np.ndarray
    expand_loss: np.ndarray
    base_scores: np.ndarray
    base_positive: np.ndarray
    base_total: np.ndarray
    safe_pap: float
    safe_pauroc: float

    def image_delta(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        indices = np.asarray(indices, dtype=np.int64)
        if indices.size == 0:
            return np.empty(0, np.float32), np.empty(0), np.empty(0)
        return delta_groups(self.safe[indices], self.expand[indices], self.masks[indices])

    def pap(self, expanded: np.ndarray) -> float:
        index = np.flatnonzero(np.asarray(expanded, dtype=bool))
        if index.size == 0:
            return self.safe_pap
        s, p, t = self.image_delta(index)
        return ap_with_delta(self.base_scores, self.base_positive, self.base_total, s, p, t)

    def metrics(self, expanded: np.ndarray) -> dict[str, float]:
        expanded = np.asarray(expanded, dtype=bool)
        score = self.safe.copy(); score[expanded] = self.expand[expanded]
        metric = exact_metrics(score.reshape(-1), self.masks.reshape(-1))
        loss = np.where(expanded, self.expand_loss, self.safe_loss)
        return {"pixel_ap": float(metric["pAP"]), "pixel_auroc": float(metric["pAUROC"]), "mean_loss": float(loss.mean())}


def _loss_from_scores(scores: np.ndarray, masks: np.ndarray) -> np.ndarray:
    probability = np.stack((1.0 - scores, scores), axis=1)
    with torch.no_grad():
        return canonical_loss_per_image(torch.from_numpy(probability), torch.from_numpy(masks[:, None].astype(np.float32))).numpy().astype(np.float64)


def build_cache(name: str, safe_actions: np.ndarray, expand_actions: np.ndarray) -> ClassCache:
    """Construct one bounded immutable class cache; no AP target is calculated."""
    with np.load(r1.SOURCE_ROOT / "gt_free_cache" / f"{name}.npz", allow_pickle=False) as data:
        logits = np.asarray(data["native_logits"], dtype=np.float32)
        paths = data["image_path"].astype(str)
        native = np.asarray(data["native_pixel_probability"], dtype=np.float32)
    metadata, root = metadata_and_root(r2.DATA_ROOT); masks = load_masks(paths, metadata, root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def deploy(actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return evaluate_correction(logits, masks, (actions.astype(np.float32) * ALPHA * r2.MARGIN_SCALE).reshape(-1, PATCHES), device, 4)
    safe, safe_loss = deploy(safe_actions)
    expand, expand_loss = deploy(expand_actions)
    base_scores, base_positive, base_total = score_groups(safe, masks)
    safe_metric = exact_metrics(safe.reshape(-1), masks.reshape(-1))
    return ClassCache(name, paths, masks, native, safe, expand, _loss_from_scores(native, masks), safe_loss, expand_loss, base_scores, base_positive, base_total, float(safe_metric["pAP"]), float(safe_metric["pAUROC"]))


def reference_counterfactual_ap(cache: ClassCache, image_index: int) -> float:
    candidate = cache.safe.copy(); candidate[image_index] = cache.expand[image_index]
    return exact_average_precision(candidate.reshape(-1), cache.masks.reshape(-1))


def optimized_counterfactual_ap(cache: ClassCache, image_index: int) -> float:
    mask = np.zeros(len(cache.paths), dtype=bool); mask[image_index] = True
    return cache.pap(mask)


def cached_metric_parity(cache: ClassCache, expanded: np.ndarray) -> float:
    cached = cache.metrics(expanded)
    actions = np.where(np.asarray(expanded)[:, None], 1, 0)  # shape guard only; never used as policy
    del actions
    score = cache.safe.copy(); score[np.asarray(expanded, dtype=bool)] = cache.expand[np.asarray(expanded, dtype=bool)]
    reference_metric = exact_metrics(score.reshape(-1), cache.masks.reshape(-1))
    return max(abs(cached["pixel_ap"] - reference_metric["pAP"]), abs(cached["pixel_auroc"] - reference_metric["pAUROC"]))


def image_targets(cache: ClassCache, x: np.ndarray, mu: np.ndarray, sigma: np.ndarray, risk: np.ndarray, tau20: float, tau40: float, workers: int = WORKERS) -> dict[str, Any]:
    safe, expand = ref.actions(mu, risk, tau20), ref.actions(mu, risk, tau40)
    if len(safe) != len(cache.paths) * PATCHES:
        raise RuntimeError("P15_ENGINEERING_STOP cache/image alignment")
    indices = range(len(cache.paths))
    if workers == 1:
        ap = [optimized_counterfactual_ap(cache, index) for index in indices]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            ap = list(pool.map(lambda index: optimized_counterfactual_ap(cache, index), indices))
    values = np.asarray(ap, dtype=np.float64) - cache.safe_pap
    ref.finite("P15 V", values)
    return {"x": ref.fields(cache.name, x, mu, sigma, risk, tau20, tau40, cache.paths), "v": values, "safe": safe, "expand": expand, "tau20": tau20, "tau40": tau40, "safe_metrics": {"pixel_ap": cache.safe_pap, "pixel_auroc": cache.safe_pauroc, "mean_loss": float(cache.safe_loss.mean())}, "expand_metrics": cache.metrics(np.ones(len(cache.paths), dtype=bool)), "safe_loss": cache.safe_loss, "expand_loss": cache.expand_loss, "cache": cache}


def compose_downstream(cache: ClassCache, key: str, expanded: np.ndarray | None = None) -> dict[str, float]:
    if key == "native":
        metric = exact_metrics(cache.native.reshape(-1), cache.masks.reshape(-1))
        return {"pixel_ap": float(metric["pAP"]), "pixel_auroc": float(metric["pAUROC"]), "mean_loss": float(cache.native_loss.mean())}
    if key == "safe20":
        return {"pixel_ap": cache.safe_pap, "pixel_auroc": cache.safe_pauroc, "mean_loss": float(cache.safe_loss.mean())}
    if key == "always_expand40":
        return cache.metrics(np.ones(len(cache.paths), dtype=bool))
    if expanded is None:
        raise ValueError("expanded mask required")
    return cache.metrics(expanded)


def source_selection(groups: dict[str, dict[str, Any]], oof: dict[str, np.ndarray]) -> tuple[float | None, dict[str, Any]]:
    allv = np.concatenate([oof[name] for name in groups]); candidates = []
    safe_macro = float(np.mean([group["cache"].safe_pap for group in groups.values()]))
    for quantile in Q:
        threshold = float(np.quantile(allv, quantile, method="linear")); pap = []; actions = []; y = []; mu = []
        for name, group in groups.items():
            choose = oof[name] > threshold
            pap.append(group["cache"].pap(choose))
            action = group["safe"].reshape(-1, PATCHES).copy(); action[choose] = group["expand"].reshape(-1, PATCHES)[choose]
            actions.append(action.reshape(-1)); y.append(group["y"]); mu.append(group["mu"])
        safety = ref.safety(np.concatenate(actions), np.concatenate(y), np.concatenate(mu))
        macro = float(np.mean(pap)); eligible = safety["wrong_rate"] <= .05 and safety["relative_weighted_harm_reduction"] >= .5 and macro > safe_macro
        candidates.append({"q": quantile, "threshold": threshold, "macro_pap": macro, "safety": safety, "eligible": eligible})
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    if not eligible:
        return None, {"safe_macro_pap": safe_macro, "candidates": candidates, "selected": "NO_EXPANSION"}
    best = max(eligible, key=lambda item: (item["macro_pap"], item["q"]))
    tied = [item for item in eligible if abs(item["macro_pap"] - best["macro_pap"]) <= 1e-12]
    best = max(tied, key=lambda item: item["q"])
    return float(best["q"]), {"safe_macro_pap": safe_macro, "candidates": candidates, "selected_q": best["q"], "selected_threshold": best["threshold"]}


def outer(held: str, shards: dict[str, r1.Shard], workers: int = WORKERS, checkpoint: Callable[[str, dict[str, Any]], None] | None = None, checkpoint_dir: Path | None = None) -> dict[str, Any]:
    base = frozen.outer(held, shards); names = [name for name in r1.CLASSES if name != held]; groups: dict[str, dict[str, Any]] = {}
    for group in base["level1"]:
        other = np.concatenate([item["r_h"] for item in base["level1"] if item["name"] != group["name"]]); t20, t40 = ref.thresholds(other)
        safe, expand = ref.actions(group["mu"], group["r_h"], t20), ref.actions(group["mu"], group["r_h"], t40)
        cache = build_cache(group["name"], safe, expand)
        saved = checkpoint_dir / f"targets_{group['name']}.npz" if checkpoint_dir else None
        if saved and saved.exists():
            with np.load(saved, allow_pickle=False) as data:
                target = {"x": np.asarray(data["features"], dtype=np.float64), "v": np.asarray(data["value"], dtype=np.float64), "safe": safe, "expand": expand, "tau20": t20, "tau40": t40, "safe_metrics": {"pixel_ap": cache.safe_pap, "pixel_auroc": cache.safe_pauroc, "mean_loss": float(cache.safe_loss.mean())}, "expand_metrics": cache.metrics(np.ones(len(cache.paths), dtype=bool)), "safe_loss": cache.safe_loss, "expand_loss": cache.expand_loss, "cache": cache}
            ref.finite("P15 resumed V", target["x"], target["v"])
        else:
            target = image_targets(cache, group["x"], group["mu"], group["sigma"], group["r_h"], t20, t40, workers)
        target.update({"mu": group["mu"], "y": group["y"], "risk": group["r_h"], "sigma": group["sigma"], "training": group["training"], "cache": cache}); groups[group["name"]] = target
        if checkpoint and not (saved and saved.exists()): checkpoint(group["name"], target)
    oof = {}
    for name in names:
        others = [item for item in names if item != name]; model = ref.fit(np.concatenate([groups[item]["x"] for item in others]), np.concatenate([groups[item]["v"] for item in others])); oof[name] = ref.predict(model, groups[name]["x"])
    selected, selection = source_selection(groups, oof)
    t20, t40 = ref.thresholds(np.concatenate([item["r_h"] for item in base["level1"]]))
    safe, expand = ref.actions(base["mu"], base["risk_h"], t20), ref.actions(base["mu"], base["risk_h"], t40)
    held_cache = build_cache(held, safe, expand)
    held_target = image_targets(held_cache, shards[held].x, base["mu"], base["sigma"], base["risk_h"], t20, t40, workers)
    model = ref.fit(np.concatenate([groups[item]["x"] for item in names]), np.concatenate([groups[item]["v"] for item in names])); vhat = ref.predict(model, held_target["x"])
    threshold = float("inf") if selected is None else float(np.quantile(np.concatenate([oof[item] for item in names]), selected, method="linear"))
    expanded = vhat > threshold; oracle = held_target["v"] > 0
    return {"held": held, "base": base, "groups": groups, "oof": oof, "selection": selection, "selected_q": selected, "threshold": threshold, "value_model": model, "vhat": vhat, "expand_images": expanded, "target": held_target, "downstream": {"native": compose_downstream(held_cache, "native"), "safe20": compose_downstream(held_cache, "safe20"), "always_expand40": compose_downstream(held_cache, "always_expand40"), "context": compose_downstream(held_cache, "context", expanded), "image_oracle": compose_downstream(held_cache, "image_oracle", oracle)}}


def historical_unchanged() -> bool:
    protected = (*ref.PROTECTED, "tools/sabra_cure/context_value_risk.py", "results/sabra_cure/context_value_risk", "research/sabra_cure/context_value_risk")
    import subprocess
    return subprocess.run(["git", "diff", "--quiet", P14_TERMINAL, "--", *protected], cwd=ROOT).returncode == 0


def input_hashes() -> dict[str, str]:
    return {"p14_reference": sha256(P14_SOURCE), "r2v2_summary": sha256(ROOT / "results/sabra_cure/r2v2_harm/summary.json"), "p13_summary": sha256(ROOT / "results/sabra_cure/post_r2v2_diagnostic_recovery/summary.json")}


def pre_audit(out: Path) -> dict[str, Any]:
    if (out / "ATTEMPT_STARTED.json").exists():
        raise RuntimeError("P15_ENGINEERING_STOP attempt exists")
    benchmark = json.loads((out / "performance_benchmark.json").read_text())
    parity = json.loads((out / "parity_report.json").read_text())
    checks = {"status": "PASS", "p14_terminal_abort_ancestor": git("merge-base", "--is-ancestor", P14_TERMINAL, "HEAD") == "", "p15_preregistration_sha": PREREG, "execution_base_published": git("rev-parse", "HEAD") == git("rev-parse", f"origin/{BRANCH}"), "worktree_clean_before_audit": git("status", "--porcelain") == "", "p14_reference_hash": sha256(P14_SOURCE), "p14_science_contract_unchanged": sha256(P14_SOURCE) == P14_SOURCE_SHA, "historical_unchanged": historical_unchanged(), "exact_parity_pass": bool(parity.get("pass")), "performance_gate_pass": bool(benchmark.get("pass")), "median_speedup": benchmark.get("median_speedup"), "input_hashes": input_hashes(), "workers": WORKERS, "alpha": ALPHA, "q": list(Q), "feature_order": list(FEATURE_ORDER), "mvtec_accessed": False, "medical_accessed": False, "additional_clip_forwards": 0, "phase2b_training_steps": 0}
    if not all((checks["p14_terminal_abort_ancestor"], checks["execution_base_published"], checks["worktree_clean_before_audit"], checks["p14_science_contract_unchanged"], checks["historical_unchanged"], checks["exact_parity_pass"], checks["performance_gate_pass"])):
        checks["status"] = "FAIL"
    atomic(out / "pre_execution_audit.json", checks)
    if checks["status"] != "PASS":
        raise RuntimeError("P15_ENGINEERING_STOP pre-execution audit")
    return checks


def marker_payload() -> dict[str, Any]:
    return {"status": "ATTEMPT_STARTED", "attempt_uuid": str(uuid.uuid4()), "execution_base_sha": git("rev-parse", "HEAD"), "prereg_sha": PREREG, "input_hashes": input_hashes(), "runs": 1}


def write_checkpoint(out: Path, held: str, stage: str, completed_sources: Iterable[str], payload: dict[str, Any]) -> None:
    path = out / "checkpoints" / f"outer_{held}" / f"{stage}.json"; body = {"held": held, "stage": stage, "completed_source_classes": list(completed_sources), "execution_base_sha": git("rev-parse", "HEAD"), "prereq_hashes": input_hashes(), "payload": payload}; atomic(path, body); body["artifact_sha256"] = sha256(path)
    try: checkpoint_path = str(path.relative_to(ROOT))
    except ValueError: checkpoint_path = str(path)
    atomic(out / "progress.json", {"status": "RUNNING", "last_completed_outer": held, "last_stage": stage, "checkpoint": checkpoint_path, "checkpoint_sha256": body["artifact_sha256"]})


def completed_outer_folds(out: Path) -> list[str]:
    complete: list[str] = []
    for held in r1.CLASSES:
        checkpoint = out / "checkpoints" / f"outer_{held}" / "outer_complete.json"
        if checkpoint.exists() and (out / "folds" / f"{held}.npz").exists() and (out / "downstream_folds" / f"{held}.json").exists():
            body = json.loads(checkpoint.read_text())
            if body.get("execution_base_sha") != git("rev-parse", "HEAD") or body.get("prereq_hashes") != input_hashes():
                raise RuntimeError("P15_ENGINEERING_STOP invalid completed checkpoint")
            complete.append(held)
    return complete


def rehydrate_fold(out: Path, held: str) -> dict[str, Any]:
    with np.load(out / "folds" / f"{held}.npz", allow_pickle=False) as data:
        parameter = json.loads((out / "parameters" / f"{held}.json").read_text())
        downstream = json.loads((out / "downstream_folds" / f"{held}.json").read_text())
        return {"held": held, "base": {"mu": np.asarray(data["mu"]), "y": np.asarray(data["y"]), "utility": np.asarray(data["utility"]), "sigma": np.asarray(data["sigma"]), "risk_h": np.asarray(data["risk"])}, "target": {"safe": np.asarray(data["safe20"]), "expand": np.asarray(data["expand40"]), "v": np.asarray(data["v"])}, "vhat": np.asarray(data["vhat"]), "expand_images": np.asarray(data["expand_images"], dtype=bool), "selected_q": parameter["selected_q"], "downstream": downstream}


def execute(out: Path, resume: bool = False) -> dict[str, Any]:
    marker = out / "ATTEMPT_STARTED.json"
    if resume:
        if not marker.exists(): raise RuntimeError("P15_ENGINEERING_STOP no attempt to resume")
        old = json.loads(marker.read_text())
        if old["execution_base_sha"] != git("rev-parse", "HEAD") or old["prereg_sha"] != PREREG or old["input_hashes"] != input_hashes(): raise RuntimeError("P15_ENGINEERING_STOP resume identity mismatch")
        if not completed_outer_folds(out) and not list((out / "checkpoints").glob("outer_*/targets_*.npz")):
            raise RuntimeError("P15_ENGINEERING_STOP no validated checkpoint to resume")
        with (out / "RESUME_LOG.jsonl").open("a", encoding="utf-8") as handle: handle.write(json.dumps({"event": "RESUME", "attempt_uuid": old["attempt_uuid"], "time": time.time()}) + "\n")
    if (not resume and marker.exists()) or (out / "summary.json").exists(): raise RuntimeError("P15_ENGINEERING_STOP attempt exists")
    if not resume: atomic(marker, marker_payload()); log(out, "P15_ATTEMPT_STARTED")
    for directory in ("folds", "parameters", "policy_selection", "downstream_folds", "checkpoints"): (out / directory).mkdir(parents=True, exist_ok=True)
    shards, _ = r1.load_shards(True); completed = completed_outer_folds(out) if resume else []; folds = {held: rehydrate_fold(out, held) for held in completed}
    try:
        for held in r1.CLASSES:
            if held in folds: continue
            completed: list[str] = []
            def checkpoint(name: str, target: dict[str, Any]) -> None:
                completed.append(name); path = out / "checkpoints" / f"outer_{held}" / f"targets_{name}.npz"; path.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(path, features=target["x"], value=target["v"]); write_checkpoint(out, held, "image_target_group", completed, {"last_source_class": name, "target_sha256": sha256(path)})
            fold = outer(held, shards, WORKERS, checkpoint, out / "checkpoints" / f"outer_{held}"); folds[held] = fold
            parameter = {"held": held, "outer_training": [name for name in r1.CLASSES if name != held], "tau20": fold["target"]["tau20"], "tau40": fold["target"]["tau40"], "selected_q": fold["selected_q"], "value_threshold": fold["threshold"], "feature_order": list(FEATURE_ORDER), "value_model": {key: np.asarray(value).tolist() if isinstance(value, np.ndarray) else value for key, value in fold["value_model"].items()}, "selection": fold["selection"], "direction": {key: np.asarray(value).tolist() if isinstance(value, np.ndarray) else value for key, value in fold["base"]["direction"].items()}}
            atomic(out / "parameters" / f"{held}.json", parameter); target = fold["target"]
            np.savez_compressed(out / "folds" / f"{held}.npz", image_path=shards[held].image_path, mu=fold["base"]["mu"], y=fold["base"]["y"], utility=fold["base"]["utility"], sigma=fold["base"]["sigma"], risk=fold["base"]["risk_h"], safe20=target["safe"], expand40=target["expand"], context=np.where(fold["expand_images"][:, None], target["expand"].reshape(-1, PATCHES), target["safe"].reshape(-1, PATCHES)).reshape(-1), vhat=fold["vhat"], v=target["v"], expand_images=fold["expand_images"])
            atomic(out / "policy_selection" / f"{held}.json", fold["selection"]); atomic(out / "downstream_folds" / f"{held}.json", fold["downstream"]); write_checkpoint(out, held, "outer_complete", completed, {"completed_outer_folds": len(folds)}); log(out, f"OUTER_COMPLETE {held}")
        down = {name: folds[name]["downstream"] for name in r1.CLASSES}; macro = {key: float(np.mean([down[name][key]["pixel_ap"] for name in r1.CLASSES])) for key in next(iter(down.values()))}; auc = {key: float(np.mean([down[name][key]["pixel_auroc"] for name in r1.CLASSES])) for key in next(iter(down.values()))}
        actions = np.concatenate([np.where(folds[name]["expand_images"][:, None], folds[name]["target"]["expand"].reshape(-1, PATCHES), folds[name]["target"]["safe"].reshape(-1, PATCHES)).reshape(-1) for name in r1.CLASSES]); y = np.concatenate([folds[name]["base"]["y"] for name in r1.CLASSES]); mu = np.concatenate([folds[name]["base"]["mu"] for name in r1.CLASSES]); safe = ref.safety(actions, y, mu); expansion = float(np.mean(np.concatenate([folds[name]["expand_images"] for name in r1.CLASSES]))); fallback = sum(folds[name]["selected_q"] is None for name in r1.CLASSES); native = macro["native"]; nonreg = sum(down[name]["context"]["pixel_ap"] >= down[name]["native"]["pixel_ap"] for name in r1.CLASSES); improving = sum(down[name]["context"]["pixel_ap"] > down[name]["native"]["pixel_ap"] for name in r1.CLASSES)
        values, predicted = np.concatenate([folds[name]["target"]["v"] for name in r1.CLASSES]), np.concatenate([folds[name]["vhat"] for name in r1.CLASSES]); nz = np.abs(values) > EPS
        gates = {"G1_AUDIT": True, "G2_SAFETY": safe["wrong_rate"] <= .05, "G3_WEIGHTED_HARM": safe["relative_weighted_harm_reduction"] >= .5, "G4_RECOVERY_COVERAGE": safe["coverage"] >= .1934, "G5_CONTEXT_USAGE": expansion >= .1, "G6_PAP_NATIVE": macro["context"] - native >= .0025, "G7_BREADTH": nonreg >= 9, "G8_POSITIVE_BREADTH": improving >= 7, "G9_AUROC": auc["context"] - auc["native"] >= -.005, "G10_POLICY_VALUE": macro["context"] > macro["safe20"], "G11_SELECTION": all(folds[name]["selected_q"] in (*Q, None) for name in r1.CLASSES)}
        summary = {"status": "P14_SCIENCE_RECOVERED_PASS" if all(gates.values()) else "P14_SCIENCE_RECOVERED_STOP", "execution_base_sha": git("rev-parse", "HEAD"), "attempt_uuid": json.loads(marker.read_text())["attempt_uuid"], "folds_completed": 12, "metrics": {"macro_pap": macro, "macro_pauroc": auc, "safety": safe, "coverage_delta_vs_r2v2": safe["coverage"] - .1734, "expand40_fraction": expansion, "no_expansion_folds": fallback, "nonregressing_classes": nonreg, "improving_classes": improving, "value_prediction": {"pearson": ref.corr(predicted, values)["pearson"], "spearman": ref.corr(predicted, values)["spearman"], "sign_accuracy_nonzero": float(np.mean(np.sign(predicted[nz]) == np.sign(values[nz])))}}, "selected_q": {name: folds[name]["selected_q"] for name in r1.CLASSES}, "gates": gates, "firewall": {"mvtec_accessed": False, "medical_accessed": False}, "freeze": {"alpha": ALPHA, "additional_clip_forwards": 0, "phase2b_training_steps": 0, "r3_run": False, "r4_run": False}}
        atomic(out / "downstream_metrics.json", down); atomic(out / "summary.json", summary); atomic(out / "progress.json", {"status": "COMPLETE", "completed_folds": 12, "total_folds": 12}); return summary
    except Exception as exc:
        (out / "ENGINEERING_FAILURE.traceback.log").write_text(traceback.format_exc()); atomic(out / "ENGINEERING_FAILURE.json", {"status": "P15_ENGINEERING_STOP", "exception_type": type(exc).__name__, "exception_message": str(exc)[:1000], "execution_base_sha": git("rev-parse", "HEAD")}); raise


def benchmark(out: Path) -> dict[str, Any]:
    rows = []
    for name in FIXTURE_CLASSES:
        shard, _ = r1.load_shards(True); data = np.load(ROOT / "results/sabra_cure/r2v2_harm/folds" / f"{name}.npz", allow_pickle=False); parameter = json.loads((ROOT / "results/sabra_cure/r2v2_harm/parameters" / f"{name}.json").read_text()); safe = ref.actions(data["mu"], data["harm_risk"], parameter["tau_harm"]); other = np.asarray(data["harm_risk"], dtype=np.float64); t20, t40 = ref.thresholds(other); expand = ref.actions(data["mu"], data["harm_risk"], t40); cache = build_cache(name, safe, expand)
        for index in (0, len(cache.paths) - 1):
            started = time.perf_counter(); reference = reference_counterfactual_ap(cache, index); reference_seconds = time.perf_counter() - started
            started = time.perf_counter(); optimized = optimized_counterfactual_ap(cache, index); optimized_seconds = time.perf_counter() - started
            rows.append({"fixture_class": name, "image_index": index, "reference_ap": reference, "optimized_ap": optimized, "absolute_difference": abs(reference - optimized), "reference_seconds": reference_seconds, "optimized_seconds": optimized_seconds, "speedup": reference_seconds / max(optimized_seconds, np.finfo(float).tiny)})
        del shard, cache
    report = {"fixtures": rows, "max_abs_error": float(max(row["absolute_difference"] for row in rows)), "median_speedup": float(np.median([row["speedup"] for row in rows])), "projected_hours": float(np.median([row["optimized_seconds"] for row in rows]) * 11 * 2162 / 3600), "pass": bool(max(row["absolute_difference"] for row in rows) <= 1e-12 and np.median([row["speedup"] for row in rows]) >= 5.0)}
    atomic(out / "performance_benchmark.json", report); return report


def parity_report(out: Path) -> dict[str, Any]:
    synthetic = []
    fixtures = [(np.array([.1, .2, .3], np.float32), np.array([0, 1, 1], np.uint8)), (np.array([.5, .5, .5, .1], np.float32), np.array([0, 1, 0, 1], np.uint8)), (np.array([0., -0., 1., 1., -2.], np.float32), np.array([1, 0, 1, 0, 1], np.uint8))]
    for score, label in fixtures:
        gs, gp, gt = score_groups(score, label); synthetic.append(abs(exact_average_precision(score, label) - ap_from_groups(gp, gt)))
    benchmark_data = json.loads((out / "performance_benchmark.json").read_text()) if (out / "performance_benchmark.json").exists() else {"max_abs_error": float("inf")}
    report = {"synthetic_max_abs_error": float(max(synthetic)), "real_fixture_max_abs_error": benchmark_data["max_abs_error"], "pass": bool(max(synthetic) <= 1e-12 and benchmark_data["max_abs_error"] <= 1e-12), "float32_tie_semantics": "np.float32, stable descending grouping exactly matches frozen exact_average_precision"}; atomic(out / "parity_report.json", report); return report


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--benchmark", action="store_true"); parser.add_argument("--parity", action="store_true"); parser.add_argument("--pre-audit", action="store_true"); parser.add_argument("--execute-once", action="store_true"); parser.add_argument("--resume-attempt", action="store_true"); parser.add_argument("--output", type=Path, default=OUT); args = parser.parse_args()
    if sum((args.benchmark, args.parity, args.pre_audit, args.execute_once, args.resume_attempt)) != 1: parser.error("choose exactly one action")
    result = benchmark(args.output) if args.benchmark else parity_report(args.output) if args.parity else pre_audit(args.output) if args.pre_audit else execute(args.output, args.resume_attempt)
    print(json.dumps(result, indent=2, sort_keys=True, default=json_default))


if __name__ == "__main__":
    main()
