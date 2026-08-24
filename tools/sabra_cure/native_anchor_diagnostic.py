"""P21 exact native-anchored action-space diagnostic.

This module is deliberately separate from the frozen P14--P20 runners.  Its
fast path stores only exact sparse score-group *count* deltas; it never adds
scalar AP deltas.  P21 remains pre-marker until ``--run`` creates its marker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from tools.sabra_car.r0_direction import exact_metrics
from tools.sabra_cure import context_value_risk as p14
from tools.sabra_cure import context_value_risk_recovery as p15
from tools.sabra_cure import r1, r2v2_harm as frozen

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/sabra_cure/native_anchor_diagnostic"
P20 = ROOT / "results/sabra_cure/context_value_risk_json_sentinel"
PARENT = "fcbd218c28cdd5c58f54e5d187f8d8ff9bdaa63f"
PREREG = "baa1715af18dd2444f2871321dfc05cca7e22aa1"
INTERPRETATION = "634208b"
EPS = 1e-12
PATCHES = 1369
A0 = ("NATIVE", "SAFE20", "EXPAND40")
A1 = ("NATIVE", "SAFE20", "SAFE30", "EXPAND40")


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def atomic(path: Path, payload: Any) -> None:
    """Strict, atomic JSON -- non-finite values are an engineering error."""
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


def fold_dir(held: str) -> Path:
    return P20 / "folds" / held


def load_fold(held: str) -> dict[str, np.ndarray]:
    path = fold_dir(held) / "fold.npz"
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def grouped_counts(scores: np.ndarray, labels: np.ndarray, union: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Put frozen P15 score groups into one precomputed descending score axis."""
    score, positive, total = p15.score_groups(scores, labels)
    index = np.searchsorted(-union, -score)
    out_positive = np.zeros(len(union), dtype=np.float64)
    out_total = np.zeros(len(union), dtype=np.float64)
    np.add.at(out_positive, index, positive)
    np.add.at(out_total, index, total)
    return out_positive, out_total


@dataclass(frozen=True)
class SparseDelta:
    """Exact count change on the one frozen descending union-score axis."""

    index: np.ndarray
    positive: np.ndarray
    total: np.ndarray


EMPTY_DELTA = SparseDelta(
    np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
)


class NativeAnchorEngine:
    """Exact native-base AP state with sparse per-image/action grouped deltas.

    The one union-score axis is exact float32 P15 semantics. Image/action
    deltas are indexed and created on demand; no dense image-by-score-bin
    cache is ever retained.
    """

    def __init__(self, native: np.ndarray, actions: dict[str, np.ndarray], masks: np.ndarray):
        self.native = np.asarray(native, dtype=np.float32)
        self.actions = {name: np.asarray(score, dtype=np.float32) for name, score in actions.items()}
        self.masks = np.asarray(masks, dtype=np.uint8)
        if tuple(self.actions)[:1] != ("NATIVE",) or not np.array_equal(self.actions["NATIVE"], self.native):
            raise RuntimeError("P21_ENGINEERING_STOP native action contract")
        if any(score.shape != self.native.shape for score in self.actions.values()) or self.masks.shape != self.native.shape:
            raise RuntimeError("P21_ENGINEERING_STOP score/mask alignment")
        self.names = tuple(self.actions)
        self.n_images = len(self.native)
        self.union = np.unique(np.concatenate([score.reshape(-1) for score in self.actions.values()])).astype(np.float32)[::-1]
        self.base_positive, self.base_total = grouped_counts(self.native, self.masks, self.union)
        self.delta_cache: Path | None = None

    def image_delta(self, images: np.ndarray | int, action: str) -> SparseDelta:
        if action == "NATIVE":
            return EMPTY_DELTA
        selected = np.atleast_1d(np.asarray(images, dtype=np.int64))
        if not len(selected):
            return EMPTY_DELTA
        if self.delta_cache is not None and len(selected) == 1:
            path = self.delta_cache / f"{action}_{int(selected[0]):04d}.npz"
            if not path.exists():
                raise RuntimeError("P21_ENGINEERING_STOP missing delta cache")
            with np.load(path, allow_pickle=False) as data:
                return SparseDelta(np.asarray(data["index"], dtype=np.int64), np.asarray(data["positive"], dtype=np.float64), np.asarray(data["total"], dtype=np.float64))
        score, positive, total = p15.delta_groups(self.native[selected], self.actions[action][selected], self.masks[selected])
        index = np.searchsorted(-self.union, -score)
        keep = (positive != 0) | (total != 0)
        return SparseDelta(index[keep], positive[keep], total[keep])

    def build_delta_cache(self, directory: Path) -> None:
        """Persist exact indexed deltas once, outside RAM, for the active class."""
        directory.mkdir(parents=True, exist_ok=True)
        if any(directory.iterdir()):
            raise RuntimeError("P21_ENGINEERING_STOP nonempty delta cache directory")
        previous = self.delta_cache; self.delta_cache = None
        try:
            for action in self.names:
                if action == "NATIVE":
                    continue
                for image in range(self.n_images):
                    delta = self.image_delta(image, action)
                    path = directory / f"{action}_{image:04d}.npz"
                    # Every per-image count is an exactly representable int32;
                    # compressed storage is a memory-lifetime choice only.
                    np.savez_compressed(path, index=delta.index.astype(np.uint32), positive=delta.positive.astype(np.int32), total=delta.total.astype(np.int32))
                    if hasattr(os, "posix_fadvise"):
                        with path.open("rb") as handle:
                            os.posix_fadvise(handle.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except Exception:
            shutil.rmtree(directory, ignore_errors=True)
            self.delta_cache = previous
            raise
        self.delta_cache = directory

    def clear_delta_cache(self) -> None:
        if self.delta_cache is not None:
            shutil.rmtree(self.delta_cache, ignore_errors=True)
            self.delta_cache = None

    @staticmethod
    def apply(positive: np.ndarray, total: np.ndarray, delta: SparseDelta, sign: float) -> None:
        if len(delta.index):
            np.add.at(positive, delta.index, sign * delta.positive)
            np.add.at(total, delta.index, sign * delta.total)

    def counts(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        positive, total = self.base_positive.copy(), self.base_total.copy()
        for action in self.names:
            selected = np.flatnonzero(state == action)
            if action != "NATIVE" and len(selected):
                self.apply(positive, total, self.image_delta(selected, action), 1.0)
        return positive, total

    @staticmethod
    def ap(positive: np.ndarray, total: np.ndarray) -> float:
        keep = total != 0
        if np.any(total[keep] < 0) or not np.array_equal(total[keep], np.rint(total[keep])):
            raise RuntimeError("P21_ENGINEERING_STOP invalid grouped count state")
        return p15.ap_from_groups(positive[keep], total[keep])

    def candidate(self, positive: np.ndarray, total: np.ndarray, image: int, old: str, candidate: str) -> tuple[np.ndarray, np.ndarray, float]:
        if old == candidate:
            return positive, total, self.ap(positive, total)
        out_positive, out_total = positive.copy(), total.copy()
        self.apply(out_positive, out_total, self.image_delta(image, old), -1.0)
        self.apply(out_positive, out_total, self.image_delta(image, candidate), 1.0)
        return out_positive, out_total, self.ap(out_positive, out_total)

    def compose(self, state: np.ndarray) -> np.ndarray:
        score = self.native.copy()
        for name in self.names:
            selected = np.flatnonzero(state == name)
            if len(selected):
                score[selected] = self.actions[name][selected]
        return score


def gpu_ap(positive: torch.Tensor, total: torch.Tensor) -> float:
    """Exact float64 grouped AP candidate; accepted only after strict parity."""
    if positive.device.type != "cuda" or total.device.type != "cuda":
        raise RuntimeError("P21_ENGINEERING_STOP GPU AP device")
    keep = total != 0
    p, t = positive[keep], total[keep]
    if bool(torch.any(t < 0).item()):
        raise RuntimeError("P21_ENGINEERING_STOP GPU AP count")
    all_positive = torch.sum(p)
    if not bool(all_positive > 0):
        raise RuntimeError("P21_ENGINEERING_STOP GPU AP labels")
    precision = torch.cumsum(p, 0) / torch.cumsum(t, 0)
    return float(torch.sum((p / all_positive) * precision).item())


def gpu_ap_batch(positive: torch.Tensor, total: torch.Tensor) -> np.ndarray:
    """Vectorized E3 candidate AP with the identical per-row arithmetic."""
    if positive.ndim != 2 or total.shape != positive.shape:
        raise RuntimeError("P21_ENGINEERING_STOP GPU AP batch shape")
    cumulative_positive = torch.cumsum(positive, dim=1)
    cumulative_total = torch.cumsum(total, dim=1)
    precision = cumulative_positive / torch.where(cumulative_total == 0.0, torch.ones_like(cumulative_total), cumulative_total)
    all_positive = torch.sum(positive, dim=1)
    return torch.sum((positive / all_positive[:, None]) * precision, dim=1).detach().cpu().numpy().astype(np.float64)


def gpu_apply(positive: torch.Tensor, total: torch.Tensor, delta: SparseDelta, sign: float) -> None:
    if len(delta.index):
        index = torch.as_tensor(delta.index, dtype=torch.int64, device="cuda")
        positive.index_add_(0, index, torch.as_tensor(sign * delta.positive, dtype=torch.float64, device="cuda"))
        total.index_add_(0, index, torch.as_tensor(sign * delta.total, dtype=torch.float64, device="cuda"))


def coordinate_gpu(engine: NativeAnchorEngine, seed: np.ndarray, action_order: tuple[str, ...] = A0, max_sweeps: int = 10) -> dict[str, Any]:
    """GPU implementation of the same frozen sequential trajectory."""
    if not torch.cuda.is_available():
        raise RuntimeError("P21_ENGINEERING_STOP CUDA unavailable")
    state = np.asarray(seed, dtype="<U16").copy()
    base_positive, base_total = engine.counts(state)
    positive = torch.as_tensor(base_positive, dtype=torch.float64, device="cuda")
    total = torch.as_tensor(base_total, dtype=torch.float64, device="cuda")
    current, total_changes = gpu_ap(positive, total), 0
    for sweep in range(1, max_sweeps + 1):
        changes = 0
        for image in range(engine.n_images):
            old = str(state[image]); best, best_ap = old, current
            best_positive, best_total = positive, total
            choices = [candidate for candidate in action_order if candidate != old]
            local = {name: engine.image_delta(image, name) for name in action_order if name != "NATIVE"}
            candidate_positive = positive.repeat((len(choices), 1)); candidate_total = total.repeat((len(choices), 1))
            for row, candidate in enumerate(choices):
                gpu_apply(candidate_positive[row], candidate_total[row], local.get(old, EMPTY_DELTA), -1.0)
                gpu_apply(candidate_positive[row], candidate_total[row], local.get(candidate, EMPTY_DELTA), 1.0)
            values = gpu_ap_batch(candidate_positive, candidate_total)
            for row, (candidate, value) in enumerate(zip(choices, values.tolist())):
                if value > best_ap + EPS:
                    best, best_ap, best_positive, best_total = candidate, value, candidate_positive[row].clone(), candidate_total[row].clone()
            if best != old:
                state[image], current, positive, total = best, best_ap, best_positive, best_total
                changes += 1; total_changes += 1
        if changes == 0:
            return {"state": state, "pap": current, "sweeps": sweep, "converged": True, "changes": total_changes}
    return {"state": state, "pap": current, "sweeps": max_sweeps, "converged": False, "changes": total_changes}


def coordinate(engine: NativeAnchorEngine, seed: np.ndarray, action_order: tuple[str, ...] = A0, max_sweeps: int = 10) -> dict[str, Any]:
    """Frozen sequential coordinate trajectory with strict-improvement updates."""
    state = np.asarray(seed, dtype="<U16").copy()
    positive, total = engine.counts(state)
    current = engine.ap(positive, total)
    total_changes = 0
    for sweep in range(1, max_sweeps + 1):
        changes = 0
        for image in range(engine.n_images):
            old = str(state[image])
            best, best_ap = old, current
            best_positive, best_total = positive, total
            # Action order plus strict update is the prescribed conservative tie rule.
            for candidate in action_order:
                if candidate == old:
                    continue
                candidate_positive, candidate_total, value = engine.candidate(positive, total, image, old, candidate)
                if value > best_ap + EPS:
                    best, best_ap, best_positive, best_total = candidate, value, candidate_positive, candidate_total
            if best != old:
                state[image], current, positive, total = best, best_ap, best_positive, best_total
                changes += 1
                total_changes += 1
        if changes == 0:
            return {"state": state, "pap": current, "sweeps": sweep, "converged": True, "changes": total_changes}
    return {"state": state, "pap": current, "sweeps": max_sweeps, "converged": False, "changes": total_changes}


def p20_oracle_seed(fold: dict[str, np.ndarray]) -> np.ndarray:
    return np.where(np.asarray(fold["v"], dtype=np.float64) > 0.0, "EXPAND40", "SAFE20").astype("<U16")


def action_vector(state: np.ndarray, fold: dict[str, np.ndarray]) -> np.ndarray:
    action = np.zeros_like(fold["safe20"], dtype=np.int8).reshape(-1, PATCHES)
    for name, key in (("SAFE20", "safe20"), ("EXPAND40", "expand40")):
        selected = np.flatnonzero(state == name)
        if len(selected):
            action[selected] = fold[key].reshape(-1, PATCHES)[selected]
    return action.reshape(-1)


def safety(state: np.ndarray, fold: dict[str, np.ndarray]) -> dict[str, float]:
    return p14.safety(action_vector(state, fold), fold["y"], fold["mu"])


def choose_seed(results: Iterable[dict[str, Any]], family: tuple[str, ...] = A0) -> dict[str, Any]:
    """Frozen pAP/coverage/lexicographic multi-start resolution."""
    values = list(results)
    best_pap = max(float(item["pap"]) for item in values)
    tied = [item for item in values if abs(float(item["pap"]) - best_pap) <= EPS]
    coverage = [float(np.mean(item["state"] != "NATIVE")) for item in tied]
    tied = [item for item, value in zip(tied, coverage) if value == min(coverage)]
    rank = {name: index for index, name in enumerate(family)}
    return min(tied, key=lambda item: tuple(rank[str(value)] for value in item["state"]))


def class_cache(held: str, fold: dict[str, np.ndarray]) -> p15.ClassCache:
    """One exact cache-only re-deployment of P20's persisted S20/E40 actions."""
    cache = p15.build_cache(held, fold["safe20"], fold["expand40"])
    if not np.array_equal(cache.paths.astype(str), fold["image_path"].astype(str)):
        raise RuntimeError("P21_ENGINEERING_STOP P20 image ordering")
    return cache


def release_unused_p15_base_groups(cache: p15.ClassCache) -> None:
    """P21 starts from NATIVE, so P15's retained SAFE20 AP base is unused.

    The immutable score maps and scalar SAFE20 metrics remain intact.  Releasing
    these three large NumPy buffers before constructing the native base avoids
    concurrent full-class grouped-count residency without changing arithmetic.
    """
    for name in ("base_scores", "base_positive", "base_total"):
        if hasattr(cache, name):
            delattr(cache, name)


def witness_a0(held: str) -> dict[str, Any]:
    fold = load_fold(held)
    cache = class_cache(held, fold)
    release_unused_p15_base_groups(cache)
    engine = NativeAnchorEngine(cache.native, {"NATIVE": cache.native, "SAFE20": cache.safe, "EXPAND40": cache.expand}, cache.masks)
    started = time.monotonic()
    engine.build_delta_cache(Path(tempfile.mkdtemp(prefix=f"p21_{held}_")))
    try:
        runs = [coordinate(engine, np.full(cache.n_images, "NATIVE", dtype="<U16")), coordinate(engine, p20_oracle_seed(fold))]
    finally:
        engine.clear_delta_cache()
    chosen = choose_seed(runs, A0)
    score = engine.compose(chosen["state"])
    reference = exact_metrics(score.reshape(-1), cache.masks.reshape(-1))
    fast_error = abs(float(chosen["pap"]) - float(reference["pAP"]))
    if fast_error > EPS:
        raise RuntimeError(f"P21_ENGINEERING_STOP fast/reference pAP {fast_error}")
    compact_runs = []
    for run in runs:
        compact_runs.append({
            "pap": float(run["pap"]), "sweeps": int(run["sweeps"]), "converged": bool(run["converged"]), "changes": int(run["changes"]),
            "action_counts": {name: int(np.sum(run["state"] == name)) for name in A0},
        })
    return {
        "held": held,
        "label": "POST_HOC_MULTI_START_COORDINATE_WITNESS_A0",
        "seeds": compact_runs,
        "assignment": chosen["state"].tolist(),
        "pap": float(chosen["pap"]),
        "pauroc": float(reference["pAUROC"]),
        "sweeps": int(chosen["sweeps"]),
        "converged": bool(chosen["converged"]),
        "action_counts": {name: int(np.sum(chosen["state"] == name)) for name in A0},
        "safety": safety(chosen["state"], fold),
        "fast_reference_error": fast_error,
        "runtime_seconds": time.monotonic() - started,
        "input_hashes": {"p20_fold": sha256(fold_dir(held) / "fold.npz")},
    }


def source_tau30(held: str) -> tuple[np.ndarray, float]:
    """Reconstruct the frozen, held-excluded risk context for SAFE30 only."""
    shards, _ = r1.load_shards(True)
    base = frozen.outer(held, shards)
    fold = load_fold(held)
    if np.max(np.abs(np.asarray(base["risk_h"]) - fold["risk"])) > 1e-10:
        raise RuntimeError("P21_ENGINEERING_STOP SAFE30 P20 harm-risk parity")
    values = np.concatenate([np.asarray(group["r_h"], dtype=np.float64) for group in base["level1"]])
    return p14.actions(base["mu"], base["risk_h"], float(np.quantile(values, .30, method="linear"))), float(np.quantile(values, .30, method="linear"))


def action_maps(held: str, family: tuple[str, ...]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    """Build each allowed deployed map once; no map is recomputed in sweeps."""
    fold = load_fold(held)
    cache = class_cache(held, fold)
    release_unused_p15_base_groups(cache)
    maps = {"NATIVE": cache.native, "SAFE20": cache.safe, "EXPAND40": cache.expand}
    vectors = {"NATIVE": np.zeros_like(fold["safe20"], dtype=np.int8), "SAFE20": fold["safe20"], "EXPAND40": fold["expand40"]}
    auxiliary: dict[str, Any] = {"fold": fold, "cache": cache}
    if "SAFE30" in family:
        safe30, tau30 = source_tau30(held)
        score, masks, _, _ = p14.deploy(held, safe30)
        if not np.array_equal(masks, cache.masks):
            raise RuntimeError("P21_ENGINEERING_STOP SAFE30 mask alignment")
        maps["SAFE30"] = score
        vectors["SAFE30"] = safe30
        auxiliary["tau30"] = tau30
    return maps, vectors, auxiliary


def state_action_vector(state: np.ndarray, vectors: dict[str, np.ndarray]) -> np.ndarray:
    out = np.zeros_like(next(iter(vectors.values())), dtype=np.int8).reshape(-1, PATCHES)
    for name, vector in vectors.items():
        if name == "NATIVE":
            continue
        selected = np.flatnonzero(state == name)
        if len(selected):
            out[selected] = vector.reshape(-1, PATCHES)[selected]
    return out.reshape(-1)


def generic_witness(held: str, family: tuple[str, ...], seed2: np.ndarray, label: str) -> dict[str, Any]:
    maps, vectors, extra = action_maps(held, family)
    fold, cache = extra["fold"], extra["cache"]
    engine = NativeAnchorEngine(cache.native, {name: maps[name] for name in family}, cache.masks)
    started = time.monotonic()
    engine.build_delta_cache(Path(tempfile.mkdtemp(prefix=f"p21_{held}_")))
    try:
        runs = [coordinate(engine, np.full(cache.n_images, "NATIVE", dtype="<U16"), family), coordinate(engine, seed2, family)]
    finally:
        engine.clear_delta_cache()
    chosen = choose_seed(runs, family)
    score = engine.compose(chosen["state"])
    reference = exact_metrics(score.reshape(-1), cache.masks.reshape(-1))
    error = abs(float(chosen["pap"]) - float(reference["pAP"]))
    if error > EPS:
        raise RuntimeError("P21_ENGINEERING_STOP generic witness reference parity")
    record = {
        "held": held, "label": label, "family": list(family), "assignment": chosen["state"].tolist(),
        "pap": float(chosen["pap"]), "pauroc": float(reference["pAUROC"]), "fast_reference_error": error,
        "sweeps": int(chosen["sweeps"]), "converged": bool(chosen["converged"]), "changes": int(chosen["changes"]),
        "action_counts": {name: int(np.sum(chosen["state"] == name)) for name in family},
        "safety": p14.safety(state_action_vector(chosen["state"], vectors), fold["y"], fold["mu"]),
        "runtime_seconds": time.monotonic() - started, "input_hashes": {"p20_fold": sha256(fold_dir(held) / "fold.npz")},
        "seeds": [{"pap": float(run["pap"]), "sweeps": int(run["sweeps"]), "converged": bool(run["converged"]), "changes": int(run["changes"]), "action_counts": {name: int(np.sum(run["state"] == name)) for name in family}} for run in runs],
    }
    if "tau30" in extra:
        record["tau30"] = float(extra["tau30"])
    # Private in-process values are consumed by the aggregator and never
    # serialized into a result artifact.
    record["_action"] = state_action_vector(chosen["state"], vectors)
    record["_y"] = fold["y"]
    record["_mu"] = fold["mu"]
    return record


def run_action_space(family: tuple[str, ...], seed_from_a0: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """One deterministic Stage-B or Stage-C pass; its outcome is persisted later."""
    per_class: dict[str, dict[str, Any]] = {}
    all_actions: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_mu: list[np.ndarray] = []
    native_pap: list[float] = []
    native_auc: list[float] = []
    for held in r1.CLASSES:
        fold = load_fold(held)
        seed = p20_oracle_seed(fold) if seed_from_a0 is None else np.asarray(seed_from_a0[held]["assignment"], dtype="<U16")
        record = generic_witness(held, family, seed, f"POST_HOC_MULTI_START_COORDINATE_WITNESS_{'A0' if family == A0 else 'A1'}")
        if not record["converged"]:
            raise RuntimeError("P21_ENGINEERING_STOP coordinate nonconvergence")
        all_actions.append(record.pop("_action")); all_y.append(record.pop("_y")); all_mu.append(record.pop("_mu"))
        per_class[held] = record
        stored = json.loads((fold_dir(held) / "downstream.json").read_text(encoding="utf-8"))
        native_pap.append(float(stored["native"]["pixel_ap"])); native_auc.append(float(stored["native"]["pixel_auroc"]))
        atomic(OUT / "progress.json", {"status": "ACTION_SPACE_RUNNING", "stage": "A0" if family == A0 else "A1", "completed_classes": len(per_class), "total_classes": len(r1.CLASSES), "last_completed_class": held})
        print(f"[P21][CLASS_DONE] held={held} stage={'A0' if family == A0 else 'A1'} sweeps={record['sweeps']} parity={record['fast_reference_error']:.1e}", flush=True)
    pap = float(np.mean([row["pap"] for row in per_class.values()]))
    auc = float(np.mean([row["pauroc"] for row in per_class.values()]))
    safety_all = p14.safety(np.concatenate(all_actions), np.concatenate(all_y), np.concatenate(all_mu))
    nonreg = sum(row["pap"] >= native_pap[index] for index, row in enumerate(per_class.values()))
    improve = sum(row["pap"] > native_pap[index] for index, row in enumerate(per_class.values()))
    headroom = {
        "H1_wrong_sign": safety_all["wrong_rate"] <= .05,
        "H2_weighted_harm": safety_all["relative_weighted_harm_reduction"] >= .50,
        "H3_macro_pap": pap >= float(np.mean(native_pap)) + .0025,
        "H4_nonregressing": nonreg >= 9,
        "H5_improving": improve >= 7,
        "H6_pauroc_guardrail": auc - float(np.mean(native_auc)) >= -.005,
    }
    images = sum(sum(row["action_counts"].values()) for row in per_class.values())
    return {
        "family": list(family), "per_class": per_class, "macro_pap": pap, "macro_pauroc": auc,
        "native_macro_pap": float(np.mean(native_pap)), "native_macro_pauroc": float(np.mean(native_auc)),
        "delta_vs_native": pap - float(np.mean(native_pap)), "nonregressing_classes": int(nonreg), "improving_classes": int(improve),
        "safety": safety_all, "action_fractions": {name: sum(row["action_counts"].get(name, 0) for row in per_class.values()) / images for name in family},
        "headroom_gates": headroom, "headroom_strong": bool(all(headroom.values())),
    }


def stable_rank_spearman(left: np.ndarray, right: np.ndarray) -> float:
    """The P21 F1 stable-order correlation, with no implicit tie averaging."""
    a, b = np.asarray(left).reshape(-1), np.asarray(right).reshape(-1)
    if len(a) != len(b) or len(a) < 2:
        raise RuntimeError("P21_ENGINEERING_STOP F1 rank shape")
    ar = np.empty(len(a), dtype=np.float64); br = np.empty(len(b), dtype=np.float64)
    ar[np.argsort(-a, kind="mergesort")] = np.arange(len(a), dtype=np.float64)
    br[np.argsort(-b, kind="mergesort")] = np.arange(len(b), dtype=np.float64)
    ar -= ar.mean(); br -= br.mean()
    den = float(np.sqrt(np.dot(ar, ar) * np.dot(br, br)))
    if not np.isfinite(den) or den == 0.0:
        raise RuntimeError("P21_ENGINEERING_STOP F1 constant rank")
    return float(np.dot(ar, br) / den)


def f1_features(native: np.ndarray, maps: dict[str, np.ndarray], family: tuple[str, ...]) -> np.ndarray:
    """Exactly four GT-free score-impact features per non-native action."""
    non_native = [name for name in family if name != "NATIVE"]
    out = np.empty((len(native), 4 * len(non_native)), dtype=np.float64)
    top = int(np.ceil(.10 * native.shape[1] * native.shape[2]))
    for image in range(len(native)):
        baseline = native[image].reshape(-1)
        baseline_top = np.zeros(len(baseline), dtype=bool)
        baseline_top[np.argsort(-baseline, kind="mergesort")[:top]] = True
        for action_index, name in enumerate(non_native):
            score = maps[name][image].reshape(-1)
            delta = score - baseline
            selected_top = np.zeros(len(score), dtype=bool)
            selected_top[np.argsort(-score, kind="mergesort")[:top]] = True
            offset = 4 * action_index
            out[image, offset:offset + 4] = (
                float(np.mean(delta)), float(np.quantile(np.abs(delta), .90, method="linear")),
                float(np.mean(selected_top != baseline_top)), stable_rank_spearman(baseline, score),
            )
    if not np.isfinite(out).all():
        raise RuntimeError("P21_ENGINEERING_STOP nonfinite F1")
    return out


def opportunity(held: str, family: tuple[str, ...]) -> dict[str, Any]:
    """Native-anchored one-image action opportunities and strictly GT-free F0/F1."""
    maps, _, extra = action_maps(held, family)
    cache, fold = extra["cache"], extra["fold"]
    engine = NativeAnchorEngine(cache.native, {name: maps[name] for name in family}, cache.masks)
    engine.build_delta_cache(Path(tempfile.mkdtemp(prefix=f"p21_opportunity_{held}_")))
    try:
        base_positive, base_total = engine.counts(np.full(cache.n_images, "NATIVE", dtype="<U16"))
        base_ap = engine.ap(base_positive, base_total)
        values = np.zeros((cache.n_images, len(family)), dtype=np.float64)
        for image in range(cache.n_images):
            for action_index, name in enumerate(family[1:], start=1):
                _, _, candidate_ap = engine.candidate(base_positive, base_total, image, "NATIVE", name)
                values[image, action_index] = candidate_ap - base_ap
    finally:
        engine.clear_delta_cache()
    target = np.maximum(0.0, np.max(values[:, 1:], axis=1))
    oracle_index = np.zeros(cache.n_images, dtype=np.int64)
    for image in range(cache.n_images):
        best = 0
        for action_index in range(1, len(family)):
            if values[image, action_index] > values[image, best] + EPS:
                best = action_index
        oracle_index[image] = best
    shards, _ = r1.load_shards(True)
    params = json.loads((fold_dir(held) / "parameters.json").read_text(encoding="utf-8"))
    f0 = p14.fields(held, shards[held].x, fold["mu"], fold["sigma"], fold["risk"], float(params["tau20"]), float(params["tau40"]), fold["image_path"].astype(str))
    if f0.shape[1] != 16 or not np.isfinite(f0).all():
        raise RuntimeError("P21_ENGINEERING_STOP F0 contract")
    f1 = f1_features(cache.native, maps, family)
    return {
        "held": held, "f0": f0, "f1": f1, "opportunity": target, "action_opportunity": values,
        "oracle_action": oracle_index, "family": family, "base_native_pap": base_ap,
        "image_path": fold["image_path"].astype(str),
    }


def scale_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median, iqr = frozen.scaler(np.asarray(x, dtype=np.float64))
    return median, iqr


def rank_pairs(feature: np.ndarray, value: np.ndarray, classes: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    delta: list[np.ndarray] = []
    sign: list[np.ndarray] = []
    for index in classes:
        left, right = np.triu_indices(len(index), 1)
        value_delta = value[index[left]] - value[index[right]]
        keep = np.abs(value_delta) > EPS
        delta.append(feature[index[left][keep]] - feature[index[right][keep]])
        sign.append(np.sign(value_delta[keep]))
    if not delta or not sum(len(x) for x in sign):
        raise RuntimeError("P21_ENGINEERING_STOP no RankNet pairs")
    return np.concatenate(delta).astype(np.float64), np.concatenate(sign).astype(np.float64)


def fit_ranknet(feature: np.ndarray, value: np.ndarray, classes: list[np.ndarray]) -> tuple[np.ndarray, dict[str, float]]:
    pair_x, pair_y = rank_pairs(feature, value, classes)
    torch.set_num_threads(1)
    delta = torch.from_numpy(pair_x)
    sign = torch.from_numpy(pair_y)
    weight = torch.zeros(feature.shape[1], dtype=torch.float64, requires_grad=True)
    solver = torch.optim.LBFGS([weight], lr=1.0, max_iter=500, max_eval=1000, tolerance_grad=1e-10, tolerance_change=1e-15, history_size=10, line_search_fn="strong_wolfe")
    def closure() -> torch.Tensor:
        solver.zero_grad()
        loss = torch.nn.functional.softplus(-sign * (delta @ weight)).sum() + .5 * (weight * weight).sum()
        loss.backward()
        return loss
    loss = solver.step(closure)
    result = weight.detach().cpu().numpy().astype(np.float64, copy=True)
    if not np.isfinite(result).all() or not np.isfinite(float(loss.detach())):
        raise RuntimeError("P21_ENGINEERING_STOP RankNet solver nonfinite")
    return result, {"pairs": float(len(pair_y)), "iterations": float(solver.state[weight].get("n_iter", 0)), "objective": float(loss.detach())}


def probe_metrics(prediction: dict[str, np.ndarray], target: dict[str, np.ndarray]) -> dict[str, Any]:
    global_prediction = np.concatenate([prediction[name] for name in r1.CLASSES])
    global_target = np.concatenate([target[name] for name in r1.CLASSES])
    correlation = p14.corr(global_prediction, global_target)
    per = {name: p14.corr(prediction[name], target[name]) for name in r1.CLASSES}
    eligible = np.abs(global_target) > EPS
    sign = float(np.mean(np.sign(global_prediction[eligible]) == np.sign(global_target[eligible]))) if eligible.any() else None
    top_capture: list[float] = []
    for name in r1.CLASSES:
        order = np.argsort(-prediction[name], kind="mergesort")
        k = int(np.ceil(.20 * len(order)))
        positive = target[name] > 0.0
        top_capture.append(float(positive[order[:k]].sum() / max(1, int(positive.sum()))))
    held_spearman = [per[name]["spearman"] for name in r1.CLASSES]
    if any(value is None for value in held_spearman):
        raise RuntimeError("P21_ENGINEERING_STOP undefined held Spearman")
    result = {
        "global_pearson": correlation["pearson"], "global_spearman": correlation["spearman"],
        "median_held_spearman": float(np.median(np.asarray(held_spearman, dtype=np.float64))),
        "positive_spearman_classes": int(sum(value > 0.0 for value in held_spearman)), "sign_accuracy": sign,
        "top20_positive_benefit_capture": float(np.mean(top_capture)), "per_class": per,
    }
    result["floors_pass"] = bool(result["median_held_spearman"] >= .20 and result["positive_spearman_classes"] >= 9 and sign is not None and sign >= .60)
    return result


def run_probes(family: tuple[str, ...]) -> dict[str, Any]:
    """Stage D's three and only three strict outer-LOCO diagnostic probes."""
    packages = {held: opportunity(held, family) for held in r1.CLASSES}
    prediction = {"P0": {}, "P1": {}, "P2": {}}
    solver: dict[str, dict[str, Any]] = {"P1": {}, "P2": {}}
    target = {held: np.asarray(packages[held]["opportunity"], dtype=np.float64) for held in r1.CLASSES}
    for held in r1.CLASSES:
        names = [name for name in r1.CLASSES if name != held]
        for probe, feature_key in (("P0", "f0"), ("P1", "f0"), ("P2", "f01")):
            train_feature = np.concatenate([packages[name]["f0"] if feature_key == "f0" else np.column_stack((packages[name]["f0"], packages[name]["f1"])) for name in names])
            held_feature = packages[held]["f0"] if feature_key == "f0" else np.column_stack((packages[held]["f0"], packages[held]["f1"]))
            train_value = np.concatenate([target[name] for name in names])
            median, iqr = scale_fit(train_feature)
            train_scaled, held_scaled = (train_feature - median) / iqr, (held_feature - median) / iqr
            if probe == "P0":
                beta, intercept = frozen.ridge(train_scaled, train_value)
                prediction[probe][held] = held_scaled @ beta + intercept
            else:
                offsets: list[np.ndarray] = []
                start = 0
                for name in names:
                    stop = start + len(target[name]); offsets.append(np.arange(start, stop, dtype=np.int64)); start = stop
                weight, info = fit_ranknet(train_scaled, train_value, offsets)
                prediction[probe][held] = held_scaled @ weight
                solver[probe][held] = info
    output = {"family": list(family), "probes": {name: probe_metrics(prediction[name], target) for name in prediction}, "solver": solver}
    output["diagnosis"] = {
        "RANK_OBJECTIVE_MISMATCH": (not output["probes"]["P0"]["floors_pass"]) and output["probes"]["P1"]["floors_pass"],
        "ACTION_IMPACT_FEATURE_GAP": (not output["probes"]["P1"]["floors_pass"]) and output["probes"]["P2"]["floors_pass"],
        "IMAGE_VALUE_NOT_GT_FREE_PREDICTABLE": not any(output["probes"][name]["floors_pass"] for name in ("P0", "P1", "P2")),
        "GROUP_SHIFT_LIMIT": output["probes"]["P2"]["global_spearman"] is not None and output["probes"]["P2"]["global_spearman"] > 0.0 and not output["probes"]["P2"]["floors_pass"],
    }
    np.savez_compressed(OUT / "opportunity_targets.npz", **{f"{held}_opportunity": packages[held]["opportunity"] for held in r1.CLASSES})
    return output


def fixture() -> dict[str, Any]:
    """Non-outcome-bearing exact grouped-count and trajectory fixture."""
    labels = np.array([[[1, 0]], [[0, 1]]], dtype=np.uint8)
    native = np.array([[[.1, .2]], [[.3, .4]]], dtype=np.float32)
    safe = np.array([[[.2, .1]], [[.3, .5]]], dtype=np.float32)
    expand = np.array([[[.3, .0]], [[.2, .6]]], dtype=np.float32)
    engine = NativeAnchorEngine(native, {"NATIVE": native, "SAFE20": safe, "EXPAND40": expand}, labels)
    result = coordinate(engine, np.array(["NATIVE", "NATIVE"]))
    direct = exact_metrics(engine.compose(result["state"]).reshape(-1), labels.reshape(-1))
    return {"pap": float(result["pap"]), "reference": float(direct["pAP"]), "error": abs(float(result["pap"]) - float(direct["pAP"])), "state": result["state"].tolist()}


def historical_parity() -> dict[str, Any]:
    """Stage A: independently reproduce all five persisted P20 comparators.

    This intentionally uses direct frozen metrics as a historical parity audit,
    never an input to the P21 witness trajectory.
    """
    expected = json.loads((P20 / "summary.json").read_text(encoding="utf-8"))["metrics"]["macro_pap"]
    observed: dict[str, list[float]] = {name: [] for name in expected}
    per_class: dict[str, dict[str, float]] = {}
    audit_ok = True
    for held in r1.CLASSES:
        fold = load_fold(held)
        audit = json.loads((fold_dir(held) / "fold_audit_summary.json").read_text(encoding="utf-8"))
        audit_ok = audit_ok and audit.get("status") == "PASS"
        cache = class_cache(held, fold)
        release_unused_p15_base_groups(cache)
        oracle_action = fold["safe20"].reshape(-1, PATCHES).copy()
        oracle_action[np.asarray(fold["v"], dtype=np.float64) > 0.0] = fold["expand40"].reshape(-1, PATCHES)[np.asarray(fold["v"], dtype=np.float64) > 0.0]
        variants = {
            "native": cache.native,
            "safe20": cache.safe,
            "always_expand40": cache.expand,
            "context": p14.deploy(held, fold["context"])[0],
            "image_oracle": p14.deploy(held, oracle_action.reshape(-1))[0],
        }
        stored = json.loads((fold_dir(held) / "downstream.json").read_text(encoding="utf-8"))
        row: dict[str, float] = {}
        for name, score in variants.items():
            pap = float(exact_metrics(score.reshape(-1), cache.masks.reshape(-1))["pAP"])
            row[name] = pap
            observed[name].append(pap)
            if abs(pap - float(stored[name]["pixel_ap"])) > EPS:
                raise RuntimeError(f"P21_ENGINEERING_STOP historical {held} {name} parity")
        per_class[held] = row
    macro = {name: float(np.mean(values)) for name, values in observed.items()}
    errors = {name: abs(macro[name] - float(expected[name])) for name in expected}
    result = {
        "status": "PASS" if audit_ok and max(errors.values()) <= EPS else "FAIL",
        "held_order": list(r1.CLASSES),
        "folds": len(per_class),
        "historical_fold_audits_pass": audit_ok,
        "expected_macro_pap": expected,
        "recomputed_macro_pap": macro,
        "max_abs_errors": errors,
        "per_class_pap": per_class,
    }
    if result["status"] != "PASS":
        raise RuntimeError("P21_ENGINEERING_STOP historical parity")
    return result


def git(*args: str) -> str:
    return r1.git(*args)


def inputs() -> dict[str, str]:
    return {
        "p20_summary": sha256(P20 / "summary.json"),
        "p20_audit": sha256(P20 / "post_execution_audit.json"),
        "p14_source": sha256(ROOT / "tools/sabra_cure/context_value_risk.py"),
        "p15_engine": sha256(ROOT / "tools/sabra_cure/context_value_risk_recovery.py"),
        "p21_runner": sha256(Path(__file__)),
    }


def rss_bytes() -> int:
    return int(Path("/proc/self/statm").read_text(encoding="utf-8").split()[1]) * os.sysconf("SC_PAGE_SIZE")


def benchmark() -> dict[str, Any]:
    """Pre-marker engineering benchmark; it never runs a coordinate trajectory."""
    started = time.perf_counter()
    report = fixture()
    held = "candle"
    before_rss = rss_bytes()
    deploy_start = time.perf_counter()
    fold = load_fold(held); cache = class_cache(held, fold); release_unused_p15_base_groups(cache)
    deploy_seconds = time.perf_counter() - deploy_start
    engine = NativeAnchorEngine(cache.native, {"NATIVE": cache.native, "SAFE20": cache.safe, "EXPAND40": cache.expand}, cache.masks)
    union_build_seconds = time.perf_counter() - deploy_start - deploy_seconds
    cache_start = time.perf_counter(); engine.build_delta_cache(Path(tempfile.mkdtemp(prefix="p21_benchmark_"))); delta_cache_seconds = time.perf_counter() - cache_start
    candidate_start = time.perf_counter()
    positive, total = engine.counts(np.full(engine.n_images, "NATIVE"))
    _, _, candidate_safe = engine.candidate(positive, total, 0, "NATIVE", "SAFE20")
    _, _, candidate_expand = engine.candidate(positive, total, 0, "NATIVE", "EXPAND40")
    candidate_seconds = (time.perf_counter() - candidate_start) / 2.0
    gpu_seconds: float | None = None
    gpu_error: float | None = None
    gpu_selected = False
    if torch.cuda.is_available():
        # Controller ownership keeps this state resident across image/action
        # candidates; the one-time upload is not part of candidate throughput.
        gpu_positive = torch.as_tensor(positive, dtype=torch.float64, device="cuda")
        gpu_total = torch.as_tensor(total, dtype=torch.float64, device="cuda")
        _ = gpu_ap(gpu_positive, gpu_total)
        torch.cuda.synchronize(); gpu_start = time.perf_counter()
        safe_delta, expand_delta = engine.image_delta(0, "SAFE20"), engine.image_delta(0, "EXPAND40")
        candidate_positive, candidate_total = gpu_positive.repeat((2, 1)), gpu_total.repeat((2, 1))
        gpu_apply(candidate_positive[0], candidate_total[0], safe_delta, 1.0)
        gpu_apply(candidate_positive[1], candidate_total[1], expand_delta, 1.0)
        gpu_candidates = gpu_ap_batch(candidate_positive, candidate_total)
        torch.cuda.synchronize(); gpu_seconds = (time.perf_counter() - gpu_start) / 2.0
        gpu_error = max(abs(float(gpu_candidates[0]) - candidate_safe), abs(float(gpu_candidates[1]) - candidate_expand))
        gpu_selected = gpu_error <= EPS and candidate_seconds / max(gpu_seconds, 1e-12) >= 2.0
    engine.clear_delta_cache()
    projected_candidates = 12 * 2 * 10 * 200 * 2
    selected_seconds = gpu_seconds if gpu_selected and gpu_seconds is not None else candidate_seconds
    projected_full_minutes = float(2.0 * projected_candidates * selected_seconds / 60.0 + 30.0)
    status = "PASS" if report["error"] == 0.0 and np.isfinite(candidate_safe) and np.isfinite(candidate_expand) and projected_full_minutes <= 180.0 else "FAIL"
    result = {
        "status": status, "fixture_ap_error": report["error"], "real_class": held, "real_deploy_seconds": deploy_seconds, "real_union_build_seconds": union_build_seconds, "delta_cache_seconds": delta_cache_seconds,
        "real_union_score_count": int(len(engine.union)), "real_rss_delta_bytes": int(rss_bytes() - before_rss), "candidate_seconds": candidate_seconds,
        "candidate_evaluations_per_second": 1.0 / max(candidate_seconds, 1e-12), "gpu_candidate_seconds": gpu_seconds,
        "gpu_candidate_error": gpu_error, "gpu_ap_selected": gpu_selected,
        "selected_class_workers": 1,
        "worker_selection": "1: 31 GiB host RAM with existing swap use; 2/4 rejected pre-marker to preserve no-swap-growth safety",
        "gpu_path": "exact grouped AP selected" if gpu_selected else "CPU grouped AP; GPU rejected by parity/speed gate",
        "ranknet": "CPU float64 PyTorch LBFGS only", "projected_stage_a_minutes": float(projected_candidates * selected_seconds / 60.0),
        "projected_stage_b_minutes": float(projected_candidates * selected_seconds / 60.0), "projected_full_minutes": projected_full_minutes,
        "hard_three_hour_gate": projected_full_minutes <= 180.0, "wall_seconds": time.perf_counter() - started,
        "firewall": {"mvtec": 0, "medical": 0, "clip": 0, "phase2b_steps": 0},
    }
    atomic(OUT / "performance_benchmark.json", result)
    if status != "PASS":
        raise RuntimeError("P21_PERFORMANCE_NO_GO")
    return result


def exactness_parity() -> dict[str, Any]:
    result = fixture()
    gpu: dict[str, Any] = {"available": bool(torch.cuda.is_available()), "ap_error": None, "trajectory_error": None}
    if torch.cuda.is_available():
        labels = np.array([[[1, 0]], [[0, 1]]], dtype=np.uint8)
        native = np.array([[[.1, .2]], [[.3, .4]]], dtype=np.float32)
        safe = np.array([[[.2, .1]], [[.3, .5]]], dtype=np.float32)
        expand = np.array([[[.3, .0]], [[.2, .6]]], dtype=np.float32)
        engine = NativeAnchorEngine(native, {"NATIVE": native, "SAFE20": safe, "EXPAND40": expand}, labels)
        positive, total = engine.counts(np.array(["NATIVE", "SAFE20"]))
        cpu = engine.ap(positive, total)
        gpu_value = gpu_ap(torch.as_tensor(positive, dtype=torch.float64, device="cuda"), torch.as_tensor(total, dtype=torch.float64, device="cuda"))
        left, right = coordinate(engine, np.array(["NATIVE", "NATIVE"])), coordinate_gpu(engine, np.array(["NATIVE", "NATIVE"]))
        gpu["ap_error"] = abs(cpu - gpu_value)
        gpu["trajectory_error"] = abs(float(left["pap"]) - float(right["pap"]))
        gpu["trajectory_match"] = left["state"].tolist() == right["state"].tolist()
    output = {"status": "PASS" if result["error"] == 0.0 and (not gpu["available"] or (gpu["ap_error"] <= EPS and gpu["trajectory_error"] <= EPS and gpu["trajectory_match"])) else "FAIL", "cpu_fixture": result, "gpu": gpu, "no_scalar_ap_delta_composition": True}
    atomic(OUT / "exactness_parity.json", output)
    if output["status"] != "PASS":
        raise RuntimeError("P21_ENGINEERING_STOP exactness")
    return output


def performance_stop() -> dict[str, Any]:
    if (OUT / "ATTEMPT_STARTED.json").exists():
        raise RuntimeError("P21_ENGINEERING_STOP marker exists")
    benchmark_result = json.loads((OUT / "performance_benchmark.json").read_text(encoding="utf-8"))
    parity = json.loads((OUT / "exactness_parity.json").read_text(encoding="utf-8"))
    if benchmark_result.get("status") != "FAIL" or parity.get("status") != "PASS":
        raise RuntimeError("P21_ENGINEERING_STOP invalid performance-stop evidence")
    output = {
        "status": "P21_PERFORMANCE_NO_GO", "attempt_created": False, "completed_folds": 0,
        "reason": "even with exact compressed disk-backed deltas, measured candidate throughput projects above the frozen three-hour hard ceiling",
        "performance": benchmark_result, "exactness": parity,
        "firewall": {"mvtec_accessed": False, "medical_accessed": False, "additional_clip_forwards": 0, "phase2b_training_steps": 0, "r2v3_run": False, "r3_run": False, "r4_run": False},
    }
    atomic(OUT / "pre_execution_audit.json", {"status": "P21_PERFORMANCE_NO_GO", "parent_sha": PARENT, "prereg_sha": PREREG, "checks": {"exactness": parity["status"] == "PASS", "performance": False, "attempt_marker_absent": not (OUT / "ATTEMPT_STARTED.json").exists(), "historical_execution_not_started": True}, "firewall": output["firewall"]})
    atomic(OUT / "summary.json", output)
    atomic(OUT / "progress.json", {"status": "P21_PERFORMANCE_NO_GO", "completed_classes": 0, "total_classes": 12})
    (ROOT / "research/sabra_cure/native_anchor_diagnostic/P21_FINAL_DECISION.md").write_text("# P21 Final Decision\n\n`P21_PERFORMANCE_NO_GO`: no attempt marker or P21 scientific outcome was created. Exact compressed-delta candidate throughput failed the frozen pre-marker three-hour ceiling; stop for explicit user review.\n", encoding="utf-8")
    return output


def pre_execution_audit() -> dict[str, Any]:
    if (OUT / "ATTEMPT_STARTED.json").exists():
        raise RuntimeError("P21_ENGINEERING_STOP marker exists")
    parity = historical_parity()
    performance = json.loads((OUT / "performance_benchmark.json").read_text(encoding="utf-8"))
    p20_audit = json.loads((P20 / "post_execution_audit.json").read_text(encoding="utf-8"))
    checks = {
        "parent_ancestor": git("merge-base", "--is-ancestor", PARENT, "HEAD") == "",
        "p20_status": json.loads((P20 / "summary.json").read_text(encoding="utf-8"))["status"] == "P14_SCIENCE_RECOVERED_STOP",
        "p20_audit": p20_audit["status"] == "PASS" and p20_audit["science_folds_complete"] == 12 and p20_audit["fold_audits_pass"] == 12,
        "historical_parity": parity["status"] == "PASS", "performance": performance["status"] == "PASS",
        "fold_inventory": all((fold_dir(held) / "fold.npz").exists() and (fold_dir(held) / "fold_audit_summary.json").exists() for held in r1.CLASSES),
        "firewall": True,
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL", "parent_sha": PARENT, "prereg_sha": PREREG,
        "protocol_notes": [INTERPRETATION, "f9d3f8f"], "checks": checks, "historical_parity": parity,
        "performance": performance, "input_hashes": inputs(), "classes": list(r1.CLASSES),
        "firewall": {"mvtec_accessed": False, "medical_accessed": False, "additional_clip_forwards": 0, "phase2b_training_steps": 0},
    }
    atomic(OUT / "pre_execution_audit.json", result)
    if result["status"] != "PASS":
        raise RuntimeError("P21_ENGINEERING_STOP pre-execution audit")
    return result


def marker() -> dict[str, Any]:
    return {
        "status": "ATTEMPT_STARTED", "attempt_uuid": str(uuid.uuid4()), "execution_base_sha": git("rev-parse", "HEAD"),
        "prereg_sha": PREREG, "protocol_notes": [INTERPRETATION, "f9d3f8f"], "input_hashes": inputs(), "runs": 1,
    }


def selected_audit(stage: dict[str, Any], family: tuple[str, ...]) -> dict[str, Any]:
    """Independent direct-score audit of stored selected assignments only."""
    maximum = 0.0
    held_order: list[str] = []
    for held in r1.CLASSES:
        row = stage["per_class"][held]
        maps, _, extra = action_maps(held, family)
        engine = NativeAnchorEngine(extra["cache"].native, {name: maps[name] for name in family}, extra["cache"].masks)
        score = engine.compose(np.asarray(row["assignment"], dtype="<U16"))
        metric = exact_metrics(score.reshape(-1), extra["cache"].masks.reshape(-1))
        maximum = max(maximum, abs(float(metric["pAP"]) - float(row["pap"])), abs(float(metric["pAUROC"]) - float(row["pauroc"])))
        held_order.append(held)
    result = {"status": "PASS" if maximum <= EPS and held_order == list(r1.CLASSES) else "FAIL", "held_order": held_order, "max_metric_error": maximum, "firewall": {"mvtec": 0, "medical": 0, "clip": 0, "phase2b_steps": 0}}
    atomic(OUT / "selected_result_audit.json", result)
    if result["status"] != "PASS":
        raise RuntimeError("P21_ENGINEERING_STOP selected result audit")
    return result


def final_diagnosis(a0: dict[str, Any], a1: dict[str, Any] | None, probes: dict[str, Any] | None) -> str:
    if not a0["headroom_strong"] and a1 is not None and not a1["headroom_strong"]:
        return "P21_CONTEXTUAL_ACTION_SPACE_INSUFFICIENT"
    if probes is None:
        return "P21_MIXED_FAILURE"
    flag = probes["diagnosis"]
    if flag["IMAGE_VALUE_NOT_GT_FREE_PREDICTABLE"]:
        return "P21_IMAGE_VALUE_NOT_GT_FREE_PREDICTABLE"
    if flag["RANK_OBJECTIVE_MISMATCH"]:
        return "P21_RANK_OBJECTIVE_MISMATCH_SUPPORTED"
    if flag["ACTION_IMPACT_FEATURE_GAP"]:
        return "P21_ACTION_IMPACT_FEATURE_GAP_SUPPORTED"
    if flag["GROUP_SHIFT_LIMIT"]:
        return "P21_GROUP_SHIFT_LIMIT"
    if probes["probes"]["P2"]["floors_pass"]:
        return "P21_FINAL_RANK_CONTROLLER_JUSTIFIED"
    return "P21_NATIVE_FALLBACK_HEADROOM_SUPPORTED" if a0["headroom_strong"] else "P21_BUDGET_DISCRETIZATION_SUPPORTED"


def execute_once() -> dict[str, Any]:
    if (OUT / "ATTEMPT_STARTED.json").exists() or (OUT / "summary.json").exists():
        raise RuntimeError("P21_ENGINEERING_STOP attempt exists")
    pre = json.loads((OUT / "pre_execution_audit.json").read_text(encoding="utf-8"))
    if pre.get("status") != "PASS":
        raise RuntimeError("P21_ENGINEERING_STOP missing pre-audit")
    attempt = marker(); atomic(OUT / "ATTEMPT_STARTED.json", attempt)
    atomic(OUT / "progress.json", {"status": "STARTED", "attempt_uuid": attempt["attempt_uuid"], "completed_classes": 0, "total_classes": 12})
    started = time.monotonic()
    try:
        a0 = run_action_space(A0)
        atomic(OUT / "action_space_A0.json", a0)
        a1: dict[str, Any] | None = None
        probes: dict[str, Any] | None = None
        active = A0
        if a0["headroom_strong"]:
            atomic(OUT / "progress.json", {"status": "STAGE_C_SKIPPED", "reason": "A0_HEADROOM_STRONG", "completed_classes": 12, "total_classes": 12})
        else:
            a1 = run_action_space(A1, a0["per_class"])
            atomic(OUT / "action_space_A1.json", a1)
            active = A1
        if a0["headroom_strong"] or (a1 is not None and a1["headroom_strong"]):
            atomic(OUT / "progress.json", {"status": "STAGE_D_RUNNING", "family": list(active), "completed_classes": 12, "total_classes": 12})
            probes = run_probes(active)
            for name in ("P0", "P1", "P2"):
                atomic(OUT / f"probe_{name}.json", probes["probes"][name])
        else:
            atomic(OUT / "progress.json", {"status": "STAGE_D_SKIPPED", "reason": "ACTION_SPACE_INSUFFICIENT", "completed_classes": 12, "total_classes": 12})
        selected = a0 if a0["headroom_strong"] else a1
        assert selected is not None
        audit = selected_audit(selected, active)
        diagnosis = final_diagnosis(a0, a1, probes)
        summary = {"status": "P21_COMPLETE", "primary_diagnosis": diagnosis, "attempt": attempt, "runtime_seconds": time.monotonic() - started, "historical_parity": pre["historical_parity"], "A0": a0, "A1": a1, "probes": probes, "selected_result_audit": audit, "firewall": {"mvtec_accessed": False, "medical_accessed": False, "additional_clip_forwards": 0, "phase2b_training_steps": 0, "r2v3_run": False, "r3_run": False, "r4_run": False}}
        atomic(OUT / "summary.json", summary)
        atomic(OUT / "progress.json", {"status": "COMPLETE", "completed_classes": 12, "total_classes": 12, "primary_diagnosis": diagnosis})
        doc = ROOT / "research/sabra_cure/native_anchor_diagnostic/P21_FINAL_DECISION.md"
        doc.write_text(f"# P21 Final Decision\n\nPrimary diagnosis: `{diagnosis}`. One authorized P21 diagnostic attempt only; explicit user review is required before any new preregistration.\n", encoding="utf-8")
        return summary
    except Exception as exc:
        atomic(OUT / "ENGINEERING_FAILURE.json", {"status": "P21_ENGINEERING_STOP", "exception_type": type(exc).__name__, "exception_message": str(exc)[:1000], "attempt": attempt, "execution_base_sha": git("rev-parse", "HEAD")})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--historical-parity", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--exactness", action="store_true")
    parser.add_argument("--performance-stop", action="store_true")
    parser.add_argument("--pre-audit", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--held", choices=r1.CLASSES)
    args = parser.parse_args()
    if sum((args.fixture, args.historical_parity, args.benchmark, args.exactness, args.performance_stop, args.pre_audit, args.run, args.held is not None)) != 1:
        parser.error("choose exactly one operation")
    if args.fixture:
        result = fixture()
    elif args.historical_parity:
        result = historical_parity()
    elif args.benchmark:
        result = benchmark()
    elif args.exactness:
        result = exactness_parity()
    elif args.performance_stop:
        result = performance_stop()
    elif args.pre_audit:
        result = pre_execution_audit()
    elif args.run:
        result = execute_once()
    else:
        result = witness_a0(str(args.held))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False, default=json_default))


if __name__ == "__main__":
    main()
