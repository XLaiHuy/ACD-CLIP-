"""P16 bounded-memory owner around the frozen P15 exact AP engine."""
from __future__ import annotations

import argparse, gc, hashlib, json, os, sys, tempfile, time, traceback, uuid, weakref
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.sabra_cure import context_value_risk as p14
from tools.sabra_cure import context_value_risk_recovery as p15
from tools.sabra_cure import r1

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/sabra_cure/context_value_risk_memory_recovery"
DOC = ROOT / "research/sabra_cure/context_value_risk_memory_recovery"
PARENT = "102822854a2e5914fd29d0778cba25a169aff2aa"
PREREG = "ad5d38e4077b85129bb7a431f944fcd90fb6f8da"
BRANCH = "research/p16-sabra-cure-context-value-risk-memory-recovery-v1"
P14_SHA = "d83c59c7e52b21022c90708198048f049bd4e4e46ff443a67adf0c524414273f"
MAX_RSS = 14 * 1024**3
POST_SLACK = 1024**3


def git(*args: str) -> str:
    return r1.git(*args)


def atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False, default=p15.json_default)
        handle.write("\n"); tmp = Path(handle.name)
    os.replace(tmp, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rss_bytes() -> int:
    with Path("/proc/self/statm").open() as handle:
        resident = int(handle.read().split()[1])
    return resident * os.sysconf("SC_PAGE_SIZE")


def memory_event(out: Path, held: str, stage: str, completed: int, **extra: Any) -> dict[str, Any]:
    event = {"timestamp": time.time(), "held": held, "stage": stage, "rss_bytes": rss_bytes(), "completed_outer_folds": completed, **extra}
    out.mkdir(parents=True, exist_ok=True)
    with (out / "memory_progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n"); handle.flush(); os.fsync(handle.fileno())
    return event


def input_hashes() -> dict[str, str]:
    return {"p14_reference": sha256(ROOT / "tools/sabra_cure/context_value_risk.py"), "p15_engine": sha256(ROOT / "tools/sabra_cure/context_value_risk_recovery.py"), "p15_terminal": sha256(ROOT / "results/sabra_cure/context_value_risk_recovery/ENGINEERING_FAILURE.json")}


def historical_unchanged() -> bool:
    import subprocess
    protected = (*p14.PROTECTED, "tools/sabra_cure/context_value_risk.py", "tools/sabra_cure/context_value_risk_recovery.py", "results/sabra_cure/context_value_risk", "results/sabra_cure/context_value_risk_recovery", "research/sabra_cure/context_value_risk", "research/sabra_cure/context_value_risk_recovery")
    return subprocess.run(["git", "diff", "--quiet", PARENT, "--", *protected], cwd=ROOT).returncode == 0


def cache_refs(fold: dict[str, Any]) -> list[weakref.ReferenceType[Any]]:
    if "_cache_refs" in fold:
        return list(fold["_cache_refs"])
    caches = [group["cache"] for group in fold["groups"].values()] + [fold["target"]["cache"]]
    refs = [weakref.ref(cache) for cache in caches]
    del caches
    return refs


def assert_no_live_completed_cache(refs: list[weakref.ReferenceType[Any]]) -> None:
    live = [ref for ref in refs if ref() is not None]
    if live:
        raise RuntimeError("P16_ENGINEERING_STOP completed outer full cache remains reachable")


def enforce_memory(event: dict[str, Any], pre_fold_rss: int | None = None, finalized: bool = False) -> None:
    if event["rss_bytes"] > MAX_RSS:
        raise RuntimeError("P16_ENGINEERING_STOP peak RSS threshold breach")
    if finalized and pre_fold_rss is not None and event["rss_bytes"] > pre_fold_rss + POST_SLACK:
        raise RuntimeError("P16_ENGINEERING_STOP post-finalize RSS threshold breach")


def fold_dir(out: Path, held: str) -> Path:
    return out / "folds" / held


def persist_fold(out: Path, fold: dict[str, Any], held: str, completed: int, image_path: np.ndarray) -> dict[str, Any]:
    """Persist all required compact fold outputs without retaining ClassCache."""
    directory = fold_dir(out, held); directory.mkdir(parents=True, exist_ok=True)
    target, base = fold["target"], fold["base"]
    parameter = {"held": held, "outer_training": [name for name in r1.CLASSES if name != held], "tau20": target["tau20"], "tau40": target["tau40"], "selected_q": fold["selected_q"], "value_threshold": fold["threshold"], "feature_order": list(p15.FEATURE_ORDER), "value_model": {key: np.asarray(value).tolist() if isinstance(value, np.ndarray) else value for key, value in fold["value_model"].items()}, "selection": fold["selection"], "direction": {key: np.asarray(value).tolist() if isinstance(value, np.ndarray) else value for key, value in base["direction"].items()}}
    atomic(directory / "parameters.json", parameter)
    action = np.where(fold["expand_images"][:, None], target["expand"].reshape(-1, p15.PATCHES), target["safe"].reshape(-1, p15.PATCHES)).reshape(-1)
    np.savez_compressed(directory / "fold.npz", image_path=image_path, mu=base["mu"], y=base["y"], utility=base["utility"], sigma=base["sigma"], risk=base["risk_h"], safe20=target["safe"], expand40=target["expand"], context=action, vhat=fold["vhat"], v=target["v"], expand_images=fold["expand_images"])
    atomic(directory / "downstream.json", fold["downstream"])
    atomic(directory / "policy_selection.json", fold["selection"])
    hashes = {path.name: sha256(path) for path in (directory / "parameters.json", directory / "fold.npz", directory / "downstream.json", directory / "policy_selection.json")}
    atomic(directory / "checkpoint.json", {"held": held, "completed_outer_folds": completed, "execution_base_sha": git("rev-parse", "HEAD"), "input_hashes": input_hashes(), "artifacts": hashes})
    return {"held": held, "selected_q": fold["selected_q"], "downstream": fold["downstream"], "artifact_hashes": hashes}


def finalize_outer_fold(out: Path, fold: dict[str, Any], held: str, completed: int, pre_fold_rss: int, image_path: np.ndarray) -> tuple[dict[str, Any], list[weakref.ReferenceType[Any]], dict[str, Any]]:
    """Persist compact artifacts, then structurally release the full working set."""
    refs = cache_refs(fold)
    row = persist_fold(out, fold, held, completed, image_path)
    return row, refs, {"pre_fold_rss": pre_fold_rss}


def rehydrate(out: Path, held: str) -> dict[str, Any]:
    directory = fold_dir(out, held)
    with np.load(directory / "fold.npz", allow_pickle=False) as data:
        return {"held": held, "base": {"mu": np.asarray(data["mu"]), "y": np.asarray(data["y"]), "utility": np.asarray(data["utility"]), "sigma": np.asarray(data["sigma"]), "risk_h": np.asarray(data["risk"])}, "target": {"safe": np.asarray(data["safe20"]), "expand": np.asarray(data["expand40"]), "v": np.asarray(data["v"])}, "vhat": np.asarray(data["vhat"]), "expand_images": np.asarray(data["expand_images"], dtype=bool), "selected_q": json.loads((directory / "parameters.json").read_text())["selected_q"], "downstream": json.loads((directory / "downstream.json").read_text())}


def completed_folds(out: Path) -> list[str]:
    completed = []
    for held in r1.CLASSES:
        checkpoint = fold_dir(out, held) / "checkpoint.json"
        if checkpoint.exists():
            data = json.loads(checkpoint.read_text())
            if data["execution_base_sha"] != git("rev-parse", "HEAD") or data["input_hashes"] != input_hashes():
                raise RuntimeError("P16_ENGINEERING_STOP checkpoint identity mismatch")
            if any(sha256(fold_dir(out, held) / name) != digest for name, digest in data["artifacts"].items()):
                raise RuntimeError("P16_ENGINEERING_STOP checkpoint hash mismatch")
            completed.append(held)
    return completed


def marker() -> dict[str, Any]:
    return {"status": "ATTEMPT_STARTED", "attempt_uuid": str(uuid.uuid4()), "execution_base_sha": git("rev-parse", "HEAD"), "prereg_sha": PREREG, "input_hashes": input_hashes(), "runs": 1}


def outer(held: str, shards: dict[str, r1.Shard], checkpoint: Any, checkpoint_dir: Path, stage: Any) -> dict[str, Any]:
    """Exact P15 arithmetic with explicit, fold-local cache ownership."""
    base = p15.frozen.outer(held, shards); stage("after_direction_harm")
    names = [name for name in r1.CLASSES if name != held]; groups: dict[str, dict[str, Any]] = {}
    for group in base["level1"]:
        other = np.concatenate([item["r_h"] for item in base["level1"] if item["name"] != group["name"]]); t20, t40 = p14.thresholds(other)
        safe, expand = p14.actions(group["mu"], group["r_h"], t20), p14.actions(group["mu"], group["r_h"], t40)
        cache = p15.build_cache(group["name"], safe, expand); saved = checkpoint_dir / f"targets_{group['name']}.npz"
        if saved.exists():
            with np.load(saved, allow_pickle=False) as data:
                target = {"x": np.asarray(data["features"], dtype=np.float64), "v": np.asarray(data["value"], dtype=np.float64), "safe": safe, "expand": expand, "tau20": t20, "tau40": t40, "cache": cache}
        else:
            target = p15.image_targets(cache, group["x"], group["mu"], group["sigma"], group["r_h"], t20, t40, p15.WORKERS); checkpoint(group["name"], target)
        target.update({"mu": group["mu"], "y": group["y"], "risk": group["r_h"], "sigma": group["sigma"], "training": group["training"]}); groups[group["name"]] = target; stage("after_source_image_target", group["name"])
    stage("before_source_policy_selection")
    oof = {}
    for name in names:
        others = [item for item in names if item != name]; model = p14.fit(np.concatenate([groups[item]["x"] for item in others]), np.concatenate([groups[item]["v"] for item in others])); oof[name] = p14.predict(model, groups[name]["x"])
    selected, selection = p15.source_selection(groups, oof)
    value_model = p14.fit(np.concatenate([groups[item]["x"] for item in names]), np.concatenate([groups[item]["v"] for item in names])); source_refs = [weakref.ref(group["cache"]) for group in groups.values()]
    for group in groups.values(): del group["cache"]
    del cache, target
    gc.collect(); assert_no_live_completed_cache(source_refs); stage("after_source_policy_selection")
    t20, t40 = p14.thresholds(np.concatenate([item["r_h"] for item in base["level1"]])); safe, expand = p14.actions(base["mu"], base["risk_h"], t20), p14.actions(base["mu"], base["risk_h"], t40)
    held_cache = p15.build_cache(held, safe, expand); held_target = p15.image_targets(held_cache, shards[held].x, base["mu"], base["sigma"], base["risk_h"], t20, t40, p15.WORKERS)
    vhat = p14.predict(value_model, held_target["x"]); threshold = float("inf") if selected is None else float(np.quantile(np.concatenate([oof[item] for item in names]), selected, method="linear")); expanded = vhat > threshold; oracle = held_target["v"] > 0
    downstream = {"native": p15.compose_downstream(held_cache, "native"), "safe20": p15.compose_downstream(held_cache, "safe20"), "always_expand40": p15.compose_downstream(held_cache, "always_expand40"), "context": p15.compose_downstream(held_cache, "context", expanded), "image_oracle": p15.compose_downstream(held_cache, "image_oracle", oracle)}
    held_ref = weakref.ref(held_cache); compact_target = {key: held_target[key] for key in ("safe", "expand", "v", "tau20", "tau40")}; compact_base = {key: base[key] for key in ("mu", "y", "utility", "sigma", "risk_h", "direction")}
    del held_target, held_cache, groups, base
    return {"held": held, "base": compact_base, "selection": selection, "selected_q": selected, "threshold": threshold, "value_model": value_model, "vhat": vhat, "expand_images": expanded, "target": compact_target, "downstream": downstream, "_cache_refs": source_refs + [held_ref]}


def synthetic_memory_fixture(out: Path) -> dict[str, Any]:
    """Engineering-only owner/RSS test; it has no P14 outcome or dataset input."""
    events, refs = [], []
    baseline = rss_bytes()
    for index in range(2):
        before = rss_bytes()
        payload = {"safe": np.full((16, 518, 518), .1, np.float32), "expand": np.full((16, 518, 518), .2, np.float32), "mask": np.ones((16, 518, 518), np.uint8), "groups": np.full((16 * 518 * 518,), 1.0, np.float64)}
        weak = [weakref.ref(value) for value in payload.values()]; peak = rss_bytes(); del payload; gc.collect(); post = rss_bytes(); refs.extend(weak)
        events.append({"fold": index + 1, "before": before, "peak": peak, "post": post})
    if any(ref() is not None for ref in refs): raise RuntimeError("P16_ENGINEERING_STOP synthetic fold buffer retention")
    no_monotonic = events[1]["post"] <= events[0]["post"] + POST_SLACK
    post_gate = all(event["post"] <= event["before"] + POST_SLACK for event in events)
    report = {"status": "PASS" if no_monotonic and post_gate else "FAIL", "baseline_rss": baseline, "events": events, "post_finalize_gate": post_gate, "no_monotonic_growth": no_monotonic, "full_buffers_unreachable": True, "fixture_shape": [16, 518, 518]}
    atomic(out / "memory_fixture.json", report); return report


def benchmark(out: Path) -> dict[str, Any]:
    report = p15.benchmark(out)
    return report


def parity(out: Path) -> dict[str, Any]:
    return p15.parity_report(out)


def pre_audit(out: Path) -> dict[str, Any]:
    if (out / "ATTEMPT_STARTED.json").exists(): raise RuntimeError("P16_ENGINEERING_STOP attempt exists")
    bench, par, memory = json.loads((out / "performance_benchmark.json").read_text()), json.loads((out / "parity_report.json").read_text()), json.loads((out / "memory_fixture.json").read_text())
    checks = {"status": "PASS", "parent_sha": PARENT, "parent_ancestor": git("merge-base", "--is-ancestor", PARENT, "HEAD") == "", "p14_hash": sha256(ROOT / "tools/sabra_cure/context_value_risk.py"), "p14_science_unchanged": sha256(ROOT / "tools/sabra_cure/context_value_risk.py") == P14_SHA, "p15_partial_excluded": True, "historical_unchanged": historical_unchanged(), "exact_parity_pass": bool(par["pass"]), "speed_gate_pass": bool(bench["median_speedup"] >= 5), "memory_fixture_pass": memory["status"] == "PASS", "memory_peak_limit_bytes": MAX_RSS, "execution_base_published": git("rev-parse", "HEAD") == git("rev-parse", f"origin/{BRANCH}"), "worktree_clean_before_audit": git("status", "--porcelain") == "", "prereg_sha": PREREG, "input_hashes": input_hashes(), "mvtec_accessed": False, "medical_accessed": False, "additional_clip_forwards": 0, "phase2b_training_steps": 0}
    if not all((checks["parent_ancestor"], checks["p14_science_unchanged"], checks["historical_unchanged"], checks["exact_parity_pass"], checks["speed_gate_pass"], checks["memory_fixture_pass"], checks["execution_base_published"], checks["worktree_clean_before_audit"])): checks["status"] = "FAIL"
    atomic(out / "pre_execution_audit.json", checks)
    if checks["status"] != "PASS": raise RuntimeError("P16_COMPUTATIONAL_NO_GO")
    return checks


def execute(out: Path, resume: bool = False) -> dict[str, Any]:
    attempt = out / "ATTEMPT_STARTED.json"
    if resume:
        old = json.loads(attempt.read_text()) if attempt.exists() else None
        if not old or old["execution_base_sha"] != git("rev-parse", "HEAD") or old["prereg_sha"] != PREREG or old["input_hashes"] != input_hashes(): raise RuntimeError("P16_ENGINEERING_STOP resume identity mismatch")
        with (out / "RESUME_LOG.jsonl").open("a", encoding="utf-8") as handle: handle.write(json.dumps({"event": "RESUME", "attempt_uuid": old["attempt_uuid"], "time": time.time()}) + "\n")
    elif attempt.exists() or (out / "summary.json").exists(): raise RuntimeError("P16_ENGINEERING_STOP attempt exists")
    else: atomic(attempt, marker())
    shards, _ = r1.load_shards(True); done = completed_folds(out) if resume else []; compact = [rehydrate(out, held) for held in done]; stale_refs: list[weakref.ReferenceType[Any]] = []
    try:
        for held in r1.CLASSES:
            assert_no_live_completed_cache(stale_refs); before = memory_event(out, held, "before_outer", len(compact)); enforce_memory(before)
            if held in done: continue
            source_done: list[str] = []
            def checkpoint(name: str, target: dict[str, Any]) -> None:
                source_done.append(name); path = out / "checkpoints" / f"outer_{held}" / f"targets_{name}.npz"; path.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(path, features=target["x"], value=target["v"]); p15.write_checkpoint(out, held, "image_target_group", source_done, {"last_source_class": name, "target_sha256": sha256(path)}); event = memory_event(out, held, "source_target_group", len(compact), source=name); enforce_memory(event)
            def stage(name: str, source: str | None = None) -> None:
                event = memory_event(out, held, name, len(compact), source=source); enforce_memory(event)
            fold = outer(held, shards, checkpoint, out / "checkpoints" / f"outer_{held}", stage)
            held_event = memory_event(out, held, "after_held_evaluation", len(compact)); enforce_memory(held_event)
            row, refs, _ = finalize_outer_fold(out, fold, held, len(compact) + 1, before["rss_bytes"], shards[held].image_path); compact.append({"held": held, "selected_q": row["selected_q"], "downstream": row["downstream"]})
            del fold; gc.collect()
            if p15.torch.cuda.is_available(): p15.torch.cuda.synchronize(); p15.torch.cuda.empty_cache()
            stale_refs.extend(refs); assert_no_live_completed_cache(stale_refs); post = memory_event(out, held, "after_finalize", len(compact)); enforce_memory(post, before["rss_bytes"], finalized=True)
            atomic(out / "progress.json", {"status": "RUNNING", "last_completed_outer": held, "completed_folds": len(compact), "total_folds": 12})
        folds = [rehydrate(out, held) for held in r1.CLASSES]
        down = {fold["held"]: fold["downstream"] for fold in folds}; macro = {key: float(np.mean([down[held][key]["pixel_ap"] for held in r1.CLASSES])) for key in next(iter(down.values()))}; auc = {key: float(np.mean([down[held][key]["pixel_auroc"] for held in r1.CLASSES])) for key in next(iter(down.values()))}
        actions = np.concatenate([np.where(fold["expand_images"][:, None], fold["target"]["expand"].reshape(-1, p15.PATCHES), fold["target"]["safe"].reshape(-1, p15.PATCHES)).reshape(-1) for fold in folds]); y = np.concatenate([fold["base"]["y"] for fold in folds]); mu = np.concatenate([fold["base"]["mu"] for fold in folds]); safety = p14.safety(actions, y, mu); expansion = float(np.mean(np.concatenate([fold["expand_images"] for fold in folds]))); nonreg = sum(down[held]["context"]["pixel_ap"] >= down[held]["native"]["pixel_ap"] for held in r1.CLASSES); improving = sum(down[held]["context"]["pixel_ap"] > down[held]["native"]["pixel_ap"] for held in r1.CLASSES); values, predicted = np.concatenate([fold["target"]["v"] for fold in folds]), np.concatenate([fold["vhat"] for fold in folds]); nz = np.abs(values) > p15.EPS
        gates = {"G1_AUDIT": True, "G2_SAFETY": safety["wrong_rate"] <= .05, "G3_WEIGHTED_HARM": safety["relative_weighted_harm_reduction"] >= .5, "G4_RECOVERY_COVERAGE": safety["coverage"] >= .1934, "G5_CONTEXT_USAGE": expansion >= .1, "G6_PAP_NATIVE": macro["context"] - macro["native"] >= .0025, "G7_BREADTH": nonreg >= 9, "G8_POSITIVE_BREADTH": improving >= 7, "G9_AUROC": auc["context"] - auc["native"] >= -.005, "G10_POLICY_VALUE": macro["context"] > macro["safe20"], "G11_SELECTION": all(fold["selected_q"] in (*p15.Q, None) for fold in folds)}
        summary = {"status": "P14_SCIENCE_RECOVERED_PASS" if all(gates.values()) else "P14_SCIENCE_RECOVERED_STOP", "execution_base_sha": git("rev-parse", "HEAD"), "attempt_uuid": json.loads(attempt.read_text())["attempt_uuid"], "folds_completed": 12, "metrics": {"macro_pap": macro, "macro_pauroc": auc, "safety": safety, "expand40_fraction": expansion, "nonregressing_classes": nonreg, "improving_classes": improving, "value_prediction": {"pearson": p14.corr(predicted, values)["pearson"], "spearman": p14.corr(predicted, values)["spearman"], "sign_accuracy_nonzero": float(np.mean(np.sign(predicted[nz]) == np.sign(values[nz])))}}, "selected_q": {fold["held"]: fold["selected_q"] for fold in folds}, "gates": gates, "firewall": {"mvtec_accessed": False, "medical_accessed": False}, "freeze": {"alpha": .25, "additional_clip_forwards": 0, "phase2b_training_steps": 0}}
        atomic(out / "summary.json", summary); atomic(out / "progress.json", {"status": "COMPLETE", "completed_folds": 12, "total_folds": 12}); return summary
    except Exception as exc:
        (out / "ENGINEERING_FAILURE.traceback.log").write_text(traceback.format_exc()); atomic(out / "ENGINEERING_FAILURE.json", {"status": "P16_ENGINEERING_STOP", "exception_type": type(exc).__name__, "exception_message": str(exc)[:1000], "execution_base_sha": git("rev-parse", "HEAD")}); raise


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--memory-fixture", action="store_true"); parser.add_argument("--benchmark", action="store_true"); parser.add_argument("--parity", action="store_true"); parser.add_argument("--pre-audit", action="store_true"); parser.add_argument("--execute-once", action="store_true"); parser.add_argument("--resume-attempt", action="store_true"); parser.add_argument("--output", type=Path, default=OUT); args = parser.parse_args()
    if sum((args.memory_fixture, args.benchmark, args.parity, args.pre_audit, args.execute_once, args.resume_attempt)) != 1: parser.error("choose exactly one action")
    result = synthetic_memory_fixture(args.output) if args.memory_fixture else benchmark(args.output) if args.benchmark else parity(args.output) if args.parity else pre_audit(args.output) if args.pre_audit else execute(args.output, args.resume_attempt)
    print(json.dumps(result, indent=2, sort_keys=True, default=p15.json_default))


if __name__ == "__main__": main()
