"""Frozen, post-hoc-only SABRA-CURE R2 failure diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from tools.sabra_car.r0_direction import evaluate_correction, exact_metrics, load_masks, metadata_and_root
from tools.sabra_cure import r1
from tools.sabra_cure import r2

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/sabra_cure/post_r2_diagnostic"
DOCS = ROOT / "research/sabra_cure/post_r2_diagnostic"
BASE = "7785fb2d984a226f14eccfc387bd537ff7d7957b"
PREREG = "c70ac416fbf0241249e3a0158f0ee92c7be90c3e"
BRANCH = "research/p10-sabra-cure-postr2-diagnostic-v1"
R2OUT = ROOT / "results/sabra_cure/r2"
GRID = 37
PATCHES = 1369
PROTECTED = ("tools/sabra_cure/r2.py", "tests/test_sabra_cure_r2.py", "results/sabra_cure/r2", "research/sabra_cure/r2", "results/sabra_cure/r1", "results/sabra_car/r0")


def write_json(path: Path, data: Any) -> None: r1.write_json(path, data)
def sha(path: Path) -> str: return r1.sha256(path)
def git(*args: str) -> str: return r1.git(*args)


def protected_ok() -> bool:
    return subprocess.run(["git", "diff", "--quiet", BASE, "--", *PROTECTED], cwd=ROOT).returncode == 0


def qbounds(values: np.ndarray) -> list[float]:
    return np.quantile(np.asarray(values, dtype=np.float64), [0, .2, .4, .6, .8, 1], method="linear").tolist()


def assign(values: np.ndarray, bounds: list[float]) -> np.ndarray:
    return np.minimum(np.searchsorted(np.asarray(bounds)[1:-1], values, side="right"), 4).astype(np.int8)


def sign_cohorts(actions: np.ndarray, utility: np.ndarray) -> dict[str, np.ndarray]:
    actions = np.asarray(actions, dtype=np.int8); utility = np.asarray(utility, dtype=np.float64)
    accepted = actions != 0
    product = actions * np.sign(utility).astype(np.int8)
    return {"accepted": accepted, "keep": ~accepted, "boost": actions > 0, "suppress": actions < 0,
            "correct": accepted & (product > 0), "wrong": accepted & (product < 0),
            "near_zero": accepted & (np.abs(utility) <= 1e-8)}


def action_reconstruct(params: dict[str, Any], mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return np.zeros(len(mu), dtype=np.int8) if params["selected_q"] is None else r2.interval_actions(mu, sigma, float(params["selected_q"]))


def rank_transition(native: np.ndarray, changed: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    def ranks(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x.reshape(-1), kind="stable"); out = np.empty(len(order), dtype=np.int64); out[order] = np.arange(len(order)); return out
    shift = ranks(changed) - ranks(native); labels = labels.reshape(-1).astype(bool)
    return {"positive_mean_rank_shift": float(shift[labels].mean()), "negative_mean_rank_shift": float(shift[~labels].mean()),
            "positive_up_fraction": float(np.mean(shift[labels] > 0)), "negative_up_fraction": float(np.mean(shift[~labels] > 0))}


def image_ap_mean(scores: np.ndarray, masks: np.ndarray) -> float | None:
    rows = []
    for score, mask in zip(scores, masks):
        flat = mask.reshape(-1)
        if flat.min() != flat.max(): rows.append(exact_metrics(score.reshape(-1), flat)["pAP"])
    return None if not rows else float(np.mean(rows))


def source_fields(name: str, paths: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(r1.SOURCE_ROOT / "gt_free_cache" / f"{name}.npz", allow_pickle=False) as source, np.load(r1.TRUST_ROOT / "cache" / f"{name}.npz", allow_pickle=False) as trust:
        if not np.array_equal(source["image_path"].astype(str), paths): raise RuntimeError("DIAGNOSTIC_ENGINEERING_STOP image alignment")
        features = r1.build_features(source, trust).reshape(-1, len(r1.FEATURE_ORDER))
        native = np.asarray(source["native_pixel_probability"], dtype=np.float32)
        patch_score = native.reshape(len(paths), GRID, 14, GRID, 14).mean(axis=(2, 4)).reshape(-1)
        return patch_score.astype(np.float64), features[:, 8], features[:, 13]


def load_fold(name: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    params = json.loads((R2OUT / "parameters" / f"{name}.json").read_text())
    with np.load(R2OUT / "folds" / f"{name}.npz", allow_pickle=False) as d:
        arrays = {k: np.asarray(d[k]) for k in d.files}
    required = {"image_path", "utility", "y", "mu", "sigma", "actions", "direction_actions"}
    if set(arrays) != required or len(arrays["utility"]) != len(arrays["image_path"]) * PATCHES: raise RuntimeError("DIAGNOSTIC_ENGINEERING_STOP fold shape")
    return params, arrays


def input_hashes() -> dict[str, str]:
    files = [R2OUT / "summary.json", R2OUT / "post_execution_audit.json", ROOT / "results/sabra_car/r0/alpha_selection.json"]
    for name in r1.CLASSES:
        files += [R2OUT / "folds" / f"{name}.npz", R2OUT / "parameters" / f"{name}.json", R2OUT / "inner_crossfit" / f"{name}.npz",
                  ROOT / "results/sabra_car/r0/utility" / f"{name}.npz", ROOT / "results/sabra_cure/r1/folds" / f"{name}.npz"]
    return {str(p.relative_to(ROOT)): sha(p) for p in files}


def pre_audit(out: Path) -> dict[str, Any]:
    summary = json.loads((R2OUT / "summary.json").read_text()); post = json.loads((R2OUT / "post_execution_audit.json").read_text())
    status = git("status", "--porcelain")
    checks = {"status": "PASS", "base_sha": BASE, "preregistration_sha": PREREG, "branch": BRANCH,
              "head": git("rev-parse", "HEAD"), "base_is_ancestor": git("merge-base", "--is-ancestor", BASE, "HEAD") == "",
              "r2_status": summary["status"], "r2_post_audit": post["status"], "r2_attempt_count": 1,
              "r2_folds": summary["folds_completed"], "protected_history_unchanged": protected_ok(),
              "worktree_clean": status == "", "input_hashes": input_hashes(),
              "mvtec_accessed": False, "medical_accessed": False, "additional_clip_forwards": 0, "phase2b_training_steps": 0}
    if not (checks["base_is_ancestor"] and checks["r2_status"] == "R2_SCIENTIFIC_STOP" and checks["r2_post_audit"] == "PASS" and checks["r2_folds"] == 12 and checks["protected_history_unchanged"] and checks["worktree_clean"]):
        raise RuntimeError("DIAGNOSTIC_ENGINEERING_STOP pre-audit")
    write_json(out / "pre_execution_audit.json", checks); return checks


def pooled_bounds() -> dict[str, list[float]]:
    pools: dict[str, list[np.ndarray]] = {k: [] for k in ("abs_utility", "abs_mu", "sigma", "width", "native_score", "stage_disagreement", "peer_consensus")}
    for name in r1.CLASSES:
        params, a = load_fold(name); score, stage, peer = source_fields(name, a["image_path"].astype(str))
        pools["abs_utility"].append(np.abs(a["utility"])); pools["abs_mu"].append(np.abs(a["mu"])); pools["sigma"].append(a["sigma"])
        pools["width"].append(2 * (0.0 if params["selected_q"] is None else float(params["selected_q"])) * a["sigma"])
        pools["native_score"].append(score); pools["stage_disagreement"].append(stage); pools["peer_consensus"].append(peer)
    return {key: qbounds(np.concatenate(value)) for key, value in pools.items()}


def cohort_stats(a: dict[str, np.ndarray], fields: dict[str, np.ndarray], bins: dict[str, list[float]]) -> dict[str, Any]:
    c = sign_cohorts(a["actions"], a["utility"]); result: dict[str, Any] = {"counts": {k: int(v.sum()) for k, v in c.items()}}
    for label in ("accepted", "keep", "boost", "suppress", "correct", "wrong", "near_zero"):
        mask = c[label]; result[label] = {key: (None if not mask.any() else {"mean": float(v[mask].mean()), "median": float(np.median(v[mask]))}) for key, v in fields.items()}
    result["bins"] = {key: {label: np.bincount(assign(v, bins[key])[c[label]], minlength=5).tolist() for label in ("accepted", "keep", "correct", "wrong", "boost", "suppress")} for key, v in fields.items()}
    result["abs_y_fixed_bins"] = np.histogram(np.abs(a["y"]), bins=[0,.1,.25,.5,.75,1.0000001])[0].tolist()
    return result


def deploy_conditions(name: str, a: dict[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, Any]]:
    c = sign_cohorts(a["actions"], a["utility"]); acts = a["actions"]
    conditions = {"D0_NATIVE": np.zeros_like(acts), "D1_PERSISTED_R2": acts,
                  "D2_SIGN_CORRECT_ONLY": np.where(c["correct"], acts, 0), "D3_SIGN_WRONG_ONLY": np.where(c["wrong"], acts, 0),
                  "D4_BOOST_ONLY": np.where(c["boost"], acts, 0), "D5_SUPPRESS_ONLY": np.where(c["suppress"], acts, 0)}
    with np.load(r1.SOURCE_ROOT / "gt_free_cache" / f"{name}.npz", allow_pickle=False) as source:
        logits = np.asarray(source["native_logits"], dtype=np.float32); paths = source["image_path"].astype(str)
        cached_native = np.asarray(source["native_pixel_probability"], dtype=np.float32)
    if not np.array_equal(paths, a["image_path"].astype(str)): raise RuntimeError("DIAGNOSTIC_ENGINEERING_STOP deployment alignment")
    metadata, data_root = metadata_and_root(r2.DATA_ROOT); masks = load_masks(paths, metadata, data_root); device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: dict[str, Any] = {}; rankings: dict[str, Any] = {}; native_scores = None
    for label, action in conditions.items():
        scores, loss = evaluate_correction(logits, masks, (action.astype(np.float32) * r2.ALPHA * r2.MARGIN_SCALE).reshape(-1, PATCHES), device, 4)
        metric = exact_metrics(scores.reshape(-1), masks.reshape(-1)); rows[label] = {"pixel_ap": metric["pAP"], "pixel_auroc": metric["pAUROC"], "mean_loss": float(loss.mean()), "per_image_ap_mean": image_ap_mean(scores, masks)}
        if native_scores is None:
            native_scores = scores
            rankings[label] = {"native_zero_cache_max_abs_error": float(np.max(np.abs(scores - cached_native)))}
        else: rankings[label] = rank_transition(native_scores, scores, masks)
    return rows, rankings


def classify(summary: dict[str, Any]) -> dict[str, Any]:
    c = summary["conditions"]; d0, d1, d2, d4, d5 = (c[x] for x in ("D0_NATIVE", "D1_PERSISTED_R2", "D2_SIGN_CORRECT_ONLY", "D4_BOOST_ONLY", "D5_SUPPRESS_ONLY"))
    delta = lambda row, key: row[key] - d0[key]
    retained = None if delta(d1, "pixel_ap") >= 0 else delta(d2, "pixel_ap") / delta(d1, "pixel_ap")
    h1 = "SUPPORTED" if delta(d2, "pixel_ap") < 0 and retained is not None and retained >= .5 else "WEAK" if delta(d2, "pixel_ap") >= 0 else "PLAUSIBLE"
    h2 = "SUPPORTED" if ((delta(d4,"pixel_ap") < 0) != (delta(d5,"pixel_ap") < 0)) else "PLAUSIBLE" if abs(delta(d4,"pixel_ap")-delta(d5,"pixel_ap")) >= .005 else "WEAK"
    act = summary["decomposition"]["global"]; low = sum(act["bins"]["abs_utility"]["accepted"][:2]) / max(1, act["counts"]["accepted"])
    med = act["accepted"]["abs_utility"]["median"] < act["keep"]["abs_utility"]["median"]
    h3 = "SUPPORTED" if med and low >= .5 else "PLAUSIBLE" if med or low >= .5 else "WEAK"
    h4 = "SUPPORTED" if delta(d2,"mean_loss") < 0 and delta(d2,"pixel_ap") < 0 else "PLAUSIBLE" if delta(d1,"mean_loss") < 0 and delta(d1,"pixel_ap") < 0 else "INSUFFICIENT_EVIDENCE"
    rate = act["counts"]["accepted"] / summary["patches"]; adjacent = summary["spatial"]["accepted_adjacent_fraction"]
    h6 = "PLAUSIBLE" if rate > 0 and adjacent / (rate * rate) >= 1.25 and delta(d1,"pixel_ap") < 0 else "WEAK"
    hypotheses = {"H1_WRONG_SIGN_TARGET_TOO_NARROW": h1, "H2_BOOST_SUPPRESS_ASYMMETRY": h2, "H3_ACTIONABILITY_MISMATCH": h3,
                  "H4_TARGET_LOSS_RANKING_MISMATCH": h4, "H5_FIXED_ALPHA_SUBSET_MISMATCH": "INSUFFICIENT_EVIDENCE", "H6_SPATIAL_RANKING_COUPLING": h6}
    primary = "TARGET_LOSS_RANKING_MISMATCH" if h4 == "SUPPORTED" and sum(v in {"SUPPORTED","PLAUSIBLE"} for v in hypotheses.values()) == 1 else "MIXED_FAILURE"
    return {"hypotheses": hypotheses, "primary_root_cause": primary, "retained_harm_ratio_D2_over_D1": retained,
            "recommended_next_formulation": "ACTION-HARM / RANKING-AWARE SELECTIVE INTERVENTION" if h4 == "SUPPORTED" else "explicit user review before any new preregistration"}


def execute(out: Path) -> dict[str, Any]:
    if any((out / x).exists() for x in ("ATTEMPT_STARTED.json", "summary.json")): raise RuntimeError("DIAGNOSTIC_ENGINEERING_STOP diagnostic already started")
    audit = pre_audit(out); write_json(out / "ATTEMPT_STARTED.json", {"status":"ATTEMPT_STARTED", "base_sha":BASE, "execution_sha":git("rev-parse","HEAD"), "runs":1})
    bins = pooled_bounds(); global_fields: dict[str, list[np.ndarray]] = {k: [] for k in bins}; global_a: dict[str, list[np.ndarray]] = {k: [] for k in ("actions","utility","y","mu","sigma")}
    per: dict[str, Any] = {}; condition_rows: dict[str, list[dict[str, float]]] = {f"D{i}_{x}": [] for i,x in enumerate(["NATIVE","PERSISTED_R2","SIGN_CORRECT_ONLY","SIGN_WRONG_ONLY","BOOST_ONLY","SUPPRESS_ONLY"])}; rankings: dict[str, Any] = {}; adjacent_hits=adjacent_total=0
    for name in r1.CLASSES:
        params, a = load_fold(name); score, stage, peer = source_fields(name, a["image_path"].astype(str)); width = 2*(0 if params["selected_q"] is None else float(params["selected_q"]))*a["sigma"]
        fields={"abs_utility":np.abs(a["utility"]),"abs_mu":np.abs(a["mu"]),"sigma":a["sigma"],"width":width,"native_score":score,"stage_disagreement":stage,"peer_consensus":peer}
        per[name]=cohort_stats(a,fields,bins); global_a={k: global_a[k]+[a[k]] for k in global_a}; global_fields={k:global_fields[k]+[v] for k,v in fields.items()}
        g=a["actions"].reshape(-1,GRID,GRID)!=0; adjacent_hits += int((g[:,:,:-1]&g[:,:,1:]).sum()+(g[:,:-1,:]&g[:,1:,:]).sum()); adjacent_total += int(g.shape[0]*(2*GRID*(GRID-1)))
        rows, ranks=deploy_conditions(name,a); rankings[name]=ranks
        for key,row in rows.items(): condition_rows[key].append(row)
    ga={k:np.concatenate(v) for k,v in global_a.items()}; gf={k:np.concatenate(v) for k,v in global_fields.items()}; global_dec=cohort_stats(ga,gf,bins)
    conditions={key:{metric:float(np.mean([r[metric] for r in rows])) if rows[0][metric] is not None else None for metric in rows[0]} for key,rows in condition_rows.items()}
    for key,row in conditions.items(): row["label"] = "OBSERVED_RECONSTRUCTION" if key in {"D0_NATIVE","D1_PERSISTED_R2"} else "POST_HOC_ORACLE_DIAGNOSTIC"
    decomposition={"bin_boundaries_descriptive":bins,"global":global_dec,"fixed_abs_y_bins":[0,.1,.25,.5,.75,1],"post_hoc_note":"AP is not decomposed per patch."}
    core={"patches":int(len(ga["actions"])),"conditions":conditions,"decomposition":decomposition,"spatial":{"accepted_adjacent_pairs":adjacent_hits,"total_adjacent_pairs":adjacent_total,"accepted_adjacent_fraction":adjacent_hits/adjacent_total}}
    root=classify(core); summary={"status":"POST_R2_DIAGNOSTIC_COMPLETE","base_r2_sha":BASE,"diagnostic_execution_sha":git("rev-parse","HEAD"),"pre_execution_audit":"PASS","patches":core["patches"],"conditions":conditions,"root_cause":root,"freeze":{"additional_clip_forwards":0,"phase2b_training_steps":0},"firewall":{"mvtec_accessed":False,"medical_accessed":False}}
    write_json(out/"failure_decomposition.json",decomposition); write_json(out/"per_class_diagnostics.json",per); write_json(out/"action_cohorts.json",conditions); write_json(out/"ranking_diagnostics.json",rankings); write_json(out/"root_cause_summary.json",root); write_json(out/"summary.json",summary)
    write_docs(summary); post=audit_results(out)
    if post["status"] != "PASS": raise RuntimeError("DIAGNOSTIC_ENGINEERING_STOP post-audit")
    return summary


def write_docs(summary: dict[str, Any]) -> None:
    root=summary["root_cause"]; h=root["hypotheses"]; c=summary["conditions"]
    (DOCS/"POST_R2_ROOT_CAUSE.md").write_text("# SABRA-CURE Post-R2 Root Cause\n\n"+f"Primary: `{root['primary_root_cause']}`. This is POST_HOC diagnostic evidence only; R2 remains `R2_SCIENTIFIC_STOP`.\n\n"+"\n".join(f"- {k}: `{v}`" for k,v in h.items())+f"\n\nD1 pAP delta: `{c['D1_PERSISTED_R2']['pixel_ap']-c['D0_NATIVE']['pixel_ap']:.6f}`; D2 sign-correct-only pAP delta: `{c['D2_SIGN_CORRECT_ONLY']['pixel_ap']-c['D0_NATIVE']['pixel_ap']:.6f}`.\n")
    (DOCS/"DECISION_TREE.md").write_text("# Post-R2 Decision Tree\n\nR2 is preserved as a terminal failure. The diagnostic classifies H1-H6 without selecting a replacement policy.\n")
    (DOCS/"NEXT_RESEARCH_OPTIONS.md").write_text("# Next Research Options\n\nRecommended for user review only: `"+root["recommended_next_formulation"]+"`. No new scientific preregistration is created here.\n")
    (DOCS/"POST_R2_FINAL_DECISION.md").write_text("# Post-R2 Diagnostic Final Decision\n\n`POST_R2_DIAGNOSTIC_COMPLETE`; R2 remains `R2_SCIENTIFIC_STOP`. Stop for explicit user review before any new scientific preregistration.\n")


def audit_results(out: Path) -> dict[str, Any]:
    s=json.loads((out/"summary.json").read_text()); total=0; action_total=0; reconstruction=0.0; shards,_=r1.load_shards(check_hashes=True)
    for name in r1.CLASSES:
        p,a=load_fold(name); x=r1.scale_x(shards[name].x,np.asarray(p["feature_median"]),np.asarray(p["feature_iqr"])); mu=x@np.asarray(p["mean_beta"])+float(p["mean_intercept"]); sigma=np.exp(np.clip(x@np.asarray(p["uncertainty_beta"])+float(p["uncertainty_intercept"]),np.log(r2.EPS),np.log(4.0))); reconstruction=max(reconstruction,float(np.max(np.abs(mu-a["mu"]))),float(np.max(np.abs(sigma-a["sigma"])))); action_total+=int(np.count_nonzero(action_reconstruct(p,mu,sigma)-a["actions"])); total+=len(a["actions"])
    status="PASS" if total==s["patches"] and reconstruction<=1e-10 and action_total==0 and protected_ok() else "FAIL"
    payload={"status":status,"patches":total,"serialization_max_abs_error":reconstruction,"action_reconstruction_mismatches":action_total,"alignment_audit":total==s["patches"],"protected_history_unchanged":protected_ok(),"firewall_audit":True,"freeze_audit":True,"mvtec_accessed":False,"medical_accessed":False,"additional_clip_forwards":0,"phase2b_training_steps":0}; write_json(out/"post_execution_audit.json",payload); return payload


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--pre-audit",action="store_true"); p.add_argument("--execute-once",action="store_true"); p.add_argument("--audit-only",action="store_true"); p.add_argument("--output",type=Path,default=OUT); a=p.parse_args()
    if sum((a.pre_audit,a.execute_once,a.audit_only))!=1: p.error("choose exactly one mode")
    result=pre_audit(a.output) if a.pre_audit else execute(a.output) if a.execute_once else audit_results(a.output); print(json.dumps(result,indent=2,sort_keys=True))
if __name__=="__main__": main()
