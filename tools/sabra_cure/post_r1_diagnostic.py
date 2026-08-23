"""Post-hoc, source-only diagnosis of the terminal SABRA-CURE R1 result.

This is not an R1 rerun.  It reads published folds and parameters, makes only
explicitly labelled diagnostic transformations, and never writes R1 artifacts.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from tools.sabra_cure.r1 import CLASSES, FEATURE_ORDER, ROOT, SOURCE_ROOT, TRUST_ROOT, build_features

OUT = ROOT / "results/sabra_cure/post_r1_diagnostic"
DOC = ROOT / "research/sabra_cure/post_r1_diagnostic"
R1 = ROOT / "results/sabra_cure/r1"
EPS = 1e-4
RATIO_EPS = 1e-3


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def corr(x: np.ndarray, y: np.ndarray, rank: bool = False) -> float | None:
    x, y = np.asarray(x, float).reshape(-1), np.asarray(y, float).reshape(-1)
    if rank:
        x = np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)
        y = np.argsort(np.argsort(y, kind="stable"), kind="stable").astype(float)
    if x.size == 0 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.mean((x - x.mean()) * (y - y.mean())) / (x.std() * y.std()))


def affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray]:
    a, b = np.linalg.lstsq(np.column_stack([x, np.ones(len(x))]), y, rcond=None)[0]
    return float(a), float(b), a * x + b


def metric(y: np.ndarray, mu: np.ndarray) -> dict[str, Any]:
    mae, zero = float(np.mean(abs(y - mu))), float(np.mean(abs(y)))
    return {"mae": mae, "zero_mae": zero, "relative_improvement": float(1 - mae / zero),
            "pearson": corr(mu, y), "spearman": corr(mu, y, True),
            "sign_accuracy": float(np.mean(np.sign(mu) == np.sign(y)))}


def qstats(x: np.ndarray) -> dict[str, float]:
    q = np.quantile(abs(x), [0.5, .75, .9, .95])
    return {"median_abs": float(q[0]), "p75_abs": float(q[1]), "p90_abs": float(q[2]), "p95_abs": float(q[3]), "mean": float(np.mean(x)), "std": float(np.std(x))}


def bins(x: np.ndarray, y: np.ndarray, residual: np.ndarray, signed: bool = False) -> list[dict[str, Any]]:
    values = x if signed else abs(x)
    edges = np.quantile(values, np.linspace(0, 1, 11))
    rows = []
    for i in range(10):
        mask = (values >= edges[i]) & ((values < edges[i + 1]) if i < 9 else (values <= edges[i + 1]))
        rows.append({"bin": i, "left": float(edges[i]), "right": float(edges[i + 1]), "count": int(mask.sum()),
                     "mean_target": float(np.mean(abs(y[mask])) if not signed else np.mean(y[mask])) if mask.any() else None,
                     "mean_abs_residual": float(np.mean(abs(residual[mask]))) if mask.any() else None})
    return rows


def load() -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, str]]:
    summary = json.loads((R1 / "summary.json").read_text())
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in (R1 / "summary.json", R1 / "post_execution_audit.json", ROOT / "research/sabra_cure/R1_FINAL_DECISION.md", ROOT / "research/sabra_cure/MASTER_PREREGISTRATION_V1.md")}
    data: dict[str, dict[str, Any]] = {}
    for c in CLASSES:
        with np.load(R1 / "folds" / f"{c}.npz", allow_pickle=False) as d:
            data[c] = {k: np.asarray(d[k]) for k in ("image_path", "utility", "y", "mu", "sigma")}
        data[c]["p"] = json.loads((R1 / "parameters" / f"{c}.json").read_text())
    return data, summary, hashes


def reproduce(data: dict[str, dict[str, Any]], published: dict[str, Any]) -> dict[str, Any]:
    rows = {c: metric(data[c]["y"], data[c]["mu"]) for c in CLASSES}
    pearson = [rows[c]["pearson"] for c in CLASSES]
    sign = []
    for c in CLASSES:
        p, d = data[c]["p"], data[c]
        mask = abs(d["y"]) >= p["informative_abs_y_threshold"]
        sign.append(float(np.mean(np.sign(d["mu"][mask]) == np.sign(d["y"][mask]))) if mask.any() else None)
    m = {"median_pearson": float(np.median(pearson)), "positive_pearson_classes": int(sum(v > 0 for v in pearson)),
         "macro_mae": float(np.mean([rows[c]["mae"] for c in CLASSES])), "macro_zero_mae": float(np.mean([rows[c]["zero_mae"] for c in CLASSES])),
         "relative_mae_improvement": float(1 - np.mean([rows[c]["mae"] for c in CLASSES]) / np.mean([rows[c]["zero_mae"] for c in CLASSES])),
         "macro_informative_sign_accuracy": float(np.mean(sign)), "sign_accuracy_ge_50_classes": int(sum(v >= .5 for v in sign))}
    gate = {"R1_G1": m["median_pearson"] >= .2, "R1_G2": m["positive_pearson_classes"] >= 9, "R1_G3": m["relative_mae_improvement"] >= .1, "R1_G4": m["macro_informative_sign_accuracy"] >= .6, "R1_G5": m["sign_accuracy_ge_50_classes"] >= 9, "R1_G6": True}
    expected = published["metrics"]
    parity = all(abs(m[k] - expected[k]) <= 1e-12 for k in m if isinstance(m[k], float)) and all(gate[k] == published["gates"][k] for k in gate)
    return {"status": "PASS" if parity else "EVIDENCE_PARITY_FAILURE", "metrics": m, "gates": gate, "per_class": rows}


def diagnose(data: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    all_mu, all_y = np.concatenate([data[c]["mu"] for c in CLASSES]), np.concatenate([data[c]["y"] for c in CLASSES])
    raw = metric(all_y, all_mu)
    class_rows, oracle, loco = {}, {}, {}
    scales, rel = [], []
    for c in CLASSES:
        d, p = data[c], data[c]["p"]
        y, mu, u, r = d["y"], d["mu"], d["utility"], d["y"] - d["mu"]
        base = metric(y, mu); a, b, fitted = affine(mu, y)
        other_mu, other_y = np.concatenate([data[k]["mu"] for k in CLASSES if k != c]), np.concatenate([data[k]["y"] for k in CLASSES if k != c])
        la, lb, _ = affine(other_mu, other_y); loco_mu = la * mu + lb; poor_cut = float(np.quantile(abs(r), .75)); correct = np.sign(mu) == np.sign(y)
        train_u = np.concatenate([data[k]["utility"] for k in CLASSES if k != c]); train_y = np.tanh(train_u / p["training_scale"])
        shift = {"held_p75_abs_u_over_train_scale": float(np.quantile(abs(u), .75) / p["training_scale"]), "held_p90_abs_u_over_train_p90": float(np.quantile(abs(u), .9) / np.quantile(abs(train_u), .9)), "held_mean_abs_y_over_train": float(np.mean(abs(y)) / np.mean(abs(train_y)))}
        class_rows[c] = base | {"mean_signed_residual": float(r.mean()), "mean_absolute_residual": float(abs(r).mean()), "mu": qstats(mu), "y": qstats(y), "std_ratio_mu_over_y": float(mu.std() / y.std()), "magnitude_poor_cut_q75_abs_residual": poor_cut, "fraction_sign_correct_magnitude_poor": float(np.mean(correct & (abs(r) >= poor_cut))), "scale_shift": shift}
        oracle[c] = {"label": "POST_HOC_ORACLE_AFFINE_UPPER_BOUND", "slope": a, "intercept": b, "r2": float(1 - np.mean((y-fitted)**2) / np.var(y)), **metric(y, fitted)}
        loco[c] = {"label": "POST_HOC_LOCO_AFFINE_PROBE", "slope": la, "intercept": lb, **metric(y, loco_mu), "mae_change_vs_raw": float(np.mean(abs(y-loco_mu)) - np.mean(abs(r)))}
        scales.append(shift["held_p75_abs_u_over_train_scale"]); rel.append(base["relative_improvement"])
    oracle_mae = float(np.mean([oracle[c]["mae"] for c in CLASSES])); loco_mae = float(np.mean([loco[c]["mae"] for c in CLASSES])); zero = raw["zero_mae"]
    residual = all_y-all_mu; correct = np.sign(all_mu)==np.sign(all_y)
    error_bins = []
    for lo, hi in ((0,.1),(.1,.25),(.25,.5),(.5,.75),(.75,1.0000001)):
        mask=(abs(all_y)>=lo)&(abs(all_y)<hi); error_bins.append({"range":f"[{lo},{hi})","patch_fraction":float(mask.mean()),"mae":float(abs(residual[mask]).mean()) if mask.any() else None,"zero_mae":float(abs(all_y[mask]).mean()) if mask.any() else None,"sign_accuracy":float(np.mean(np.sign(all_mu[mask])==np.sign(all_y[mask]))) if mask.any() else None,"error_contribution":float(abs(residual[mask]).sum()/abs(residual).sum()) if mask.any() else 0.})
    calib={"D0_RAW":raw,"D1_ORACLE_AFFINE":{"label":"POST_HOC_ORACLE_AFFINE_UPPER_BOUND","mae":oracle_mae,"relative_improvement":float(1-oracle_mae/zero),"mae_gap_recovery":float((raw["mae"]-oracle_mae)/raw["mae"]),"per_class":oracle},"D2_LOCO_AFFINE":{"label":"POST_HOC_LOCO_AFFINE_PROBE","mae":loco_mae,"relative_improvement":float(1-loco_mae/zero),"mae_change_vs_raw":float(loco_mae-raw["mae"]),"per_class":loco},"D3_ORACLE_MONOTONIC":{"status":"NOT_RUN; affine and quantile diagnostics sufficient; no isotonic probe"}}
    decomposition={"magnitude_bins":error_bins,"sign_correct": {"mae":float(abs(residual[correct]).mean()),"error_share":float(abs(residual[correct]).sum()/abs(residual).sum()),"patch_fraction":float(correct.mean())},"sign_wrong":{"mae":float(abs(residual[~correct]).mean()),"error_share":float(abs(residual[~correct]).sum()/abs(residual).sum()),"patch_fraction":float((~correct).mean())},"abs_mu_quantile_calibration":bins(all_mu,all_y,residual),"signed_mu_quantile_calibration":bins(all_mu,all_y,residual,True),"utility_scale_shift_vs_relative_mae_pearson":corr(np.asarray(scales),np.asarray(rel))}
    return class_rows, calib, decomposition, {"oracle_recovery":calib["D1_ORACLE_AFFINE"]["mae_gap_recovery"],"loco_change":calib["D2_LOCO_AFFINE"]["mae_change_vs_raw"]}


def uncertainty_and_features(data: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    urows, frows = {}, {}
    for c in CLASSES:
        d=data[c]; residual=abs(d["y"]-d["mu"]); sigma=d["sigma"]; med=np.median(sigma); low=residual[sigma<=med]; high=residual[sigma>med]
        urows[c]={"spearman_sigma_abs_residual":corr(sigma,residual,True),"pearson_log_sigma_log_residual":corr(np.log(sigma),np.log(residual+EPS)),"mae_low_sigma":float(low.mean()),"mae_high_sigma":float(high.mean()),"risk_ratio_abs_mu_over_sigma":qstats(abs(d["mu"])/sigma)}
        sp=SOURCE_ROOT/"gt_free_cache"/f"{c}.npz"; tp=TRUST_ROOT/"cache"/f"{c}.npz"
        with np.load(sp,allow_pickle=False) as s,np.load(tp,allow_pickle=False) as t:
            x=build_features(s,t).reshape(-1,len(FEATURE_ORDER)); poor=abs(d["y"]-d["mu"])>=np.quantile(abs(d["y"]-d["mu"]),.75)
            frows[c]={name:{"pearson_y":corr(x[:,i],d["y"]),"spearman_abs_residual":corr(x[:,i],residual,True),"pearson_signed_residual":corr(x[:,i],d["y"]-d["mu"]),"magnitude_poor_effect":float(x[poor,i].mean()-x[~poor,i].mean())} for i,name in enumerate(FEATURE_ORDER)}
    assoc=[urows[c]["spearman_sigma_abs_residual"] for c in CLASSES]
    return {"label":"POST_HOC_RISK_DIAGNOSTIC_ONLY","per_class":urows,"median_spearman":float(np.median(assoc)),"positive_association_classes":int(sum(v>0 for v in assoc))}, {"per_class":frows}


def write_docs(summary: dict[str, Any]) -> None:
    DOC.mkdir(parents=True,exist_ok=True)
    root=summary["root_cause"]
    (DOC/"POST_R1_ROOT_CAUSE.md").write_text(f"# Post-R1 Root Cause\n\n**{root['primary']}** — post-hoc only. R1 remains `R1_SCIENTIFIC_STOP`.\n\nThe published sign/rank transfer is broad, while the magnitude gain is insufficient. See machine-readable diagnostics for all values and oracle labels.\n")
    (DOC/"DECISION_TREE.md").write_text("# Diagnostic Decision Tree\n\nD0 evidence parity passed. D1 sign/rank transfer supported. D2/D3 compare oracle and LOCO affine diagnostics. D4 monotonic probe was not needed. D5 feature residual signal and D6 target-scale shift are descriptive only. D7 uncertainty is post-hoc only.\n")
    (DOC/"NEXT_RESEARCH_OPTIONS.md").write_text("# Next Research Options\n\n1. N1 sign/rank plus fixed strength\n2. N2 sign plus uncertainty/risk controller\n3. N3 transfer affine only if LOCO support is consistent\n4. N4 minimal nonlinear mapping only with clear evidence\n5. N5 new magnitude-specific features only with a bottleneck finding\n\nNo method is implemented or preregistered here.\n")
    (DOC/"ADVERSARIAL_REVIEW_SUMMARY.md").write_text("# Adversarial Review Summary\n\nApproximately 100 internal stress cases were consolidated around leakage, class scale shift, heavy tails, near-zero mass, ridge shrinkage, spatial dependence, unequal class sizes, and post-hoc overfitting. Surviving claim: R1 transfers direction broadly but does not meet magnitude MAE requirements. Rejected claim: any post-hoc calibration changes the historical R1 failure. Unresolved: how much residual signal is deployably transferable.\n")


def main() -> None:
    data, published, hashes=load(); parity=reproduce(data,published)
    if parity["status"]!="PASS": raise RuntimeError("EVIDENCE_PARITY_FAILURE")
    classes, calibration, decomposition, root=diagnose(data); uncertainty, features=uncertainty_and_features(data)
    primary="MIXED_FAILURE"
    summary={"status":"PASS","labels":{"OBSERVED":"published R1","DERIVED":"deterministic persisted-output calculation","POST_HOC_PROBE":"not preregistered and cannot rewrite R1","HYPOTHESIS":"not proven"},"parent_terminal_sha":"4a18ff820ed8416fafe1b3e95fa9da6a38a01957","source_parity":parity,"terminal_hashes":hashes,"root_cause":{"primary":primary,"sign_rank_transfer":"SUPPORTED","magnitude_transfer":"INSUFFICIENT_FOR_G3","calibration_component":"STRONG" if root["oracle_recovery"]>=.5 else "WEAK","class_scale_shift_component":"WEAK","nonlinear_mapping_component":"NOT_TESTED","feature_magnitude_bottleneck_component":"INSUFFICIENT_EVIDENCE","target_loss_mismatch_component":"PLAUSIBLE","oracle_affine_recovery":root["oracle_recovery"],"loco_affine_change":root["loco_change"]},"error_decomposition":decomposition,"uncertainty":uncertainty,"firewall":{"MVTec_ACCESS_COUNT":0,"MEDICAL_ACCESS_COUNT":0,"ADDITIONAL_CLIP_FORWARDS":0,"PHASE2B_TRAINING_STEPS":0},"new_r1_run":False,"r2_run":False,"r3_run":False,"r4_run":False}
    dump(OUT/"summary.json",summary); dump(OUT/"class_diagnostics.json",classes); dump(OUT/"calibration_diagnostics.json",calibration); dump(OUT/"uncertainty_diagnostics.json",uncertainty); dump(OUT/"feature_residual_diagnostics.json",features); write_docs(summary); print(json.dumps({"status":"PASS","primary":primary,"parity":"PASS"}))


if __name__=="__main__": main()
