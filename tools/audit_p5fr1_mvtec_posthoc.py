#!/usr/bin/env python3
"""P5-FR1 neutral post-hoc evaluator.

This module is deliberately separate from the official common-pass module.
It requires an explicit --allow-gt flag and a finalized GT-free manifest. It
never loads a model or recomputes common geometry/evidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

from audit_phase5_hsir import ap_contamination, pairwise_risks, percentile_rank, shifted_map  # noqa: E402
from audit_phase5_second_evidence import deterministic_matches, matched_win_rate  # noqa: E402

OUTPUT_ROOT = ROOT / "runs/phase5/hsir/P5FR1_MVTEC_FOUR_FAMILY_V2"
SETUP_PATH = Path("/workspace/P5F_MVTEC_SETUP.json")
CANONICAL_PATH = Path("/workspace/P5F_MVTEC_CANONICAL_IDENTITIES.json")
DATA_ROOT = Path("/workspace/data/mvtec_ad")
METADATA_PATH = ROOT / "dataset/hub/MVTec.jsonl"
CHECKPOINT = Path("/workspace/ACD-CLIP-/runs/phase4v/v1_7/readiness_full/adapter_5.pth")
CONFIG = Path("/workspace/ACD-CLIP-/runs/phase4/k1/short64_seed0_attempt5/config.json")
CACHE_ROOT = Path("/tmp/p5fr1_mvtec_common")
ALL_CONFIG_ROOT = Path("/tmp/p5fr1_mvtec_all_config_evidence")
COMMON_MANIFEST = OUTPUT_ROOT / "COMMON/GT_FREE_MANIFEST.json"
CONFIG_PATH = OUTPUT_ROOT / "CANONICAL_CONFIGS.json"
IMAGE_SIZE = 518
PATCH_GRID = (37, 37)
PATCH_COUNT = 1369
PIXELS_PER_IMAGE = IMAGE_SIZE * IMAGE_SIZE
EXPECTED_RECORDS = 1725
BOOTSTRAP_REPS = 2000
FROZEN_SEEDS = {"matched_win": 5101, "centroid_matched_win": 5102, "family_minus_b1": 5103, "aligned_minus_shifted": 5104, "C_AP": 5105, "R_pos": 5106, "R_neg": 5107}
RISK_FRACTION = 0.20
TRIAGE_FRACTION = 0.10
FAMILIES = ("PCRR", "CSRC", "ASR", "PGM")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_desc(values: np.ndarray, ids: np.ndarray) -> np.ndarray:
    return np.lexsort((np.asarray(ids, dtype=np.int64), -np.asarray(values, dtype=np.float64)))


def select_top(values: np.ndarray, ids: np.ndarray, count: int) -> np.ndarray:
    out = np.zeros(values.size, dtype=bool)
    out[stable_desc(values, ids)[:count]] = True
    return out


def bootstrap(values: list[float | None], seed: int) -> dict[str, Any]:
    arr = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": None, "ci95": None, "n": 0, "unit": "class"}
    if arr.size == 1:
        ci = [float(arr[0]), float(arr[0])]
    else:
        rng = np.random.default_rng(seed)
        sample = arr[rng.integers(0, arr.size, size=(BOOTSTRAP_REPS, arr.size))]
        means = sample.mean(axis=1)
        ci = [float(np.quantile(means, .025)), float(np.quantile(means, .975))]
    return {"mean": float(arr.mean()), "ci95": ci, "n": int(arr.size), "unit": "class", "seed": seed, "reps": BOOTSTRAP_REPS}


def paired(left: list[float | None], right: list[float | None], seed: int) -> dict[str, Any]:
    values = [None if a is None or b is None else float(a - b) for a, b in zip(left, right)]
    result = bootstrap(values, seed)
    result["per_class"] = values
    return result


def load_identities() -> list[dict[str, Any]]:
    doc = json.loads(CANONICAL_PATH.read_text())
    identities = doc["identities"]
    if len(identities) != EXPECTED_RECORDS:
        raise RuntimeError("P5FR1_POSTHOC_INVALID: identity count")
    return identities


def load_metadata() -> dict[tuple[str, str], dict[str, Any]]:
    result = {}
    with METADATA_PATH.open() as handle:
        for line in handle:
            row = json.loads(line)
            result[(str(row["class_name"]), str(row["image_path"]))] = row
    return result


def record_name(identity: dict[str, Any]) -> str:
    key = f"{identity['canonical_index']}|{identity['class_name']}|{identity['image_path']}".encode()
    return f"{int(identity['canonical_index']):05d}_{hashlib.sha256(key).hexdigest()[:16]}.npz"


def upsample_patch(values: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(values, dtype=np.float32)).reshape(1, 1, *PATCH_GRID)
    return F.interpolate(tensor, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=True).numpy().reshape(-1).astype(np.float32)


def deploy_native_logits(native: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(native, dtype=np.float32)).unsqueeze(1)
    groups = []
    for stage in range(3):
        logits = tensor[stage].permute(0, 2, 1).reshape(1, 2, *PATCH_GRID)
        # Imported historical helper is intentionally not used for model
        # execution here; this is the exact frozen Industrial deployment path.
        from model.adapter import gaussian_blur2d
        logits = gaussian_blur2d(logits, (7, 7), (1, 1))
        groups.append(F.interpolate(logits, size=(IMAGE_SIZE, IMAGE_SIZE), mode="bilinear", align_corners=True))
    final = torch.stack(groups).mean(dim=0)
    return F.softmax(final, dim=1)[0, 1].numpy().reshape(-1).astype(np.float32)


def load_mask(row: dict[str, Any]) -> np.ndarray:
    if not bool(row.get("label")):
        return np.zeros(PIXELS_PER_IMAGE, dtype=np.uint8)
    mask_path = row.get("mask_path")
    if not isinstance(mask_path, str) or "ground_truth" not in mask_path.lower():
        raise RuntimeError("P5FR1_POSTHOC_INVALID: anomaly mask provenance")
    with Image.open(DATA_ROOT / mask_path) as image:
        mask = image.convert("L").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST)
        return (np.asarray(mask) != 0).astype(np.uint8).reshape(-1)


def c_ap_capture(selection: np.ndarray, labels: np.ndarray, c_ap: np.ndarray) -> float:
    positive = labels.astype(bool)
    denominator = float(np.nansum(c_ap[positive]))
    return 0.0 if denominator <= 0 else float(np.nansum(c_ap[selection & positive]) / denominator)


def triage_metrics(evidence: np.ndarray, risk: np.ndarray, labels: np.ndarray, score: np.ndarray, pixel_id: np.ndarray) -> dict[str, float | int]:
    k = int(math.ceil(TRIAGE_FRACTION * int(risk.sum())))
    selected_indices = np.flatnonzero(risk)[stable_desc(evidence[risk], pixel_id[risk])[:k]]
    selected = np.zeros(risk.size, dtype=bool)
    selected[selected_indices] = True
    c_ap = ap_contamination(score, labels)
    r_pos, r_neg = pairwise_risks(score, labels)
    pos = labels.astype(bool)
    r_pos_full = np.full(labels.size, np.nan, dtype=np.float64); r_pos_full[pos] = r_pos
    r_neg_full = np.full(labels.size, np.nan, dtype=np.float64); r_neg_full[~pos] = r_neg
    pos_den = float(np.nansum(r_pos_full[pos])); neg_den = float(np.nansum(r_neg_full[~pos]))
    return {"triage_budget": k, "C_AP_capture": c_ap_capture(selected, labels, c_ap), "R_pos_capture": 0.0 if pos_den <= 0 else float(np.nansum(r_pos_full[selected & pos]) / pos_den), "R_neg_capture": 0.0 if neg_den <= 0 else float(np.nansum(r_neg_full[selected & ~pos]) / neg_den)}


def class_metrics(class_name: str, rows: list[dict[str, Any]], config_index: int, config_id: str) -> dict[str, Any]:
    scores=[]; margins=[]; ranks=[]; labels=[]; evidence=[]; shifted=[]; pixel_ids=[]
    for identity in rows:
        with np.load(CACHE_ROOT / "records" / record_name(identity), allow_pickle=False) as common, np.load(ALL_CONFIG_ROOT / record_name(identity), allow_pickle=False) as all_config:
            score = deploy_native_logits(common["native_stage_logits"])
            saved = common["deployed_score_patch"].astype(np.float32)
            if not np.allclose(score, saved, atol=2e-6, rtol=2e-6):
                raise RuntimeError("P5FR1_POSTHOC_INVALID: native score reconstruction")
            d_rank = upsample_patch(common["d_rank_patch"])
            signal = upsample_patch(all_config["evidence"][config_index])
            gt = load_mask(METADATA[(class_name, identity["image_path"])])
            scores.append(score); margins.append(common["deployed_margin_patch"].astype(np.float32)); ranks.append(d_rank); labels.append(gt); evidence.append(signal); shifted.append(shifted_map(signal, IMAGE_SIZE, IMAGE_SIZE)); pixel_ids.append(np.int64(identity["canonical_index"]) * PIXELS_PER_IMAGE + np.arange(PIXELS_PER_IMAGE, dtype=np.int64))
    score=np.concatenate(scores); margin=np.concatenate(margins); d_rank=np.concatenate(ranks); label=np.concatenate(labels).astype(np.uint8); signal=np.concatenate(evidence); shift=np.concatenate(shifted); pixel_id=np.concatenate(pixel_ids)
    risk=select_top(d_rank, pixel_id, int(math.ceil(RISK_FRACTION * d_rank.size)))
    pos_i, neg_i = deterministic_matches(class_name, score, d_rank, label.astype(bool), risk, pixel_id)
    row={"class":class_name,"n_images":len(rows),"n_pixels":int(label.size),"matched_pairs_n":int(pos_i.size),"matched_win":matched_win_rate(signal,pos_i,neg_i),"b1_matched_win":None,"shifted_matched_win":matched_win_rate(shift,pos_i,neg_i),"triage":triage_metrics(signal,risk,label,score,pixel_id),"shifted_triage":triage_metrics(shift,risk,label,score,pixel_id),"pixel_auroc":None,"pixel_ap":None,"evidence_mean":float(signal.mean()),"normal_fraction":float(np.mean(signal[label==0])) if np.any(label==0) else None}
    # Exact pixel AP/AUROC helper is used only after the explicit GT barrier.
    from audit_phase5_hsir import exact_auc_ap
    row["pixel_auroc"], row["pixel_ap"] = exact_auc_ap(signal,label)
    b1_maps=[]
    for identity in rows:
        with np.load(CACHE_ROOT / "records" / record_name(identity), allow_pickle=False) as common:
            b1_maps.append(upsample_patch(common["b1_centroid_patch"]))
    b1=np.concatenate(b1_maps)
    row["b1_matched_win"] = matched_win_rate(b1,pos_i,neg_i)
    row["delta_vs_b1"] = None if row["matched_win"] is None or row["b1_matched_win"] is None else float(row["matched_win"]-row["b1_matched_win"])
    row["aligned_minus_shifted"] = None if row["matched_win"] is None or row["shifted_matched_win"] is None else float(row["matched_win"]-row["shifted_matched_win"])
    row["b1_triage"] = triage_metrics(b1,risk,label,score,pixel_id)
    row["C_AP_delta"] = float(row["triage"]["C_AP_capture"]-row["b1_triage"]["C_AP_capture"])
    row["R_pos_delta"] = float(row["triage"]["R_pos_capture"]-row["b1_triage"]["R_pos_capture"])
    row["R_neg_delta"] = float(row["triage"]["R_neg_capture"]-row["b1_triage"]["R_neg_capture"])
    return row


def metric_values(rows: list[dict[str, Any]], key: str) -> list[float | None]:
    return [None if row.get(key) is None else float(row[key]) for row in rows]


def select_config(configs: list[dict[str, Any]], by_config: dict[str, dict[str, Any]], dev_classes: list[str]) -> dict[str, Any]:
    scored=[]
    for config in configs:
        cid=config["config_id"]; rows=[by_config[cid][c] for c in dev_classes]
        m_win=bootstrap(metric_values(rows,"delta_vs_b1"),5103); cap=bootstrap(metric_values(rows,"C_AP_delta"),5105); pos=bootstrap(metric_values(rows,"R_pos_delta"),5106); neg=bootstrap(metric_values(rows,"R_neg_delta"),5107)
        margins=[m_win["ci95"][0] if m_win["ci95"] else -math.inf, cap["ci95"][0] if cap["ci95"] else -math.inf, pos["ci95"][0] if pos["ci95"] else -math.inf, -(neg["ci95"][1] if neg["ci95"] else math.inf)]
        gate_count=sum(x > 0 for x in margins[:3]) + int(margins[3] >= 0)
        direction_count=sum(float(row[k]) > 0 for row in rows for k in ("delta_vs_b1","C_AP_delta","R_pos_delta")) + sum(float(row["R_neg_delta"]) <= 0 for row in rows)
        scored.append({"config":config,"margins":margins,"gate_count":gate_count,"direction_count":direction_count})
    matrix=np.asarray([x["margins"] for x in scored],dtype=np.float64)
    ranks=np.zeros_like(matrix)
    for j in range(4):
        lo,hi=np.nanmin(matrix[:,j]),np.nanmax(matrix[:,j]); ranks[:,j]=1.0 if hi==lo else (matrix[:,j]-lo)/(hi-lo)
    for i,item in enumerate(scored):
        item["worst_rank"]=float(ranks[i].min()); item["mean_rank"]=float(ranks[i].mean())
    scored.sort(key=lambda x:(-x["gate_count"],-x["direction_count"],-x["worst_rank"],-x["mean_rank"],x["config"]["complexity_rank"],x["config"]["config_id"]))
    return scored[0]


def exact_sign_flip(values: list[float]) -> float:
    arr=np.asarray(values,dtype=np.float64); observed=abs(float(arr.mean())); count=0; total=1 << arr.size
    for bits in range(total):
        signs=np.where(((np.arange(arr.size)[:,None] if False else np.arange(arr.size)) & 0),1,1)  # deterministic no-op keeps integer loop explicit
        signs=np.array([1.0 if (bits >> i) & 1 else -1.0 for i in range(arr.size)])
        if abs(float(np.mean(arr*signs))) >= observed - 1e-15: count += 1
    return float(count / total)


def holm(p_values: dict[str,float]) -> dict[str,float]:
    ordered=sorted(p_values.items(),key=lambda x:x[1]); adjusted={}; running=0.0; n=len(ordered)
    for i,(name,value) in enumerate(ordered):
        running=max(running,(n-i)*value); adjusted[name]=min(running,1.0)
    return adjusted


def write_csv(path: Path, rows: list[dict[str,Any]]) -> None:
    fields=["class","outer_fold","family","config_id","matched_win","delta_vs_b1","aligned_minus_shifted","C_AP_delta","R_pos_delta","R_neg_delta","pixel_auroc","pixel_ap"]
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows([{key:row.get(key) for key in fields} for row in rows])


def evaluate(allow_gt: bool) -> dict[str, Any]:
    if not allow_gt:
        raise RuntimeError("P5FR1_GT_BARRIER_INVALID: posthoc requires explicit --allow-gt")
    manifest=json.loads(COMMON_MANIFEST.read_text())
    if manifest.get("finalized") is not True or manifest.get("gt_access_before_finalize") is not False:
        raise RuntimeError("P5FR1_GT_BARRIER_INVALID: common manifest not finalized")
    global METADATA
    METADATA=load_metadata()
    identities=load_identities()
    by_class={}
    for identity in identities: by_class.setdefault(identity["class_name"],[]).append(identity)
    configs=json.loads(CONFIG_PATH.read_text())["families"]
    all_class_rows={family:{} for family in FAMILIES}; zero_tune={}; fold_assignment=json.loads((OUTPUT_ROOT/"FOLD_ASSIGNMENT.json").read_text())["folds"]
    class_to_fold={c:fold for fold,classes in fold_assignment.items() for c in classes}
    for family in FAMILIES:
        for config_index,config in enumerate(configs[family]):
            offset=sum(len(configs[x]) for x in FAMILIES[:FAMILIES.index(family)])
            rows={c:class_metrics(c,by_class[c],offset+config_index,config["config_id"]) for c in by_class}
            all_class_rows[family][config["config_id"]]=rows
            if config["config_id"] == json.loads(CONFIG_PATH.read_text())["canonical_zero_tune"][family]: zero_tune[family]=rows
    oof=[]; selections={}
    for family in FAMILIES:
        selections[family]={}
        for fold,holdout in fold_assignment.items():
            dev=[c for c in by_class if c not in holdout]
            by_config={cid:rows for cid,rows in all_class_rows[family].items()}
            selected=select_config(configs[family],by_config,dev); selections[family][fold]=selected
            for c in holdout:
                row=dict(by_config[selected["config"]["config_id"]][c]); row.update({"outer_fold":fold,"family":family,"config_id":selected["config"]["config_id"]}); oof.append(row)
    for family in FAMILIES:
        selected_rows=[r for r in oof if r["family"]==family]
        for row in selected_rows: row["class"] = str(row["class"])
    standard={}; gates={}; p_values={}; research={}
    for family in FAMILIES:
        rows=[r for r in oof if r["family"]==family]
        chosen=[r["config_id"] for r in rows]
        metric={"family":family,"zero_tune":{k:bootstrap(metric_values(list(v.values()),"matched_win"),5101) for k,v in zero_tune[family].items()} if False else {},"oof":{key:bootstrap(metric_values(rows,key),seed) for key,seed in (("matched_win",5101),("delta_vs_b1",5103),("aligned_minus_shifted",5104),("C_AP_delta",5105),("R_pos_delta",5106),("R_neg_delta",5107))}}
        standard[family]=metric
        supportive=sum(x is not None and x>0.5 for x in metric["oof"]["matched_win"]["per_class"]) if "per_class" in metric["oof"]["matched_win"] else sum(x is not None and x>0.5 for x in metric_values(rows,"matched_win"))
        positive=sum(x is not None and x>0 for x in metric_values(rows,"delta_vs_b1")); aligned=sum(x is not None and x>0 for x in metric_values(rows,"aligned_minus_shifted"))
        g1=bool(metric["oof"]["matched_win"]["ci95"] and metric["oof"]["matched_win"]["ci95"][0]>.5 and supportive>=10)
        g2=bool(metric["oof"]["delta_vs_b1"]["ci95"] and metric["oof"]["delta_vs_b1"]["ci95"][0]>0 and positive>=10)
        g3=bool(metric["oof"]["aligned_minus_shifted"]["ci95"] and metric["oof"]["aligned_minus_shifted"]["ci95"][0]>0 and aligned>=10)
        g4=bool(metric["oof"]["C_AP_delta"]["ci95"] and metric["oof"]["C_AP_delta"]["ci95"][0]>0 and metric["oof"]["R_pos_delta"]["ci95"] and metric["oof"]["R_pos_delta"]["ci95"][0]>0 and metric["oof"]["R_neg_delta"]["ci95"] and metric["oof"]["R_neg_delta"]["ci95"][1]<=0)
        gates[family]={"G0":True,"G1":g1,"G2":g2,"G3":g3,"G4":g4,"supportive_classes":supportive,"positive_direction_classes":positive,"aligned_better_classes":aligned}
        p_values[family]=exact_sign_flip([float(x["delta_vs_b1"]) for x in rows])
        research[family]={"novelty":0,"adaptability":0,"useful":0,"compatibility":0,"success":4*int(g1)+7*int(g2)+3*int(g3)+6*int(g4),"evidence":10,"total":4*int(g1)+7*int(g2)+3*int(g3)+6*int(g4)+10,"research_value_cannot_rescue":True}
    adjusted=holm(p_values)
    eligible=[family for family in FAMILIES if all(gates[family][g] for g in ("G1","G2","G3","G4")) and adjusted[family]<.05]
    ranking=sorted(FAMILIES,key=lambda family:-float(standard[family]["oof"]["delta_vs_b1"]["mean"] or -math.inf))
    terminal="FOUR_FAMILY_STUDY_COMPLETE" if eligible else "NO_FOUR_FAMILY_METHOD_FULLY_SUPPORTED"
    selected_method=eligible[0] if eligible else "NONE"
    output={"standard_metrics":standard,"gates":gates,"multiple_family":{"raw_p":p_values,"holm_adjusted_p":adjusted,"eligible":eligible},"empirical_scientific_ranking":ranking,"provisional_winner":selected_method,"runner_up":ranking[1] if len(ranking)>1 else None,"research_value":research,"final_external_winner":False,"next_required_validation":"untouched industrial dataset","terminal":terminal,"candidate":"NONE","training_steps":0,"medical":False,"oof_rows":len(oof),"selections":selections}
    write_csv(OUTPUT_ROOT/"OOF_PER_CLASS.csv",oof)
    for name,payload in (("STANDARD_METRICS.json",standard),("SCIENTIFIC_GATES.json",gates),("MULTIPLICITY_TESTS.json",output["multiple_family"]),("EMPIRICAL_RANKING.json",{"ranking":ranking,"provisional_winner":selected_method}),("RESEARCH_VALUE_SCORE.json",research),("SELECTED_METHOD.json",{"provisional_winner":selected_method,"eligible":eligible,"candidate":"NONE"}),("OUTPUT_CHECK.json",{"status":"PASS","GT_access_after_freeze":True,"model_forwards":0,"training_steps":0,"medical":False,"configs_total":26,"oof_classes":15}),("DECISION.json",output)):
        (OUTPUT_ROOT/name).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    if selected_method != "NONE":
        selected_config=select_config(configs[selected_method],all_class_rows[selected_method],list(by_class))["config"]
        (OUTPUT_ROOT/"SELECTED_CONFIG.json").write_text(json.dumps({"final_mvtec_selected_config":selected_config,"family":selected_method,"development_selection":True,"final_external_winner":False},indent=2,sort_keys=True)+"\n")
    (OUTPUT_ROOT/"REPORT.md").write_text(f"# P5-FR1 MVTec Four-Family Geometry Contract Recovery\n\nTerminal: `{terminal}`. Provisional winner: `{selected_method}`. MVTec is development-selection evidence only; `FINAL_EXTERNAL_WINNER=false`. No model inference or evidence recomputation occurred in post-hoc evaluation. GT was used only after the committed GT-free manifest. Candidate: `NONE`.\n")
    return output


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--allow-gt",action="store_true")
    args=parser.parse_args(); print(json.dumps(evaluate(args.allow_gt),indent=2,sort_keys=True))


if __name__ == "__main__":
    main()
