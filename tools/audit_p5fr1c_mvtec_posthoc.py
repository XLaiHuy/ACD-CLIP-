#!/usr/bin/env python3
"""P5-FR1C neutral GT evaluator over frozen all-config evidence.

This evaluator never loads a checkpoint or model. It reads the immutable
P5FR1 common snapshot, the committed P5FR1C all-config evidence backup, and
MVTec GT only after the P5FR1C derived-evidence commit. It evaluates one class
at a time and stores scalar per-class/config results.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from audit_phase5_hsir import ap_contamination, exact_auc_ap, pairwise_risks, shifted_map  # noqa: E402
from audit_phase5_second_evidence import deterministic_matches, matched_win_rate  # noqa: E402

NAMESPACE = ROOT / "runs/phase5/hsir/P5FR1C_MVTEC_LATE_COMPLETION"
SNAPSHOT_ROOT = Path("/workspace/P5FR1_LATE_COMPLETION_SNAPSHOT")
ALL_CONFIG_ROOT = Path("/workspace/P5FR1C_ALL_CONFIG_EVIDENCE")
OUTPUT_ROOT = NAMESPACE
CANONICAL_PATH = Path("/workspace/P5F_MVTEC_CANONICAL_IDENTITIES.json")
DATA_ROOT = Path("/workspace/data/mvtec_ad")
METADATA_PATH = ROOT / "dataset/hub/MVTec.jsonl"
CONFIG_PATH = NAMESPACE / "CANONICAL_CONFIGS.json"
FOLD_PATH = NAMESPACE / "FOLD_ASSIGNMENT.json"
MANIFEST_PATH = NAMESPACE / "GT_FREE_DERIVED_MANIFEST.json"
INPUT_LOCK_PATH = NAMESPACE / "INPUT_LOCK.json"
RUN_STATUS_PATH = OUTPUT_ROOT / "EVALUATOR_RUN.json"
IMAGE_SIZE = 518
PATCH_GRID = (37, 37)
PATCH_COUNT = 1369
PIXELS_PER_IMAGE = IMAGE_SIZE * IMAGE_SIZE
EXPECTED_RECORDS = 1725
BOOTSTRAP_REPS = 2000
SIGN_TOL = 1e-15
RISK_FRACTION = 0.20
TRIAGE_FRACTION = 0.10
FAMILIES = ("PCRR", "CSRC", "ASR", "PGM")
SEEDS = {"matched_win":5101,"b1_matched_win":5102,"delta_vs_B1":5103,"aligned_minus_shifted":5104,"C_AP_delta":5105,"R_pos_delta":5106,"R_neg_delta":5107}
METRIC_KEYS = tuple(SEEDS)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp=path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with tmp.open("wb") as f:
        f.write(json_bytes(value)); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)


def record_name(identity: dict[str, Any]) -> str:
    key=f"{identity['canonical_index']}|{identity['class_name']}|{identity['image_path']}".encode()
    return f"{int(identity['canonical_index']):05d}_{hashlib.sha256(key).hexdigest()[:16]}.npz"


def load_identities() -> list[dict[str, Any]]:
    identities=json.loads(CANONICAL_PATH.read_text())["identities"]
    if len(identities)!=EXPECTED_RECORDS: raise RuntimeError("P5FR1C_IDENTITY_COUNT_INVALID")
    return identities


def load_metadata() -> dict[tuple[str,str],dict[str,Any]]:
    out={}
    with METADATA_PATH.open() as f:
        for line in f:
            row=json.loads(line); out[(str(row["class_name"]),str(row["image_path"]))]=row
    return out


def load_configs() -> tuple[dict[str,list[dict[str,Any]]],list[tuple[str,dict[str,Any]]],list[str]]:
    doc=json.loads(CONFIG_PATH.read_text()); families=doc["families"]
    rows=[(family,cfg) for family in FAMILIES for cfg in families[family]]
    ids=[cfg["config_id"] for _,cfg in rows]
    if len(rows)!=26 or len(set(ids))!=26: raise RuntimeError("P5FR1C_CONFIG_INVALID")
    return families,rows,ids


def load_folds() -> dict[str,list[str]]:
    folds=json.loads(FOLD_PATH.read_text())["folds"]
    expected={"FOLD_0":["carpet","bottle","cable"],"FOLD_1":["grid","capsule","hazelnut"],"FOLD_2":["leather","metal_nut","pill"],"FOLD_3":["tile","screw","transistor"],"FOLD_4":["wood","toothbrush","zipper"]}
    if folds!=expected: raise RuntimeError("P5FR1C_FOLD_INVALID")
    return folds


def integrity_subchecks(config_ids: list[str]) -> dict[str, bool]:
    lock = json.loads(INPUT_LOCK_PATH.read_text())
    manifest = json.loads(MANIFEST_PATH.read_text())
    expected_ids = [cfg["config_id"] for _, cfg in load_configs()[1]]
    source_hashes = lock.get("common_record_sha256", {})
    derived_hashes = manifest.get("all_config_record_hashes", {})
    return {
        "late_completion_valid": lock.get("late_completion_validated") is True,
        "common_record_count": len(source_hashes) == EXPECTED_RECORDS,
        "common_aggregate_exact": lock.get("common_record_aggregate_sha256") == manifest.get("common_record_aggregate_hash"),
        "implementation_exact": lock.get("frozen_implementation_sha") == "64a36d2df78ffac690f35c170770721ed1069fe1",
        "derived_record_count": len(derived_hashes) == EXPECTED_RECORDS,
        "derived_config_count": manifest.get("config_count") == 26 and manifest.get("config_ids") == expected_ids and config_ids == expected_ids,
        "derived_image_count": manifest.get("image_count") == EXPECTED_RECORDS,
        "derived_model_forwards_zero": manifest.get("model_forwards") == 0,
        "derived_no_gt": manifest.get("GT_metrics_read") is False and manifest.get("labels_read") == 0 and manifest.get("masks_read") == 0,
        "derived_training_zero": manifest.get("training_steps") == 0,
        "derived_medical_false": manifest.get("medical") is False,
        "derived_finalized": manifest.get("finalized") is True,
    }


def bootstrap(values: list[float|None], seed: int) -> dict[str,Any]:
    vals=[float(x) for x in values if x is not None and np.isfinite(x)]
    arr=np.asarray(vals,dtype=np.float64)
    if arr.size==0: return {"mean":None,"ci95":None,"per_class":[],"n":0,"unit":"class","seed":seed,"reps":BOOTSTRAP_REPS}
    rng=np.random.default_rng(seed)
    sample=arr[rng.integers(0,arr.size,size=(BOOTSTRAP_REPS,arr.size))]
    means=sample.mean(axis=1)
    return {"mean":float(arr.mean()),"ci95":[float(np.quantile(means,.025)),float(np.quantile(means,.975))],"per_class":vals,"n":int(arr.size),"unit":"class","seed":seed,"reps":BOOTSTRAP_REPS}


def exact_one_sided_sign_flip(values: list[float]) -> float:
    arr=np.asarray(values,dtype=np.float64)
    if arr.size!=15 or not np.all(np.isfinite(arr)): raise ValueError("exact sign flip requires 15 finite class values")
    observed=float(arr.mean()); count=0
    for bits in range(1<<arr.size):
        signs=np.asarray([1.0 if (bits>>i)&1 else -1.0 for i in range(arr.size)])
        if float(np.mean(signs*arr)) >= observed-SIGN_TOL: count+=1
    return float(count/(1<<arr.size))


def holm(raw: dict[str,float]) -> dict[str,float]:
    ordered=sorted(raw.items(),key=lambda x:(x[1],x[0])); running=0.0; out={}; n=len(ordered)
    for i,(name,p) in enumerate(ordered):
        running=max(running,(n-i)*float(p)); out[name]=min(running,1.0)
    return out


def stable_desc(values: np.ndarray, ids: np.ndarray) -> np.ndarray:
    return np.lexsort((np.asarray(ids,dtype=np.int64),-np.asarray(values,dtype=np.float64)))


def select_top(values: np.ndarray, ids: np.ndarray, count: int) -> np.ndarray:
    out=np.zeros(values.size,dtype=bool); out[stable_desc(values,ids)[:count]]=True; return out


def upsample_patch(values: np.ndarray) -> np.ndarray:
    t=torch.from_numpy(np.asarray(values,dtype=np.float32)).reshape(1,1,*PATCH_GRID)
    return F.interpolate(t,size=(IMAGE_SIZE,IMAGE_SIZE),mode="bilinear",align_corners=True).numpy().reshape(-1).astype(np.float32)


def deploy_native_logits(native: np.ndarray) -> np.ndarray:
    # This imports only the frozen blur helper; it does not instantiate or load a model.
    from model.adapter import gaussian_blur2d
    tensor=torch.from_numpy(np.asarray(native,dtype=np.float32)).unsqueeze(1)
    groups=[]
    for stage in range(3):
        logits=tensor[stage].permute(0,2,1).reshape(1,2,*PATCH_GRID)
        logits=gaussian_blur2d(logits,(7,7),(1,1))
        groups.append(F.interpolate(logits,size=(IMAGE_SIZE,IMAGE_SIZE),mode="bilinear",align_corners=True))
    final=torch.stack(groups).mean(dim=0)
    return F.softmax(final,dim=1)[0,1].numpy().reshape(-1).astype(np.float32)


def load_mask(row: dict[str,Any]) -> np.ndarray:
    if not bool(row.get("label")): return np.zeros(PIXELS_PER_IMAGE,dtype=np.uint8)
    path=row.get("mask_path")
    if not isinstance(path,str) or "ground_truth" not in path.lower(): raise RuntimeError("P5FR1C_GT_MASK_PROVENANCE_INVALID")
    with Image.open(DATA_ROOT/path) as image:
        mask=image.convert("L").resize((IMAGE_SIZE,IMAGE_SIZE),Image.Resampling.NEAREST)
        return (np.asarray(mask)!=0).astype(np.uint8).reshape(-1)


def triage_metrics(evidence: np.ndarray, common: dict[str,np.ndarray]) -> dict[str,float|int]:
    risk=common["risk"]; ids=common["pixel_id"]; labels=common["label"]
    k=int(math.ceil(TRIAGE_FRACTION*int(risk.sum())))
    selected_indices=np.flatnonzero(risk)[stable_desc(evidence[risk],ids[risk])[:k]]
    selected=np.zeros(risk.size,dtype=bool); selected[selected_indices]=True
    positive=labels.astype(bool); neg=~positive
    cden=float(np.nansum(common["c_ap"][positive])); pden=float(np.nansum(common["r_pos_full"][positive])); nden=float(np.nansum(common["r_neg_full"][neg]))
    return {"triage_budget":k,"C_AP_capture":0.0 if cden<=0 else float(np.nansum(common["c_ap"][selected&positive])/cden),"R_pos_capture":0.0 if pden<=0 else float(np.nansum(common["r_pos_full"][selected&positive])/pden),"R_neg_capture":0.0 if nden<=0 else float(np.nansum(common["r_neg_full"][selected&neg])/nden)}


def class_common(class_name: str, rows: list[dict[str,Any]], metadata: dict[tuple[str,str],dict[str,Any]], config_ids: list[str]) -> dict[str,Any]:
    scores=[]; ranks=[]; b1s=[]; labels=[]; pixels=[]; evidence_records=[]
    for identity in rows:
        name=record_name(identity)
        with np.load(SNAPSHOT_ROOT/"records"/name,allow_pickle=False) as common, np.load(ALL_CONFIG_ROOT/"records"/name,allow_pickle=False) as derived:
            score=deploy_native_logits(common["native_stage_logits"])
            saved=common["deployed_score_patch"].astype(np.float32)
            if not np.allclose(score,saved,atol=2e-6,rtol=2e-6): raise RuntimeError("P5FR1C_SCORE_RECONSTRUCTION_INVALID")
            if list(derived["config_ids"].astype(str))!=config_ids: raise RuntimeError("P5FR1C_CONFIG_ID_INVALID")
            signal=np.asarray(derived["evidence"],dtype=np.float32)
            if signal.shape!=(26,PATCH_COUNT) or not np.all(np.isfinite(signal)): raise RuntimeError("P5FR1C_EVIDENCE_SCHEMA_INVALID")
            scores.append(score); ranks.append(upsample_patch(common["d_rank_patch"])); b1s.append(upsample_patch(common["b1_centroid_patch"]))
            labels.append(load_mask(metadata[(class_name,identity["image_path"])]) )
            pixels.append(np.int64(identity["canonical_index"])*PIXELS_PER_IMAGE+np.arange(PIXELS_PER_IMAGE,dtype=np.int64))
            evidence_records.append(signal)
    score=np.concatenate(scores); d_rank=np.concatenate(ranks); b1=np.concatenate(b1s); label=np.concatenate(labels).astype(np.uint8); pixel_id=np.concatenate(pixels)
    risk=select_top(d_rank,pixel_id,int(math.ceil(RISK_FRACTION*d_rank.size)))
    pos_i,neg_i=deterministic_matches(class_name,score,d_rank,label.astype(bool),risk,pixel_id)
    c_ap=ap_contamination(score,label); r_pos,r_neg=pairwise_risks(score,label)
    positive=label.astype(bool); rpos_full=np.full(label.size,np.nan); rpos_full[positive]=r_pos; rneg_full=np.full(label.size,np.nan); rneg_full[~positive]=r_neg
    common={"score":score,"d_rank":d_rank,"b1":b1,"label":label,"pixel_id":pixel_id,"risk":risk,"pos_i":pos_i,"neg_i":neg_i,"c_ap":c_ap,"r_pos_full":rpos_full,"r_neg_full":rneg_full}
    common["b1_triage"]=triage_metrics(b1,common)
    return {"common":common,"evidence_records":evidence_records,"n_images":len(rows)}


def signal_arrays(patches: list[np.ndarray], shift: bool=False) -> np.ndarray:
    chunks=[]
    for patch in patches:
        image=upsample_patch(patch)
        chunks.append(shifted_map(image,IMAGE_SIZE,IMAGE_SIZE) if shift else image)
    return np.concatenate(chunks)


def evaluate_config(class_name: str, data: dict[str,Any], config_index: int, config_id: str) -> dict[str,Any]:
    common=data["common"]; patches=[x[config_index] for x in data["evidence_records"]]
    signal=signal_arrays(patches,False); shifted=signal_arrays(patches,True)
    triage=triage_metrics(signal,common); b1win=matched_win_rate(common["b1"],common["pos_i"],common["neg_i"])
    row={"class":class_name,"n_images":data["n_images"],"n_pixels":int(common["label"].size),"config_id":config_id,"matched_win":matched_win_rate(signal,common["pos_i"],common["neg_i"]),"b1_matched_win":b1win,"delta_vs_b1":None,"aligned_minus_shifted":None,"triage":triage,"b1_triage":common["b1_triage"],"shifted_triage":triage_metrics(shifted,common),"pixel_auroc":None,"pixel_ap":None,"evidence_mean":float(signal.mean()),"normal_fraction":float(np.mean(signal[common["label"]==0])) if np.any(common["label"]==0) else None}
    row["delta_vs_b1"]=float(row["matched_win"]-row["b1_matched_win"]); row["aligned_minus_shifted"]=float(row["matched_win"]-matched_win_rate(shifted,common["pos_i"],common["neg_i"]))
    row["C_AP_delta"]=float(triage["C_AP_capture"]-common["b1_triage"]["C_AP_capture"]); row["R_pos_delta"]=float(triage["R_pos_capture"]-common["b1_triage"]["R_pos_capture"]); row["R_neg_delta"]=float(triage["R_neg_capture"]-common["b1_triage"]["R_neg_capture"])
    row["pixel_auroc"],row["pixel_ap"]=exact_auc_ap(signal,common["label"])
    return row


def metric_values(rows: list[dict[str,Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows]


def select_config(configs: list[dict[str,Any]], by_config: dict[str,dict[str,dict[str,Any]]], dev_classes: list[str]) -> dict[str,Any]:
    scored=[]
    for cfg in configs:
        cid=cfg["config_id"]; rows=[by_config[cid][c] for c in dev_classes]
        metrics={"delta_vs_B1":bootstrap(metric_values(rows,"delta_vs_b1"),SEEDS["delta_vs_B1"]),"C_AP_delta":bootstrap(metric_values(rows,"C_AP_delta"),SEEDS["C_AP_delta"]),"R_pos_delta":bootstrap(metric_values(rows,"R_pos_delta"),SEEDS["R_pos_delta"]),"R_neg_delta":bootstrap(metric_values(rows,"R_neg_delta"),SEEDS["R_neg_delta"])}
        margins=[metrics["delta_vs_B1"]["ci95"][0],metrics["C_AP_delta"]["ci95"][0],metrics["R_pos_delta"]["ci95"][0],-metrics["R_neg_delta"]["ci95"][1]]
        gate_pass=sum(x>0 for x in margins[:3])+int(margins[3]>=0)
        direction=sum(float(row[k])>0 for row in rows for k in ("delta_vs_b1","C_AP_delta","R_pos_delta"))+sum(float(row["R_neg_delta"])<=0 for row in rows)
        scored.append({"config":cfg,"metrics":metrics,"margins":margins,"DEV_GATE_PASS_COUNT":gate_pass,"DEV_CLASS_DIRECTION_COUNT":direction})
    mat=np.asarray([x["margins"] for x in scored],dtype=np.float64); ranks=np.zeros_like(mat)
    for j in range(4):
        lo,hi=float(mat[:,j].min()),float(mat[:,j].max()); ranks[:,j]=1.0 if hi==lo else (mat[:,j]-lo)/(hi-lo)
    for i,item in enumerate(scored): item["WORST_MARGIN_RANK"]=float(ranks[i].min()); item["MEAN_MARGIN_RANK"]=float(ranks[i].mean())
    scored.sort(key=lambda x:(-x["DEV_GATE_PASS_COUNT"],-x["DEV_CLASS_DIRECTION_COUNT"],-x["WORST_MARGIN_RANK"],-x["MEAN_MARGIN_RANK"],x["config"]["complexity_rank"],x["config"]["config_id"]))
    return {"selected_config_id":scored[0]["config"]["config_id"],"selected_config":scored[0]["config"],"candidate_scores":scored}


def summarize(rows: list[dict[str,Any]]) -> dict[str,Any]:
    out={"n_classes":len(rows),"classes":[r["class"] for r in rows],"metrics":{}}
    for key,seed in SEEDS.items():
        field={"matched_win":"matched_win","b1_matched_win":"b1_matched_win","delta_vs_B1":"delta_vs_b1","aligned_minus_shifted":"aligned_minus_shifted","C_AP_delta":"C_AP_delta","R_pos_delta":"R_pos_delta","R_neg_delta":"R_neg_delta"}[key]
        out["metrics"][key]=bootstrap(metric_values(rows,field),seed)
    return out


def gate_summary(summary: dict[str,Any], g0: bool, g0_subchecks: dict[str,bool]) -> dict[str,Any]:
    m=summary["metrics"]; supportive=sum(x>0.5 for x in m["matched_win"]["per_class"]); positive=sum(x>0 for x in m["delta_vs_B1"]["per_class"]); aligned=sum(x>0 for x in m["aligned_minus_shifted"]["per_class"])
    g1=bool(m["matched_win"]["ci95"][0]>.5 and supportive>=10); g2=bool(m["delta_vs_B1"]["ci95"][0]>0 and positive>=10); g3=bool(m["aligned_minus_shifted"]["ci95"][0]>0 and aligned>=10); g4=bool(m["C_AP_delta"]["ci95"][0]>0 and m["R_pos_delta"]["ci95"][0]>0 and m["R_neg_delta"]["ci95"][1]<=0)
    return {"G0":g0,"G0_subchecks":g0_subchecks,"G1":g1,"G2":g2,"G3":g3,"G4":g4,"supportive_classes":supportive,"positive_direction_classes":positive,"aligned_better_classes":aligned}


def sensitivity(family: str, configs: list[dict[str,Any]], rows_by_config: dict[str,dict[str,dict[str,Any]]], selections: dict[str,Any]) -> dict[str,Any]:
    result={"family":family,"selected_config_frequency":{},"distinct_selected_configs":len({v["selected_config_id"] for v in selections.values()}),"configs":{}}
    for fold,sel in selections.items(): result["selected_config_frequency"][sel["selected_config_id"]]=result["selected_config_frequency"].get(sel["selected_config_id"],0)+1
    for cfg in configs:
        cid=cfg["config_id"]; rows=list(rows_by_config[cid].values()); entry={}
        for key in ("delta_vs_b1","C_AP_delta","R_pos_delta","R_neg_delta"):
            vals=np.asarray(metric_values(rows,key)); entry[key]={"mean":float(vals.mean()),"median":float(np.median(vals)),"best":float(vals.max()),"worst":float(vals.min()),"Q10":float(np.quantile(vals,.10)),"Q90":float(np.quantile(vals,.90)),"Q90_minus_Q10":float(np.quantile(vals,.90)-np.quantile(vals,.10)),"best_minus_median":float(vals.max()-np.median(vals))}
        result["configs"][cid]=entry
    for key in ("delta_vs_b1","C_AP_delta","R_pos_delta","R_neg_delta"):
        means={cid:v[key]["mean"] for cid,v in result["configs"].items()}; result[key+"_best_config"]=max(means,key=means.get); result[key+"_worst_config"]=min(means,key=means.get)
    return result


def research_scores(family: str, standard: dict[str,Any], zero: dict[str,Any], gates: dict[str,Any], sens: dict[str,Any], holm_p: float|None, manifest_ok: bool) -> dict[str,Any]:
    priors=json.loads((NAMESPACE/"DESIGN_PRIOR_SCORE.json").read_text())["components"][family]
    zgate=gate_summary(zero, gates["G0"], gates["G0_subchecks"]); selected_freq=max(sens["selected_config_frequency"].values(),default=0)/5
    selected=sens["delta_vs_b1_best_config"]; median=float(sens["configs"][selected]["delta_vs_b1"]["median"]); best=float(sens["configs"][selected]["delta_vs_b1"]["mean"]); sensitivity_score=5 if best-median>=.10 else 3 if best-median>=.05 else 1 if best>median else 0
    adaptability={"canonical_zero_tune_strength":5 if zgate["G2"] and zgate["G4"] else 3 if zgate["G2"] or zgate["G4"] else 0,"config_sensitivity":sensitivity_score,"outer_class_consistency":5 if selected_freq>=.60 else 3 if selected_freq>=.40 else 1 if selected_freq>=.20 else 0,"no_target_calibration_simplicity":5}
    useful={"C_AP":7 if standard["metrics"]["C_AP_delta"]["ci95"][0]>0 else 0,"R_pos":7 if standard["metrics"]["R_pos_delta"]["ci95"][0]>0 else 0,"R_neg":6 if standard["metrics"]["R_neg_delta"]["ci95"][1]<=0 else 0}
    success={"G1":4*int(gates["G1"]),"G2":7*int(gates["G2"]),"G3":3*int(gates["G3"]),"G4":6*int(gates["G4"])}
    evidence={"all_config_reconstruction":3*int(manifest_ok),"alignment_grounding":3*int(gates["G3"]),"class_direction_consistency":2*int(gates["supportive_classes"]>=10 and gates["positive_direction_classes"]>=10 and gates["aligned_better_classes"]>=10),"cv_integrity":1,"multiplicity_confirmation":1*int(holm_p is not None and holm_p<.05)}
    total=sum(priors.values())+sum(adaptability.values())+sum(useful.values())+sum(success.values())+sum(evidence.values())
    return {"family":family,"novelty":priors["novelty"],"compatibility_design":priors["compatibility_design"],"adaptability":adaptability,"useful":useful,"success":success,"evidence":evidence,"total":int(total),"scientific_eligibility_independent":bool(all(gates[g] for g in ("G0","G1","G2","G3","G4")) and holm_p is not None and holm_p<.05)}


def write_csv(path: Path, rows: list[dict[str,Any]]) -> None:
    fields=["class","outer_fold","family","config_id","matched_win","b1_matched_win","delta_vs_b1","aligned_minus_shifted","C_AP_delta","R_pos_delta","R_neg_delta","pixel_auroc","pixel_ap"]
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows({k:r.get(k) for k in fields} for r in rows)


def evaluate() -> dict[str,Any]:
    if not MANIFEST_PATH.is_file() or not json.loads(MANIFEST_PATH.read_text()).get("finalized"): raise RuntimeError("P5FR1C_GT_FREE_MANIFEST_REQUIRED")
    families,config_rows,config_ids=load_configs(); folds=load_folds(); identities=load_identities(); metadata=load_metadata()
    g0_subchecks = integrity_subchecks(config_ids)
    g0 = bool(all(g0_subchecks.values()))
    by_class={}
    for x in identities: by_class.setdefault(x["class_name"],[]).append(x)
    configs_by_family={f:[c for fam,c in config_rows if fam==f] for f in FAMILIES}
    rows_by_family={f:{} for f in FAMILIES}; index_by_id={c["config_id"]:i for i,(_,c) in enumerate(config_rows)}
    for class_name in sorted(by_class):
        data=class_common(class_name,by_class[class_name],metadata,config_ids)
        for family in FAMILIES:
            for cfg in configs_by_family[family]:
                cid=cfg["config_id"]; rows_by_family[family].setdefault(cid,{})[class_name]=evaluate_config(class_name,data,index_by_id[cid],cid)
    # Strip helper sentinel and calculate fold-specific selection.
    fold_selections={}; oof=[]
    for family in FAMILIES:
        fold_selections[family]={}
        class_rows={cid:rows_by_family[family][cid] for cid in [c["config_id"] for c in configs_by_family[family]]}
        for fold,holdout in folds.items():
            dev=[c for c in sorted(by_class) if c not in holdout]
            selected=select_config(configs_by_family[family],class_rows,dev)
            fold_selections[family][fold]={"selected_config_id":selected["selected_config_id"],"selected_config":selected["selected_config"],"dev_classes":dev,"holdout_classes":holdout,"candidate_scores":selected["candidate_scores"]}
            for cls in holdout:
                row=dict(class_rows[selected["selected_config_id"]][cls]); row.update({"outer_fold":fold,"family":family,"config_id":selected["selected_config_id"]}); oof.append(row)
    standard={}; gates={}; zero={}; sensitivity_out={}
    for family in FAMILIES:
        fam_oof=[r for r in oof if r["family"]==family]; standard[family]=summarize(fam_oof); gates[family]=gate_summary(standard[family],g0,g0_subchecks); canonical=json.loads(CONFIG_PATH.read_text())["canonical_zero_tune"][family]; zero_summary=summarize(list(rows_by_family[family][canonical].values())); zero[family]={"family":family,"config_id":canonical,"result":zero_summary,"gates":gate_summary(zero_summary,g0,g0_subchecks)}
        sensitivity_out[family]=sensitivity(family,configs_by_family[family],rows_by_family[family],fold_selections[family])
    raw_p={f:exact_one_sided_sign_flip(standard[f]["metrics"]["delta_vs_B1"]["per_class"]) for f in FAMILIES}; holm_p=holm(raw_p)
    eligible=[f for f in FAMILIES if all(gates[f][g] for g in ("G0","G1","G2","G3","G4")) and holm_p[f]<.05]
    ranking=sorted(FAMILIES,key=lambda f:(-standard[f]["metrics"]["delta_vs_B1"]["mean"],f)); eligible_ranked=[f for f in ranking if f in eligible]
    winner=eligible_ranked[0] if len(eligible_ranked)==1 else "NONE"; runner="NONE"
    head_raw={}; head_holm={}; head_results={}
    if len(eligible_ranked)>1:
        best=eligible_ranked[0]; competitors=eligible_ranked[1:]
        for comp in competitors:
            a={r["class"]:r for r in oof if r["family"]==best}; b={r["class"]:r for r in oof if r["family"]==comp}; vals=[a[c]["matched_win"]-b[c]["matched_win"] for c in sorted(a)]; head_raw[f"{best}_vs_{comp}"]=exact_one_sided_sign_flip(vals); head_results[f"{best}_vs_{comp}"]={"best":best,"competitor":comp,"per_class":vals,"raw_p":head_raw[f"{best}_vs_{comp}"]}
        head_holm=holm(head_raw); winner_status="DECISIVE_PROVISIONAL_WINNER" if all(p<.05 for p in head_holm.values()) else "BEST_OBSERVED_NOT_SEPARATED"; winner=best; runner=eligible_ranked[1]
    elif len(eligible_ranked)==1: winner_status="DECISIVE_PROVISIONAL_WINNER"
    else: winner_status="NONE"
    research={f:research_scores(f,standard[f],zero[f]["result"],gates[f],sensitivity_out[f],holm_p[f],True) for f in FAMILIES}; research_rank=sorted(FAMILIES,key=lambda f:(-research[f]["total"],f)); selected_config=None
    if winner!="NONE": selected_config=select_config(configs_by_family[winner],{cid:rows_by_family[winner][cid] for cid in rows_by_family[winner] if cid!="__classes__"},sorted(by_class))["selected_config"]
    output={"schema_version":"P5FR1C_DECISION_V1","P5FR1C_terminal":"P5FR1C_LATE_COMPLETION_RECONCILED","scientific_terminal":"FOUR_FAMILY_METHOD_SUPPORTED" if eligible else "NO_FOUR_FAMILY_METHOD_FULLY_SUPPORTED","fully_eligible_families":eligible,"empirical_scientific_ranking":ranking,"empirical_provisional_winner":winner,"runner_up":runner,"winner_status":winner_status,"candidate":"NONE","final_external_winner":False,"final_mvtec_selected_config":selected_config,"model_forwards":0,"training_steps":0,"medical":False}
    atomic_json(OUTPUT_ROOT/"FOLD_SELECTIONS.json",fold_selections); atomic_json(OUTPUT_ROOT/"CONFIG_METRICS.json",rows_by_family); atomic_json(OUTPUT_ROOT/"ZERO_TUNE_RESULT.json",zero); atomic_json(OUTPUT_ROOT/"SENSITIVITY.json",sensitivity_out); atomic_json(OUTPUT_ROOT/"STANDARD_METRICS.json",standard); atomic_json(OUTPUT_ROOT/"SCIENTIFIC_GATES.json",gates); atomic_json(OUTPUT_ROOT/"MULTIPLICITY_TESTS.json",{"raw_one_sided_p":raw_p,"holm_adjusted_p":holm_p,"alternative":"mean(delta_vs_B1)>0","sign_tolerance":SIGN_TOL}); atomic_json(OUTPUT_ROOT/"HEAD_TO_HEAD.json",{"raw_one_sided_p":head_raw,"holm_adjusted_p":head_holm,"comparisons":head_results,"winner_status":winner_status}); atomic_json(OUTPUT_ROOT/"EMPIRICAL_RANKING.json",{"ranking":ranking,"provisional_winner":winner,"runner_up":runner}); atomic_json(OUTPUT_ROOT/"RESEARCH_VALUE_SCORE.json",{"ranking":research_rank,"highest_research_value_family":research_rank[0],"families":research}); atomic_json(OUTPUT_ROOT/"SELECTED_METHOD.json",output); atomic_json(OUTPUT_ROOT/"DECISION.json",output); atomic_json(OUTPUT_ROOT/"OUTPUT_CHECK.json",{"status":"PENDING_INDEPENDENT_CHECK","model_forwards":0,"training_steps":0,"medical":False})
    write_csv(OUTPUT_ROOT/"OOF_PER_CLASS.csv",oof)
    report=f"# P5-FR1C MVTec four-family evaluation\n\nHistorical terminals remain `P5F_AUDIT_INVALID` and `P5FR1_AUDIT_INVALID`. This is late completion of the already-started P5FR1 process; P5FR1C used zero model forwards and frozen all-config evidence.\n\nScientific terminal: `{output['scientific_terminal']}`. Provisional winner: `{winner}`. Winner status: `{winner_status}`. Final external winner: `false`. Candidate: `NONE`.\n"
    (OUTPUT_ROOT/"REPORT.md").write_text(report)
    return output


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--allow-gt",action="store_true"); args=p.parse_args()
    if not args.allow_gt: raise SystemExit("P5FR1C_GT_BARRIER_INVALID: explicit --allow-gt required")
    start=time.time(); status={"schema_version":"P5FR1C_EVALUATOR_RUN_V1","command":"tools/audit_p5fr1c_mvtec_posthoc.py --allow-gt","start_time":now_iso(),"end_time":None,"elapsed_seconds":None,"exit_code":None,"completion_status":"RUNNING","model_forwards":0,"gt_read":False}
    atomic_json(RUN_STATUS_PATH,status)
    try:
        result=evaluate(); status.update({"end_time":now_iso(),"elapsed_seconds":time.time()-start,"exit_code":0,"completion_status":"PASS","gt_read":True}); atomic_json(RUN_STATUS_PATH,status); print(json.dumps(result,indent=2,sort_keys=True))
    except Exception as exc:
        status.update({"end_time":now_iso(),"elapsed_seconds":time.time()-start,"exit_code":1,"completion_status":"FAIL","exception":repr(exc)}); atomic_json(RUN_STATUS_PATH,status); raise

if __name__=='__main__': main()
