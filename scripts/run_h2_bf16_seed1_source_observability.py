#!/usr/bin/env python3
"""Read-only, VisA-only observability audit for frozen H2 BF16 Seed-1 E15.

The runner deliberately has no training code, accepts no target dataset, and
writes large maps outside the repository.  Compact manifests/tables are the
only intended Git artifacts.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
from torchmetrics.functional import auroc, average_precision

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from dataset import CLASS_NAMES, DOMAINS, get_text_and_image_dataset
from h2_clean.exact_metrics import ExactBinaryAccumulator
from model.adapter import ACDCLIP
from model.clip import create_model
from utils import get_hard_anchor_single_class_text_embedding, get_multiple_adapted_text_embedding

MANIFEST_PATH = REPO / "results/H2_BF16_SEED1_E15_MANIFEST.json"
AUDIT = REPO / "audit"
LARGE = Path("/workspace/h2_bf16_screening/fresh_seed1_h_a_e15/source_observability")
IMG = 518
BOUNDARY_WIDTH = 3  # output pixels; fixed before looking at either arm.
TOP_K_FRACTION = 0.01  # fixed diagnostic definition, never a decision threshold.


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def as_json(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


class Moments:
    """Exact count/mean/std plus a deterministic, fixed-stride quantile sample."""
    def __init__(self):
        self.n = 0; self.sum = 0.0; self.sumsq = 0.0; self.sample = []
    def add(self, x):
        a = np.asarray(x, dtype=np.float32).reshape(-1)
        if not a.size: return
        self.n += int(a.size); self.sum += float(a.sum(dtype=np.float64)); self.sumsq += float(np.square(a, dtype=np.float64).sum())
        # A stable 1/257 subsample, offset by the population position.
        first = (- (self.n - a.size)) % 257
        self.sample.extend(a[first::257].tolist())
    def result(self):
        if not self.n: return {"count": 0}
        s = np.asarray(self.sample, dtype=np.float32)
        q = np.quantile(s, [.05,.25,.5,.75,.95]) if s.size else [np.nan]*5
        mean = self.sum / self.n
        return {"count":self.n,"mean":mean,"std":max(0.0,self.sumsq/self.n-mean*mean)**.5,
                "p05":float(q[0]),"p25":float(q[1]),"median":float(q[2]),"p75":float(q[3]),"p95":float(q[4]),
                "iqr":float(q[3]-q[1]),"quantile_method":"deterministic_stride_257"}


def entropy(w):
    return -(w * w.clamp_min(1e-12).log()).sum(-1)


def raw_pre_smoothing_map(model, vision, text):
    """The existing native fusion path through the point immediately before blur."""
    b, patches, _ = vision.shape[1:]
    side = int(math.sqrt(patches)); groups = []
    for stage in range(model.n_groups):
        fused_text = model._vision_text_attention_fusion(vision[stage], text.permute(1,0,2,3), stage)
        logits = torch.matmul(10 * vision[stage], fused_text).permute(0,2,1).view(b,2,side,side)
        groups.append(F.interpolate(logits, size=IMG, mode="bilinear", align_corners=True))
    return F.softmax(torch.stack(groups).mean(0), dim=1)[:,1]


def native_dfg_components(model, vision, text):
    """Collect the model's native GAP/SS2D softmax components without changing it."""
    group_text = text.permute(1,0,2,3); rows = []
    for stage in range(model.n_groups):
        feat = vision[stage]; vgap = feat.mean(1)
        vss = model.image_adapter["dfg_ss2d_branches"][stage](feat)
        kn = model.image_adapter["vision_text_k"][stage](group_text[...,0]).float()
        ka = model.image_adapter["vision_text_k"][stage](group_text[...,1]).float()
        q = model.image_adapter["vision_text_q"][stage]
        qg, qs = q(vgap).float(), q(vss).float(); scale = math.sqrt(model.dfg_attn_dim)*model.dfg_attn_tau
        gn, ga = F.softmax(torch.einsum("bd,bnd->bn",qg,kn)/scale,1), F.softmax(torch.einsum("bd,bnd->bn",qg,ka)/scale,1)
        sn, sa = F.softmax(torch.einsum("bd,bnd->bn",qs,kn)/scale,1), F.softmax(torch.einsum("bd,bnd->bn",qs,ka)/scale,1)
        fn, fa = (1-model.dfg_beta)*gn+model.dfg_beta*sn, (1-model.dfg_beta)*ga+model.dfg_beta*sa
        rows.append({"stage":stage+1,"gap_normal":gn,"gap_abnormal":ga,"ss2d_normal":sn,"ss2d_abnormal":sa,
                     "final_normal":fn,"final_abnormal":fa,"gap_ss2d_l1":(gn-sn).abs().sum(1),
                     "routing_l1":(fn-fa).abs().sum(1),"entropy_normal":entropy(fn),"entropy_abnormal":entropy(fa)})
    return rows


def load(model, payload):
    model.image_adapter.load_state_dict(payload["image_adapter"], strict=True)
    model.text_adapter.load_state_dict(payload["text_adapter"], strict=True)
    model.soft_prompt.load_state_dict(payload["soft_prompt"], strict=True)
    model.dfg_beta = float(payload["dfg_beta_current"])
    model.prompt_mode = payload.get("prompt_mode", "hybrid")
    model.use_hybrid_soft_prompt = bool(payload.get("use_hybrid_soft_prompt", True))
    model.use_soft_prompt = bool(payload.get("use_soft_prompt", False))
    model.hybrid_alpha_current = float(payload.get("hybrid_alpha_current", .2))


def stat_metrics(scores, labels):
    y = np.asarray(labels, dtype=np.uint8).reshape(-1); x = np.asarray(scores, dtype=np.float32).reshape(-1)
    if y.min() == y.max(): return {"auroc":None,"ap":None}
    xt=torch.from_numpy(x); yt=torch.from_numpy(y)
    return {"auroc":float(auroc(xt,yt,task="binary")), "ap":float(average_precision(xt,yt,task="binary"))}


def map_metrics(scores, mask):
    return stat_metrics(scores, mask)


def morphology_partitions(mask):
    """Fixed 3-output-pixel square-structuring-element morphology."""
    x=torch.from_numpy(mask.astype(np.float32))[None,None]
    kernel=2*BOUNDARY_WIDTH+1
    dilated=F.max_pool2d(x,kernel,stride=1,padding=BOUNDARY_WIDTH).bool()[0,0].numpy()
    eroded=(1-F.max_pool2d(1-x,kernel,stride=1,padding=BOUNDARY_WIDTH)).bool()[0,0].numpy()
    pos=mask.astype(bool); neg=~pos
    return pos & ~eroded, eroded, dilated & neg


def state_drift(current, reference, prefix):
    rows=[]
    for name, value in current.items():
        if name not in reference or not torch.is_tensor(value): continue
        a=value.float(); b=reference[name].float(); d=(a-b).norm().item(); r=b.norm().item()
        family=name.split(".")[0]
        if "lora_adapters" in name: family="conv_lora"
        rows.append({"kind":prefix,"name":name,"family":family,"l2_drift":d,"reference_norm":r,
                     "normalized_drift":d/max(r,1e-12),"final_norm":a.norm().item()})
    return rows


def parse_anchor_telemetry(log_path):
    rows=[]
    pattern=re.compile(r"anchor_family_step epoch=(\d+) batch=(\d+) metrics=(\{.*\})$")
    for line in log_path.read_text(errors="replace").splitlines():
        m=pattern.search(line)
        if not m: continue
        data=json.loads(m.group(3))
        for family, values in data["families"].items():
            rows.append({"epoch":int(m.group(1)),"batch":int(m.group(2)),"family":family, **values})
    return rows


def verify_and_freeze_source_manifest(manifest):
    out={"audit_git_branch":subprocess.check_output(["git","branch","--show-current"],cwd=REPO,text=True).strip(),
         "audit_git_sha":subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip(),
         "working_tree_clean":not subprocess.check_output(["git","status","--porcelain"],cwd=REPO,text=True).strip(),
         "protocol_id":manifest["protocol_id"],"checks":{}}
    if out["audit_git_branch"] != "research/h2-bf16-screening-seed1" or not out["working_tree_clean"]:
        raise RuntimeError("Phase 0 branch/clean-tree invariant failed")
    for key in ("H_E15","A_E15","shared_e1"):
        spec=manifest["checkpoints"][key]; p=Path(spec["path"]); got=sha(p)
        if got != spec["sha256"]: raise RuntimeError(f"checkpoint sha mismatch: {key}")
        payload=torch.load(p,map_location="cpu",weights_only=False)
        fields={k:payload.get(k) for k in ("precision","amp_enabled","gradscaler_enabled","tf32_enabled","clip_sha256","config_sha256","dataset_manifest_sha256")}
        expected={k:spec[k] for k in fields}
        if fields != expected or payload.get("scaler_state") != {}: raise RuntimeError(f"checkpoint metadata mismatch: {key}")
        out["checks"][key]={"path":str(p),"sha256":got,**fields,"scaler_state_empty":True}
    visa=REPO/"dataset/hub/VisA.jsonl"; expected=manifest["checkpoints"]["H_E15"]["dataset_manifest_sha256"]
    if sha(visa)!=expected: raise RuntimeError("VisA manifest SHA mismatch")
    out["visa_manifest_sha256"]=expected
    metas=[json.loads(x) for x in visa.read_text().splitlines()]
    # get_text_and_image_dataset("VisA", ..., "test") enumerates exactly this
    # category-major order, with JSONL order retained inside every category.
    metas=[m for category in CLASS_NAMES["VisA"] for m in metas if m["class_name"] == category]
    records=[]
    for i,m in enumerate(metas):
        records.append({"index":i,"image_id":m["image_path"],"category":m["class_name"],"label":int(m["label"]),
                        "mask_available":bool(m["label"]),"mask_path":m.get("mask_path"),"valid_pixels":IMG*IMG})
    out["source_split"]={"dataset":"VisA","stage":"test","count":len(records),"order":"dataset/hub/VisA.jsonl file order","records":records,
                         "boundary_definition":f"mask pixels with Euclidean distance to background <= {BOUNDARY_WIDTH} resized output pixels"}
    return out, metas


def main():
    AUDIT.mkdir(exist_ok=True); LARGE.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(MANIFEST_PATH.read_text()); frozen, metas=verify_and_freeze_source_manifest(manifest)
    # Save before loading/evaluating either arm; this is the immutable diagnostic contract.
    (AUDIT/"H2_BF16_SEED1_SOURCE_DIAGNOSTIC_MANIFEST.json").write_text(json.dumps(frozen,indent=2,default=as_json)+"\n")
    payloads={k:torch.load(v["path"],map_location="cpu",weights_only=False) for k,v in manifest["checkpoints"].items()}
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("Frozen audit requires the project CUDA evaluator")
    clip=create_model("ViT-L-14-336",img_size=IMG,device=device,pretrained="openai",require_pretrained=True); clip.eval()
    model=ACDCLIP(clip_model=clip,n_groups=3,image_adapt_weight=.2,text_adapt_weight=.2,conv_lora_rank=8,conv_lora_alpha=2.,conv_kernel_size_list=[3,5],lora_rank=16,lora_alpha=2.,dfg_mode="attn",dfg_attn_dim=256,dfg_attn_tau=8.,use_ss2d_dfg=True,dfg_gamma_max=.2,dfg_ss2d_fusion="weight_residual",dfg_beta=.1,dfg_beta_schedule="warmup010",dfg_beta_target=.1,dfg_beta_current=.1,dfg_weight_residual_fp32=True,use_soft_prompt=False,soft_prompt_ctx_len=4,soft_prompt_init="phrase",soft_prompt_init_phrase="a photo of a").to(device).eval()
    datasets=get_text_and_image_dataset("VisA",IMG,"test")
    order=[]
    for category in CLASS_NAMES["VisA"]:
        for local, m in enumerate(datasets[category].meta): order.append((category,local,m))
    # preserve repository evaluator ordering (category order, then JSONL order inside category)
    n=len(order); shape=(n,IMG,IMG)
    arrays={}
    for arm in ("H","A"):
        arrays[arm]={k:np.lib.format.open_memmap(LARGE/f"{arm}_{k}.npy",mode="w+",dtype=d,shape=s)
                     for k,d,s in (("raw_maps","float32",shape),("final_maps","float32",shape),("masks","uint8",shape),("image_scores","float32",(n,)),("labels","uint8",(n,)))}
    rows=[]; dfg_rows=[]; feature_rows=[]; proto_rows=[]
    moments={arm:defaultdict(Moments) for arm in ("H","A")}
    global_acc={arm:ExactBinaryAccumulator(LARGE/f"{arm}_global_metric_spool") for arm in ("H","A")}
    cat_acc={arm:{c:ExactBinaryAccumulator(LARGE/f"{arm}_{c}_metric_spool") for c in CLASS_NAMES["VisA"]} for arm in ("H","A")}
    area_acc={arm:{s:ExactBinaryAccumulator(LARGE/f"{arm}_area_{s}_metric_spool") for s in ("small","medium","large")} for arm in ("H","A")}
    # Boundaries are frozen solely from source GT masks before any arm is used.
    areas=[]
    for _,_,m in order:
        if int(m["label"]):
            mask=np.asarray(Image.open(Path(datasets[m["class_name"]].data_path)/m["mask_path"]).convert("L").resize((IMG,IMG),resample=Image.Resampling.NEAREST))!=0
            areas.append(mask.mean())
    q=np.quantile(areas,[1/3,2/3]); frozen["source_split"]["area_boundaries"]={"small_le":float(q[0]),"medium_le":float(q[1]),"large_gt":float(q[1])}
    index=0
    for category in CLASS_NAMES["VisA"]:
        loader=DataLoader(datasets[category],batch_size=6,shuffle=False,num_workers=2,pin_memory=True)
        for batch in loader:
            image=batch["image"].to(device,non_blocking=True); mask=batch["mask"].to(device); label=batch["label"].to(device)
            states={}; text_cache={}
            for arm,key in (("H","H_E15"),("A","A_E15"),("E1","shared_e1")):
                load(model,payloads[key])
                with torch.no_grad():
                    text_cache[arm]=get_multiple_adapted_text_embedding(model,"VisA",device)[category]
                    seg,det=model(image); vision=torch.stack(seg); detect=torch.stack(det)
                    raw=raw_pre_smoothing_map(model,vision,text_cache[arm])
                    final=model.vision_text_fusion_gate_seg(vision,text_cache[arm].unsqueeze(1).repeat(1,image.shape[0],1,1),test_mode=True,domain=DOMAINS["VisA"])
                    cls=torch.stack([torch.matmul(detect[i].unsqueeze(1),text_cache[arm][i].unsqueeze(0).repeat(image.shape[0],1,1)).squeeze(1) for i in range(3)]).mean(0)
                    image_score=.9*F.softmax(cls,1)[:,1]+.1*final.flatten(1).max(1).values
                    states[arm]={"vision":vision.detach(),"det":detect.detach(),"raw":raw.detach(),"final":final.detach(),"image":image_score.detach(),"dfg":native_dfg_components(model,vision,text_cache[arm].unsqueeze(1).repeat(1,image.shape[0],1,1))}
                    if arm != "E1":
                        for stage,d in enumerate(states[arm]["dfg"],1):
                            for component in ("gap_normal","gap_abnormal","ss2d_normal","ss2d_abnormal","final_normal","final_abnormal"):
                                v=d[component].detach().cpu().numpy()
                                dfg_rows.append({"arm":arm,"category":category,"stage":stage,"component":component,"mean":float(v.mean()),"std":float(v.std()),"min":float(v.min()),"max":float(v.max()),"count":int(v.size)})
                            for field in ("entropy_normal","entropy_abnormal","routing_l1","gap_ss2d_l1"):
                                v=d[field].detach().cpu().numpy(); dfg_rows.append({"arm":arm,"category":category,"stage":stage,"component":field,"mean":float(v.mean()),"std":float(v.std()),"min":float(v.min()),"max":float(v.max()),"count":int(v.size)})
                torch.cuda.empty_cache()
            mask_np=mask[:,0].cpu().numpy().astype(np.uint8); labels=label.cpu().numpy().astype(np.uint8)
            for bi in range(len(labels)):
                pos=mask_np[bi].astype(bool); neg=~pos
                # deterministic boundary/interior/background partitions.
                boundary,interior,ring=morphology_partitions(pos)
                area=float(pos.mean()) if labels[bi] else 0.0
                stratum=None if not labels[bi] else ("small" if area<=q[0] else "medium" if area<=q[1] else "large")
                for arm in ("H","A"):
                    final=states[arm]["final"][bi].float().cpu().numpy(); raw=states[arm]["raw"][bi].float().cpu().numpy()
                    arrays[arm]["raw_maps"][index]=raw; arrays[arm]["final_maps"][index]=final; arrays[arm]["masks"][index]=mask_np[bi]; arrays[arm]["image_scores"][index]=float(states[arm]["image"][bi]); arrays[arm]["labels"][index]=labels[bi]
                    global_acc[arm].update(torch.from_numpy(final),torch.from_numpy(mask_np[bi])); cat_acc[arm][category].update(torch.from_numpy(final),torch.from_numpy(mask_np[bi]))
                    if stratum: area_acc[arm][stratum].update(torch.from_numpy(final),torch.from_numpy(mask_np[bi]))
                    moments[arm]["pixel_positive"].add(final[pos]); moments[arm]["pixel_negative"].add(final[neg]); moments[arm]["image_anomalous" if labels[bi] else "image_normal"].add([float(states[arm]["image"][bi])])
                    moments[arm]["boundary"].add(final[boundary]); moments[arm]["interior"].add(final[interior]); moments[arm]["background"].add(final[neg]); moments[arm]["near_boundary_background"].add(final[ring]); moments[arm]["far_background"].add(final[neg&~ring])
                    im=map_metrics(final,pos) if labels[bi] and pos.any() and neg.any() else {"auroc":None,"ap":None}
                    rows.append({"arm":arm,"index":index,"image_id":batch["file_name"][bi],"category":category,"label":int(labels[bi]),"anomaly_pixels":int(pos.sum()),"area_ratio":area,"stratum":stratum,"pixel_auroc":im["auroc"],"pixel_ap":im["ap"],"image_score":float(states[arm]["image"][bi]),"top_1pct_normal_quantile":float(np.quantile(final[neg],1-TOP_K_FRACTION))})
                # paired feature distances at each frozen stage, including E1.
                patchmask=F.interpolate(mask[bi:bi+1].float(),size=(37,37),mode="nearest")[0,0].bool()
                for stage in range(3):
                    h,a,e=(states[x]["vision"][stage,bi] for x in ("H","A","E1"))
                    for name,x,y in (("H_A",h,a),("H_E1",h,e),("A_E1",a,e)):
                        cos=(1-F.cosine_similarity(x.float(),y.float(),dim=-1)); l2=(x.float()-y.float()).norm(dim=-1)
                        for region,sel in (("all",torch.ones_like(patchmask)),("anomalous_pixel",patchmask),("normal_pixel",~patchmask)):
                            if sel.any(): feature_rows.append({"comparison":name,"stage":stage+1,"region":region,"category":category,"cosine_distance":float(cos[sel].mean()),"l2_distance":float(l2[sel].mean())})
                index+=1
            # prototype and feature-to-prototype distributions, aggregated per batch/state.
            for arm in ("H","A","E1"):
                t=text_cache[arm]; hard=get_hard_anchor_single_class_text_embedding(model,"VisA",category,device) if arm=="E1" else None
                for stage in range(3):
                    pn,pa=t[stage,:,0],t[stage,:,1]
                    proto_rows.append({"arm":arm,"category":category,"stage":stage+1,"normal_abnormal_cos":float(F.cosine_similarity(pn,pa,dim=0)),"normal_norm":float(pn.norm()),"abnormal_norm":float(pa.norm()),"hard_reference_available":hard is not None})
    if index!=n: raise RuntimeError(f"source ordering mismatch {index}!={n}")
    for group in arrays.values():
        for a in group.values(): a.flush()
    frozen["source_split"]["records"]=[{**r,"anomaly_pixel_count":int(arrays["H"]["masks"][i].sum()),"anomaly_area_ratio":float(arrays["H"]["masks"][i].mean())} for i,r in enumerate(frozen["source_split"]["records"])]
    (AUDIT/"H2_BF16_SEED1_SOURCE_DIAGNOSTIC_MANIFEST.json").write_text(json.dumps(frozen,indent=2,default=as_json)+"\n")
    result={"frozen_state":frozen,"raw_artifact_root":str(LARGE),"maps":{"dtype":"float32","raw":"pre-smoothing, pre-group-blur after native interpolation","final":"native evaluator post-processing","shape":[n,IMG,IMG]},"arms":{},"dfg_rows":dfg_rows,"feature_rows":feature_rows,"prototype_rows":proto_rows,"per_image":rows}
    for arm in ("H","A"):
        auc,ap=global_acc[arm].compute(); global_acc[arm].cleanup()
        cats={}; strata={}
        for c,a in cat_acc[arm].items():
            x,y=a.compute(); a.cleanup(); cats[c]={"pixel_auroc":x,"pixel_ap":y}
        for s,a in area_acc[arm].items():
            x,y=a.compute(); a.cleanup(); strata[s]={"pixel_auroc":x,"pixel_ap":y}
        image_scores=arrays[arm]["image_scores"]; image_labels=arrays[arm]["labels"]
        result["arms"][arm]={"pixel_global":{"auroc":auc,"ap":ap},"image_global":stat_metrics(image_scores,image_labels),"distributions":{k:v.result() for k,v in moments[arm].items()},"per_category":cats,"area":strata}
    # Parameter drift uses the shared E1 checkpoint, never a reconstructed reference.
    for arm,key in (("H","H_E15"),("A","A_E15")):
        result.setdefault("parameter_drift",[]).extend(state_drift(payloads[key]["image_adapter"],payloads["shared_e1"]["image_adapter"],arm+"_image_adapter"))
        result["parameter_drift"].extend(state_drift(payloads[key]["text_adapter"],payloads["shared_e1"]["text_adapter"],arm+"_text_adapter"))
    telemetry=parse_anchor_telemetry(Path(manifest["checkpoints"]["A_E15"]["path"]).parent/"train.log")
    result["anchor_family_telemetry"]={"available":bool(telemetry),"samples":len(telemetry),"rows":telemetry}
    # Compact CSV artifacts.
    def write_csv(name, data):
        p=AUDIT/name; keys=sorted({k for r in data for k in r}) if data else ["status"]
        with p.open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(data or [{"status":"UNKNOWN"}])
    score=[]
    for arm in ("H","A"):
        for cohort,vals in result["arms"][arm]["distributions"].items(): score.append({"arm":arm,"cohort":cohort,**vals})
    write_csv("H2_BF16_SEED1_SOURCE_SCORE_DISTRIBUTIONS.csv",score)
    write_csv("H2_BF16_SEED1_SOURCE_DFG.csv",dfg_rows)
    write_csv("H2_BF16_SEED1_SOURCE_FEATURE_DRIFT.csv",result["parameter_drift"]+feature_rows+proto_rows)
    area_rows=[]
    for arm in ("H","A"):
        for s,v in result["arms"][arm]["area"].items(): area_rows.append({"arm":arm,"kind":"area","stratum":s,**v})
        for cohort in ("boundary","interior","background","near_boundary_background","far_background"):
            area_rows.append({"arm":arm,"kind":"boundary","stratum":cohort,**result["arms"][arm]["distributions"][cohort]})
    write_csv("H2_BF16_SEED1_SOURCE_AREA_BOUNDARY.csv",area_rows)
    (AUDIT/"H2_BF16_SEED1_SOURCE_OBSERVABILITY.json").write_text(json.dumps(result,indent=2,default=as_json)+"\n")
    h,a=result["arms"]["H"],result["arms"]["A"]
    md=["# H2 BF16 Seed1 source-only observability audit","", "## Scope and frozen state","", "VisA test split only; no training, target evaluation, or configuration changes were performed.","",f"- H pixel AUROC/AP: {h['pixel_global']['auroc']:.8f} / {h['pixel_global']['ap']:.8f}",f"- A pixel AUROC/AP: {a['pixel_global']['auroc']:.8f} / {a['pixel_global']['ap']:.8f}",f"- H image AUROC/AP: {h['image_global']['auroc']:.8f} / {h['image_global']['ap']:.8f}",f"- A image AUROC/AP: {a['image_global']['auroc']:.8f} / {a['image_global']['ap']:.8f}","", "Raw float32 pre-smoothing and final maps are retained under the external artifact root recorded in JSON and are intentionally excluded from Git. Quantile estimates use the documented fixed stride-257 source-only sample; means/std and AUROC/AP are full-resolution.","", "## Interpretation","", "This audit establishes source behavior of the two BF16 arms. It cannot by itself establish BF16 degradation relative to a matched valid non-BF16 training protocol, and it cannot establish target generalization because target data were prohibited.",""]
    (AUDIT/"H2_BF16_SEED1_SOURCE_OBSERVABILITY.md").write_text("\n".join(md))
    print(json.dumps({"status":"PASS","H":h["pixel_global"],"A":a["pixel_global"],"artifact_root":str(LARGE)}))

if __name__ == "__main__": main()
