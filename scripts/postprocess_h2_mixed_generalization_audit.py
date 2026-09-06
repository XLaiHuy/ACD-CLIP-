#!/usr/bin/env python3
"""Derive compact source curves, morphology, geometry, and optimizer summaries."""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy import stats

REPO=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(REPO))
from dataset import CLASS_NAMES
from h2_clean.precision import PrecisionPolicy
from model.adapter import ACDCLIP
from model.clip import create_model
from utils import get_multiple_adapted_text_embedding
from scripts.run_h2_mixed_generalization_audit import IMG, ARMS, MAP_ARMS, disable_later_islands, load_arm, parameter_family

AUDIT=REPO/"audit"; LARGE=Path("/workspace/h2_mixed_generalization_audit_v1"); BINS=65536


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def hist_curve(scores,masks,indices=None,bins=BINS):
    pos=np.zeros(bins,np.int64); neg=np.zeros(bins,np.int64)
    use=range(len(scores)) if indices is None else indices
    for i in use:
        x=np.asarray(scores[i],np.float32).reshape(-1); y=np.asarray(masks[i],np.uint8).reshape(-1)
        b=np.minimum((x*(bins-1)).astype(np.int64),bins-1)
        pos += np.bincount(b[y==1],minlength=bins); neg += np.bincount(b[y==0],minlength=bins)
    tp=np.cumsum(pos[::-1]); fp=np.cumsum(neg[::-1]); P=max(int(pos.sum()),1); N=max(int(neg.sum()),1)
    recall=tp/P; precision=tp/np.maximum(tp+fp,1); tpr=recall; fpr=fp/N
    ap=float(np.sum((tp-np.r_[0,tp[:-1]])/P*precision)); auc=float(np.trapezoid(np.r_[0,tpr],np.r_[0,fpr]))
    threshold=np.arange(bins-1,-1,-1)/(bins-1)
    return {"auroc":auc,"ap":ap,"threshold":threshold,"tpr":tpr,"fpr":fpr,"precision":precision,"recall":recall,
            "positive_pixels":int(pos.sum()),"negative_pixels":int(neg.sum()),"bins":bins}


def dispersion(x):
    x=x[np.isfinite(x).all(1)]; mu=x.mean(0)
    return float(np.sqrt(np.mean(np.sum((x-mu)**2,axis=1)))),mu


def optimizer_family_rows(payloads):
    rows=[]
    for arm in ARMS:
        p=payloads[arm]; opt=p["optimizer_state"]
        by_group={g["name"]:g for g in opt["param_groups"]}
        for module_key,group_name in (("text_adapter","text_adapter"),("image_adapter","image_adapter"),("soft_prompt","soft_prompt")):
            names=list(p[module_key]); ids=by_group[group_name]["params"]
            if len(names)!=len(ids): raise RuntimeError(f"optimizer mapping mismatch {arm} {module_key}")
            groups=defaultdict(list)
            for name,pid in zip(names,ids):
                fam=parameter_family(name) if module_key=="image_adapter" else module_key
                groups[fam].append((name,p[module_key][name].float(),opt["state"].get(pid,{})))
            for fam,items in groups.items():
                w2=m2=v2=u2=0.; count=0; active=0
                lr=float(by_group[group_name]["lr"]); eps=float(by_group[group_name].get("eps",1e-8))
                for _,w,s in items:
                    w2+=float(w.square().sum()); count+=w.numel()
                    if "exp_avg" in s:
                        m=s["exp_avg"].float(); v=s["exp_avg_sq"].float(); upd=lr*m/(v.sqrt()+eps)
                        m2+=float(m.square().sum()); v2+=float(v.square().sum()); u2+=float(upd.square().sum()); active+=1
                rows.append({"arm":arm,"family":fam,"parameter_count":count,"parameter_norm":w2**.5,"adam_m_norm":m2**.5,
                             "adam_v_norm":v2**.5,"effective_step_norm":u2**.5,"update_to_weight_ratio":u2**.5/max(w2**.5,1e-12),
                             "lr":lr,"state_tensors":active,"epoch":p["epoch"],"global_step":p["global_step"]})
    return rows


def parameter_geometry(payloads):
    rows=[]
    for arm in MAP_ARMS:
        for module_key in ("image_adapter","text_adapter","soft_prompt"):
            groups=defaultdict(list)
            for name,x in payloads[arm][module_key].items():
                fam=parameter_family(name) if module_key=="image_adapter" else module_key
                groups[fam].append((x.float(),payloads["E1"][module_key][name].float()))
            for fam,items in groups.items():
                cur=torch.cat([x.reshape(-1) for x,_ in items]); ref=torch.cat([r.reshape(-1) for _,r in items]); delta=cur-ref
                rows.append({"arm":arm,"family":fam,"parameter_count":cur.numel(),"parameter_norm":float(cur.norm()),
                             "l2_drift":float(delta.norm()),"relative_drift":float(delta.norm()/ref.norm().clamp_min(1e-12)),
                             "parameter_cosine_to_E1":float(F.cosine_similarity(cur,ref,dim=0)),
                             "drift_energy_share_pending":float(delta.square().sum())})
        total=sum(r["drift_energy_share_pending"] for r in rows if r["arm"]==arm)
        for r in rows:
            if r["arm"]==arm: r["drift_energy_share"]=r.pop("drift_energy_share_pending")/max(total,1e-12)
    return rows


def semantic_geometry(payloads,index_df):
    device=torch.device("cuda:0")
    clip=create_model("ViT-L-14-336",img_size=IMG,device=device,pretrained="openai",require_pretrained=True).eval()
    model=ACDCLIP(clip_model=clip,n_groups=3,image_adapt_weight=.2,text_adapt_weight=.2,conv_lora_rank=8,conv_lora_alpha=2.,
        conv_kernel_size_list=[3,5],lora_rank=16,lora_alpha=2.,dfg_mode="attn",dfg_attn_dim=256,dfg_attn_tau=8.,use_ss2d_dfg=True,
        dfg_gamma_max=.2,dfg_ss2d_fusion="weight_residual",dfg_beta=.1,dfg_beta_schedule="warmup010",dfg_beta_target=.1,
        dfg_beta_current=.1,dfg_weight_residual_fp32=True,use_soft_prompt=False,soft_prompt_ctx_len=4,
        soft_prompt_init="phrase",soft_prompt_init_phrase="a photo of a").to(device).eval(); disable_later_islands(model)
    rows=[]
    policy=PrecisionPolicy("fp16")
    categories=index_df.sort_values("index")["category"].to_numpy()
    for arm in ARMS:
        load_arm(model,payloads[arm]); feats=np.load(LARGE/f"{arm}_feature_means.npy",mmap_mode="r")
        text={}
        with torch.no_grad(), policy.autocast(device):
            for cat in CLASS_NAMES["VisA"]: text[cat]=get_multiple_adapted_text_embedding(model,"VisA",device)[cat].float().cpu().numpy()
        for stage in range(3):
            margins={"normal_patch":[],"anomalous_patch":[]}
            for i,cat in enumerate(categories):
                for ri,region in ((1,"normal_patch"),(2,"anomalous_patch")):
                    x=np.asarray(feats[i,stage,ri],np.float32)
                    if not np.isfinite(x).all(): continue
                    t=text[cat][stage]; xn=x/(np.linalg.norm(x)+1e-12)
                    margins[region].append(float(xn@(t[:,1]/np.linalg.norm(t[:,1]))-xn@(t[:,0]/np.linalg.norm(t[:,0]))))
            normal=np.asarray(feats[:,stage,1],np.float32); abnormal=np.asarray(feats[:,stage,2],np.float32)
            wn,nmu=dispersion(normal); wa,amu=dispersion(abnormal)
            rows.append({"arm":arm,"stage":stage+1,"normal_prototype_margin_mean":float(np.mean(margins["normal_patch"])),
                         "anomaly_prototype_margin_mean":float(np.mean(margins["anomalous_patch"])),
                         "prototype_margin_separation":float(np.mean(margins["anomalous_patch"])-np.mean(margins["normal_patch"])),
                         "within_normal_dispersion":wn,"within_anomaly_dispersion":wa,"between_region_centroid_l2":float(np.linalg.norm(amu-nmu)),
                         "between_region_centroid_cosine":float(np.dot(amu,nmu)/(np.linalg.norm(amu)*np.linalg.norm(nmu)+1e-12))})
    return rows


def routing_derived(image_df):
    rdf=pd.read_csv(AUDIT/"H2_MIXED_DFG_ROUTING.csv")
    base=rdf[rdf["index"].notna()].copy(); base["index"]=base["index"].astype(int)
    normal=base[base.branch=="normal"].set_index(["arm","index","stage"])
    abnormal=base[base.branch=="abnormal"].set_index(["arm","index","stage"])
    rows=[]
    for key in normal.index.intersection(abnormal.index):
        n=normal.loc[key]; a=abnormal.loc[key]
        wn=np.asarray([n[f"weight_final_g{i}"] for i in range(1,4)],float); wa=np.asarray([a[f"weight_final_g{i}"] for i in range(1,4)],float)
        m=.5*(wn+wa); js=.5*np.sum(wn*np.log(np.maximum(wn,1e-12)/np.maximum(m,1e-12)))+.5*np.sum(wa*np.log(np.maximum(wa,1e-12)/np.maximum(m,1e-12)))
        row={"record_type":"paired_branch","arm":key[0],"index":key[1],"stage":key[2],"category":a["category"],"label":int(a["label"]),
             "area_ratio":float(a["area_ratio"]),"normal_abnormal_routing_l1":float(np.abs(wn-wa).sum()),"normal_abnormal_routing_js":float(js),
             "gap_branch_score_margin":float(np.mean([a[f"score_gap_g{i}"]-n[f"score_gap_g{i}"] for i in range(1,4)])),
             "ss2d_branch_score_margin":float(np.mean([a[f"score_ss2d_g{i}"]-n[f"score_ss2d_g{i}"] for i in range(1,4)])),
             "gap_ss2d_disagreement":float(.5*(a["gap_ss2d_l1"]+n["gap_ss2d_l1"])),
             "routing_entropy":float(.5*(a["entropy_final"]+n["entropy_final"]))}
        im=image_df[(image_df.arm==key[0])&(image_df["index"]==key[1])]
        row["pixel_ap"]=None if im.empty else float(im.iloc[0].pixel_ap) if pd.notna(im.iloc[0].pixel_ap) else None
        rows.append(row)
    assoc=[]
    for arm in MAP_ARMS:
        for stage in range(1,4):
            sub=[r for r in rows if r["arm"]==arm and r["stage"]==stage and r["label"]==1]
            for response in ("gap_branch_score_margin","ss2d_branch_score_margin","routing_entropy","gap_ss2d_disagreement","pixel_ap"):
                pairs=[(r["area_ratio"],r[response]) for r in sub if r[response] is not None]
                x=np.asarray([p[0] for p in pairs]); y=np.asarray([p[1] for p in pairs])
                assoc.append({"record_type":"gap_dilution_association","arm":arm,"stage":stage,"covariate":"area_ratio","response":response,"n":len(x),
                              "pearson_r":float(stats.pearsonr(x,y).statistic),"spearman_rho":float(stats.spearmanr(x,y).statistic)})
    all_rows=rdf.to_dict("records")+rows+assoc
    write_csv(AUDIT/"H2_MIXED_DFG_ROUTING.csv",all_rows)
    return {"paired_rows":rows,"associations":assoc}


def main():
    protocol=json.loads((AUDIT/"H2_MIXED_DIAGNOSTIC_PROTOCOL.json").read_text())
    paths={"E1":"E1","H":"H_E15","A":"A_E15"}; payloads={a:torch.load(REPO/protocol["frozen_checkpoints"][k]["path"],map_location="cpu",weights_only=False) for a,k in paths.items()}
    df=pd.read_csv(AUDIT/"H2_MIXED_SOURCE_PIXEL_RANKING.csv")
    images=df[df["index"].notna()].copy(); images["index"]=images["index"].astype(int)
    index_df=images[images.arm=="A"].sort_values("index").drop_duplicates("index")
    masks=np.load(LARGE/"masks.npy",mmap_mode="r")
    derived=[]; curve_manifest={}
    for arm in MAP_ARMS:
        for kind in ("raw","final"):
            scores=np.load(LARGE/f"{arm}_{kind}.npy",mmap_mode="r"); c=hist_curve(scores,masks)
            curve_manifest[f"{arm}_{kind}"]={k:v for k,v in c.items() if not isinstance(v,np.ndarray)}
            np.savez_compressed(LARGE/f"{arm}_{kind}_roc_pr_curve_65536.npz",threshold=c["threshold"],tpr=c["tpr"],fpr=c["fpr"],precision=c["precision"],recall=c["recall"])
            pick=np.linspace(0,BINS-1,129).round().astype(int)
            for j in pick: derived.append({"record_type":"curve","arm":arm,"map":kind,"threshold":float(c["threshold"][j]),
                "tpr":float(c["tpr"][j]),"fpr":float(c["fpr"][j]),"precision":float(c["precision"][j]),"recall":float(c["recall"][j])})
        sub=images[(images.arm==arm)&(images.label==1)]
        score=np.load(LARGE/f"{arm}_final.npy",mmap_mode="r")
        for stratum in ("small","medium","large"):
            s=sub[sub.size_stratum==stratum]; margins=[]; tail_margins=[]; compact=[]; comp_areas=[]
            for i in s["index"].astype(int):
                m=np.asarray(masks[i],bool); x=np.asarray(score[i],np.float32); boundary=m&~ndimage.binary_erosion(m,structure=np.ones((7,7),bool))
                margins.append(float(x[m].mean()-x[~m].mean())); tail_margins.append(float(np.quantile(x[m],.01)-np.quantile(x[~m],.99)))
                perimeter=max(int(boundary.sum()),1); compact.append(float(4*math.pi*m.sum()/perimeter**2))
                lab,n=ndimage.label(m,structure=np.ones((3,3),int)); comp_areas.extend(np.bincount(lab.reshape(-1))[1:].tolist())
            derived.append({"record_type":"size_summary","arm":arm,"size_stratum":stratum,"image_count":len(s),
                "macro_image_pixel_auroc":float(s.pixel_auroc.mean()),"macro_image_pixel_ap":float(s.pixel_ap.mean()),
                "positive_negative_mean_margin":float(np.mean(margins)),"p01_positive_minus_p99_negative":float(np.mean(tail_margins)),
                "component_count":len(comp_areas),"component_area_mean_pixels":float(np.mean(comp_areas)),
                "compactness_mean":float(np.mean(compact)),"boundary_area_ratio_mean":float(s.boundary_area_ratio.mean())})
    routing=routing_derived(images); opt=optimizer_family_rows(payloads); param=parameter_geometry(payloads); semantic=semantic_geometry(payloads,index_df)
    write_csv(AUDIT/"H2_MIXED_SOURCE_PIXEL_RANKING_DERIVED.csv",derived)
    write_csv(AUDIT/"H2_MIXED_ADAPTER_UTILIZATION.csv",param+opt)
    write_csv(AUDIT/"H2_MIXED_SEMANTIC_GEOMETRY.csv",semantic)
    out={"curve_manifest":curve_manifest,"size_and_curve_rows":derived,"routing_derived":routing,"optimizer_family":opt,"parameter_geometry":param,"semantic_geometry":semantic}
    (AUDIT/"H2_MIXED_DERIVED_DIAGNOSTICS.json").write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({"status":"PASS","derived_rows":len(derived),"external_curve_root":str(LARGE)}))

if __name__=="__main__": main()
