#!/usr/bin/env python3
"""Fixed source perturbation and local-sensitivity probes for frozen A-E15.

No gradients, optimizer, target data, or parameter updates are used.  The
selection and perturbation constants are frozen in H2_MIXED_DIAGNOSTIC_PROTOCOL.
"""
from __future__ import annotations

import csv
import io
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import stats
from torchvision.transforms.functional import gaussian_blur

REPO=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(REPO))
from dataset import CLASS_NAMES, get_text_and_image_dataset
from h2_clean.precision import PrecisionPolicy
from model.adapter import ACDCLIP
from model.clip import create_model
from utils import get_multiple_adapted_text_embedding
from scripts.run_h2_mixed_generalization_audit import (
    IMG, PATCH, binary_metrics, disable_later_islands, load_arm, maps_from_features, routing_components,
)

AUDIT=REPO/"audit"; LARGE=Path("/workspace/h2_mixed_generalization_audit_v1")
MEAN=torch.tensor([.48145466,.4578275,.40821073])[:,None,None]
STD=torch.tensor([.26862954,.26130258,.27577711])[:,None,None]


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def to_unit(x): return (x.cpu()*STD+MEAN).clamp(0,1)
def normalize(x): return ((x-MEAN)/STD)


def perturb(x,name,global_indices):
    u=to_unit(x)
    if name=="brightness": u=(u*1.1).clamp(0,1)
    elif name=="contrast":
        mu=u.mean((-2,-1),keepdim=True); u=((u-mu)*1.1+mu).clamp(0,1)
    elif name=="gamma": u=u.clamp_min(1e-8).pow(1.1)
    elif name=="saturation":
        gray=(u[:,0:1]*.2989+u[:,1:2]*.5870+u[:,2:3]*.1140); u=((u-gray)*1.1+gray).clamp(0,1)
    elif name=="gaussian_blur": u=gaussian_blur(u,[5,5],[1.,1.])
    elif name=="gaussian_noise":
        out=[]
        for sample,gi in zip(u,global_indices):
            gen=torch.Generator().manual_seed(91001+int(gi)); out.append((sample+torch.randn(sample.shape,generator=gen)*.02).clamp(0,1))
        u=torch.stack(out)
    elif name=="jpeg":
        out=[]
        for sample in u:
            arr=(sample.permute(1,2,0).numpy()*255).round().astype(np.uint8); buf=io.BytesIO()
            Image.fromarray(arr).save(buf,format="JPEG",quality=80); buf.seek(0)
            out.append(torch.from_numpy(np.asarray(Image.open(buf).convert("RGB"),dtype=np.float32).copy()).permute(2,0,1)/255.)
        u=torch.stack(out)
    elif name=="resize_cycle":
        u=F.interpolate(F.interpolate(u,size=(466,466),mode="bilinear",align_corners=False),size=(IMG,IMG),mode="bilinear",align_corners=False)
    else: raise ValueError(name)
    return normalize(u)


def metric(x,y):
    x=np.asarray(x,float).reshape(-1); y=np.asarray(y,float).reshape(-1)
    return {"l1":float(np.abs(x-y).mean()),"rmse":float(np.sqrt(np.mean((x-y)**2))),
            "pearson":float(stats.pearsonr(x,y).statistic),"spearman":float(stats.spearmanr(x,y).statistic),
            "top1_overlap":float(len(set(np.argpartition(x,-max(1,len(x)//100))[-max(1,len(x)//100):]) & set(np.argpartition(y,-max(1,len(y)//100))[-max(1,len(y)//100):]))/max(1,len(x)//100))}


def pooled_metrics(x,y):
    from torchmetrics.functional import auroc,average_precision
    x=torch.as_tensor(np.asarray(x),dtype=torch.float32).flatten(); y=torch.as_tensor(np.asarray(y),dtype=torch.uint8).flatten()
    return float(auroc(x,y,task="binary")),float(average_precision(x,y,task="binary"))


def cka(x,y):
    x=x-x.mean(0); y=y-y.mean(0); xy=x.T@y
    return float((xy*xy).sum()/((x.T@x).square().sum().sqrt()*(y.T@y).square().sum().sqrt()).clamp_min(1e-12))


def effective_rank(x):
    x=x-x.mean(0); eig=torch.linalg.eigvalsh(x@x.T).clamp_min(0); p=eig/eig.sum().clamp_min(1e-12); p=p[p>0]
    return float(torch.exp(-(p*p.log()).sum()))


def raw_weighted(model,vision,text,weights):
    b,p,_=vision.shape[1:]; side=int(math.sqrt(p)); gt=text.unsqueeze(1).repeat(1,b,1,1).permute(1,0,2,3); logits=[]
    for s in range(3):
        fused=model._vision_text_attention_fusion(vision[s],gt,s)
        z=torch.matmul(10*vision[s],fused).permute(0,2,1).view(b,2,side,side)
        logits.append(F.interpolate(z,(IMG,IMG),mode="bilinear",align_corners=True))
    return F.softmax(sum(w*z for w,z in zip(weights,logits)),dim=1)[:,1]


def main():
    protocol=json.loads((AUDIT/"H2_MIXED_DIAGNOSTIC_PROTOCOL.json").read_text())
    p=REPO/protocol["frozen_checkpoints"]["A_E15"]["path"]
    payload=torch.load(p,map_location="cpu",weights_only=False)
    device=torch.device("cuda:0"); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    policy=PrecisionPolicy("fp16")
    clip=create_model("ViT-L-14-336",img_size=IMG,device=device,pretrained="openai",require_pretrained=True).eval()
    model=ACDCLIP(clip_model=clip,n_groups=3,image_adapt_weight=.2,text_adapt_weight=.2,conv_lora_rank=8,conv_lora_alpha=2.,
        conv_kernel_size_list=[3,5],lora_rank=16,lora_alpha=2.,dfg_mode="attn",dfg_attn_dim=256,dfg_attn_tau=8.,
        use_ss2d_dfg=True,dfg_gamma_max=.2,dfg_ss2d_fusion="weight_residual",dfg_beta=.1,dfg_beta_schedule="warmup010",
        dfg_beta_target=.1,dfg_beta_current=.1,dfg_weight_residual_fp32=True,use_soft_prompt=False,soft_prompt_ctx_len=4,
        soft_prompt_init="phrase",soft_prompt_init_phrase="a photo of a").to(device).eval()
    load_arm(model,payload); disable_later_islands(model)
    datasets=get_text_and_image_dataset("VisA",IMG,"test")
    offsets={}; k=0
    for cat in CLASS_NAMES["VisA"]: offsets[cat]=k; k+=len(datasets[cat])
    selected={}
    for cat in CLASS_NAMES["VisA"]:
        normal=[i for i,m in enumerate(datasets[cat].meta) if not m["label"]][:4]
        anomalous=[i for i,m in enumerate(datasets[cat].meta) if m["label"]][:4]
        selected[cat]=normal+anomalous
    transforms=["brightness","contrast","gamma","saturation","gaussian_blur","gaussian_noise","jpeg","resize_cycle"]
    rows=[]; sensitivity_maps=defaultdict(list); sensitivity_masks=[]; feature_bank=defaultdict(list)
    for cat in CLASS_NAMES["VisA"]:
        samples=[datasets[cat][i] for i in selected[cat]]
        images=torch.stack([s["image"] for s in samples]); masks=torch.stack([s["mask"] for s in samples])[:,0].numpy().astype(np.uint8)
        gis=[offsets[cat]+i for i in selected[cat]]
        with torch.no_grad(), policy.autocast(device):
            if not torch.is_autocast_enabled("cuda") or torch.get_autocast_dtype("cuda") != torch.float16:
                raise RuntimeError("historical FP16 autocast identity is not active")
            text=get_multiple_adapted_text_embedding(model,"VisA",device)[cat]
            clean=images.to(device); seg,_=model(clean); vision=torch.stack(seg)
            raw0,final0,_=maps_from_features(model,vision,text); rc0=routing_components(model,vision,text.unsqueeze(1).repeat(1,len(images),1,1))
            clean_feature=torch.stack([v.float().mean(1) for v in vision],1)
            for beta in (.08,.10,.12):
                model.dfg_beta=beta; sensitivity_maps[f"beta:{beta}"].append(raw_weighted(model,vision,text,[1/3]*3).cpu().numpy())
            model.dfg_beta=.1
            for tau in (7.2,8.,8.8):
                model.dfg_attn_tau=tau; sensitivity_maps[f"tau:{tau}"].append(raw_weighted(model,vision,text,[1/3]*3).cpu().numpy())
            model.dfg_attn_tau=8.
            for wi,w in enumerate(protocol["local_sensitivity"]["stage_weight_vectors"]):
                sensitivity_maps[f"stage:{wi+1}"].append(raw_weighted(model,vision,text,w).cpu().numpy())
            sensitivity_maps["stage:base"].append(raw_weighted(model,vision,text,[1/3]*3).cpu().numpy())
            sensitivity_masks.append(masks)
        clean_np=final0.float().cpu().numpy()
        for name in transforms:
            with torch.no_grad(), policy.autocast(device):
                px=perturb(images,name,gis).to(device); pseg,_=model(px); pv=torch.stack(pseg)
                _,pfinal,_=maps_from_features(model,pv,text); prc=routing_components(model,pv,text.unsqueeze(1).repeat(1,len(images),1,1))
                pfeat=torch.stack([v.float().mean(1) for v in pv],1)
            pnp=pfinal.float().cpu().numpy()
            feature_bank[(name,"clean")].append(clean_feature.cpu()); feature_bank[(name,"perturbed")].append(pfeat.cpu())
            for bi,gi in enumerate(gis):
                mm=metric(clean_np[bi],pnp[bi]); feat_cos=float(F.cosine_similarity(clean_feature[bi],pfeat[bi],dim=-1).mean())
                routing_l1=np.mean([float((rc0[s]["weight_final_abnormal"][bi]-prc[s]["weight_final_abnormal"][bi]).abs().sum()) for s in range(3)])
                score_l1=np.mean([float((rc0[s][f"score_{source}_{branch}"][bi]-prc[s][f"score_{source}_{branch}"][bi]).abs().mean())
                    for s in range(3) for source in ("gap","ss2d") for branch in ("normal","abnormal")])
                ent_delta=np.mean([abs(float((-(rc0[s]["weight_final_abnormal"][bi]*rc0[s]["weight_final_abnormal"][bi].clamp_min(1e-12).log()).sum())-
                                              (-(prc[s]["weight_final_abnormal"][bi]*prc[s]["weight_final_abnormal"][bi].clamp_min(1e-12).log()).sum()))) for s in range(3)])
                dis_delta=np.mean([abs(float((rc0[s]["weight_gap_abnormal"][bi]-rc0[s]["weight_ss2d_abnormal"][bi]).abs().sum()-
                                              (prc[s]["weight_gap_abnormal"][bi]-prc[s]["weight_ss2d_abnormal"][bi]).abs().sum())) for s in range(3)])
                cm=[]; pm=[]
                for s in range(3):
                    cm.append(float(F.cosine_similarity(clean_feature[bi,s],text[s,:,1],dim=0)-F.cosine_similarity(clean_feature[bi,s],text[s,:,0],dim=0)))
                    pm.append(float(F.cosine_similarity(pfeat[bi,s],text[s,:,1],dim=0)-F.cosine_similarity(pfeat[bi,s],text[s,:,0],dim=0)))
                pix0=binary_metrics(clean_np[bi],masks[bi]) if masks[bi].any() else {"auroc":None,"ap":None}
                pix1=binary_metrics(pnp[bi],masks[bi]) if masks[bi].any() else {"auroc":None,"ap":None}
                rows.append({"transform":name,"index":gi,"category":cat,"label":int(samples[bi]["label"]),**mm,
                             "feature_cosine":feat_cos,"routing_abnormal_l1":routing_l1,"routing_raw_score_l1":score_l1,
                             "routing_entropy_abs_delta":ent_delta,"gap_ss2d_disagreement_abs_delta":dis_delta,
                             "clean_margin":float(np.mean(cm)),"perturbed_margin":float(np.mean(pm)),"margin_abs_delta":float(np.mean(np.abs(np.asarray(cm)-pm))),
                             "stage_margin_rank_preserved":float(np.array_equal(np.argsort(cm),np.argsort(pm))),
                             "clean_pixel_ap":pix0["ap"],"perturbed_pixel_ap":pix1["ap"],
                             "pixel_ap_delta":None if pix0["ap"] is None else pix1["ap"]-pix0["ap"]})
        print(f"stability_category={cat}",flush=True)
    smask=np.concatenate(sensitivity_masks); sens=[]
    for key,chunks in sensitivity_maps.items():
        maps=np.concatenate(chunks); auc,ap=pooled_metrics(maps,smask); sens.append({"setting":key,"pixel_auroc":auc,"pixel_ap":ap})
    lookup={r["setting"]:r for r in sens}
    sens += [
        {"setting":"derivative:beta","pixel_auroc":(lookup["beta:0.12"]["pixel_auroc"]-lookup["beta:0.08"]["pixel_auroc"])/.04,
         "pixel_ap":(lookup["beta:0.12"]["pixel_ap"]-lookup["beta:0.08"]["pixel_ap"])/.04},
        {"setting":"derivative:tau","pixel_auroc":(lookup["tau:8.8"]["pixel_auroc"]-lookup["tau:7.2"]["pixel_auroc"])/1.6,
         "pixel_ap":(lookup["tau:8.8"]["pixel_ap"]-lookup["tau:7.2"]["pixel_ap"])/1.6},
    ]
    summary=[]
    for name in transforms:
        sub=[r for r in rows if r["transform"]==name]
        summary.append({"transform":name,"n":len(sub),**{k:float(np.mean([r[k] for r in sub])) for k in ("l1","rmse","pearson","spearman","top1_overlap","feature_cosine","routing_abnormal_l1","routing_raw_score_l1","routing_entropy_abs_delta","gap_ss2d_disagreement_abs_delta","margin_abs_delta","stage_margin_rank_preserved")},
                        "anomalous_pixel_ap_delta":float(np.mean([r["pixel_ap_delta"] for r in sub if r["pixel_ap_delta"] is not None]))})
        clean=torch.cat(feature_bank[(name,"clean")]); pert=torch.cat(feature_bank[(name,"perturbed")])
        for stage in range(3):
            summary.append({"transform":name,"record_type":"feature_geometry","stage":stage+1,"n":len(clean),
                            "linear_CKA":cka(clean[:,stage],pert[:,stage]),"clean_effective_rank":effective_rank(clean[:,stage]),
                            "perturbed_effective_rank":effective_rank(pert[:,stage]),
                            "effective_rank_ratio":effective_rank(pert[:,stage])/max(effective_rank(clean[:,stage]),1e-12)})
    write_csv(AUDIT/"H2_MIXED_DFG_STABILITY.csv",summary+rows+sens)
    (AUDIT/"H2_MIXED_STABILITY.json").write_text(json.dumps({"selection_count":sum(map(len,selected.values())),"summary":summary,"sensitivity":sens},indent=2)+"\n")
    print(json.dumps({"status":"PASS","selection_count":sum(map(len,selected.values())),"rows":len(rows)}))

if __name__=="__main__": main()
