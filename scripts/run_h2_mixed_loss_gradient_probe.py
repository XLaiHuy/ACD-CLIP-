#!/usr/bin/env python3
"""Read-only, fixed-source-batch loss-gradient conflict audit for A-E15."""
from __future__ import annotations

import csv
import itertools
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

REPO=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(REPO))
from dataset import CLASS_NAMES, get_text_and_image_dataset
from h2_clean.contract import SafeImageAdapterAnchor
from h2_clean.precision import PrecisionPolicy
from model.adapter import ACDCLIP
from model.clip import create_model
from train import compute_hybrid_k_regularization
from utils import (
    dice_loss, focal_loss, get_hybrid_soft_prompt_single_class_text_embedding,
)
from scripts.run_h2_mixed_generalization_audit import IMG, disable_later_islands, load_arm

AUDIT=REPO/"audit"
COEFFICIENTS={"classification":1.,"focal":1.,"dice_normal":1.,"dice_abnormal":1.,"KG":.01,"K":.002,
              "Anchor":.0021633926715180626}


def family(module,name):
    if module=="text_adapter": return "text_adapter"
    if module=="soft_prompt": return "soft_prompt"
    if "lora_adapters" in name: return "conv_lora"
    if "vision_text_q" in name or "vision_text_k" in name: return "dfg_qk"
    if "dfg_ss2d_branches" in name or "dfg_raw_gamma" in name: return "ss2d"
    return "image_projection"


def dot_norm(left,right,indices):
    dot=torch.zeros((),device="cpu"); ls=torch.zeros((),device="cpu"); rs=torch.zeros((),device="cpu")
    overlap=0
    for i in indices:
        a,b=left[i],right[i]
        if a is not None: ls += a.float().cpu().square().sum()
        if b is not None: rs += b.float().cpu().square().sum()
        if a is not None and b is not None:
            dot += (a.float().cpu()*b.float().cpu()).sum(); overlap+=1
    cosine=float(dot/(ls.sqrt()*rs.sqrt())) if ls>0 and rs>0 else None
    return float(ls.sqrt()),float(rs.sqrt()),cosine,overlap


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)


def main():
    protocol=json.loads((AUDIT/"H2_MIXED_DIAGNOSTIC_PROTOCOL.json").read_text())
    apath=REPO/protocol["frozen_checkpoints"]["A_E15"]["path"]
    epath=REPO/protocol["frozen_checkpoints"]["E1"]["path"]
    payload=torch.load(apath,map_location="cpu",weights_only=False)
    device=torch.device("cuda:0"); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    clip=create_model("ViT-L-14-336",img_size=IMG,device=device,pretrained="openai",require_pretrained=True)
    clip.set_grad_checkpointing(True)
    model=ACDCLIP(clip_model=clip,n_groups=3,image_adapt_weight=.2,text_adapt_weight=.2,conv_lora_rank=8,conv_lora_alpha=2.,
        conv_kernel_size_list=[3,5],lora_rank=16,lora_alpha=2.,dfg_mode="attn",dfg_attn_dim=256,dfg_attn_tau=8.,
        use_ss2d_dfg=True,dfg_gamma_max=.2,dfg_ss2d_fusion="weight_residual",dfg_beta=.1,dfg_beta_schedule="warmup010",
        dfg_beta_target=.1,dfg_beta_current=.1,dfg_weight_residual_fp32=True,use_soft_prompt=False,soft_prompt_ctx_len=4,
        soft_prompt_init="phrase",soft_prompt_init_phrase="a photo of a").to(device).eval()
    load_arm(model,payload); disable_later_islands(model)
    model.requires_grad_(False); model.image_adapter.requires_grad_(True); model.text_adapter.requires_grad_(True); model.soft_prompt.requires_grad_(True)
    anchor=SafeImageAdapterAnchor.from_checkpoint(epath,device)
    named=[]
    for module_name,module in (("image_adapter",model.image_adapter),("text_adapter",model.text_adapter),("soft_prompt",model.soft_prompt)):
        for name,p in module.named_parameters(): named.append((f"{module_name}.{name}",p,family(module_name,name)))
    names=[x[0] for x in named]; params=[x[1] for x in named]
    families={f:[i for i,x in enumerate(named) if x[2]==f] for f in ("conv_lora","image_projection","dfg_qk","ss2d","text_adapter","soft_prompt")}
    datasets=get_text_and_image_dataset("VisA",IMG,"test")
    rows=[]; policy=PrecisionPolicy("fp16")
    for batch_id,cat in enumerate(CLASS_NAMES["VisA"][:3],1):
        normal=[i for i,m in enumerate(datasets[cat].meta) if not m["label"]][:2]
        anomalous=[i for i,m in enumerate(datasets[cat].meta) if m["label"]][:2]
        samples=[datasets[cat][i] for i in normal+anomalous]
        image=torch.stack([s["image"] for s in samples]).to(device); mask=torch.stack([s["mask"] for s in samples]).to(device)
        labels=torch.tensor([s["label"] for s in samples],device=device,dtype=torch.long)
        text,kg,_,comp=get_hybrid_soft_prompt_single_class_text_embedding(model,"VisA",cat,device,return_kg=True,return_components=True)
        kval,_=compute_hybrid_k_regularization(model,comp["hard_text"],comp["soft_text"],float(model.hybrid_alpha_current))
        text_batch=text.unsqueeze(1).repeat(1,len(samples),1,1)
        with policy.autocast(device):
            seg,det=model(image); seg=torch.stack(seg); det=torch.stack(det)
            cls=torch.stack([torch.matmul(det[s].unsqueeze(1),text_batch[s]).squeeze(1) for s in range(3)]).mean(0)
            segpred=model.vision_text_fusion_gate_seg(seg,text_batch,cir_training=False)
            terms={"classification":F.cross_entropy(cls,labels),"focal":focal_loss(segpred,mask),
                   "dice_normal":dice_loss(segpred[:,0],1-mask),"dice_abnormal":dice_loss(segpred[:,1],mask),
                   "KG":kg,"K":kval,"Anchor":anchor.loss(model.image_adapter)}
        grads={}
        for ti,(term_name,term) in enumerate(terms.items()):
            grads[term_name]=torch.autograd.grad(term,params,retain_graph=ti<len(terms)-1,allow_unused=True)
        for term_name,g in grads.items():
            for fam,idx in families.items():
                norm,_,_,active=dot_norm(g,g,idx)
                rows.append({"row_type":"norm","batch":batch_id,"category":cat,"term":term_name,"family":fam,
                             "coefficient":COEFFICIENTS[term_name],"raw_gradient_norm":norm,
                             "weighted_gradient_norm":norm*COEFFICIENTS[term_name],"active_parameter_tensors":active,
                             "loss_value":float(terms[term_name].detach().float())})
        for left,right in itertools.combinations(terms,2):
            for fam,idx in families.items():
                ln,rn,cos,active=dot_norm(grads[left],grads[right],idx)
                rows.append({"row_type":"cosine","batch":batch_id,"category":cat,"term":left,"term_right":right,"family":fam,
                             "cosine":cos,"left_norm":ln,"right_norm":rn,"active_parameter_tensors":active})
        del grads,terms,seg,det,segpred; torch.cuda.empty_cache(); print(f"gradient_batch={batch_id}/3",flush=True)
    # Compact aggregation preserves nulls for structurally disconnected loss/family pairs.
    agg=[]
    keys=sorted({(r["row_type"],r["term"],r.get("term_right"),r["family"]) for r in rows})
    for key in keys:
        sub=[r for r in rows if (r["row_type"],r["term"],r.get("term_right"),r["family"])==key]
        out={"row_type":"aggregate_"+key[0],"term":key[1],"term_right":key[2],"family":key[3],"batch_count":len(sub)}
        for field in ("raw_gradient_norm","weighted_gradient_norm","cosine","loss_value"):
            vals=[r[field] for r in sub if r.get(field) is not None]
            out[field]=sum(vals)/len(vals) if vals else None
        agg.append(out)
    write_csv(AUDIT/"H2_MIXED_LOSS_GRADIENT_CONFLICT.csv",agg+rows)
    (AUDIT/"H2_MIXED_LOSS_GRADIENT_CONFLICT.json").write_text(json.dumps({"protocol":protocol["gradient_probe"],"coefficients":COEFFICIENTS,
        "anchor_runtime_note":"nominal coefficient reported; historical per-family gradient budget can cap the applied Anchor vector at 0.1 of task-family norm",
        "aggregates":agg},indent=2)+"\n")
    print(json.dumps({"status":"PASS","batches":3,"rows":len(rows)}))

if __name__=="__main__": main()
