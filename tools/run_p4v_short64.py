#!/usr/bin/env python3
"""Bounded, deterministic Phase4-V K=1 A/B/C/D mechanism experiment."""
from __future__ import annotations
import argparse, inspect, json, math, sys
from collections import defaultdict
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]; sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
from audit_p4_k1_oracle_utility import DeterministicVisATrainDataset, _sha256
from model.adapter import ACDCLIP
from model.clip import create_model
from utils import calculate_seg_loss, configure_canonical_fp32, get_phase2b_global_text_features

VARIANTS = ("BASE", "VISUAL_RESIDUAL", "SEMANTIC_VISUAL", "FACTORIZED")

def build(config, device):
    direct = inspect.signature(ACDCLIP.__init__).parameters
    kw = {n: config[n] for n,p in direct.items() if n not in {"self","clip_model","kwargs"} and p.kind is not inspect.Parameter.VAR_KEYWORD and n in config}
    kw.update(h6_progress=1, h6_num_factors=1, h6_top_k=1, h6_progress_version="P4V-K1", h6_local_factor_mode="legacy_mix", h6_expert_enabled=False, h6_role_topology="flat", h6_intrinsic_factor_responsibility=False, h6_cluster_responsibility=False, h6_router_boundary_mode="none", phase4v_bottleneck=64, phase4v_lambda=.05)
    clip = create_model(config["model_name"], img_size=config["img_size"], device=device, pretrained="openai", require_pretrained=True, precision="fp32")
    if config.get("grad_checkpointing"): clip.set_grad_checkpointing(True)
    m = ACDCLIP(clip_model=clip, **kw).to(device); m.requires_grad_(False)
    m.image_adapter.requires_grad_(True); m.text_adapter.requires_grad_(True); m.h6.conditional_semantic_core.requires_grad_(True); m.h6.visual_adapter.requires_grad_(True)
    m.h6.router.requires_grad_(False); m.h6.semantic_core.requires_grad_(False); m.soft_prompt.requires_grad_(False)
    return m

def forward(m, image, mask, label, classes, config, variant, active):
    visual = m(image, return_phase4_features=True); original = torch.stack(visual["seg_tokens"])
    text = get_phase2b_global_text_features(m, "VisA", classes, image.device, use_hybrid_soft_prompt=True, use_soft_prompt=False).float()
    base, logits, _ = m.vision_text_fusion_gate_seg(original, text, img_size=config["img_size"], return_details=True)
    outputs = None
    if active and variant != "BASE":
        state = m.h6.phase4v_state_code(m, visual)["semantic_code"]
        raw_gate = torch.softmax(logits.float(), -1)[...,1].detach()
        semantic = variant != "VISUAL_RESIDUAL"; spatial = variant == "FACTORIZED"
        outputs = [m.h6.phase4v_adapt(visual["seg_tokens"][g], state, raw_gate[g], enabled=True, semantic_conditioning=semantic, spatial_gating=spatial) for g in range(m.n_groups)]
        pred = m.vision_text_fusion_gate_seg(torch.stack([x["adapted"] for x in outputs]), text, img_size=config["img_size"])
    else: pred = base
    det = torch.stack(visual["det_tokens"]); cls = torch.stack([det[g].unsqueeze(1).matmul(text[g]).squeeze(1) for g in range(m.n_groups)]).mean(0)
    loss = calculate_seg_loss(pred.float(), mask) + F.cross_entropy(cls.float(), label)
    return loss, pred, base, outputs, original, text

def region_metrics(pred, mask, classes, records):
    prob = pred[:,1].detach().float(); y = mask[:,0].detach().float(); eps=1e-6
    bce = F.binary_cross_entropy(prob.clamp(eps,1-eps), y, reduction="none")
    for region, sel in (("normal", y < .5), ("anomaly", y >= .5)):
        if sel.any():
            vals=bce[sel]; records[region].append(float(vals.mean())); records["count_"+region].append(int(sel.sum()))
    for name in classes: records["classes"].append(name)

def _pixel_metrics(target, score):
    y=target.detach().float().flatten(); s=score.detach().float().flatten(); pos=y.sum(); neg=y.numel()-pos
    if pos <= 0 or neg <= 0: return None, None
    ranked=y[torch.argsort(s,descending=True)]; precision=ranked.cumsum(0)/torch.arange(1,ranked.numel()+1,device=y.device); ap=float((precision*ranked).sum()/pos)
    ascending=y[torch.argsort(s)]; ranks=torch.arange(1,ascending.numel()+1,device=y.device,dtype=torch.float32); auc=float(((ranks*ascending).sum()-pos*(pos+1)/2)/(pos*neg))
    return ap,auc


def _mean_defined(values):
    values=[value for value in values if value is not None]
    return None if not values else float(torch.tensor(values).mean())

@torch.no_grad()
def evaluate(m, loader, config, variant):
    m.eval(); m.clipmodel.eval()
    rec=defaultdict(list); correction=[]; per_class=defaultdict(lambda: defaultdict(list)); per_image=[]; geometry=defaultdict(list); aps=[]; aucs=[]
    for raw in loader:
        image=raw["image"].cuda().float(); mask=raw["mask"].cuda().float(); label=raw["label"].cuda(); classes=list(raw["class_name"])
        _,pred,_,out,orig,text=forward(m,image,mask,label,classes,config,variant,True)
        region_metrics(pred,mask,classes,rec)
        if out: correction.extend([float(x["correction"].float().norm(dim=-1).mean()) for x in out])
        prob=pred[:,1].float(); target=mask[:,0].float(); bce=F.binary_cross_entropy(prob.clamp(1e-6,1-1e-6),target,reduction="none")
        name=classes[0]; ap,auc=_pixel_metrics(target,prob); aps.append(ap); aucs.append(auc); per_class[name]["pixel_ap"].append(ap); per_class[name]["pixel_auc"].append(auc)
        for region,sel in (("normal",target<.5),("anomaly",target>=.5)):
            if sel.any(): per_class[name][region+"_bce"].append(float(bce[sel].mean()))
        per_image.append({"class":name,"file":raw["file_name"][0],"normal_bce":None if not (target<.5).any() else float(bce[target<.5].mean()),"anomaly_bce":None if not (target>=.5).any() else float(bce[target>=.5].mean()),"pixel_ap":ap,"pixel_auc":auc})
        features=torch.stack([x["adapted"] for x in out]) if out else orig
        for g in range(m.n_groups):
            side=math.isqrt(features.shape[2]); patch_y=F.adaptive_avg_pool2d(target.unsqueeze(1),(side,side)).flatten(1)[0]
            feats=features[g,0].float(); normal_anchor=text[g,0,:,0]; anomaly_anchor=text[g,0,:,1]
            sim_n=F.cosine_similarity(feats,normal_anchor.unsqueeze(0),dim=-1); sim_a=F.cosine_similarity(feats,anomaly_anchor.unsqueeze(0),dim=-1)
            for region,sel in (("normal",patch_y<.5),("anomaly",patch_y>=.5)):
                if sel.any(): geometry[region+"_sim_normal"].append(float(sim_n[sel].mean())); geometry[region+"_sim_anomaly"].append(float(sim_a[sel].mean()))
    pcs={name:{"pixel_ap_macro":_mean_defined(v["pixel_ap"]),"pixel_auc_macro":_mean_defined(v["pixel_auc"]),"normal_bce":_mean_defined(v["normal_bce"]),"anomaly_bce":_mean_defined(v["anomaly_bce"])} for name,v in per_class.items()}
    return {"normal_bce":_mean_defined(rec["normal"]),"anomaly_bce":_mean_defined(rec["anomaly"]),"normal_pixels":sum(rec["count_normal"]),"anomaly_pixels":sum(rec["count_anomaly"]),"pixel_ap_macro":_mean_defined(aps),"pixel_auc_macro":_mean_defined(aucs),"per_class":pcs,"per_image":per_image,"geometry":{k:_mean_defined(v) for k,v in geometry.items()},"per_class_seen":sorted(set(rec["classes"])),"mean_relative_correction":_mean_defined(correction) if correction else 0.0}
def run(config, manifest, variant, out):
    torch.manual_seed(0); torch.cuda.manual_seed_all(0); device=torch.device("cuda:0"); m=build(config,device)
    ds=DeterministicVisATrainDataset(manifest,config["img_size"]); loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=0); opt=torch.optim.Adam([{"params":m.image_adapter.parameters(),"lr":config["image_lr"]},{"params":m.text_adapter.parameters(),"lr":config["text_lr"]},{"params":m.h6.conditional_semantic_core.parameters(),"lr":1e-4},{"params":m.h6.visual_adapter.parameters(),"lr":1e-4}])
    rows=[]; it=iter(loader)
    for epoch in range(1,9):
        m.train(); m.clipmodel.eval(); active=epoch>7
        for micro in range(8):
            try: raw=next(it)
            except StopIteration: it=iter(loader); raw=next(it)
            image=raw["image"].to(device).float(); mask=raw["mask"].to(device).float(); label=raw["label"].to(device); classes=list(raw["class_name"])
            opt.zero_grad(set_to_none=True); loss,pred,base,outs,orig,_=forward(m,image,mask,label,classes,config,variant,active); loss.backward(); torch.nn.utils.clip_grad_norm_([p for q in opt.param_groups for p in q["params"]],1.0); opt.step()
            rows.append({"epoch":epoch,"microbatch":len(rows)+1,"active":active,"loss":float(loss.detach()),"finite":bool(torch.isfinite(loss.detach())),"relative_correction":0.0 if not outs else float((torch.stack([x["correction"].float().norm(dim=-1).mean() for x in outs]).mean()/orig.float().norm(dim=-1).mean()).detach())})
    metrics=evaluate(m,loader,config,variant); metrics.update({"variant":variant,"microbatches":64,"warmup_epoch":7,"post_activation_microbatches":8,"all_finite":all(x["finite"] for x in rows),"max_relative_correction":max(x["relative_correction"] for x in rows),"rows":rows})
    out.mkdir(parents=True,exist_ok=True); (out/"METRICS.json").write_text(json.dumps(metrics,indent=2)+"\n"); return metrics

def main():
 p=argparse.ArgumentParser(); p.add_argument("--config",type=Path,default=Path("runs/phase4/k1/short64_seed0_attempt5/config.json")); p.add_argument("--manifest",type=Path,default=Path("runs/phase4/k1/stage1_7r/visa_train_audit_manifest.json")); p.add_argument("--output",type=Path,default=Path("runs/phase4v/short64")); a=p.parse_args(); configure_canonical_fp32(); c=json.loads(a.config.read_text()); m=json.loads(a.manifest.read_text()); a.output.mkdir(parents=True,exist_ok=True); (a.output/"RUN_MANIFEST.json").write_text(json.dumps({"initialization":"fresh OpenAI CLIP only","precision":"strict FP32","variants":VARIANTS,"warmup_evidence_epoch":7,"schedule":"8 epochs x 8 deterministic train microbatches; activation only epoch 8","config_sha256":_sha256(a.config),"audit_manifest_sha256":_sha256(a.manifest)},indent=2)+"\n"); allm={v:run(c,m,v,a.output/v) for v in VARIANTS}; (a.output/"ABLATION_METRICS.json").write_text(json.dumps(allm,indent=2)+"\n"); print(json.dumps({k:{x:y for x,y in v.items() if x not in {"rows","per_class_seen"}} for k,v in allm.items()},indent=2))
if __name__ == "__main__": main()
