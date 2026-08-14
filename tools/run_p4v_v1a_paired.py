#!/usr/bin/env python3
"""Branch-A paired run: base-only gradients plus isolated visual correction."""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/"tools")]
from audit_p4_k1_oracle_utility import DeterministicVisATrainDataset, _sha256
from run_p4v_short64 import build
from run_p4v_v15_paired import batch, evaluate, optimizer
from utils import calculate_seg_loss, configure_canonical_fp32, get_phase2b_global_text_features

WARMUP_STEPS=56;ACTIVE_STEPS=32;ETA=1.0

def base_and_isolated(model,image,mask,label,classes,config):
    visual=model(image,return_phase4_features=True);v=torch.stack(visual["seg_tokens"])
    text=get_phase2b_global_text_features(model,"VisA",classes,image.device,use_hybrid_soft_prompt=True,use_soft_prompt=False).float()
    base,logits,_=model.vision_text_fusion_gate_seg(v,text,img_size=config["img_size"],return_details=True)
    det=torch.stack(visual["det_tokens"]);cls=torch.stack([det[g].unsqueeze(1).matmul(text[g]).squeeze(1) for g in range(model.n_groups)]).mean(0)
    base_loss=calculate_seg_loss(base.float(),mask)+F.cross_entropy(cls.float(),label)
    detached={key:([item.detach() for item in value] if isinstance(value,list) else value.detach()) for key,value in visual.items()};state=model.h6.phase4v_state_code(model,detached)["semantic_code"];gate=torch.softmax(logits.detach().float(),-1)[...,1]
    adapted=[];anchors=[]
    for g in range(model.n_groups):
        out=model.h6.phase4v_adapt(v[g].detach(),state,gate[g],enabled=True,semantic_conditioning=True,spatial_gating=True);adapted.append(out["adapted"])
        with torch.no_grad():
            group_text=text.detach().permute(1,0,2,3);weights=model.compute_dfg_weights(v[g].detach(),group_text,g);anchors.append(model.apply_dfg_weights(group_text,weights["normal"],weights["abnormal"]).detach())
    corr_logits=[]
    for g,features in enumerate(adapted):
        side=math.isqrt(features.shape[1]);corr_logits.append((10*features).matmul(anchors[g]).permute(0,2,1).view(features.shape[0],2,side,side))
    corr=F.softmax(torch.stack([F.interpolate(x,size=config["img_size"],mode="bilinear",align_corners=True) for x in corr_logits]).mean(0),dim=1)
    return base_loss,calculate_seg_loss(corr.float(),mask),base,corr,adapted

def base_loss_only(model,image,mask,label,classes,config):
    visual=model(image,return_phase4_features=True);v=torch.stack(visual["seg_tokens"]);text=get_phase2b_global_text_features(model,"VisA",classes,image.device,use_hybrid_soft_prompt=True,use_soft_prompt=False).float();pred=model.vision_text_fusion_gate_seg(v,text,img_size=config["img_size"]);det=torch.stack(visual["det_tokens"]);cls=torch.stack([det[g].unsqueeze(1).matmul(text[g]).squeeze(1) for g in range(model.n_groups)]).mean(0);return calculate_seg_loss(pred.float(),mask)+F.cross_entropy(cls.float(),label)

def steps(model,opt,data,start,count,config,isolated):
    model.train();model.clipmodel.eval();rows=[]
    for i in range(count):
        image,mask,label,classes,_=batch(data,start+i,next(model.parameters()).device);opt.zero_grad(set_to_none=True)
        if isolated:
            base,corr,_,_,adapted=base_and_isolated(model,image,mask,label,classes,config);loss=base+ETA*corr
        else: base=base_loss_only(model,image,mask,label,classes,config);corr=torch.zeros_like(base);loss=base
        loss.backward();torch.nn.utils.clip_grad_norm_([p for group in opt.param_groups for p in group["params"]],1.0);opt.step();rows.append({"step":i+1,"base_loss":float(base.detach()),"corr_loss":float(corr.detach()),"corr_to_base":float((corr/base.detach().clamp_min(1e-6)).detach()),"finite":bool(torch.isfinite(loss.detach()))})
    return rows

def save_adapter(model,path):torch.save({"image_adapter":model.image_adapter.state_dict(),"text_adapter":model.text_adapter.state_dict(),"h6":model.h6.state_dict()},path)

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,default=Path("runs/phase4/k1/short64_seed0_attempt5/config.json"));p.add_argument("--manifest",type=Path,default=Path("runs/phase4/k1/stage1_7r/visa_train_audit_manifest.json"));p.add_argument("--output",type=Path,default=Path("runs/phase4v/v1a/paired"));a=p.parse_args();configure_canonical_fp32();config=json.loads(a.config.read_text());manifest=json.loads(a.manifest.read_text());torch.manual_seed(1601);torch.cuda.manual_seed_all(1601);device=torch.device("cuda:0");data=DeterministicVisATrainDataset(manifest,config["img_size"]);a.output.mkdir(parents=True,exist_ok=True)
    warm=build(config,device);wo=optimizer(warm,config);warm_rows=steps(warm,wo,data,0,WARMUP_STEPS,config,False);warm_path=a.output/"warmup_adapter_state.pth";torch.save({"image_adapter":warm.image_adapter.state_dict(),"text_adapter":warm.text_adapter.state_dict(),"h6":warm.h6.state_dict(),"optimizer":wo.state_dict()},warm_path);del warm,wo;torch.cuda.empty_cache();state=torch.load(warm_path,map_location="cpu",weights_only=False)
    def branch(isolated):
        torch.manual_seed(1601);torch.cuda.manual_seed_all(1601);m=build(config,device);m.image_adapter.load_state_dict(state["image_adapter"]);m.text_adapter.load_state_dict(state["text_adapter"]);m.h6.load_state_dict(state["h6"]);o=optimizer(m,config);o.load_state_dict(state["optimizer"]);return m,steps(m,o,data,WARMUP_STEPS,ACTIVE_STEPS,config,isolated)
    base,base_rows=branch(False);base_report=evaluate(base,data,config,"OFF");save_adapter(base,a.output/"base_adapter_state.pth");del base;torch.cuda.empty_cache()
    candidate,candidate_rows=branch(True);active_report=evaluate(candidate,data,config,"ACTIVE");off_report=evaluate(candidate,data,config,"OFF");save_adapter(candidate,a.output/"v1a_adapter_state.pth")
    report={"protocol":{"initialization":"fresh OpenAI CLIP shared warmup","precision":"strict FP32","warmup_steps":WARMUP_STEPS,"active_steps":ACTIVE_STEPS,"eta":ETA,"config_sha256":_sha256(a.config),"manifest_sha256":_sha256(a.manifest)},"finite":{"warmup":all(x["finite"] for x in warm_rows),"base":all(x["finite"] for x in base_rows),"v1a":all(x["finite"] for x in candidate_rows)},"loss_scale":{"first_active_corr_to_base":candidate_rows[0]["corr_to_base"],"mean_active_corr_to_base":sum(x["corr_to_base"] for x in candidate_rows)/len(candidate_rows)},"reports":{"BASE":base_report,"ACTIVE":active_report,"OFF":off_report},"local_checkpoints":[str(warm_path),str(a.output/"base_adapter_state.pth"),str(a.output/"v1a_adapter_state.pth")]}
    (a.output/"V1A_PAIRED.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps({"finite":report["finite"],"loss_scale":report["loss_scale"],"base":{k:base_report[k] for k in ("normal_bce","anomaly_bce","pixel_ap_macro","pixel_auc_macro")},"active":{k:active_report[k] for k in ("normal_bce","anomaly_bce","pixel_ap_macro","pixel_auc_macro")}},indent=2))
if __name__=="__main__":main()
