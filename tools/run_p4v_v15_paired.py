#!/usr/bin/env python3
"""One Branch-E paired V1 replay with retained ON/OFF causal evidence."""
from __future__ import annotations
import argparse, copy, json, math, sys
from collections import defaultdict
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT),str(ROOT/"tools")]
from audit_p4_k1_oracle_utility import DeterministicVisATrainDataset, _sha256
from run_p4v_short64 import build, _pixel_metrics, _mean_defined
from utils import calculate_seg_loss, configure_canonical_fp32, get_phase2b_global_text_features

WARMUP_STEPS=56; ACTIVE_STEPS=32

def optimizer(model, config):
    return torch.optim.Adam([{"params":model.image_adapter.parameters(),"lr":config["image_lr"]},{"params":model.text_adapter.parameters(),"lr":config["text_lr"]},{"params":model.h6.conditional_semantic_core.parameters(),"lr":1e-4},{"params":model.h6.visual_adapter.parameters(),"lr":1e-4}])

def batch(dataset, index, device):
    raw=dataset[index % len(dataset)]
    image_path=str(raw["file_name"])
    provenance={"image_path":image_path,"file_name":Path(image_path).name,"class_name":str(raw["class_name"]),"label":int(raw["label"].item())}
    return raw["image"].unsqueeze(0).to(device).float(),raw["mask"].unsqueeze(0).to(device).float(),raw["label"].view(1).to(device),[raw["class_name"]],provenance

def phase4_forward(model,image,mask,label,classes,config,mode):
    visual=model(image,return_phase4_features=True); v=torch.stack(visual["seg_tokens"])
    text=get_phase2b_global_text_features(model,"VisA",classes,image.device,use_hybrid_soft_prompt=True,use_soft_prompt=False).float()
    base,logits,_=model.vision_text_fusion_gate_seg(v,text,img_size=config["img_size"],return_details=True)
    if mode=="OFF": return base,base,v,v,None,text
    state=model.h6.phase4v_state_code(model,visual)["semantic_code"]; gate=torch.softmax(logits.float(),-1)[...,1].detach()
    outs=[]
    for g in range(model.n_groups):
        outs.append(model.h6.phase4v_adapt(v[g],state,torch.zeros_like(gate[g]) if mode=="ZERO_GATE" else gate[g],enabled=True,semantic_conditioning=True,spatial_gating=True,force_delta_zero=(mode=="ZERO_DELTA")))
    adapted=torch.stack([out["adapted"] for out in outs]); pred=model.vision_text_fusion_gate_seg(adapted,text,img_size=config["img_size"])
    return pred,base,v,adapted,outs,text

def task_loss(model,image,mask,label,classes,config,active):
    pred,_,_,_,_,text=phase4_forward(model,image,mask,label,classes,config,"ACTIVE" if active else "OFF")
    visual=model(image,return_phase4_features=True); det=torch.stack(visual["det_tokens"])
    cls=torch.stack([det[g].unsqueeze(1).matmul(text[g]).squeeze(1) for g in range(model.n_groups)]).mean(0)
    return calculate_seg_loss(pred.float(),mask)+F.cross_entropy(cls.float(),label)

def train_steps(model,opt,dataset,start,count,config,active):
    model.train(); model.clipmodel.eval(); rows=[]
    for step in range(count):
        image,mask,label,classes,_=batch(dataset,start+step,next(model.parameters()).device)
        opt.zero_grad(set_to_none=True); loss=task_loss(model,image,mask,label,classes,config,active); loss.backward(); torch.nn.utils.clip_grad_norm_([p for x in opt.param_groups for p in x["params"]],1.0); opt.step()
        rows.append({"step":step+1,"loss":float(loss.detach()),"finite":bool(torch.isfinite(loss.detach()))})
    return rows

def group_prediction(model,features,text,g,img_size):
    group_text=text.permute(1,0,2,3); weights=model.compute_dfg_weights(features[g],group_text,g); anchors=model.apply_dfg_weights(group_text,weights["normal"],weights["abnormal"])
    logits=(10*features[g]).matmul(anchors); side=math.isqrt(features.shape[2]); pred=F.softmax(F.interpolate(logits.permute(0,2,1).view(1,2,side,side),size=img_size,mode="bilinear",align_corners=True),dim=1)
    return pred,anchors

@torch.no_grad()
def evaluate(model,dataset,config,mode):
    model.eval(); model.clipmodel.eval(); rec=defaultdict(list); group=defaultdict(lambda:defaultdict(list)); classes=defaultdict(lambda:defaultdict(list)); images=[]; geometry=defaultdict(list)
    for idx in range(len(dataset)):
        image,mask,label,names,provenance=batch(dataset,idx,next(model.parameters()).device); pred,base,v,adapted,outs,text=phase4_forward(model,image,mask,label,names,config,mode); target=mask[:,0]; prob=pred[:,1]; bce=F.binary_cross_entropy(prob.clamp(1e-6,1-1e-6),target,reduction="none")
        ap,auc=_pixel_metrics(target,prob); name=names[0]; row={**provenance,"normal_bce":None if not (target<.5).any() else float(bce[target<.5].mean()),"anomaly_bce":None if not (target>=.5).any() else float(bce[target>=.5].mean()),"pixel_ap":ap,"pixel_auc":auc}; images.append(row)
        for key,sel in (("normal_bce",target<.5),("anomaly_bce",target>=.5)):
            if sel.any(): rec[key].append(float(bce[sel].mean())); classes[name][key].append(float(bce[sel].mean()))
        rec["pixel_ap"].append(ap);rec["pixel_auc"].append(auc);classes[name]["pixel_ap"].append(ap);classes[name]["pixel_auc"].append(auc)
        after=v if adapted is None else adapted
        for g in range(model.n_groups):
            gp,anchors=group_prediction(model,after,text,g,config["img_size"]); gbce=F.binary_cross_entropy(gp[:,1].clamp(1e-6,1-1e-6),target,reduction="none"); gap,gauc=_pixel_metrics(target,gp[:,1]); group[str(g)]["pixel_ap"].append(gap);group[str(g)]["pixel_auc"].append(gauc)
            for key,sel in (("normal_bce",target<.5),("anomaly_bce",target>=.5)):
                if sel.any(): group[str(g)][key].append(float(gbce[sel].mean()))
            if outs is not None:
                raw=outs[g]["delta_v"].float(); correction=outs[g]["correction"].float(); before=v[g].float(); corrected=after[g].float(); n=F.normalize(anchors[0,:,0],dim=-1); a=F.normalize(anchors[0,:,1],dim=-1); vh=F.normalize(before,dim=-1); radial=(raw*vh).sum(-1,keepdim=True)*vh; tangent=raw-radial; angle=torch.acos((F.normalize(before,dim=-1)*F.normalize(corrected,dim=-1)).sum(-1).clamp(-1,1)); mb=(before*a).sum(-1)-(before*n).sum(-1); ma=(corrected*a).sum(-1)-(corrected*n).sum(-1); side=math.isqrt(before.shape[1]); patch=F.adaptive_avg_pool2d(target.unsqueeze(1),(side,side)).flatten(1)
                for region,sel in (("normal",patch<.5),("anomaly",patch>=.5)):
                    if sel.any():
                        geometry[region+"_angle"].append(float(angle[sel].mean()));geometry[region+"_delta_norm"].append(float(raw.norm(dim=-1)[sel].mean()));geometry[region+"_correction_norm"].append(float(correction.norm(dim=-1)[sel].mean()));geometry[region+"_relative_correction"].append(float((correction.norm(dim=-1)/before.norm(dim=-1).clamp_min(1e-6))[sel].mean()));geometry[region+"_radial_fraction"].append(float((radial.norm(dim=-1)/(raw.norm(dim=-1).clamp_min(1e-6)))[sel].mean()));geometry[region+"_tangent_fraction"].append(float((tangent.norm(dim=-1)/(raw.norm(dim=-1).clamp_min(1e-6)))[sel].mean()));geometry[region+"_delta_margin"].append(float((ma-mb)[sel].mean()));geometry[region+"_gate"].append(float(outs[g]["gate"][sel].mean()))
    compact=lambda d:{key:_mean_defined(vals) for key,vals in d.items()}; return {"mode":mode,"normal_bce":_mean_defined(rec["normal_bce"]),"anomaly_bce":_mean_defined(rec["anomaly_bce"]),"pixel_ap_macro":_mean_defined(rec["pixel_ap"]),"pixel_auc_macro":_mean_defined(rec["pixel_auc"]),"per_image":images,"per_class":{key:compact(value) for key,value in classes.items()},"per_group":{key:compact(value) for key,value in group.items()},"geometry":compact(geometry)}

def delta(left,right):
    return {key:left[key]-right[key] for key in ("normal_bce","anomaly_bce","pixel_ap_macro","pixel_auc_macro")}

def main():
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,default=Path("runs/phase4/k1/short64_seed0_attempt5/config.json"));p.add_argument("--manifest",type=Path,default=Path("runs/phase4/k1/stage1_7r/visa_train_audit_manifest.json"));p.add_argument("--output",type=Path,default=Path("runs/phase4v/v1_5/paired"));a=p.parse_args();configure_canonical_fp32();config=json.loads(a.config.read_text());manifest=json.loads(a.manifest.read_text());a.output.mkdir(parents=True,exist_ok=True);torch.manual_seed(1501);torch.cuda.manual_seed_all(1501);device=torch.device("cuda:0");data=DeterministicVisATrainDataset(manifest,config["img_size"])
    warm=build(config,device);warmopt=optimizer(warm,config);warm_rows=train_steps(warm,warmopt,data,0,WARMUP_STEPS,config,False);warm_path=a.output/"warmup_adapter_state.pth";torch.save({"image_adapter":warm.image_adapter.state_dict(),"text_adapter":warm.text_adapter.state_dict(),"h6":warm.h6.state_dict(),"optimizer":warmopt.state_dict()},warm_path);del warm,warmopt;torch.cuda.empty_cache();state=torch.load(warm_path,map_location="cpu",weights_only=False)
    def branch(active):
        torch.manual_seed(1501);torch.cuda.manual_seed_all(1501);m=build(config,device);m.image_adapter.load_state_dict(state["image_adapter"]);m.text_adapter.load_state_dict(state["text_adapter"]);m.h6.load_state_dict(state["h6"]);o=optimizer(m,config);o.load_state_dict(state["optimizer"]);rows=train_steps(m,o,data,WARMUP_STEPS,ACTIVE_STEPS,config,active);return m,rows
    base,base_rows=branch(False);base_report=evaluate(base,data,config,"OFF");torch.save({"image_adapter":base.image_adapter.state_dict(),"text_adapter":base.text_adapter.state_dict(),"h6":base.h6.state_dict()},a.output/"base_adapter_state.pth");del base;torch.cuda.empty_cache()
    candidate,candidate_rows=branch(True);torch.save({"image_adapter":candidate.image_adapter.state_dict(),"text_adapter":candidate.text_adapter.state_dict(),"h6":candidate.h6.state_dict()},a.output/"current_v1_adapter_state.pth")
    reports={"BASE":base_report,"ACTIVE":evaluate(candidate,data,config,"ACTIVE"),"OFF":evaluate(candidate,data,config,"OFF"),"ZERO_DELTA":evaluate(candidate,data,config,"ZERO_DELTA"),"ZERO_GATE":evaluate(candidate,data,config,"ZERO_GATE")};identity={key:{"total_delta":delta(reports["ACTIVE"],reports["BASE"])[key],"intervention_effect":delta(reports["ACTIVE"],reports["OFF"])[key],"optimization_drift":delta(reports["OFF"],reports["BASE"])[key]} for key in ("normal_bce","anomaly_bce","pixel_ap_macro","pixel_auc_macro")}
    for value in identity.values(): value["reconstruction_error"]=value["total_delta"]-value["intervention_effect"]-value["optimization_drift"]
    report={"protocol":{"initialization":"fresh OpenAI CLIP shared 56-update warmup","precision":"strict FP32","warmup_steps":WARMUP_STEPS,"active_steps":ACTIVE_STEPS,"seed":1501,"manifest_sha256":_sha256(a.manifest),"config_sha256":_sha256(a.config)},"finite":{"warmup":all(x["finite"] for x in warm_rows),"base":all(x["finite"] for x in base_rows),"candidate":all(x["finite"] for x in candidate_rows)},"reports":reports,"causal_identity":identity,"local_checkpoints":{"warmup":str(warm_path),"current_v1":str(a.output/"current_v1_adapter_state.pth")}}
    (a.output/"V1_5_PAIRED_CAUSAL.json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps({"finite":report["finite"],"causal_identity":identity},indent=2))
if __name__=="__main__":main()
