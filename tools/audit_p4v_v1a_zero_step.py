#!/usr/bin/env python3
"""Strict FP32 zero-step proof for Branch-A gradient isolation."""
from __future__ import annotations
import json, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/"tools")]
from audit_p4_k1_oracle_utility import DeterministicVisATrainDataset, _sha256
from run_p4v_short64 import build
from run_p4v_v15_paired import phase4_forward
from run_p4v_v1a_paired import base_and_isolated
from utils import configure_canonical_fp32

def norm(module):
    values=[p.grad.detach().float().norm() for p in module.parameters() if p.grad is not None]
    return 0.0 if not values else float(torch.stack(values).norm())

def main():
    config_path=ROOT/"runs/phase4/k1/short64_seed0_attempt5/config.json";manifest_path=ROOT/"runs/phase4/k1/stage1_7r/visa_train_audit_manifest.json";config=json.loads(config_path.read_text());manifest=json.loads(manifest_path.read_text());configure_canonical_fp32();torch.manual_seed(1701);device=torch.device("cuda:0");model=build(config,device);raw=next(iter(DataLoader(DeterministicVisATrainDataset(manifest,config["img_size"]),batch_size=1,shuffle=False,num_workers=0)));image=raw["image"].to(device).float();mask=raw["mask"].to(device).float();label=raw["label"].to(device);classes=list(raw["class_name"]);model.train();model.clipmodel.eval()
    base_pred=phase4_forward(model,image,mask,label,classes,config,"OFF")[0];zero_delta=phase4_forward(model,image,mask,label,classes,config,"ZERO_DELTA")[0];zero_gate=phase4_forward(model,image,mask,label,classes,config,"ZERO_GATE")[0]
    base,corr,_,_,_=base_and_isolated(model,image,mask,label,classes,config);model.zero_grad(set_to_none=True);corr.backward();corr_base_grad=norm(model.image_adapter)+norm(model.text_adapter);corr_cops_grad=norm(model.h6.conditional_semantic_core);corr_visual_grad=norm(model.h6.visual_adapter);router_grad=norm(model.h6.router);legacy_grad=norm(model.h6.semantic_core)
    model.zero_grad(set_to_none=True);base.backward();base_main_grad=norm(model.image_adapter)+norm(model.text_adapter)
    exact={"off":0.0,"zero_delta":float((zero_delta-base_pred).abs().max().detach()),"zero_gate":float((zero_gate-base_pred).abs().max().detach())};checks={"off_exact":exact["off"]==0.0,"zero_delta_exact":exact["zero_delta"]==0.0,"zero_gate_exact":exact["zero_gate"]==0.0,"finite":bool(torch.isfinite(base) and torch.isfinite(corr)),"corr_branch_base_grad_exact_zero":corr_base_grad==0.0,"base_main_grad_nonzero":base_main_grad>0,"cops_grad_nonzero":corr_cops_grad>0,"visual_grad_nonzero":corr_visual_grad>0,"router_grad_zero":router_grad==0.0,"legacy_factor_grad_zero":legacy_grad==0.0,"act_absent":model.h6.act_head is None}
    passed=all(checks.values());report={"decision":"V1A_GRADIENT_ISOLATION_ZERO_STEP_PASS" if passed else "V1A_GRADIENT_ISOLATION_ZERO_STEP_FAIL","checks":checks,"exact_errors":exact,"gradients":{"corr_branch_base":corr_base_grad,"corr_cops":corr_cops_grad,"corr_visual":corr_visual_grad,"base_main":base_main_grad,"router":router_grad,"legacy":legacy_grad},"provenance":{"config_sha256":_sha256(config_path),"manifest_sha256":_sha256(manifest_path),"initialization":"fresh OpenAI CLIP only","precision":"strict FP32","optimizer_steps":0}}
    out=ROOT/"runs/phase4v/v1a/V1A_ZERO_STEP.json";out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report,indent=2))
if __name__=="__main__":main()
