#!/usr/bin/env python3
"""Source-only matched A_SHORT/A_FUNC_SHORT bounded mechanism runner.

The control and candidate restore the same historical shared-E1 full state.
The candidate's sole extra term is the frozen-E1 stage-2/3 token-cosine loss.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

REPO = Path(__file__).resolve().parents[1]
os.sys.path.insert(0, str(REPO))
from dataset import get_text_and_image_dataset
from h2_clean.contract import (
    SafeImageAdapterAnchor, EpochWorkerInit, apply_family_safe_anchor_budget,
    make_dataloader_generator, sha256_file,
)
from h2_clean.functional_anchor import (
    FUNCTIONAL_ANCHOR_STAGES, bounded_config_mismatches, freeze_e1_teacher,
    functional_feature_anchor_loss, lambda_from_gradient_norms, require_source_only,
)
from h2_clean.precision import PrecisionPolicy
from model.adapter import ACDCLIP
from model.clip import create_model
from train import (
    apply_soft_prompt_lr_policy, calculate_seg_loss, get_dfg_beta_for_epoch,
    get_hybrid_alpha_for_epoch, has_non_finite_grad,
    optimizer_state_is_finite,
)
from utils import get_hybrid_soft_prompt_single_class_text_embedding

IMG = 518
SEED = 0
TARGET_AUX_RATIO = 0.10
CALIBRATION_BATCHES = 8
MAX_ATTEMPTS = 500
ANCHOR_LAMBDA = 0.0021633926715180626


def json_dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(path.name + ".tmp")
    staging.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(staging, path)


def tensor_hash(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def make_model(payload: dict, device: torch.device, *, checkpointing: bool = True) -> ACDCLIP:
    clip = create_model("ViT-L-14-336", img_size=IMG, device=device, pretrained="openai", require_pretrained=True)
    if checkpointing:
        clip.set_grad_checkpointing(True)
    model = ACDCLIP(
        clip_model=clip, n_groups=3, image_adapt_weight=.2, text_adapt_weight=.2,
        conv_lora_rank=8, conv_lora_alpha=2., conv_kernel_size_list=[3, 5],
        lora_rank=16, lora_alpha=2., dfg_mode="attn", dfg_attn_dim=256,
        dfg_attn_tau=8., use_ss2d_dfg=True, dfg_gamma_max=.2,
        dfg_ss2d_fusion="weight_residual", dfg_beta=.1,
        dfg_beta_schedule="warmup010", dfg_beta_target=.1,
        dfg_beta_current=.1, dfg_weight_residual_fp32=True,
        use_soft_prompt=False, soft_prompt_ctx_len=4,
        soft_prompt_init="phrase", soft_prompt_init_phrase="a photo of a",
    ).to(device).eval()
    for module in model.modules():
        if hasattr(module, "enable_fp16_numerical_islands"):
            module.enable_fp16_numerical_islands = False
    model.image_adapter.load_state_dict(payload["image_adapter"], strict=True)
    model.text_adapter.load_state_dict(payload["text_adapter"], strict=True)
    model.soft_prompt.load_state_dict(payload["soft_prompt"], strict=True)
    model.dfg_beta = float(payload["dfg_beta_current"])
    model.prompt_mode = "hybrid"; model.use_hybrid_soft_prompt = True; model.use_soft_prompt = False
    model.hybrid_alpha_current = 0.0; model.hybrid_alpha_max = .2; model.soft_prompt_freeze_epochs = 3
    model.requires_grad_(False); model.image_adapter.requires_grad_(True); model.text_adapter.requires_grad_(True)
    model.soft_prompt.requires_grad_(False)
    return model


def make_optimizer(model: ACDCLIP, payload: dict):
    optimizer = torch.optim.Adam([
        {"name": "text_adapter", "params": model.text_adapter.parameters(), "lr": .0005},
        {"name": "image_adapter", "params": model.image_adapter.parameters(), "lr": .001},
        {"name": "soft_prompt", "params": model.soft_prompt.parameters(), "lr": 0., "constant_lr": .00005},
    ])
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler = StepLR(optimizer, step_size=1, gamma=.9); scheduler.load_state_dict(payload["scheduler_state"])
    scaler = torch.amp.GradScaler("cuda", enabled=True); scaler.load_state_dict(payload["scaler_state"])
    return optimizer, scheduler, scaler


def loader_for_epoch(dataset, epoch: int):
    generator = make_dataloader_generator(SEED)
    generator.manual_seed(SEED + 104729 * epoch)
    worker = EpochWorkerInit(SEED); worker.set_epoch(epoch)
    return DataLoader(dataset, batch_size=6, shuffle=True, num_workers=0, pin_memory=True,
                      generator=generator, worker_init_fn=worker)


def configure_epoch(model, optimizer, epoch: int):
    model.eval(); model.image_encoder.eval(); model.clipmodel.eval()
    frozen = epoch <= 3
    model.hybrid_alpha_current = get_hybrid_alpha_for_epoch(epoch, hybrid_alpha_max=.2, soft_prompt_freeze_epochs=3)
    model.soft_prompt.requires_grad_(not frozen); model.text_adapter.requires_grad_(True)
    apply_soft_prompt_lr_policy(optimizer, frozen)
    model.set_dfg_beta(get_dfg_beta_for_epoch(epoch, "warmup010", .1, .1))
    return frozen


def text_and_task(model, image, mask, label, class_names, device, policy):
    by_class = {}; kg_losses = []; k_losses = []
    for class_name in sorted(set(class_names)):
        text, kg, _, components = get_hybrid_soft_prompt_single_class_text_embedding(
            model, "VisA", class_name, device, return_kg=True, return_components=True)
        # Mirrors the existing K regularizer's active E1/E2 branch shape.
        from train import compute_hybrid_k_regularization
        k_loss, _ = compute_hybrid_k_regularization(model, components["hard_text"], components["soft_text"], model.hybrid_alpha_current)
        by_class[class_name] = text; kg_losses.append(kg); k_losses.append(k_loss)
    text = torch.stack([by_class[name] for name in class_names]).permute(1, 0, 2, 3)
    kg_loss = torch.stack(kg_losses).mean(); k_loss = torch.stack(k_losses).mean()
    with policy.autocast(device):
        seg_tokens, det_tokens = model(image)
        seg = torch.stack(seg_tokens); det = torch.stack(det_tokens)
        cls = torch.stack([torch.matmul(det[i].unsqueeze(1), text[i]).squeeze(1) for i in range(3)]).mean(0)
        cls_loss = F.cross_entropy(cls, label)
        pred = model.vision_text_fusion_gate_seg(seg, text)
        seg_loss = calculate_seg_loss(pred, mask)
        base = cls_loss + seg_loss + .01 * kg_loss + .002 * k_loss
    return base, seg, {"classification": cls_loss, "segmentation": seg_loss, "kg": kg_loss, "k": k_loss}


def image_grad_norm(loss, model):
    params = [p for p in model.image_adapter.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    return float(sum(g.detach().float().square().sum() for g in grads if g is not None).sqrt().cpu())


def calibrate(payload, reference_path: Path, output: Path):
    device = torch.device("cuda:0"); policy = PrecisionPolicy("fp16")
    student = make_model(payload, device); teacher = freeze_e1_teacher(make_model(payload, device, checkpointing=False))
    optimizer, _, scaler = make_optimizer(student, payload); anchor = SafeImageAdapterAnchor.from_checkpoint(reference_path, device)
    dataset = get_text_and_image_dataset("VisA", IMG, "train")
    task_norms=[]; func_norms=[]; rows=[]; epoch=2; configure_epoch(student, optimizer, epoch)
    for batch_idx, batch in enumerate(loader_for_epoch(dataset, epoch)):
        if batch_idx >= CALIBRATION_BATCHES: break
        image=batch["image"].to(device); mask=batch["mask"].to(device); label=batch["label"].to(device)
        base, _, terms=text_and_task(student,image,mask,label,batch["class_name"],device,policy)
        # Deliberately discarded task-only shadow update resolves zero cosine
        # gradient at the exact E1 equality point; it never enters either arm.
        optimizer.zero_grad(set_to_none=True); scaler.scale(base).backward(); scaler.unscale_(optimizer)
        if has_non_finite_grad(optimizer): raise FloatingPointError("non-finite calibration task gradient")
        scaler.step(optimizer); scaler.update()
        with torch.no_grad(), policy.autocast(device): teacher_seg,_=teacher(image)
        base2, student_seg, _=text_and_task(student,image,mask,label,batch["class_name"],device,policy)
        func, stage_metrics=functional_feature_anchor_loss(student_seg, teacher_seg)
        task_norm=image_grad_norm(base2, student); func_norm=image_grad_norm(func, student)
        task_norms.append(task_norm); func_norms.append(func_norm)
        rows.append({"batch":batch_idx,"task_grad_norm":task_norm,"functional_grad_norm":func_norm,
                     "unweighted_func_task_ratio":func_norm/task_norm,**stage_metrics})
    lam, raw = lambda_from_gradient_norms(task_norms,func_norms,target_ratio=TARGET_AUX_RATIO)
    for row in rows: row["lambda_func"]=lam; row["effective_func_task_ratio"]=lam*row["unweighted_func_task_ratio"]
    with output.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    json_dump(output.with_suffix(".json"), {"LAMBDA_FUNC_CALIBRATION_BATCHES":CALIBRATION_BATCHES,"LAMBDA_FUNC":lam,
        "MEDIAN_INITIAL_FUNC_TASK_GRAD_RATIO":raw,"target_auxiliary_ratio":TARGET_AUX_RATIO,
        "shadow_trajectory_discarded":True,"rows":rows})
    print(json.dumps({"status":"PASS","lambda_func":lam,"median_ratio":raw}))


def run_arm(payload, reference_path: Path, root: Path, arm: str, lambda_func: float):
    candidate=arm=="A_FUNC_SHORT"; device=torch.device("cuda:0"); policy=PrecisionPolicy("fp16")
    model=make_model(payload,device); teacher=None
    if candidate: teacher=freeze_e1_teacher(make_model(payload,device,checkpointing=False))
    optimizer,scheduler,scaler=make_optimizer(model,payload); anchor=SafeImageAdapterAnchor.from_checkpoint(reference_path,device)
    dataset=get_text_and_image_dataset("VisA",IMG,"train"); attempted=successful=loss_skips=grad_skips=0; rows=[]; ids=[]
    config={"precision_protocol":"HISTORICAL_MIXED_FP16_FP32_V1","shared_e1_sha256":sha256_file(reference_path),
        "use_safe_anchor":True,"anchor_lambda":ANCHOR_LAMBDA,"anchor_gradient_budget":True,"anchor_family_budget":.1,
        "functional_anchor_stages":[2,3],"use_functional_feature_anchor":candidate,
        "functional_anchor_lambda":lambda_func if candidate else 0.,"functional_anchor_reference_sha256":sha256_file(reference_path) if candidate else None,
        "dataset":"VisA","seed":SEED,"max_attempts":MAX_ATTEMPTS}
    for epoch in (2,3):
        frozen=configure_epoch(model,optimizer,epoch)
        for batch_idx,batch in enumerate(loader_for_epoch(dataset,epoch)):
            if attempted>=MAX_ATTEMPTS: break
            attempted+=1; image=batch["image"].to(device); mask=batch["mask"].to(device); label=batch["label"].to(device)
            ids.append({"epoch":epoch,"batch":batch_idx,"image_sha256":tensor_hash(image),"mask_sha256":tensor_hash(mask),"files":list(batch["file_name"])})
            base,seg,terms=text_and_task(model,image,mask,label,batch["class_name"],device,policy)
            functional=torch.zeros((),device=device); fmetrics={"functional_anchor_total":0.,"functional_anchor_stage2":0.,"functional_anchor_stage3":0.}
            if candidate:
                with torch.no_grad(), policy.autocast(device): teacher_seg,_=teacher(image)
                functional,fmetrics=functional_feature_anchor_loss(seg,teacher_seg)
            total=base+lambda_func*functional
            anchor_loss=anchor.loss(model.image_adapter)
            if not torch.isfinite(total) or not torch.isfinite(anchor_loss): loss_skips+=1; continue
            optimizer.zero_grad(set_to_none=True); scaler.scale(total).backward(retain_graph=True); scaler.unscale_(optimizer)
            if has_non_finite_grad(optimizer): grad_skips+=1; scaler.update(); optimizer.zero_grad(set_to_none=True); continue
            pairs=[(name,p) for name,p in sorted(model.image_adapter.named_parameters()) if p.requires_grad]
            names=[x[0] for x in pairs]; pars=[x[1] for x in pairs]
            task_grads=torch.autograd.grad(total,pars,retain_graph=True,allow_unused=True)
            anchor_grads=torch.autograd.grad(anchor_loss,pars,allow_unused=True)
            ametrics=apply_family_safe_anchor_budget(model.image_adapter,sorted(model.named_parameters()),
                task_gradients=dict(zip(names,task_grads)),raw_anchor_gradients=dict(zip(names,anchor_grads)),
                anchor_lambda=ANCHOR_LAMBDA,rho=.1,total_trainable_parameters=None)
            torch.nn.utils.clip_grad_norm_(model.image_adapter.parameters(),1.0); torch.nn.utils.clip_grad_norm_(model.text_adapter.parameters(),1.0)
            if not frozen: torch.nn.utils.clip_grad_norm_(model.soft_prompt.parameters(),1.0)
            scaler.step(optimizer); scaler.update(); successful+=1
            if not optimizer_state_is_finite(optimizer): raise FloatingPointError("non-finite optimizer state")
            rows.append({"epoch":epoch,"batch":batch_idx,"attempt":attempted,"successful":successful,"base_task_loss":float(base.detach()),
                "functional_loss":float(functional.detach()),"total_loss":float(total.detach()),"anchor_loss":float(anchor_loss.detach()),
                "task_gradient_norm":image_grad_norm(total,model),"functional_gradient_norm":image_grad_norm(functional,model) if candidate else 0.,
                "functional_effective_ratio":(lambda_func*image_grad_norm(functional,model)/max(image_grad_norm(total,model),1e-12)) if candidate else 0.,
                "safe_anchor_global_effective_ratio":ametrics["global_effective_ratio"],"segmentation_loss":float(terms["segmentation"].detach()),
                "classification_loss":float(terms["classification"].detach()),**fmetrics})
        if attempted>=MAX_ATTEMPTS: break
        scheduler.step(); apply_soft_prompt_lr_policy(optimizer,epoch<=3)
    if attempted!=MAX_ATTEMPTS: raise RuntimeError(f"expected {MAX_ATTEMPTS} attempts, got {attempted}")
    out=root/arm; out.mkdir(parents=True,exist_ok=True)
    torch.save({"model_state":{"image_adapter":model.image_adapter.state_dict(),"text_adapter":model.text_adapter.state_dict(),"soft_prompt":model.soft_prompt.state_dict()},
        "optimizer_state":optimizer.state_dict(),"scheduler_state":scheduler.state_dict(),"scaler_state":scaler.state_dict(),"config":config,
        "attempted_steps":attempted,"successful_steps":successful,"nonfinite_loss_skips":loss_skips,"nonfinite_grad_skips":grad_skips},out/"final.pth")
    json_dump(out/"summary.json",{"arm":arm,"config":config,"attempted_steps":attempted,"successful_steps":successful,"nonfinite_loss_skips":loss_skips,"nonfinite_grad_skips":grad_skips,"batch_identities":ids,"metrics":rows})
    print(json.dumps({"status":"PASS","arm":arm,"attempted":attempted,"successful":successful,"grad_skips":grad_skips}))


def main():
    p=argparse.ArgumentParser(); p.add_argument("--mode",choices=("calibrate","run"),required=True); p.add_argument("--arm",choices=("A_SHORT","A_FUNC_SHORT")); p.add_argument("--shared-e1",required=True); p.add_argument("--root",required=True); p.add_argument("--calibration-csv",default=str(REPO/"audit/H2_FUNC_ANCHOR_CALIBRATION.csv")); p.add_argument("--lambda-func",type=float)
    a=p.parse_args(); require_source_only("VisA"); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    if not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    shared=Path(a.shared_e1); payload=torch.load(shared,map_location="cpu",weights_only=False)
    if payload.get("epoch")!=1 or payload.get("precision") not in ("amp","fp16"): raise RuntimeError("shared E1 identity mismatch")
    if a.mode=="calibrate": calibrate(payload,shared,Path(a.calibration_csv)); return
    if a.arm is None or a.lambda_func is None or a.lambda_func<=0: raise ValueError("run requires arm and positive frozen lambda")
    run_arm(payload,shared,Path(a.root),a.arm,a.lambda_func)


if __name__=="__main__": main()
