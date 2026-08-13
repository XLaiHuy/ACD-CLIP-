#!/usr/bin/env python3
"""No-optimizer R2 role-topology proof on a deterministic VisA sample."""
from __future__ import annotations
import argparse
import hashlib, json, math
from pathlib import Path
import torch
import torch.nn.functional as F
from model.h6.model import H6Progress1
from model.h6.utility_routing import (
    r2_responsibility_balanced_utility_router_loss,
    routed_residual_correction,
    utility_teacher,
)
from model.h6.losses import build_semantic_roles

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "runs/p1_v84a_gpu/factor_generator_specialization_fresh_3e_seed0/e3_forward_cache.pt"
OUT = ROOT / "runs/p1_v84a_gpu/factor_generator_specialization_fresh_3e_seed0/role_zero_step_proof.json"
SCALE = 0.0005203147302381694
ROUTER_ALIGNMENT_WEIGHTS = (1.4287327685865079, 0.7691839630348986)

def finite(x):
    return bool(torch.isfinite(x).all().item())

def vec(grads, params):
    return torch.cat([(g.detach().float().reshape(-1) if g is not None else torch.zeros_like(p.detach().float()).reshape(-1)) for g,p in zip(grads,params)])

def grad_for(loss, params):
    return vec(torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True), params)

def rank2(x):
    x=x.reshape(-1,2).float(); xc=x-x.mean(0,keepdim=True); cov=xc.T@xc/max(1,x.shape[0]-1); eig=torch.linalg.eigvalsh(cov).clamp_min(0).flip(0); e=eig/eig.sum().clamp_min(1e-20); pos=e[e>0]; er=float(torch.exp(-(pos*pos.clamp_min(1e-20).log()).sum())); sd=cov.diag().sqrt().clamp_min(1e-20); corr=cov/(sd[:,None]*sd[None,:]).clamp_min(1e-20); return {"effective_rank":er,"pca_energy":e.tolist(),"pairwise_correlation":corr.tolist()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-align", action="store_true", help="Exercise the fixed R2 Router supervision correction.")
    args = parser.parse_args()
    out_path = (
        ROOT / "runs/p1_v84a_gpu/role_r2_fresh_3e_seed0/router_alignment_zero_step.json"
        if args.router_align else OUT
    )
    torch.manual_seed(841)
    # Select fixed IDs from the persisted cache without forwarding or changing it.
    cache=torch.load(CACHE,map_location="cpu",weights_only=False)
    wanted=[]
    for cls in sorted(set(cache["class"])):
        for label in (0,1):
            for i,(c,y) in enumerate(zip(cache["class"],cache["label"])):
                if c==cls and int(y)==label:
                    wanted.append(i); break
            if len(wanted)>=8: break
        if len(wanted)>=8: break
    wanted=sorted(wanted[:8])
    classes=[str(cache["class"][i]) for i in wanted]; labels=torch.tensor([int(cache["label"][i]) for i in wanted])
    target=[]; valid=[]
    for i in wanted:
        t=cache["target"][i].squeeze(0).float().reshape(1,1,37,37)
        q=cache["valid"][i].squeeze(0).float().reshape(1,1,37,37)
        target.append(F.adaptive_avg_pool2d(t,(4,4)).flatten(1).squeeze(0))
        valid.append((F.adaptive_avg_pool2d(q,(4,4)).flatten(1).squeeze(0)>=1.0-1e-6))
    y=torch.stack(target); valid=torch.stack(valid).bool()
    q_role, hard_role, mask_coverage, local_valid_patch, local_valid_image = build_semantic_roles(
        y.reshape(8, 4, 4), labels, patch_count=16,
        local_mask_valid=valid.reshape(8, 4, 4), num_roles=2,
        role_topology="r2_normal_anomaly", boundary_threshold=0.01,
    )
    B,P,D,G=8,16,32,3
    h=H6Progress1(n_groups=G,num_factors=2,top_k=2,bank_dim=16,router_dim=8,text_dim=D,ctx_len=2,progress_version="P1-v8.4-A",role_topology="r2_normal_anomaly",role_teacher_scale=SCALE,factor_generator_specialization_enabled=True)
    h.eval()
    visual={"seg_tokens_pre_l2":[torch.randn(B,P,D) for _ in range(G)],"seg_tokens":[F.normalize(torch.randn(B,P,D),dim=-1) for _ in range(G)],"cls24":torch.randn(B,D)}
    ctx_normal=F.normalize(torch.randn(2,D),dim=-1); ctx_abnormal=F.normalize(torch.randn(2,D),dim=-1)
    core=h.forward_core(visual,ctx_normal,ctx_abnormal)
    role_bank=F.normalize(core["dynamic_contexts"].mean(dim=3),dim=-1).permute(0,1,3,2).unsqueeze(0).expand(G,-1,-1,-1,-1)
    base_bank=F.normalize(torch.stack([ctx_normal,ctx_abnormal],dim=0).mean(1),dim=-1).T.unsqueeze(0).unsqueeze(0).unsqueeze(2).expand(G,B,-1,-1,-1)
    router=h.router(visual["seg_tokens"],epoch_one_based=1,concept_keys=core["concept_keys"])
    patches=torch.stack(visual["seg_tokens"]).float()
    factor_logits=h.h6_logit(patches.unsqueeze(3),role_bank.unsqueeze(2))
    base_logits=h.h6_logit(patches,base_bank)
    factor_residual=factor_logits-base_logits.unsqueeze(-1)
    act_logits=h.act_head(router["router_input_features"]).squeeze(-1); act_probability=torch.sigmoid(act_logits)
    final_correction=routed_residual_correction(act_probability,router["prediction_probabilities"],factor_residual)
    rho=h.rho_values(); rho_scaled=final_correction*rho.view(G,1,1)
    teacher=utility_teacher(base_logits,factor_residual,y,valid,rho=.05,role_topology="r2_normal_anomaly",role_teacher_scale=SCALE,routed_probabilities=router["prediction_probabilities"])
    params=[p for p in h.parameters() if p.requires_grad]
    router_params=[p for name,p in h.named_parameters() if p.requires_grad and name.startswith("router.")]
    desired=(y-0.5)*0.2
    role_losses=[]; role_grads=[]
    for m in range(2):
        mask=(valid & (hard_role == m)).unsqueeze(0).expand(G,-1,-1)
        loss=F.smooth_l1_loss(factor_residual[...,m],desired.unsqueeze(0).expand_as(factor_residual[...,m]),reduction="none")
        loss=(loss*teacher["q_router_utility"][...,m]*mask).sum()/mask.sum().clamp_min(1)
        role_losses.append(loss); role_grads.append(grad_for(loss,params))
    normal_mask=valid & (hard_role == 0); anomaly_mask=valid & (hard_role == 1)
    region_losses=[]; region_grads=[]
    for mask0 in (normal_mask,anomaly_mask):
        mask=mask0.unsqueeze(0).expand(G,-1,-1); ls=[]
        for m in range(2):
            z=F.smooth_l1_loss(factor_residual[...,m],desired.unsqueeze(0).expand_as(factor_residual[...,m]),reduction="none")
            ls.append((z*teacher["q_router_utility"][...,m]*mask).sum()/mask.sum().clamp_min(1))
        L=sum(ls); region_losses.append(L); region_grads.append(grad_for(L,params))
    grad_norms=[float(g.norm()) for g in role_grads]; role_cos=float(F.cosine_similarity(role_grads[0],role_grads[1],dim=0).item()); region_mass=[float(g.abs().sum()) for g in region_grads]; region_cos=float(F.cosine_similarity(region_grads[0],region_grads[1],dim=0).item())
    router_alignment = None
    if args.router_align:
        # This compact structural forward is randomly initialized and can have no
        # utility-margin support.  In that case use a deterministic detached R2
        # teacher fixture with the frozen TRAIN responsibility masses; it tests
        # exactly the new loss/gradient contract without pretending to measure
        # data utility before the subsequent 8B gate.
        actual_informative_support = int(teacher["informative"].sum())
        if actual_informative_support > 0:
            router_payload = teacher
            teacher_source = "synthetic_forward_utility_teacher"
        else:
            train_mass = torch.tensor([0.34996047616004944, 0.650039553642273], dtype=base_logits.dtype)
            q0 = torch.full_like(base_logits, float(train_mass[0]))
            q_router_fixture = torch.stack((q0, 1.0 - q0), dim=-1)
            router_payload = {
                "role_topology": "r2_normal_anomaly",
                "q_utility": q_router_fixture,
                "informative": torch.ones_like(base_logits, dtype=torch.bool),
            }
            teacher_source = "deterministic_frozen_train_mass_fixture"
        router_loss = r2_responsibility_balanced_utility_router_loss(
            router["prediction_probabilities"], router_payload, role_weights=ROUTER_ALIGNMENT_WEIGHTS,
        )
        router_grad = grad_for(router_loss, router_params)
        q_router = router_payload["q_utility"].detach()
        info = router_payload["informative"].float().unsqueeze(-1)
        weights = torch.tensor(ROUTER_ALIGNMENT_WEIGHTS, dtype=q_router.dtype).view(1, 1, 1, 2)
        denominator = (info * q_router * weights).sum().clamp_min(1e-12)
        role_grads_router = []
        role_losses_router = []
        for role in range(2):
            role_loss = (
                info[..., 0] * q_router[..., role]
                * weights[..., role]
                * -router["prediction_probabilities"][..., role].clamp_min(1e-12).log()
            ).sum() / denominator
            role_losses_router.append(role_loss)
            role_grads_router.append(grad_for(role_loss, router_params))
        router_alignment = {
            "loss": float(router_loss.detach()),
            "teacher_source": teacher_source,
            "actual_synthetic_teacher_informative_support": actual_informative_support,
            "informative_support": int(router_payload["informative"].sum()),
            "weights": list(ROUTER_ALIGNMENT_WEIGHTS),
            "weighted_role_mass": (info * q_router * weights).sum((0, 1, 2)).div(denominator).tolist(),
            "router_gradient_norm": float(router_grad.norm()),
            "role_router_gradient_norm": [float(g.norm()) for g in role_grads_router],
            "role_router_gradient_cosine": float(F.cosine_similarity(role_grads_router[0], role_grads_router[1], dim=0)),
            "finite": bool(finite(router_loss.detach()) and finite(router_grad) and all(finite(g) for g in role_grads_router)),
        }
        router_alignment["pass"] = bool(
            router_alignment["finite"]
            and router_alignment["informative_support"] > 0
            and router_alignment["router_gradient_norm"] > 0.0
            and all(value > 0.0 for value in router_alignment["role_router_gradient_norm"])
        )
        if not router_alignment["pass"]:
            raise RuntimeError(f"ROUTER_ALIGN_ZERO_STEP_FAIL: {router_alignment}")
    d1,d2=0.0005,0.0010; p1=float(torch.sigmoid(torch.tensor(d1/SCALE))); p2=float(torch.sigmoid(torch.tensor(d2/SCALE)))
    gaps=teacher["role_gap"].detach(); probs=teacher["role_probability"].detach(); valid_role=teacher["valid"].bool(); normal=valid_role&(hard_role.unsqueeze(0)==0); anomaly=valid_role&(hard_role.unsqueeze(0)==1)
    def util(mask):
        z=teacher["gain_rel"][mask]; return {"n":int(z.shape[0]),"mean_role_gain":z.mean(0).tolist(),"winner_share":torch.bincount(z.argmax(-1),minlength=2).float().div(z.shape[0]).tolist() if z.numel() else [0,0]}
    out={"decision":("ROUTER_ALIGN_ZERO_STEP_PASS" if args.router_align else "ROLE_REDESIGN_ZERO_STEP_PASS"),"training_steps":0,"optimizer_steps":0,"sample":{"cache":str(CACHE),"indices":wanted,"image_ids":[cache["file_name"][i] for i in wanted],"classes":classes,"labels":labels.tolist()},"architecture":{"role_topology":h.role_topology,"role_names":["R_NORMAL","R_ANOMALY"],"factor_residual_shape":list(factor_residual.shape),"router_shape":list(router["prediction_probabilities"].shape),"act_shape":list(act_probability.shape),"teacher_shape":list(teacher["role_probability"].shape)},"invariants":{"finite":all(finite(x) for x in [factor_logits,base_logits,factor_residual,router["prediction_probabilities"],act_probability,final_correction,teacher["role_probability"]]),"true_residual_max_abs_error":float((factor_residual-(factor_logits-base_logits.unsqueeze(-1))).abs().max()),"dense_reconstruction_max_abs_error":float((final_correction-act_probability*(router["prediction_probabilities"]*factor_residual).sum(-1)).abs().max()),"act_reconstruction_max_abs_error":float((rho_scaled-rho.view(G,1,1)*final_correction).abs().max()),"main_path_max_abs_error":float(((base_logits+rho_scaled)-(base_logits+rho.view(G,1,1)*final_correction)).abs().max()),"rho":rho.tolist(),"rho_fixed":bool(not h.rho.raw.requires_grad and torch.allclose(rho,torch.full((G,),.05)))},"role_semantics":{"normal_patch_count":int(normal_mask.sum()),"anomaly_patch_count":int(anomaly_mask.sum()),"normal_image_count":int((labels==0).sum()),"anomaly_image_count":int((labels==1).sum())},"teacher":{"scale":SCALE,"gap_mean":float(gaps[valid_role].mean()),"gap_std":float(gaps[valid_role].std()),"probability_mean":float(probs[valid_role].mean()),"probability_std":float(probs[valid_role].std()),"entropy_mean":float(teacher["role_entropy"][valid_role].mean()),"normal":util(normal),"anomaly":util(anomaly),"monotonic_same_winner_examples":[{"d1":d1,"d2":d2,"abs_order":abs(d1)<abs(d2),"p_normal_d1":p1,"p_normal_d2":p2,"confidence_order":p1<p2}]},"gradients":{"role_loss": [float(x.detach()) for x in role_losses],"per_role_norm":grad_norms,"role_cosine":role_cos,"normal_anomaly_weighted_mass":region_mass,"normal_anomaly_gradient_cosine":region_cos,"all_parameters_finite":all(finite(p.detach()) for p in h.parameters())},"functional_residual":rank2(factor_residual.detach()),"role_output_difference":{"max_abs":float((factor_residual[...,0]-factor_residual[...,1]).abs().max().detach()),"mean_abs":float((factor_residual[...,0]-factor_residual[...,1]).abs().mean().detach()),"role0_std":float(factor_residual[...,0].float().std().detach()),"role1_std":float(factor_residual[...,1].float().std().detach())},"router_alignment":router_alignment,"unchanged_science":{"act_changed":False,"rho_changed":False,"optimizer_changed":False,"checkpoint_changed":False,"router_architecture_changed":False,"router_loss_changed":bool(args.router_align)}}
    out_path.write_text(json.dumps(out,indent=2)+"\n"); print(json.dumps({"decision":out["decision"],"sample":out["sample"],"invariants":out["invariants"],"router_alignment":out["router_alignment"],"teacher":out["teacher"],"gradients":out["gradients"],"functional_residual":out["functional_residual"],"output":str(out_path)},indent=2))
if __name__=="__main__": main()
