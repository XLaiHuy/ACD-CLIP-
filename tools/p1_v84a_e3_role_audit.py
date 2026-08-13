#!/usr/bin/env python3
"""One-pass inference-only E3 audit/cache for the trained R2 topology."""
from __future__ import annotations
import argparse, json, math, time, sys
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import get_text_and_image_dataset
from model.h6.utility_routing import build_patch_targets, utility_teacher
from tools.audit_p1_v84a_post300 import _IndexedDataset, _seed, _state_hash, _effective_rank, _functional_correlation, _region_utility
from tools.audit_p1_v83_semantics import _model_from_checkpoint
from utils import get_phase2b_global_text_features, make_dataloader_generator, seed_worker

ROOT=Path(__file__).resolve().parents[1]
SCALE=0.0005203147302381694

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--progress-every',type=int,default=100)
    args=ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    torch.set_float32_matmul_precision('highest'); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; _seed(args.seed)
    device=torch.device('cuda:0'); ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=False); cfg=ckpt.get('h6_config',{})
    required={'progress_version':cfg.get('progress_version')=='P1-v8.4-A','role_topology':cfg.get('role_topology')=='r2_normal_anomaly','num_factors':cfg.get('num_factors')==2,'role_scale':abs(float(cfg.get('role_teacher_scale',0))-SCALE)<1e-15,'seed':ckpt.get('seed')==args.seed,'img_size':ckpt.get('img_size')==518,'batch_size':ckpt.get('batch_size')==1,'grad_accum_steps':ckpt.get('grad_accum_steps')==6,'precision':ckpt.get('precision')=='fp32','tf32_off':ckpt.get('tf32_enabled') is False,'amp_off':ckpt.get('amp_enabled') is False,'rho_fixed':cfg.get('rho_fixed') is True}
    if not all(required.values()): raise RuntimeError(f'contract failure: {[k for k,v in required.items() if not v]}')
    model=_model_from_checkpoint(ckpt,device); model.requires_grad_(False); model.eval(); model.clipmodel.eval(); before=_state_hash(model); grad_before=all(p.grad is None for p in model.parameters())
    ds=_IndexedDataset(get_text_and_image_dataset('VisA',518,'train')); loader=DataLoader(ds,batch_size=1,shuffle=False,num_workers=0,pin_memory=True,worker_init_fn=seed_worker,generator=make_dataloader_generator(args.seed))
    # Audit-only capture retains Q/K and affine residual outputs from this same forward.
    fields={k:[] for k in ['residual','factor_abs','z0','dense','router_logits','qk_logits','boundary_raw_delta','boundary_delta','qk_dense','act','actual','qk_actual','rho_actual','target','valid']}; classes=[]; image_indices=[]; labels=[]; residual_err=0.; dense_err=0.; act_err=0.; boundary_err=0.; probability_err=0.; started=time.monotonic()
    for bn,s in enumerate(loader,1):
        image=s['image'].to(device,non_blocking=True); mask=s['mask'].to(device,non_blocking=True); local=s['local_mask_valid'].to(device,non_blocking=True); cls=[s['class_name'][0]]
        with torch.inference_mode():
            visual=model(image,return_phase4_features=True); h6=model.h6.build_batch(model,'VisA',cls,visual,hybrid_alpha=float(ckpt['hybrid_alpha_current']),update_load_bias=False)
            seg=torch.stack(visual['seg_tokens'],dim=0); text=get_phase2b_global_text_features(model,'VisA',cls,device,use_hybrid_soft_prompt=True,use_soft_prompt=False).to(dtype=seg.dtype)
            _,_,z0=model.vision_text_fusion_gate_seg(seg,text,img_size=518,h6_patch_logits=h6['h6_logits'],return_details=True)
            P=int(h6['factor_residual_logits'].shape[2]); y,v=build_patch_targets(mask,P,local); residual=h6['factor_residual_logits'].float(); factor_abs=h6['factor_patch_logits'].float(); dense=h6['prediction_probabilities'].float(); act=h6['act_probability'].float(); actual=h6['h6_logits'].float()
            residual_err=max(residual_err,float((residual-(factor_abs-h6['noop_reference_logit'].float().unsqueeze(-1))).abs().max().item()))
            dense_err=max(dense_err,float((h6['h6_logits'].float()-(act*(dense*residual).sum(-1))).abs().max().item()))
            if 'rho_scaled_actual_correction' in h6 and 'rho' in h6:
                rho = h6['rho'].float().view(3, 1, 1)
                expected_rho_scaled = rho * h6['h6_logits'].float()
                act_err = max(
                    act_err,
                    float((h6['rho_scaled_actual_correction'].float() - expected_rho_scaled).abs().max().item()),
                )
            qk_logits=h6['qk_logits'].float(); boundary_raw_delta=h6['boundary_raw_residual_logits'].float(); boundary_delta=h6['boundary_residual_logits'].float(); qk_dense=h6['qk_probabilities'].float(); qk_actual=h6['qk_h6_logits'].float(); rho_actual=h6['rho_scaled_actual_correction'].float()
            boundary_err=max(boundary_err,float((h6['prediction_logits'].float()-(qk_logits+boundary_delta)).abs().max().item()))
            probability_err=max(probability_err,float((dense.sum(dim=-1)-1.0).abs().max().item()))
            fields['residual'].append(residual.cpu()); fields['factor_abs'].append(factor_abs.cpu()); fields['z0'].append(z0.float().cpu()); fields['dense'].append(dense.cpu()); fields['router_logits'].append(h6['prediction_logits'].float().cpu()); fields['qk_logits'].append(qk_logits.cpu()); fields['boundary_raw_delta'].append(boundary_raw_delta.cpu()); fields['boundary_delta'].append(boundary_delta.cpu()); fields['qk_dense'].append(qk_dense.cpu()); fields['act'].append(act.cpu()); fields['actual'].append(actual.cpu()); fields['qk_actual'].append(qk_actual.cpu()); fields['rho_actual'].append(rho_actual.cpu()); fields['target'].append(y.cpu()); fields['valid'].append(v.cpu())
        classes.append(cls[0]); image_indices.append(int(s['dataset_index'].item())); labels.append(int(s['label'].item()))
        if bn%args.progress_every==0: print(json.dumps({'images':bn,'total':len(ds),'elapsed_seconds':round(time.monotonic()-started,1)}),flush=True)
    if len(classes)!=len(ds): raise RuntimeError(f'expected {len(ds)} images, got {len(classes)}')
    residual=torch.cat(fields['residual'],dim=1); factor_abs=torch.cat(fields['factor_abs'],dim=1); z0=torch.cat(fields['z0'],dim=1); dense=torch.cat(fields['dense'],dim=1); router_logits=torch.cat(fields['router_logits'],dim=1); act=torch.cat(fields['act'],dim=1); actual=torch.cat(fields['actual'],dim=1); target=torch.cat(fields['target'],dim=0); valid=torch.cat(fields['valid'],dim=0)
    targets=target.unsqueeze(0).expand_as(z0).float(); validg=valid.unsqueeze(0).expand_as(z0); base_loss=F.binary_cross_entropy_with_logits(z0,targets,reduction='none'); cand=z0.unsqueeze(-1)+0.05*residual; per=F.binary_cross_entropy_with_logits(cand,targets.unsqueeze(-1).expand_as(cand),reduction='none'); gain=(base_loss.unsqueeze(-1)-per)/base_loss.unsqueeze(-1).clamp_min(.1)
    _region_utility.residual=residual; regions={name:_region_utility(mask,base_loss,per,gain,dense,targets,z0) for name,mask in {'overall':validg,'normal':validg&(targets<.5),'anomaly':validg&(targets>=.5)}.items()}
    teacher=utility_teacher(z0,residual,target,valid,rho=.05,router_confidence_mode='margin_rel',router_margin_rel_threshold=.1,router_target_mode='patch_zscore_softmax',role_topology='r2_normal_anomaly',role_teacher_scale=SCALE)
    roleq=teacher['q_router_utility']; rtop=dense.argmax(-1); ttop=roleq.argmax(-1); agree=((rtop==ttop)&validg).float().sum()/validg.sum().clamp_min(1)
    flat=residual[validg]; rank=_effective_rank(flat); corr=_functional_correlation(flat); by_class={}; cls_t=torch.tensor([hash(c)%2147483647 for c in classes])
    for c in sorted(set(classes)):
        sel=torch.tensor([x==c for x in classes]).unsqueeze(0).unsqueeze(-1).expand_as(validg); sel=validg&sel
        by_class[c]={'patches':int(sel.sum()),'winner_shares':_region_utility(region=sel,base_loss=base_loss,per_factor_loss=per,gain_rel=gain,dense=dense,targets=targets,z0=z0)['factor_winner_shares']}
    after=_state_hash(model); grad_after=all(p.grad is None for p in model.parameters())
    cache={**{k:fields[k] for k in fields},'class':classes,'image_index':image_indices,'file_name':[str(ds.dataset.meta[i]['image_path']) for i in image_indices],'label':labels}
    args.output_dir.mkdir(parents=True,exist_ok=True); torch.save(cache,args.output_dir/'e3_forward_cache.pt')
    out={'status':'PASS','decision':'ROLE_R2_E3_CACHE_READY','audit_kind':'ONE_FORWARD_E3','checkpoint':str(args.checkpoint.resolve()),'checkpoint_sha256':__import__('hashlib').sha256(args.checkpoint.read_bytes()).hexdigest(),'git_sha':ckpt.get('git_sha'),'contract':required,'images':len(classes),'valid_group_patches':int(validg.sum()),'optimizer_steps':0,'backward':False,'model_state_unchanged':before==after,'all_grad_none_before':grad_before,'all_grad_none_after':grad_after,'invariants':{'true_residual_max_abs_error':residual_err,'dense_reconstruction_max_abs_error':dense_err,'rho_scaled_reconstruction_max_abs_error':act_err,'boundary_logit_reconstruction_max_abs_error':boundary_err,'router_probability_sum_max_abs_error':probability_err},'regions':regions,'factor_residual':{'effective_rank':rank,'correlation':corr,'winner_shares':regions['overall']['factor_winner_shares']},'teacher':{'scale':SCALE,'entropy_mean':float(teacher['role_entropy'][validg].mean()),'max_probability_mean':float(roleq[validg].max(-1).values.mean()),'router_agreement':float(agree),'winner_shares':regions['overall']['factor_winner_shares']},'per_class':by_class,'runtime_seconds':time.monotonic()-started,'cache':str((args.output_dir/'e3_forward_cache.pt').resolve())}
    (args.output_dir/'e3_role_audit.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps({'status':out['status'],'images':len(classes),'valid_group_patches':int(validg.sum()),'rank':rank,'router_agreement':float(agree),'cache':out['cache'],'runtime_seconds':round(out['runtime_seconds'],1)}),flush=True)
    if not (out['model_state_unchanged'] and out['all_grad_none_before'] and out['all_grad_none_after'] and residual_err==0.0 and dense_err==0.0): raise RuntimeError('E3 invariants failed')

if __name__=='__main__': main()
