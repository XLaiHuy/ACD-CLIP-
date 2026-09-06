#!/usr/bin/env python3
"""Frozen source-only diagnostics for the historical mixed H2 Seed-0 checkpoints.

This program contains no optimizer or training loop.  It evaluates immutable
E1/H-E15/A-E15 checkpoints on VisA test, saves large tensors outside Git, and
writes compact, auditable CSV/JSON summaries in ``audit/``.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage, stats
from torch.utils.data import DataLoader
from torchmetrics.functional import auroc, average_precision

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from dataset import CLASS_NAMES, DOMAINS, get_text_and_image_dataset
from h2_clean.exact_metrics import ExactBinaryAccumulator
from h2_clean.precision import PrecisionPolicy
from model.adapter import ACDCLIP
from model.clip import create_model
from utils import (
    get_hybrid_soft_prompt_single_class_text_embedding,
    get_multiple_adapted_text_embedding,
)

IMG = 518
PATCH = 37
AUDIT = REPO / "audit"
LARGE = Path("/workspace/h2_mixed_generalization_audit_v1")
PROTOCOL = AUDIT / "H2_MIXED_DIAGNOSTIC_PROTOCOL.json"
ARMS = ("E1", "H", "A")
MAP_ARMS = ("H", "A")
CKPT_KEYS = {"E1": "E1", "H": "H_E15", "A": "A_E15"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def json_default(x):
    if isinstance(x, (np.integer, np.floating, np.bool_)):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x).__name__)


def write_csv(path: Path, rows: list[dict]):
    fields = list(dict.fromkeys(k for row in rows for k in row)) if rows else ["status"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows or [{"status": "UNKNOWN"}])


def durable_json(path: Path, payload):
    """Atomically persist a compact recovery artifact before expensive next work."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(payload, f, indent=2, default=json_default)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def disable_later_islands(model):
    count = 0
    for module in model.modules():
        if hasattr(module, "enable_fp16_numerical_islands"):
            module.enable_fp16_numerical_islands = False
            count += 1
    return count


def load_arm(model, payload):
    model.image_adapter.load_state_dict(payload["image_adapter"], strict=True)
    model.text_adapter.load_state_dict(payload["text_adapter"], strict=True)
    model.soft_prompt.load_state_dict(payload["soft_prompt"], strict=True)
    model.dfg_beta = float(payload["dfg_beta_current"])
    model.dfg_attn_tau = float(payload["dfg_attn_tau"])
    model.prompt_mode = payload.get("prompt_mode", "hybrid")
    model.use_hybrid_soft_prompt = bool(payload.get("use_hybrid_soft_prompt", True))
    model.use_soft_prompt = bool(payload.get("use_soft_prompt", False))
    model.hybrid_alpha_current = float(payload.get("hybrid_alpha_current", 0.2))


def binary_metrics(scores, labels):
    x = np.asarray(scores, np.float32).reshape(-1)
    y = np.asarray(labels, np.uint8).reshape(-1)
    if not len(y) or y.min() == y.max():
        return {"auroc": None, "ap": None}
    order = np.argsort(-x, kind="stable")
    sy = y[order]
    tp = np.cumsum(sy, dtype=np.float64); fp = np.cumsum(1-sy, dtype=np.float64)
    positives=float(tp[-1]); negatives=float(fp[-1])
    ap=float((tp/(tp+fp)*sy).sum()/positives)
    sx=x[order]; ends=np.r_[np.flatnonzero(sx[1:] != sx[:-1]), len(sx)-1]
    tpr=np.r_[0.,tp[ends]/positives]; fpr=np.r_[0.,fp[ends]/negatives]
    return {"auroc": float(np.trapezoid(tpr,fpr)), "ap": ap}


def entropy(w):
    return -(w * w.clamp_min(1e-12).log()).sum(-1)


def js_divergence(a, b):
    m = 0.5 * (a + b)
    return 0.5 * ((a * (a.clamp_min(1e-12) / m.clamp_min(1e-12)).log()).sum(-1) +
                  (b * (b.clamp_min(1e-12) / m.clamp_min(1e-12)).log()).sum(-1))


def routing_components(model, vision, text):
    group_text = text.permute(1, 0, 2, 3)
    out = []
    for stage in range(model.n_groups):
        feat = vision[stage]
        vgap = feat.mean(1)
        vss = model.image_adapter["dfg_ss2d_branches"][stage](feat)
        kn = model.image_adapter["vision_text_k"][stage](group_text[..., 0]).float()
        ka = model.image_adapter["vision_text_k"][stage](group_text[..., 1]).float()
        q = model.image_adapter["vision_text_q"][stage]
        qg, qs = q(vgap).float(), q(vss).float()
        scale = math.sqrt(model.dfg_attn_dim) * model.dfg_attn_tau
        sg_n = torch.einsum("bd,bnd->bn", qg, kn) / scale
        sg_a = torch.einsum("bd,bnd->bn", qg, ka) / scale
        ss_n = torch.einsum("bd,bnd->bn", qs, kn) / scale
        ss_a = torch.einsum("bd,bnd->bn", qs, ka) / scale
        g_n, g_a = F.softmax(sg_n, 1), F.softmax(sg_a, 1)
        s_n, s_a = F.softmax(ss_n, 1), F.softmax(ss_a, 1)
        f_n = (1 - model.dfg_beta) * g_n + model.dfg_beta * s_n
        f_a = (1 - model.dfg_beta) * g_a + model.dfg_beta * s_a
        out.append({"score_gap_normal": sg_n, "score_gap_abnormal": sg_a,
                    "score_ss2d_normal": ss_n, "score_ss2d_abnormal": ss_a,
                    "weight_gap_normal": g_n, "weight_gap_abnormal": g_a,
                    "weight_ss2d_normal": s_n, "weight_ss2d_abnormal": s_a,
                    "weight_final_normal": f_n, "weight_final_abnormal": f_a})
    return out


def maps_from_features(model, vision, text):
    b, patches, _ = vision.shape[1:]
    side = int(math.sqrt(patches))
    group_text = text.unsqueeze(1).repeat(1, b, 1, 1).permute(1, 0, 2, 3)
    stage_logits = []
    for stage in range(model.n_groups):
        fused = model._vision_text_attention_fusion(vision[stage], group_text, stage)
        logits = torch.matmul(10 * vision[stage], fused).permute(0, 2, 1).view(b, 2, side, side)
        stage_logits.append(F.interpolate(logits, (IMG, IMG), mode="bilinear", align_corners=True))
    stage_logits = torch.stack(stage_logits)
    stage_probs = F.softmax(stage_logits, dim=2)[:, :, 1]
    raw = F.softmax(stage_logits.mean(0), dim=1)[:, 1]
    final = model.vision_text_fusion_gate_seg(
        vision, text.unsqueeze(1).repeat(1, b, 1, 1), img_size=IMG,
        test_mode=True, domain=DOMAINS["VisA"])
    return raw, final, stage_probs


def morphology(mask):
    structure = np.ones((7, 7), bool)
    eroded = ndimage.binary_erosion(mask, structure=structure)
    dilated = ndimage.binary_dilation(mask, structure=structure)
    boundary = mask & ~eroded
    interior = eroded
    near = dilated & ~mask
    far = ~dilated
    cc = int(ndimage.label(mask, structure=np.ones((3, 3), int))[1]) if mask.any() else 0
    return boundary, interior, near, far, cc


class Moments:
    def __init__(self):
        self.n = 0; self.s = 0.0; self.ss = 0.0; self.sample = []

    def add(self, values):
        a = np.asarray(values, np.float32).reshape(-1)
        if not a.size:
            return
        start = self.n
        self.n += a.size
        self.s += float(a.sum(dtype=np.float64))
        self.ss += float(np.square(a, dtype=np.float64).sum())
        self.sample.extend(a[(-start) % 257::257].tolist())

    def result(self):
        if not self.n:
            return {"count": 0}
        qv = np.quantile(np.asarray(self.sample), [.01, .05, .25, .5, .75, .95, .99])
        mean = self.s / self.n
        return {"count": int(self.n), "mean": mean,
                "std": max(0, self.ss / self.n - mean * mean) ** .5,
                **{f"p{int(q*100):02d}": float(v) for q, v in zip([.01,.05,.25,.5,.75,.95,.99], qv)},
                "quantile_method": "deterministic_stride_257"}


def feature_region_means(feature, patch_mask):
    rows = []
    for b in range(feature.shape[0]):
        all_mean = feature[b].float().mean(0)
        normal = feature[b][~patch_mask[b]].float().mean(0)
        abnormal = feature[b][patch_mask[b]].float().mean(0) if patch_mask[b].any() else torch.full_like(all_mean, float("nan"))
        rows.append(torch.stack([all_mean, normal, abnormal]))
    return torch.stack(rows)


def parameter_family(name):
    if "lora_adapters" in name: return "conv_lora"
    if "seg_proj" in name or "det_proj" in name or "layer_norm" in name or "adapt_weights" in name: return "image_projection"
    if "vision_text_q" in name or "vision_text_k" in name: return "dfg_qk"
    if "dfg_ss2d_branches" in name or "dfg_raw_gamma" in name: return "ss2d"
    return "other_image"


def drift_rows(payloads):
    rows = []
    for arm in ("H", "A"):
        for module_key in ("image_adapter", "text_adapter", "soft_prompt"):
            acc = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
            for name, value in payloads[arm][module_key].items():
                if not torch.is_tensor(value): continue
                ref = payloads["E1"][module_key][name].float(); cur = value.float()
                fam = parameter_family(name) if module_key == "image_adapter" else module_key
                d = cur - ref
                acc[fam][0] += float((d*d).sum()); acc[fam][1] += float((ref*ref).sum())
                acc[fam][2] += float((cur*cur).sum()); acc[fam][3] += d.numel()
            for fam, (d2, r2, c2, n) in acc.items():
                rows.append({"arm": arm, "family": fam, "parameter_count": n,
                             "l2_drift": d2**.5, "reference_norm": r2**.5,
                             "final_norm": c2**.5, "relative_drift": d2**.5/max(r2**.5, 1e-12)})
    return rows


def optimizer_rows(payloads):
    rows = []
    for arm in MAP_ARMS:
        opt = payloads[arm]["optimizer_state"]
        for group in opt["param_groups"]:
            vals = []; steps = []
            for pid in group["params"]:
                state = opt["state"].get(pid, {})
                if "exp_avg" not in state: continue
                m, v = state["exp_avg"].float(), state["exp_avg_sq"].float()
                vals.append((m.abs()/(v.sqrt()+float(group.get("eps",1e-8)))).reshape(-1))
                step = state.get("step", 0); steps.append(float(step.item() if torch.is_tensor(step) else step))
            x = torch.cat(vals) if vals else torch.empty(0)
            rows.append({"arm": arm, "group": group.get("name"), "epoch": payloads[arm]["epoch"],
                         "global_step": payloads[arm]["global_step"], "lr": group["lr"],
                         "parameter_tensors_with_state": len(vals), "state_step_min": min(steps) if steps else None,
                         "state_step_max": max(steps) if steps else None,
                         "adam_direction_abs_mean": float(x.mean()) if x.numel() else None,
                         "effective_update_abs_mean": float(x.mean()*group["lr"]) if x.numel() else None,
                         "effective_update_abs_p95": float(x.quantile(.95)*group["lr"]) if x.numel() else None})
    return rows


def prompt_rows(model, payloads, device):
    rows = []
    with torch.no_grad():
        for arm in ARMS:
            load_arm(model, payloads[arm])
            for category in CLASS_NAMES["VisA"]:
                main, _, stat, comp = get_hybrid_soft_prompt_single_class_text_embedding(
                    model, "VisA", category, device, return_kg=True, return_components=True)
                hard, soft = comp["hard_text"], comp["soft_text"]
                for stage in range(3):
                    for branch, idx in (("normal",0),("abnormal",1)):
                        rows.append({"arm":arm,"category":category,"stage":stage+1,"branch":branch,
                                     "hard_norm":float(hard[stage,:,idx].norm()),"soft_norm":float(soft[stage,:,idx].norm()),
                                     "main_norm":float(main[stage,:,idx].norm()),
                                     "soft_hard_cos":float(F.cosine_similarity(soft[stage,:,idx],hard[stage,:,idx],dim=0)),
                                     "main_hard_cos":float(F.cosine_similarity(main[stage,:,idx],hard[stage,:,idx],dim=0)),
                                     "normal_abnormal_cos":float(F.cosine_similarity(main[stage,:,0],main[stage,:,1],dim=0)),
                                     "kg_loss":float(1-stat["soft_hard_cos_mean"])})
    return rows


def linear_cka(x, y):
    x = x - x.mean(0); y = y - y.mean(0)
    xy = x.T @ y
    return float((xy*xy).sum() / ((x.T@x).square().sum().sqrt()*(y.T@y).square().sum().sqrt()).clamp_min(1e-12))


def geometry_rows(feature_arrays, labels):
    rows = []
    for arm in ARMS:
        for stage in range(3):
            for ri, region in enumerate(("all","normal_patch","anomalous_patch")):
                x = np.asarray(feature_arrays[arm][:,stage,ri], np.float32)
                valid = np.isfinite(x).all(1); x = x[valid]
                centered = x - x.mean(0)
                gram = centered @ centered.T / max(1, centered.shape[0]-1)
                eig = np.linalg.eigvalsh(gram); eig = np.maximum(eig, 0); nz=eig[eig>max(eig.max()*1e-10,1e-12)]
                p = eig/eig.sum() if eig.sum() else eig
                erank = float(np.exp(-(p[p>0]*np.log(p[p>0])).sum())) if p.size else None
                cond = float(nz.max()/nz.min()) if nz.size else None
                ref = np.asarray(feature_arrays["E1"][:,stage,ri],np.float32)[valid]
                cos = 1 - np.sum(x*ref,1)/(np.linalg.norm(x,axis=1)*np.linalg.norm(ref,axis=1)+1e-12)
                rows.append({"arm":arm,"stage":stage+1,"region":region,"sample_count":len(x),
                             "feature_norm_mean":float(np.linalg.norm(x,axis=1).mean()),
                             "cosine_drift_from_E1":float(cos.mean()),"linear_CKA_to_E1":linear_cka(torch.from_numpy(x),torch.from_numpy(ref)),
                             "effective_rank":erank,"covariance_trace":float(eig.sum()),"condition_number_nonzero":cond,
                             "anisotropy_mean_cosine":float(np.mean((x@x.T)/(np.linalg.norm(x,axis=1)[:,None]*np.linalg.norm(x,axis=1)[None,:]+1e-12))),
                             "spectrum_top1_share":float(eig[-1]/eig.sum()) if eig.sum() else None,
                             "spectrum_top10_share":float(eig[-10:].sum()/eig.sum()) if eig.sum() else None})
    return rows


def main():
    AUDIT.mkdir(exist_ok=True); LARGE.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(PROTOCOL.read_text())
    ckpts = {arm: REPO / protocol["frozen_checkpoints"][CKPT_KEYS[arm]]["path"] for arm in ARMS}
    verified = {}
    payloads = {}
    for arm, path in ckpts.items():
        got = sha256(path); expected = protocol["frozen_checkpoints"][CKPT_KEYS[arm]]["sha256"]
        if got != expected: raise RuntimeError(f"{arm} hash mismatch")
        payloads[arm] = torch.load(path, map_location="cpu", weights_only=False)
        p = payloads[arm]
        verified[arm] = {"path":str(path.relative_to(REPO)),"sha256":got,"checkpoint_version":p["checkpoint_version"],
                         "epoch":p["epoch"],"global_step":p["global_step"],"precision":p["precision"],
                         "amp_enabled":p["amp_enabled"],"scaler_state_present":bool(p.get("scaler_state")),
                         "tf32_enabled":p["tf32_enabled"],"config_sha256":p["config_sha256"],
                         "clip_sha256":p["clip_sha256"],"dataset_manifest_sha256":p["dataset_manifest_sha256"],
                         "implementation_git_sha":p["implementation_git_sha"]}
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("CUDA required for historical evaluator")
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    policy = PrecisionPolicy("fp16")
    clip = create_model("ViT-L-14-336", img_size=IMG, device=device, pretrained="openai", require_pretrained=True).eval()
    model = ACDCLIP(clip_model=clip,n_groups=3,image_adapt_weight=.2,text_adapt_weight=.2,
        conv_lora_rank=8,conv_lora_alpha=2.,conv_kernel_size_list=[3,5],lora_rank=16,lora_alpha=2.,
        dfg_mode="attn",dfg_attn_dim=256,dfg_attn_tau=8.,use_ss2d_dfg=True,dfg_gamma_max=.2,
        dfg_ss2d_fusion="weight_residual",dfg_beta=.1,dfg_beta_schedule="warmup010",dfg_beta_target=.1,
        dfg_beta_current=.1,dfg_weight_residual_fp32=True,use_soft_prompt=False,soft_prompt_ctx_len=4,
        soft_prompt_init="phrase",soft_prompt_init_phrase="a photo of a").to(device).eval()
    islands = disable_later_islands(model)
    datasets = get_text_and_image_dataset("VisA", IMG, "test")
    order = [(cat, i, m) for cat in CLASS_NAMES["VisA"] for i,m in enumerate(datasets[cat].meta)]
    n = len(order)
    # Freeze mask-derived covariates and tertiles before the first forward pass.
    mask_meta=[]; anomaly_areas=[]
    for cat, _, m in order:
        if m["label"]:
            mask=np.asarray(Image.open(Path(datasets[cat].data_path)/m["mask_path"]).convert("L").resize((IMG,IMG),Image.Resampling.NEAREST))!=0
        else: mask=np.zeros((IMG,IMG),bool)
        bnd, interior, near, far, cc=morphology(mask); area=float(mask.mean())
        if m["label"]: anomaly_areas.append(area)
        mask_meta.append((area,cc,float(bnd.sum()/max(mask.sum(),1))))
    tertiles=np.quantile(anomaly_areas,[1/3,2/3])
    frozen={"protocol_id":protocol["protocol_id"],"audit_git_sha":subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip(),
            "audit_git_branch":subprocess.check_output(["git","branch","--show-current"],cwd=REPO,text=True).strip(),
            "verified_checkpoints":verified,"later_fp16_islands_disabled_count":islands,
            "source_count":n,"area_tertiles":[float(x) for x in tertiles],"external_artifact_root":str(LARGE)}
    (AUDIT/"H2_MIXED_FROZEN_STATE.json").write_text(json.dumps(frozen,indent=2)+"\n")
    shape=(n,IMG,IMG)
    masks=np.lib.format.open_memmap(LARGE/"masks.npy",mode="w+",dtype="uint8",shape=shape)
    arrays={}; features={}
    for arm in ARMS:
        features[arm]=np.lib.format.open_memmap(LARGE/f"{arm}_feature_means.npy",mode="w+",dtype="float32",shape=(n,3,3,768))
    for arm in MAP_ARMS:
        arrays[arm]={"raw":np.lib.format.open_memmap(LARGE/f"{arm}_raw.npy",mode="w+",dtype="float32",shape=shape),
                     "final":np.lib.format.open_memmap(LARGE/f"{arm}_final.npy",mode="w+",dtype="float32",shape=shape),
                     "stage":np.lib.format.open_memmap(LARGE/f"{arm}_stage.npy",mode="w+",dtype="float16",shape=(n,3,IMG,IMG)),
                     "image":np.lib.format.open_memmap(LARGE/f"{arm}_image.npy",mode="w+",dtype="float32",shape=(n,))}
    acc={(arm,kind):ExactBinaryAccumulator(LARGE/f"spool_{arm}_{kind}") for arm in MAP_ARMS for kind in ("raw","final")}
    cat_acc={(arm,cat):ExactBinaryAccumulator(LARGE/f"spool_{arm}_{cat}") for arm in MAP_ARMS for cat in CLASS_NAMES["VisA"]}
    stage_acc={(arm,s):ExactBinaryAccumulator(LARGE/f"spool_{arm}_stage{s}") for arm in MAP_ARMS for s in range(3)}
    moments={arm:defaultdict(Moments) for arm in MAP_ARMS}; routing=[]; image_rows=[]; labels=np.zeros(n,np.uint8)
    index=0
    for category in CLASS_NAMES["VisA"]:
        loader=DataLoader(datasets[category],batch_size=40,shuffle=False,num_workers=2,pin_memory=True)
        for batch in loader:
            image=batch["image"].to(device,non_blocking=True); mask=batch["mask"].to(device); label=batch["label"].to(device)
            bs=len(label); mask_np=mask[:,0].cpu().numpy().astype(np.uint8); labels[index:index+bs]=label.cpu().numpy()
            masks[index:index+bs]=mask_np
            for arm in ARMS:
                load_arm(model,payloads[arm])
                with torch.no_grad(), policy.autocast(device):
                    if not torch.is_autocast_enabled("cuda") or torch.get_autocast_dtype("cuda") != torch.float16:
                        raise RuntimeError("historical FP16 autocast identity is not active")
                    text=get_multiple_adapted_text_embedding(model,"VisA",device)[category]
                    seg,det=model(image); vision=torch.stack(seg); detect=torch.stack(det)
                    patchmask=F.interpolate(mask.float(),(PATCH,PATCH),mode="nearest")[:,0].bool().flatten(1)
                    if arm in MAP_ARMS:
                        raw,final,stage_maps=maps_from_features(model,vision,text)
                        cls=torch.stack([torch.matmul(detect[s].unsqueeze(1),text[s].unsqueeze(0).repeat(bs,1,1)).squeeze(1) for s in range(3)]).mean(0)
                        image_score=.9*F.softmax(cls,1)[:,1]+.1*final.flatten(1).max(1).values
                        rcomp=routing_components(model,vision,text.unsqueeze(1).repeat(1,bs,1,1))
                for s in range(3): features[arm][index:index+bs,s]=feature_region_means(vision[s],patchmask).cpu().numpy()
                if arm not in MAP_ARMS:
                    del vision, detect
                    torch.cuda.empty_cache()
                    continue
                raw_np=raw.float().cpu().numpy(); final_np=final.float().cpu().numpy(); stage_np=stage_maps.float().cpu().numpy()
                arrays[arm]["raw"][index:index+bs]=raw_np; arrays[arm]["final"][index:index+bs]=final_np
                arrays[arm]["stage"][index:index+bs]=stage_np.transpose(1,0,2,3).astype(np.float16)
                arrays[arm]["image"][index:index+bs]=image_score.float().cpu().numpy()
                acc[arm,"raw"].update(torch.from_numpy(raw_np),torch.from_numpy(mask_np)); acc[arm,"final"].update(torch.from_numpy(final_np),torch.from_numpy(mask_np))
                cat_acc[arm,category].update(torch.from_numpy(final_np),torch.from_numpy(mask_np))
                for s in range(3): stage_acc[arm,s].update(torch.from_numpy(stage_np[s]),torch.from_numpy(mask_np))
                for bi in range(bs):
                    gi=index+bi; pos=mask_np[bi].astype(bool); boundary,interior,near,far,cc=morphology(pos)
                    for kind,vals in (("positive",final_np[bi][pos]),("negative",final_np[bi][~pos]),("boundary",final_np[bi][boundary]),
                                      ("interior",final_np[bi][interior]),("near_background",final_np[bi][near]),("far_background",final_np[bi][far])):
                        moments[arm][kind].add(vals)
                    met=binary_metrics(final_np[bi],mask_np[bi]) if pos.any() else {"auroc":None,"ap":None}
                    area,cc0,bar=mask_meta[gi]; stratum="normal" if not label[bi] else "small" if area<=tertiles[0] else "medium" if area<=tertiles[1] else "large"
                    image_rows.append({"arm":arm,"index":gi,"image_id":batch["file_name"][bi],"category":category,"label":int(label[bi]),
                        "area_ratio":area,"size_stratum":stratum,"connected_components":cc0,"boundary_area_ratio":bar,
                        "pixel_auroc":met["auroc"],"pixel_ap":met["ap"],"image_score":float(image_score[bi]),
                        "normal_tail_p99":float(np.quantile(final_np[bi][~pos],.99)),"anomaly_tail_p01":float(np.quantile(final_np[bi][pos],.01)) if pos.any() else None,
                        "raw_final_l1":float(np.abs(raw_np[bi]-final_np[bi]).mean())})
                    for s,d in enumerate(rcomp,1):
                        for branch in ("normal","abnormal"):
                            g=d[f"weight_gap_{branch}"][bi]; ss=d[f"weight_ss2d_{branch}"][bi]; fw=d[f"weight_final_{branch}"][bi]
                            row={"arm":arm,"index":gi,"image_id":batch["file_name"][bi],"category":category,"label":int(label[bi]),
                                "area_ratio":area,"connected_components":cc0,"boundary_area_ratio":bar,"stage":s,"branch":branch}
                            for prefix,t in (("score_gap",d[f"score_gap_{branch}"][bi]),("score_ss2d",d[f"score_ss2d_{branch}"][bi]),
                                             ("weight_gap",g),("weight_ss2d",ss),("weight_final",fw)):
                                for j,v in enumerate(t): row[f"{prefix}_g{j+1}"]=float(v)
                            row.update({"entropy_final":float(entropy(fw)),"effective_groups":float(torch.exp(entropy(fw))),
                                        "max_weight":float(fw.max()),"gap_ss2d_l1":float((g-ss).abs().sum()),
                                        "gap_ss2d_js":float(js_divergence(g,ss))})
                            routing.append(row)
                del vision, detect, raw, final, stage_maps, rcomp
                torch.cuda.empty_cache()
            index += bs
            if index % 120 < bs: print(f"processed={index}/{n}",flush=True)
    for x in [masks,*features.values(),*(v for a in arrays.values() for v in a.values())]: x.flush()
    # The reboot recovery contract requires the expensive inference results and
    # RAM-only routing rows to be durable before exact sorting/postprocessing.
    write_csv(AUDIT/"H2_MIXED_SOURCE_PER_IMAGE.csv", image_rows)
    write_csv(AUDIT/"H2_MIXED_DFG_ROUTING.csv", routing)
    durable_json(AUDIT/"H2_MIXED_INFERENCE_COMPLETE.json", {
        "status":"PASS", "protocol":protocol["protocol_id"],
        "precision":"FP16 autocast", "parameter_dtype":"FP32",
        "dfg_weight_residual_fp32":True, "later_transformer_fp32_islands":False,
        "source_count":n, "per_image_count":len(image_rows),
        "routing_row_count":len(routing), "external_artifact_root":str(LARGE),
    })
    metric_rows=[]
    for arm in MAP_ARMS:
        for kind in ("raw","final"):
            auc,ap=acc[arm,kind].compute(); acc[arm,kind].cleanup(); metric_rows.append({"arm":arm,"cohort":kind,"auroc":auc,"ap":ap})
            durable_json(AUDIT/"H2_MIXED_EXACT_METRIC_PROGRESS.json", {"completed":metric_rows})
        im=binary_metrics(arrays[arm]["image"],labels); metric_rows.append({"arm":arm,"cohort":"image","auroc":im["auroc"],"ap":im["ap"]})
        for cat in CLASS_NAMES["VisA"]:
            auc,ap=cat_acc[arm,cat].compute(); cat_acc[arm,cat].cleanup(); metric_rows.append({"arm":arm,"cohort":f"category:{cat}","auroc":auc,"ap":ap})
            durable_json(AUDIT/"H2_MIXED_EXACT_METRIC_PROGRESS.json", {"completed":metric_rows})
        for s in range(3):
            auc,ap=stage_acc[arm,s].compute(); stage_acc[arm,s].cleanup(); metric_rows.append({"arm":arm,"cohort":f"stage:{s+1}","auroc":auc,"ap":ap})
            durable_json(AUDIT/"H2_MIXED_EXACT_METRIC_PROGRESS.json", {"completed":metric_rows})
        for cohort,m in moments[arm].items(): metric_rows.append({"arm":arm,"cohort":f"distribution:{cohort}",**m.result()})
    # Routing association uses anomalous images only and records undefined constants explicitly.
    association=[]
    for arm in MAP_ARMS:
        rr=[x for x in routing if x["arm"]==arm and x["label"]==1]
        for stage in range(1,4):
            for branch in ("normal","abnormal"):
                sub=[x for x in rr if x["stage"]==stage and x["branch"]==branch]
                for cov in ("area_ratio","connected_components","boundary_area_ratio"):
                    for response in ("entropy_final","gap_ss2d_l1","gap_ss2d_js","max_weight"):
                        x=np.asarray([r[cov] for r in sub],float); y=np.asarray([r[response] for r in sub],float)
                        association.append({"arm":arm,"stage":stage,"branch":branch,"covariate":cov,"response":response,"n":len(x),
                            "pearson_r":float(stats.pearsonr(x,y).statistic) if np.std(x)>0 and np.std(y)>0 else None,
                            "spearman_rho":float(stats.spearmanr(x,y).statistic) if np.std(x)>0 and np.std(y)>0 else None})
    geom=geometry_rows(features,labels); drift=drift_rows(payloads); opt=optimizer_rows(payloads); prompts=prompt_rows(model,payloads,device)
    write_csv(AUDIT/"H2_MIXED_SOURCE_PIXEL_RANKING.csv",metric_rows+image_rows)
    write_csv(AUDIT/"H2_MIXED_DFG_ROUTING.csv",routing+association)
    write_csv(AUDIT/"H2_MIXED_FEATURE_GEOMETRY.csv",geom+drift)
    write_csv(AUDIT/"H2_MIXED_OPTIMIZATION_AUDIT.csv",opt)
    write_csv(AUDIT/"H2_MIXED_PROMPT_GEOMETRY.csv",prompts)
    compact={"frozen":frozen,"metric_rows":metric_rows,"routing_associations":association,"geometry":geom,"parameter_drift":drift,
             "optimizer":opt,"prompt_summary":prompts,"per_image_count":len(image_rows),"routing_row_count":len(routing)}
    (AUDIT/"H2_MIXED_GENERALIZATION_DIAGNOSTICS.json").write_text(json.dumps(compact,indent=2,default=json_default)+"\n")
    print(json.dumps({"status":"PASS","source_count":n,"routing_rows":len(routing),"external":str(LARGE)}))


if __name__ == "__main__":
    main()
