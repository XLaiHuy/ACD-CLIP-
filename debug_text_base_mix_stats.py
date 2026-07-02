"""
debug_text_base_mix_stats.py
============================
Diagnostic script for Phase 2 planning of ACD-CLIP.

Purpose:
    Measure how much T_base (text features BEFORE LoRA/adaptation contribution)
    and T_mix (text features AFTER LoRA/adaptation contribution) differ in the
    current best Phase 1 V3c model.

Metrics per (level j, state s):
    - cosine(T_base_j^s, T_mix_j^s)   -- directional similarity
    - delta_ratio = ||T_mix - T_base|| / ||T_base||   -- relative perturbation magnitude
    - norm_ratio  = ||T_mix|| / ||T_base||             -- scale change
    - raw norms of T_base and T_mix

Classification guide (for manual reading after run):
    Case A  cosine_min >= 0.985, delta_ratio_max <= 0.05  --> drift very small, skip key-anchor
    Case B  cosine ~0.95-0.985,  delta_ratio ~0.05-0.15   --> moderate, test key-anchor
    Case C  cosine >= 0.97 but norm_ratio drifts            --> scale drift, test key-anchor
    Case D  non-uniform across levels/states                --> highest mismatch risk

Usage:
    conda run --no-capture-output -n torchhuy \
        python debug_text_base_mix_stats.py \
        --checkpoint phase1_v3c_weightres_betawarm010_fp32attn_tau8_g3/adapter_9.pth \
        --datasets Brain Retina Colon_Kvasir Liver Colon_clinicDB \
        --batch_size 4 \
        --max_batches 2 \
        --num_workers 0 \
    | tee phase1_v3c_weightres_betawarm010_fp32attn_tau8_g3/diagnostic_text_base_mix_e9.txt

NOTE: This script does NOT modify any source files.
      It only reads the checkpoint and runs a read-only forward pass.
"""

import argparse
import os
import sys
import warnings
from collections import defaultdict

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Ensure repo root is on sys.path so that local imports work the same as
# train.py / test.py (they are run from the repo root).
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from model.clip import create_model
from model.adapter import ACDCLIP
from model.tokenizer import tokenize
from dataset.info import CLASS_NAMES, DATA_PATH, REAL_NAMES, PROMPTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_text_prompts(dataset_name, device):
    """
    Return tokenized prompted sentences for normal and abnormal states.
    Mirrors what get_multiple_adapted_single_class_text_embedding() does,
    but returns the raw tokens for manual encode_text calls below.
    Shape: (n_state=2, n_prompts_per_state, seq_len)
    """
    class_name = CLASS_NAMES[dataset_name][0]
    if class_name == "object":
        real_name = class_name
    else:
        real_name = REAL_NAMES[dataset_name][class_name]

    prompt_normal    = PROMPTS["prompt_normal"]
    prompt_abnormal  = PROMPTS["prompt_abnormal"]
    prompt_templates = PROMPTS["prompt_templates"]

    all_tokens_by_state = []
    for state_prompts in [prompt_normal, prompt_abnormal]:
        prompted  = [state.format(real_name) for state in state_prompts]
        sentences = [template.format(s) for s in prompted for template in prompt_templates]
        tokens    = tokenize(sentences).to(device)  # (n_sentences, seq_len)
        all_tokens_by_state.append(tokens)
    return all_tokens_by_state  # list of length 2, each (n_sentences, 77)


# ---------------------------------------------------------------------------
# Core: extract T_base and T_mix from encode_text
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_tbase_tmix(model, tokens):
    """
    Run a modified encode_text that returns BOTH:
        T_base[j]: hidden state at text level j, BEFORE LoRA/adaptation blending
        T_mix[j]:  hidden state at text level j, AFTER LoRA/adaptation blending

    This replicates the logic in ACDCLIP.encode_text() (adapter.py lines 474-510)
    but captures the pre-adaptation state at each adapter insertion point.

    Returns:
        t_base_list: list of n_groups tensors [n_sentences, 768]  CLS tokens, layer-normed
        t_mix_list:  list of n_groups tensors [n_sentences, 768]  CLS tokens, layer-normed
    """
    cast_dtype = model.clipmodel.transformer.get_cast_dtype()
    x = model.clipmodel.token_embedding(tokens).to(cast_dtype)   # [B, n_ctx, d]
    x = x + model.clipmodel.positional_embedding.to(cast_dtype)
    x = x.permute(1, 0, 2)  # NLD -> LND

    t_base_list_raw = []   # x BEFORE adapt blend at each level
    t_mix_list_raw  = []   # x AFTER  adapt blend at each level

    for i in range(12):
        x, _ = model.clipmodel.transformer.resblocks[i](
            x, attn_mask=model.clipmodel.attn_mask
        )
        index = -1
        for j in range(model.n_groups):
            if i + 1 == model.text_levels[j]:
                index = j
                break
        if index != -1:
            # ---------- capture T_base (before blending) ----------
            t_base_list_raw.append(x.clone())
            # ---------- compute adaptation (same as encode_text) ----------
            adapt_out = model.text_adapter["lora_adapters"][index](x)
            adapt_out = (
                adapt_out
                * x.norm(dim=-1, keepdim=True)
                / adapt_out.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            )
            x = model.text_adapter["m_t_w"][index](x, adapt_out)
            # ---------- capture T_mix (after blending) ----------
            t_mix_list_raw.append(x.clone())

    # Extract CLS token (at EOS position = argmax of input ids)
    indices = tokens.argmax(dim=-1)  # [B]

    def _to_seq_first(raw_list):
        out = []
        for feat in raw_list:
            # feat shape: [seq_len, B, d_model]
            out.append(feat.permute(1, 0, 2))  # -> [B, seq_len, d_model]
        return out

    t_base_all = _to_seq_first(t_base_list_raw)
    t_mix_all  = _to_seq_first(t_mix_list_raw)

    # Apply layer norm and extract CLS token, same as original encode_text
    t_base_cls = []
    t_mix_cls  = []
    for j in range(model.n_groups):
        b_feat = model.text_adapter["layer_norms"][j](t_base_all[j])  # [B, seq_len, 768]
        m_feat = model.text_adapter["layer_norms"][j](t_mix_all[j])   # [B, seq_len, 768]
        b_cls  = b_feat[torch.arange(b_feat.shape[0]), indices]        # [B, 768]
        m_cls  = m_feat[torch.arange(m_feat.shape[0]), indices]        # [B, 768]
        t_base_cls.append(b_cls)
        t_mix_cls.append(m_cls)

    return t_base_cls, t_mix_cls


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_drift_metrics(t_base, t_mix):
    """
    Compute drift metrics between T_base and T_mix.
    Both tensors: [B, 768]

    Returns dict with scalar metrics.
    """
    t_base = t_base.float()
    t_mix  = t_mix.float()

    cos         = F.cosine_similarity(t_base, t_mix, dim=-1)       # [B]
    norm_base   = t_base.norm(dim=-1)                               # [B]
    norm_mix    = t_mix.norm(dim=-1)                                # [B]
    delta_norm  = (t_mix - t_base).norm(dim=-1)                     # [B]
    delta_ratio = delta_norm  / norm_base.clamp_min(1e-8)           # [B]
    norm_ratio  = norm_mix    / norm_base.clamp_min(1e-8)           # [B]

    return {
        "cosine":          cos.mean().item(),
        "cosine_min":      cos.min().item(),
        "cosine_max":      cos.max().item(),
        "delta_ratio":     delta_ratio.mean().item(),
        "delta_ratio_max": delta_ratio.max().item(),
        "norm_ratio":      norm_ratio.mean().item(),
        "norm_ratio_min":  norm_ratio.min().item(),
        "norm_ratio_max":  norm_ratio.max().item(),
        "norm_base":       norm_base.mean().item(),
        "norm_mix":        norm_mix.mean().item(),
    }


# ---------------------------------------------------------------------------
# Main diagnostic routine
# ---------------------------------------------------------------------------

def run_diagnostics(args):
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"[DIAGNOSTIC] debug_text_base_mix_stats.py")
    print(f"[DIAGNOSTIC] Checkpoint : {args.checkpoint}")
    print(f"[DIAGNOSTIC] Datasets   : {args.datasets}")
    print(f"[DIAGNOSTIC] Device     : {device}")
    print(f"[DIAGNOSTIC] batch_size : {args.batch_size}")
    print(f"[DIAGNOSTIC] max_batches: {args.max_batches}")
    print(f"{'='*70}\n")

    # ------------------------------------------------------------------
    # 1. Load CLIP backbone
    # ------------------------------------------------------------------
    print("[STEP 1] Loading CLIP backbone (ViT-L-14-336) ...")
    clip_model = create_model(
        model_name="ViT-L-14-336",
        img_size=518,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_model.eval()

    # ------------------------------------------------------------------
    # 2. Load checkpoint and reconstruct ACDCLIP
    # ------------------------------------------------------------------
    print(f"\n[STEP 2] Loading checkpoint: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    ckpt = torch.load(args.checkpoint, map_location=device)

    # Read architecture metadata from checkpoint (same logic as test.py)
    n_groups         = ckpt.get("n_groups",                 3)
    dfg_mode         = ckpt.get("dfg_mode",                 "attn")
    dfg_attn_dim     = ckpt.get("dfg_attn_dim",             256)
    dfg_attn_tau     = ckpt.get("dfg_attn_tau",             8.0)
    use_ss2d_dfg     = ckpt.get("use_ss2d_dfg",             True)
    dfg_gamma_max    = ckpt.get("dfg_gamma_max",            0.2)
    dfg_ss2d_fusion  = ckpt.get("dfg_ss2d_fusion",          "weight_residual")
    dfg_beta_current     = ckpt.get("dfg_beta_current",     0.10)
    dfg_beta_schedule    = ckpt.get("dfg_beta_schedule",    "warmup010")
    dfg_beta_target      = ckpt.get("dfg_beta_target",      0.10)
    dfg_weight_residual_fp32 = ckpt.get("dfg_weight_residual_fp32", True)

    print(f"  n_groups          = {n_groups}")
    print(f"  dfg_mode          = {dfg_mode}")
    print(f"  dfg_attn_dim      = {dfg_attn_dim}")
    print(f"  dfg_attn_tau      = {dfg_attn_tau}")
    print(f"  use_ss2d_dfg      = {use_ss2d_dfg}")
    print(f"  dfg_ss2d_fusion   = {dfg_ss2d_fusion}")
    print(f"  dfg_beta_current  = {dfg_beta_current}")
    print(f"  dfg_beta_schedule = {dfg_beta_schedule}")
    print(f"  dfg_weight_residual_fp32 = {dfg_weight_residual_fp32}")
    print(f"  checkpoint epoch  = {ckpt.get('epoch', '?')}")

    model = ACDCLIP(
        clip_model=clip_model,
        n_groups=n_groups,
        dfg_mode=dfg_mode,
        dfg_attn_dim=dfg_attn_dim,
        dfg_attn_tau=dfg_attn_tau,
        use_ss2d_dfg=use_ss2d_dfg,
        dfg_gamma_max=dfg_gamma_max,
        dfg_ss2d_fusion=dfg_ss2d_fusion,
        dfg_beta=dfg_beta_current,
        dfg_beta_schedule=dfg_beta_schedule,
        dfg_beta_target=dfg_beta_target,
        dfg_beta_current=dfg_beta_current,
        dfg_weight_residual_fp32=dfg_weight_residual_fp32,
    ).to(device)

    model.image_adapter.load_state_dict(ckpt["image_adapter"])
    model.text_adapter.load_state_dict(ckpt["text_adapter"])
    model.eval()
    model.requires_grad_(False)
    print(f"  [OK] Model loaded and frozen.")

    # ------------------------------------------------------------------
    # 3. Collect per-dataset statistics
    # ------------------------------------------------------------------
    # all_stats[dataset_name][level_idx (0-based)][state_name] = list of metric dicts
    all_stats  = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    state_names = ["normal", "abnormal"]

    for ds_name in args.datasets:
        if ds_name not in DATA_PATH:
            print(f"\n[WARN] Dataset '{ds_name}' not in DATA_PATH, skipping.")
            continue

        print(f"\n{'─'*60}")
        print(f"[DATASET] {ds_name}")
        print(f"{'─'*60}")

        # Build text prompts (text only -- no image batches needed)
        try:
            text_tokens_by_state = _build_text_prompts(ds_name, device)
        except Exception as e:
            print(f"  [WARN] Failed to build prompts for {ds_name}: {e}")
            continue

        for state_idx, state_name in enumerate(state_names):
            tokens = text_tokens_by_state[state_idx]  # (n_prompts, 77)

            all_base_by_level = [[] for _ in range(n_groups)]
            all_mix_by_level  = [[] for _ in range(n_groups)]

            for _batch_iter in range(args.max_batches):
                try:
                    t_base_list, t_mix_list = extract_tbase_tmix(model, tokens)
                except Exception as e:
                    print(f"  [WARN] extract_tbase_tmix failed for {ds_name}/{state_name}: {e}")
                    break
                for j in range(n_groups):
                    all_base_by_level[j].append(t_base_list[j])
                    all_mix_by_level[j].append(t_mix_list[j])

            for j in range(n_groups):
                if not all_base_by_level[j]:
                    continue
                base_cat = torch.cat(all_base_by_level[j], dim=0)
                mix_cat  = torch.cat(all_mix_by_level[j],  dim=0)
                metrics  = compute_drift_metrics(base_cat, mix_cat)
                all_stats[ds_name][j][state_name].append(metrics)

        # ------------------------------------------------------------------
        # 4. Print per-dataset table
        # ------------------------------------------------------------------
        print(f"\n  {'Level':<8} {'State':<10} {'cosine':<10} {'cos_min':<10} "
              f"{'delta_r':<10} {'dr_max':<10} {'norm_r':<10} "
              f"{'nr_min':<10} {'nr_max':<10} {'||T_base||':<14} {'||T_mix||':<12}")
        print(f"  {'-'*118}")

        for j in range(n_groups):
            for state_name in state_names:
                reps = all_stats[ds_name][j][state_name]
                if not reps:
                    continue
                avg = {k: sum(d[k] for d in reps) / len(reps) for k in reps[0]}
                print(
                    f"  j={j+1:<5} {state_name:<10} "
                    f"{avg['cosine']:<10.5f} {avg['cosine_min']:<10.5f} "
                    f"{avg['delta_ratio']:<10.5f} {avg['delta_ratio_max']:<10.5f} "
                    f"{avg['norm_ratio']:<10.5f} {avg['norm_ratio_min']:<10.5f} "
                    f"{avg['norm_ratio_max']:<10.5f} "
                    f"{avg['norm_base']:<14.4f} {avg['norm_mix']:<12.4f}"
                )

    # ------------------------------------------------------------------
    # 5. Global aggregate summary across all datasets / levels / states
    # ------------------------------------------------------------------
    print(f"\n\n{'='*70}")
    print("GLOBAL AGGREGATE SUMMARY (across all datasets, levels, states)")
    print(f"{'='*70}\n")

    all_cosines      = []
    all_cos_mins     = []
    all_delta_ratios = []
    all_dr_maxs      = []
    all_norm_ratios  = []
    all_nr_mins      = []
    all_nr_maxs      = []

    for ds_name in all_stats:
        for j in all_stats[ds_name]:
            for state_name in all_stats[ds_name][j]:
                for d in all_stats[ds_name][j][state_name]:
                    all_cosines.append(d["cosine"])
                    all_cos_mins.append(d["cosine_min"])
                    all_delta_ratios.append(d["delta_ratio"])
                    all_dr_maxs.append(d["delta_ratio_max"])
                    all_norm_ratios.append(d["norm_ratio"])
                    all_nr_mins.append(d["norm_ratio_min"])
                    all_nr_maxs.append(d["norm_ratio_max"])

    def _stats_of(vals):
        if not vals:
            return {}
        return {
            "min":            min(vals),
            "max":            max(vals),
            "mean":           sum(vals) / len(vals),
            "spread(max-min)":max(vals) - min(vals),
        }

    for metric_name, vals in [
        ("cosine (mean per cell)",        all_cosines),
        ("cosine_min (per-sample min)",   all_cos_mins),
        ("delta_ratio (mean per cell)",   all_delta_ratios),
        ("delta_ratio_max (per-sample)",  all_dr_maxs),
        ("norm_ratio (mean per cell)",    all_norm_ratios),
        ("norm_ratio_min",                all_nr_mins),
        ("norm_ratio_max",                all_nr_maxs),
    ]:
        s = _stats_of(vals)
        print(f"  {metric_name}:")
        for k, v in s.items():
            print(f"    {k:<28} = {v:.6f}")
        print()

    # ------------------------------------------------------------------
    # 6. Case classification hint
    # ------------------------------------------------------------------
    print(f"{'='*70}")
    print("CASE CLASSIFICATION HINT")
    print(f"{'='*70}\n")

    if all_cos_mins and all_dr_maxs and all_nr_mins and all_nr_maxs:
        global_cos_min  = min(all_cos_mins)
        global_dr_max   = max(all_dr_maxs)
        global_nr_min   = min(all_nr_mins)
        global_nr_max   = max(all_nr_maxs)
        cosine_spread   = max(all_cosines) - min(all_cosines)
        dr_spread       = max(all_delta_ratios) - min(all_delta_ratios)

        print(f"  global_cos_min  = {global_cos_min:.5f}")
        print(f"  global_dr_max   = {global_dr_max:.5f}")
        print(f"  global_nr_min   = {global_nr_min:.5f}")
        print(f"  global_nr_max   = {global_nr_max:.5f}")
        print(f"  cosine_spread   = {cosine_spread:.5f}  (max - min of per-cell means)")
        print(f"  dr_spread       = {dr_spread:.5f}")
        print()

        # --- Classification logic ---
        if global_cos_min >= 0.985 and global_dr_max <= 0.05:
            case = "A"
            desc = (
                "Drift very small and uniform. T_mix is almost identical to T_base.\n"
                "  => Do NOT prioritize key-anchor as main Phase 2 run.\n"
                "  => Consider text_adapt_weight reduction (e.g. 0.1) as a clean control,\n"
                "     or keep Phase 1 V3c as current best and postpone text-side Phase 2."
            )
        elif (cosine_spread > 0.05 or dr_spread > 0.08) and (
                global_cos_min < 0.93 or global_dr_max > 0.20):
            if global_cos_min < 0.90 and global_dr_max > 0.20:
                case = "D2"
                desc = (
                    "Non-uniform AND large drift. HIGHEST MISMATCH RISK.\n"
                    "  => Do NOT implement pure key-anchor first.\n"
                    "  => Preferred: value-side stabilization:\n"
                    "       A) K=W_K(T_mix), V=T_mix, text_adapt_weight=0.1\n"
                    "       B) norm-bound ||T_mix - T_base|| / ||T_base||\n"
                    "       C) cosine anchor loss toward T_base\n"
                    "  => Choose ONLY ONE per first run."
                )
            else:
                case = "D1"
                desc = (
                    "Non-uniform but moderate drift.\n"
                    "  => Key-anchor (Phase2-A: K=W_K(T_base), V=T_mix) can still be tested\n"
                    "     but mark as a risky ablation. Monitor for K/V mismatch loss."
                )
        elif global_cos_min >= 0.95 and global_dr_max <= 0.15:
            if global_nr_min < 0.95 or global_nr_max > 1.05:
                case = "C"
                desc = (
                    "High cosine but noticeable norm/scale drift.\n"
                    "  => Direction is similar but Q·K attention logits are not cosine-normalized.\n"
                    "     Norm drift in keys can still affect score scale and routing.\n"
                    "  => Implement Phase2-A: K=W_K(T_base), V=T_mix, text_adapt_weight=0.2"
                )
            else:
                case = "B"
                desc = (
                    "Moderate, fairly uniform drift.\n"
                    "  => LoRA/adapted text changes features but not wildly non-uniformly.\n"
                    "  => Implement Phase2-A: K=W_K(T_base), V=T_mix, text_adapt_weight=0.2"
                )
        elif cosine_spread > 0.03 or dr_spread > 0.04:
            case = "D (check sub-case)"
            desc = (
                "Non-uniform drift detected but does not clearly fit D1 or D2.\n"
                "  => Check the per-level/state Spread Table carefully.\n"
                "  => Determine D1 vs D2 manually based on cos_min and dr_max."
            )
        else:
            case = "B/C (borderline)"
            desc = (
                "Borderline between Case B and C.\n"
                "  => Check the norm_ratio column carefully.\n"
                "     If nr_min < 0.95 or nr_max > 1.05 -> treat as Case C.\n"
                "     Otherwise -> treat as Case B."
            )

        print(f"  >> Likely Case: {case}")
        print()
        print(f"  >> Interpretation:")
        for line in desc.split("\n"):
            print(f"     {line}")
    else:
        print("  [WARN] No stats collected. Please check dataset paths and retry.")

    # ------------------------------------------------------------------
    # 7. Spread table: per (level, state) averaged across datasets
    # ------------------------------------------------------------------
    print(f"\n\n{'='*70}")
    print("SPREAD TABLE: per (level, state) averaged across all datasets")
    print("(Reveals whether drift is uniform or non-uniform across axes)")
    print(f"{'='*70}\n")

    print(f"  {'Level':<8} {'State':<10} {'cos_mean':<12} {'cos_min':<12} "
          f"{'dr_mean':<12} {'dr_max':<12}")
    print(f"  {'-'*70}")

    for j in range(n_groups):
        for state_name in state_names:
            cell_cos  = []
            cell_cmin = []
            cell_dr   = []
            cell_drmax= []
            for ds_name in all_stats:
                for d in all_stats[ds_name][j][state_name]:
                    cell_cos.append(d["cosine"])
                    cell_cmin.append(d["cosine_min"])
                    cell_dr.append(d["delta_ratio"])
                    cell_drmax.append(d["delta_ratio_max"])
            if cell_cos:
                print(
                    f"  j={j+1:<5} {state_name:<10} "
                    f"{sum(cell_cos)/len(cell_cos):<12.5f} "
                    f"{min(cell_cmin):<12.5f} "
                    f"{sum(cell_dr)/len(cell_dr):<12.5f} "
                    f"{max(cell_drmax):<12.5f}"
                )

    print(f"\n{'='*70}")
    print("END OF DIAGNOSTIC REPORT")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic: measure T_base vs T_mix drift in ACD-CLIP V3c checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to adapter_N.pth checkpoint from Phase 1 V3c run.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["Brain", "Retina", "Colon_Kvasir", "Liver", "Colon_clinicDB"],
        help="List of dataset names to run diagnostics on.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="(Unused directly; kept for CLI compatibility with planned image-based version.)",
    )
    parser.add_argument(
        "--max_batches",
        type=int,
        default=2,
        help="Number of repeated text extraction passes (text encoder is deterministic at eval).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader num_workers (reserved for future image-based extension).",
    )
    parser.add_argument(
        "--cuda_device",
        type=int,
        default=0,
        help="CUDA device index.",
    )
    args = parser.parse_args()

    print(f"\n[CONFIG] checkpoint  = {args.checkpoint}")
    print(f"[CONFIG] datasets    = {args.datasets}")
    print(f"[CONFIG] batch_size  = {args.batch_size}")
    print(f"[CONFIG] max_batches = {args.max_batches}")
    print(f"[CONFIG] num_workers = {args.num_workers}")
    print(f"[CONFIG] cuda_device = {args.cuda_device}")

    run_diagnostics(args)


if __name__ == "__main__":
    main()
