import argparse
import gc
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from dataset import get_text_and_image_dataset
from model.adapter import ACDCLIP
from model.clip import create_model
from utils import get_multiple_adapted_text_embedding


def limit_dataset_by_label(dataset, max_samples_per_label):
    indices_by_label = {}
    for idx, meta in enumerate(dataset.meta):
        label = int(meta["label"])
        indices_by_label.setdefault(label, [])
        if len(indices_by_label[label]) < max_samples_per_label:
            indices_by_label[label].append(idx)

    selected_indices = []
    for label in sorted(indices_by_label):
        selected_indices.extend(indices_by_label[label])
    return Subset(dataset, selected_indices)


def load_model(args, checkpoint_path, use_ss2d_dfg):
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    ckpt_fusion = checkpoint.get("dfg_ss2d_fusion", "feature_residual")
    ckpt_beta = float(checkpoint.get("dfg_beta_current", checkpoint.get("dfg_beta", args.dfg_beta)))
    ckpt_beta_schedule = checkpoint.get("dfg_beta_schedule", "fixed")
    ckpt_beta_target = float(checkpoint.get("dfg_beta_target", checkpoint.get("dfg_beta", ckpt_beta)))
    clip_model = create_model(
        model_name=args.model_name,
        img_size=args.img_size,
        device=device,
        pretrained="openai",
        require_pretrained=True,
    )
    clip_model.eval()
    model = ACDCLIP(
        clip_model=clip_model,
        n_groups=args.n_groups,
        dfg_mode="attn",
        dfg_attn_dim=args.dfg_attn_dim,
        dfg_attn_tau=args.dfg_attn_tau,
        use_ss2d_dfg=use_ss2d_dfg,
        dfg_gamma_max=args.dfg_gamma_max,
        dfg_ss2d_fusion=ckpt_fusion if use_ss2d_dfg else "feature_residual",
        dfg_beta=ckpt_beta,
        dfg_beta_schedule=ckpt_beta_schedule,
        dfg_beta_target=ckpt_beta_target,
        dfg_beta_current=ckpt_beta,
    ).to(device)
    model.eval()

    if checkpoint.get("dfg_mode") != "attn":
        raise ValueError(f"{checkpoint_path} is not an attention DFG checkpoint")
    if checkpoint.get("n_groups", args.n_groups) != args.n_groups:
        raise ValueError(f"{checkpoint_path} n_groups mismatch")
    if checkpoint.get("dfg_attn_dim", args.dfg_attn_dim) != args.dfg_attn_dim:
        raise ValueError(f"{checkpoint_path} dfg_attn_dim mismatch")
    if abs(float(checkpoint.get("dfg_attn_tau", args.dfg_attn_tau)) - args.dfg_attn_tau) > 1e-8:
        raise ValueError(f"{checkpoint_path} dfg_attn_tau mismatch")
    if bool(checkpoint.get("use_ss2d_dfg", False)) != use_ss2d_dfg:
        raise ValueError(f"{checkpoint_path} use_ss2d_dfg mismatch")
    if use_ss2d_dfg:
        if checkpoint.get("dfg_ss2d_fusion", "feature_residual") != model.dfg_ss2d_fusion:
            raise ValueError(f"{checkpoint_path} dfg_ss2d_fusion mismatch")
        if abs(float(checkpoint.get("dfg_beta_current", checkpoint.get("dfg_beta", args.dfg_beta))) - model.dfg_beta) > 1e-8:
            raise ValueError(f"{checkpoint_path} dfg_beta mismatch")

    model.image_adapter.load_state_dict(checkpoint["image_adapter"])
    model.text_adapter.load_state_dict(checkpoint["text_adapter"])
    return model, device, checkpoint


def unload_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def entropy(weights):
    return -(weights * weights.clamp_min(1e-8).log()).sum(dim=1)


def get_attention_tensors(model, seg_features, text_features):
    group_text_features = text_features.permute(1, 0, 2, 3)
    rows = []
    for stage_idx in range(model.n_groups):
        img_feat = seg_features[stage_idx]
        v_gap = img_feat.mean(dim=1)
        v_global = v_gap
        ss2d_ratio = None
        gamma = None
        v_ss2d = None

        if model.use_ss2d_dfg:
            v_ss2d = model.image_adapter["dfg_ss2d_branches"][stage_idx](img_feat)
            raw_gamma = model.image_adapter["dfg_raw_gamma"][stage_idx]
            gamma = model.dfg_gamma_max * torch.tanh(raw_gamma)
            if model.dfg_ss2d_fusion == "feature_residual":
                v_global = v_gap + gamma * v_ss2d
            ss2d_ratio = (
                gamma.abs() * v_ss2d.norm(dim=-1)
            ) / v_gap.norm(dim=-1).clamp_min(1e-6)

        text_normal = group_text_features[..., 0]
        text_abnormal = group_text_features[..., 1]
        k_normal = model.image_adapter["vision_text_k"][stage_idx](text_normal)
        k_abnormal = model.image_adapter["vision_text_k"][stage_idx](text_abnormal)
        scale = (model.dfg_attn_dim ** 0.5) * model.dfg_attn_tau
        q_gap = model.image_adapter["vision_text_q"][stage_idx](v_gap)
        w_gap_normal = attention_weights(q_gap, k_normal, scale)
        w_gap_abnormal = attention_weights(q_gap, k_abnormal, scale)
        q = q_gap
        q_ss2d = None
        w_ss2d_normal = None
        w_ss2d_abnormal = None

        if model.use_ss2d_dfg and model.dfg_ss2d_fusion == "weight_residual":
            q_ss2d = model.image_adapter["vision_text_q"][stage_idx](v_ss2d)
            w_ss2d_normal = attention_weights(q_ss2d, k_normal, scale)
            w_ss2d_abnormal = attention_weights(q_ss2d, k_abnormal, scale)
            w_normal = (1 - model.dfg_beta) * w_gap_normal + model.dfg_beta * w_ss2d_normal
            w_abnormal = (1 - model.dfg_beta) * w_gap_abnormal + model.dfg_beta * w_ss2d_abnormal
        else:
            q = model.image_adapter["vision_text_q"][stage_idx](v_global)
            w_normal = attention_weights(q, k_normal, scale)
            w_abnormal = attention_weights(q, k_abnormal, scale)

        rows.append({
            "stage": stage_idx + 1,
            "q": q.detach().cpu(),
            "q_gap": q_gap.detach().cpu(),
            "q_ss2d": None if q_ss2d is None else q_ss2d.detach().cpu(),
            "w_normal": w_normal.detach().cpu(),
            "w_abnormal": w_abnormal.detach().cpu(),
            "w_gap_normal": w_gap_normal.detach().cpu(),
            "w_gap_abnormal": w_gap_abnormal.detach().cpu(),
            "w_ss2d_normal": None if w_ss2d_normal is None else w_ss2d_normal.detach().cpu(),
            "w_ss2d_abnormal": None if w_ss2d_abnormal is None else w_ss2d_abnormal.detach().cpu(),
            "entropy_normal": entropy(w_normal).detach().cpu(),
            "entropy_abnormal": entropy(w_abnormal).detach().cpu(),
            "sum_error_normal": (w_normal.sum(dim=1) - 1).abs().detach().cpu(),
            "sum_error_abnormal": (w_abnormal.sum(dim=1) - 1).abs().detach().cpu(),
            "ss2d_ratio": None if ss2d_ratio is None else ss2d_ratio.detach().cpu(),
            "gamma": None if gamma is None else gamma.detach().cpu().reshape(1),
        })
    return rows


def attention_weights(query, key, scale):
    return F.softmax(torch.einsum("bd,bnd->bn", query, key) / scale, dim=1)


def collect_reference(args, checkpoint_path):
    model, device, checkpoint = load_model(args, checkpoint_path, use_ss2d_dfg=False)
    print(
        "reference="
        f"{checkpoint_path} epoch={checkpoint.get('epoch')} "
        f"tau={checkpoint.get('dfg_attn_tau')} use_ss2d={checkpoint.get('use_ss2d_dfg', False)}"
    )
    records = {}
    with torch.no_grad():
        for dataset_name in args.datasets:
            text_embeddings = get_multiple_adapted_text_embedding(model, dataset_name, device)
            records[dataset_name] = collect_dataset_records(
                args, model, device, dataset_name, text_embeddings
            )
    unload_model(model)
    return records


def collect_dataset_records(args, model, device, dataset_name, text_embeddings):
    datasets = get_text_and_image_dataset(dataset_name, args.img_size, "test")
    dataset_records = {}
    for class_name, dataset in datasets.items():
        if args.max_samples_per_label is not None:
            dataset = limit_dataset_by_label(dataset, args.max_samples_per_label)
        elif args.max_samples is not None:
            dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
        class_records = []
        class_text = text_embeddings[class_name]
        for batch_idx, input_data in enumerate(loader):
            if args.max_batches is not None and batch_idx >= args.max_batches:
                break
            image = input_data["image"].to(device)
            seg_tokens, _ = model(image)
            seg_features = torch.stack(seg_tokens, dim=0)
            text_features = class_text.unsqueeze(dim=1).repeat(1, image.shape[0], 1, 1)
            class_records.append(get_attention_tensors(model, seg_features, text_features))
            if device.type == "cuda":
                torch.cuda.empty_cache()
        dataset_records[class_name] = class_records
    return dataset_records


def compare_with_ss2d(args, reference_records, checkpoint_path):
    model, device, checkpoint = load_model(args, checkpoint_path, use_ss2d_dfg=True)
    print(
        "candidate="
        f"{checkpoint_path} epoch={checkpoint.get('epoch')} "
        f"tau={checkpoint.get('dfg_attn_tau')} use_ss2d={checkpoint.get('use_ss2d_dfg', False)} "
        f"gamma_max={checkpoint.get('dfg_gamma_max')}"
    )
    print(f"max_entropy=log({args.n_groups})={math.log(args.n_groups):.4f}")
    print(
        "dataset,class,stage,n,"
        "ref_H_N,ref_H_A,cand_H_N,cand_H_A,"
        "L1_N,L1_A,query_cos,ss2d_ratio,gamma,"
        "sumerr_N,sumerr_A,"
        "within_L1_final_gap_N,within_L1_final_gap_A,"
        "within_L1_ss2d_gap_N,within_L1_ss2d_gap_A,"
        "effective_drift_N,effective_drift_A,within_query_cos,"
        "ref_w_N,ref_w_A,cand_w_N,cand_w_A"
    )

    with torch.no_grad():
        for dataset_name in args.datasets:
            text_embeddings = get_multiple_adapted_text_embedding(model, dataset_name, device)
            datasets = get_text_and_image_dataset(dataset_name, args.img_size, "test")
            for class_name, dataset in datasets.items():
                if args.max_samples_per_label is not None:
                    dataset = limit_dataset_by_label(dataset, args.max_samples_per_label)
                elif args.max_samples is not None:
                    dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))

                loader = DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=torch.cuda.is_available(),
                )
                class_text = text_embeddings[class_name]
                stage_accum = {}
                ref_batches = reference_records[dataset_name][class_name]
                for batch_idx, input_data in enumerate(loader):
                    if batch_idx >= len(ref_batches):
                        break
                    image = input_data["image"].to(device)
                    seg_tokens, _ = model(image)
                    seg_features = torch.stack(seg_tokens, dim=0)
                    text_features = class_text.unsqueeze(dim=1).repeat(1, image.shape[0], 1, 1)
                    cand_rows = get_attention_tensors(model, seg_features, text_features)
                    ref_rows = ref_batches[batch_idx]

                    for ref_row, cand_row in zip(ref_rows, cand_rows):
                        stage = ref_row["stage"]
                        acc = stage_accum.setdefault(stage, {
                            "ref_H_N": [],
                            "ref_H_A": [],
                            "cand_H_N": [],
                            "cand_H_A": [],
                            "L1_N": [],
                            "L1_A": [],
                            "query_cos": [],
                            "ss2d_ratio": [],
                            "gamma": [],
                            "sumerr_N": [],
                            "sumerr_A": [],
                            "within_L1_final_gap_N": [],
                            "within_L1_final_gap_A": [],
                            "within_L1_ss2d_gap_N": [],
                            "within_L1_ss2d_gap_A": [],
                            "effective_drift_N": [],
                            "effective_drift_A": [],
                            "within_query_cos": [],
                            "ref_w_N": [],
                            "ref_w_A": [],
                            "cand_w_N": [],
                            "cand_w_A": [],
                        })
                        acc["ref_H_N"].append(ref_row["entropy_normal"])
                        acc["ref_H_A"].append(ref_row["entropy_abnormal"])
                        acc["cand_H_N"].append(cand_row["entropy_normal"])
                        acc["cand_H_A"].append(cand_row["entropy_abnormal"])
                        acc["L1_N"].append((ref_row["w_normal"] - cand_row["w_normal"]).abs().sum(dim=1))
                        acc["L1_A"].append((ref_row["w_abnormal"] - cand_row["w_abnormal"]).abs().sum(dim=1))
                        acc["query_cos"].append(F.cosine_similarity(ref_row["q"], cand_row["q"], dim=1))
                        acc["ss2d_ratio"].append(cand_row["ss2d_ratio"])
                        acc["gamma"].append(cand_row["gamma"])
                        acc["sumerr_N"].append(cand_row["sum_error_normal"])
                        acc["sumerr_A"].append(cand_row["sum_error_abnormal"])
                        if cand_row["w_ss2d_normal"] is not None:
                            ss2d_gap_n = (cand_row["w_ss2d_normal"] - cand_row["w_gap_normal"]).abs().sum(dim=1)
                            ss2d_gap_a = (cand_row["w_ss2d_abnormal"] - cand_row["w_gap_abnormal"]).abs().sum(dim=1)
                            final_gap_n = (cand_row["w_normal"] - cand_row["w_gap_normal"]).abs().sum(dim=1)
                            final_gap_a = (cand_row["w_abnormal"] - cand_row["w_gap_abnormal"]).abs().sum(dim=1)
                            acc["within_L1_final_gap_N"].append(final_gap_n)
                            acc["within_L1_final_gap_A"].append(final_gap_a)
                            acc["within_L1_ss2d_gap_N"].append(ss2d_gap_n)
                            acc["within_L1_ss2d_gap_A"].append(ss2d_gap_a)
                            acc["effective_drift_N"].append(torch.tensor(model.dfg_beta) * ss2d_gap_n)
                            acc["effective_drift_A"].append(torch.tensor(model.dfg_beta) * ss2d_gap_a)
                            acc["within_query_cos"].append(
                                F.cosine_similarity(cand_row["q_gap"], cand_row["q_ss2d"], dim=1)
                            )
                        acc["ref_w_N"].append(ref_row["w_normal"])
                        acc["ref_w_A"].append(ref_row["w_abnormal"])
                        acc["cand_w_N"].append(cand_row["w_normal"])
                        acc["cand_w_A"].append(cand_row["w_abnormal"])

                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                for stage in sorted(stage_accum):
                    print_summary(dataset_name, class_name, stage, stage_accum[stage])
    unload_model(model)


def mean_cat(values):
    if not values:
        return float("nan")
    return torch.cat(values, dim=0).float().mean().item()


def mean_weight(values):
    return torch.cat(values, dim=0).float().mean(dim=0).tolist()


def fmt_weights(values):
    return "[" + "/".join(f"{value:.4f}" for value in values) + "]"


def print_summary(dataset_name, class_name, stage, acc):
    n = sum(value.numel() for value in acc["ref_H_N"])
    gamma = torch.cat(acc["gamma"], dim=0).float().mean().item()
    print(
        f"{dataset_name},{class_name},{stage},{n},"
        f"{mean_cat(acc['ref_H_N']):.4f},{mean_cat(acc['ref_H_A']):.4f},"
        f"{mean_cat(acc['cand_H_N']):.4f},{mean_cat(acc['cand_H_A']):.4f},"
        f"{mean_cat(acc['L1_N']):.4f},{mean_cat(acc['L1_A']):.4f},"
        f"{mean_cat(acc['query_cos']):.4f},{mean_cat(acc['ss2d_ratio']):.4f},{gamma:.4f},"
        f"{mean_cat(acc['sumerr_N']):.6f},{mean_cat(acc['sumerr_A']):.6f},"
        f"{mean_cat(acc['within_L1_final_gap_N']):.4f},{mean_cat(acc['within_L1_final_gap_A']):.4f},"
        f"{mean_cat(acc['within_L1_ss2d_gap_N']):.4f},{mean_cat(acc['within_L1_ss2d_gap_A']):.4f},"
        f"{mean_cat(acc['effective_drift_N']):.4f},{mean_cat(acc['effective_drift_A']):.4f},"
        f"{mean_cat(acc['within_query_cos']):.4f},"
        f"{fmt_weights(mean_weight(acc['ref_w_N']))},"
        f"{fmt_weights(mean_weight(acc['ref_w_A']))},"
        f"{fmt_weights(mean_weight(acc['cand_w_N']))},"
        f"{fmt_weights(mean_weight(acc['cand_w_A']))}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Pairwise diagnostic for Phase 1A attention DFG vs Phase 1B SS2D DFG."
    )
    parser.add_argument("--reference_checkpoint", default="phase1_v2_attn_tau8/adapter_13.pth")
    parser.add_argument("--candidate_checkpoint", default="phase1_v3_attn_tau8_ss2d_g02/adapter_11.pth")
    parser.add_argument("--datasets", nargs="+", default=["Retina", "Colon_Kvasir"])
    parser.add_argument("--model_name", default="ViT-L-14-336")
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_batches", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_samples_per_label", type=int, default=None)
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--n_groups", type=int, default=3)
    parser.add_argument("--dfg_attn_dim", type=int, default=256)
    parser.add_argument("--dfg_attn_tau", type=float, default=8.0)
    parser.add_argument("--dfg_gamma_max", type=float, default=0.2)
    args = parser.parse_args()

    reference_records = collect_reference(args, args.reference_checkpoint)
    compare_with_ss2d(args, reference_records, args.candidate_checkpoint)


if __name__ == "__main__":
    main()
