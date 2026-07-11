import argparse
import csv
import math
import os
from collections import defaultdict
from glob import glob
from pathlib import Path

import torch
import torch.nn.functional as F
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from torchmetrics.functional import auroc, average_precision
from tqdm import tqdm

from dataset import DOMAINS, get_text_and_image_dataset
from model.adapter import ACDCLIP
from model.clip import create_model
from test import get_epoch_from_checkpoint, limit_dataset_by_label
from utils import (
    get_hard_phase1_single_class_text_embedding,
    get_hybrid_soft_prompt_single_class_text_embedding,
)


PIXEL_DATASETS = [
    "Brain",
    "Liver",
    "Retina",
    "Colon_clinicDB",
    "Colon_colonDB",
    "Colon_Kvasir",
]
IMAGE_DATASETS = ["Brain", "Liver", "Retina"]

PROMPT_CONFIGS = {
    "phase1_hard": {"cls_mode": "hard", "cls_alpha": 0.0, "seg_alpha": 0.0},
    "current_shared": {"cls_mode": "hybrid", "cls_alpha": 0.20, "seg_alpha": 0.20},
    "split_hard_cls": {"cls_mode": "hard", "cls_alpha": 0.0, "seg_alpha": 0.20},
    "split_lowalpha_cls": {"cls_mode": "hybrid", "cls_alpha": 0.05, "seg_alpha": 0.20},
}

SCORE_RULES = [
    "cls_only",
    "0.9_cls_0.1_max",
    "0.8_cls_0.2_max",
    "0.9_cls_0.1_top1pct",
    "0.8_cls_0.2_top1pct",
    "0.5_cls_0.5_max_current_medical",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Phase2B e10 anchor diagnosis and fixed-config epoch sweep."
    )
    parser.add_argument("--mode", choices=["anchor", "sweep"], required=True)
    parser.add_argument("--save_path", required=True, help="Directory containing adapter_*.pth")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--epochs", type=int, nargs="+", required=True)
    parser.add_argument("--datasets", nargs="+", default=PIXEL_DATASETS)
    parser.add_argument("--image_datasets", nargs="+", default=IMAGE_DATASETS)
    parser.add_argument(
        "--prompt_configs",
        nargs="+",
        choices=sorted(PROMPT_CONFIGS),
        default=["current_shared", "split_hard_cls", "split_lowalpha_cls"],
    )
    parser.add_argument("--fixed_prompt_config", choices=sorted(PROMPT_CONFIGS), default=None)
    parser.add_argument("--fixed_score_rule", choices=SCORE_RULES, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--img_size", type=int, default=518)
    parser.add_argument("--model_name", type=str, default="ViT-L-14-336")
    parser.add_argument("--n_groups", type=int, default=3)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=2.0)
    parser.add_argument("--conv_lora_rank", type=int, default=8)
    parser.add_argument("--conv_lora_alpha", type=float, default=2.0)
    parser.add_argument("--conv_kernel_size_list", type=int, nargs="+", default=[3, 5])
    parser.add_argument("--soft_prompt_ctx_len", type=int, default=4)
    parser.add_argument("--soft_prompt_init", choices=["phrase", "random"], default="phrase")
    parser.add_argument("--soft_prompt_init_phrase", type=str, default="a photo of a")
    parser.add_argument("--dfg_mode", choices=["mlp", "attn"], default="attn")
    parser.add_argument("--dfg_attn_dim", type=int, default=256)
    parser.add_argument("--dfg_attn_tau", type=float, default=8.0)
    parser.add_argument("--use_ss2d_dfg", action="store_true")
    parser.add_argument("--dfg_gamma_max", type=float, default=0.2)
    parser.add_argument(
        "--dfg_ss2d_fusion",
        choices=["feature_residual", "weight_residual"],
        default="weight_residual",
    )
    parser.add_argument("--dfg_beta", type=float, default=0.10)
    parser.add_argument("--dfg_beta_schedule", choices=["fixed", "warmup010"], default="warmup010")
    parser.add_argument("--dfg_beta_target", type=float, default=0.10)
    parser.add_argument(
        "--metric_thresholds",
        type=int,
        default=None,
        help="Use binned pixel metrics. Default None is exact.",
    )
    parser.add_argument("--pixel_stride", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_samples_per_label", type=int, default=None)
    args = parser.parse_args()

    if args.mode == "anchor":
        if args.fixed_prompt_config is not None or args.fixed_score_rule is not None:
            raise ValueError("Anchor mode chooses among prompt configs and score rules; do not pass fixed options.")
    if args.mode == "sweep":
        if args.fixed_prompt_config is None or args.fixed_score_rule is None:
            raise ValueError("Sweep mode requires --fixed_prompt_config and --fixed_score_rule.")
        args.prompt_configs = [args.fixed_prompt_config]
    return args


def checkpoint_files(save_path, epochs):
    selected = set(epochs)
    files = sorted(glob(os.path.join(save_path, "adapter_*.pth")), key=get_epoch_from_checkpoint)
    files = [path for path in files if get_epoch_from_checkpoint(path) in selected]
    if len(files) != len(selected):
        found = {get_epoch_from_checkpoint(path) for path in files}
        missing = sorted(selected - found)
        raise FileNotFoundError(f"Missing checkpoints for epochs: {missing}")
    return files


def build_model(args, device):
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
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        conv_lora_rank=args.conv_lora_rank,
        conv_lora_alpha=args.conv_lora_alpha,
        conv_kernel_size_list=args.conv_kernel_size_list,
        dfg_mode=args.dfg_mode,
        dfg_attn_dim=args.dfg_attn_dim,
        dfg_attn_tau=args.dfg_attn_tau,
        use_ss2d_dfg=args.use_ss2d_dfg,
        dfg_gamma_max=args.dfg_gamma_max,
        dfg_ss2d_fusion=args.dfg_ss2d_fusion,
        dfg_beta=args.dfg_beta,
        dfg_beta_schedule=args.dfg_beta_schedule,
        dfg_beta_target=args.dfg_beta_target,
        dfg_beta_current=args.dfg_beta,
        use_soft_prompt=True,
        soft_prompt_ctx_len=args.soft_prompt_ctx_len,
        soft_prompt_init=args.soft_prompt_init,
        soft_prompt_init_phrase=args.soft_prompt_init_phrase,
    ).to(device)
    model.eval()
    return model


def validate_checkpoint_metadata(checkpoint, args):
    if checkpoint.get("dfg_mode", args.dfg_mode) != args.dfg_mode:
        raise ValueError(f"Checkpoint dfg_mode={checkpoint.get('dfg_mode')!r}, expected {args.dfg_mode!r}")
    if checkpoint.get("n_groups", args.n_groups) != args.n_groups:
        raise ValueError(f"Checkpoint n_groups={checkpoint.get('n_groups')!r}, expected {args.n_groups!r}")
    if args.dfg_mode == "attn":
        if checkpoint.get("dfg_attn_dim", args.dfg_attn_dim) != args.dfg_attn_dim:
            raise ValueError("Checkpoint dfg_attn_dim does not match.")
        ckpt_tau = float(checkpoint.get("dfg_attn_tau", args.dfg_attn_tau))
        if abs(ckpt_tau - args.dfg_attn_tau) > 1e-8:
            raise ValueError(f"Checkpoint dfg_attn_tau={ckpt_tau}, expected {args.dfg_attn_tau}")
        if bool(checkpoint.get("use_ss2d_dfg", False)) != args.use_ss2d_dfg:
            raise ValueError("Checkpoint use_ss2d_dfg does not match.")
        ckpt_fusion = checkpoint.get("dfg_ss2d_fusion", "feature_residual")
        if ckpt_fusion != args.dfg_ss2d_fusion:
            raise ValueError(f"Checkpoint dfg_ss2d_fusion={ckpt_fusion!r}, expected {args.dfg_ss2d_fusion!r}")


def load_checkpoint(model, checkpoint_path, args, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    validate_checkpoint_metadata(checkpoint, args)
    if args.dfg_mode == "attn":
        ckpt_beta = checkpoint.get("dfg_beta", 0.10)
        ckpt_beta_current = checkpoint.get("dfg_beta_current", ckpt_beta)
        model.dfg_beta_schedule = checkpoint.get("dfg_beta_schedule", "fixed")
        model.dfg_beta_target = float(checkpoint.get("dfg_beta_target", ckpt_beta))
        model.dfg_weight_residual_fp32 = bool(checkpoint.get("dfg_weight_residual_fp32", True))
        model.set_dfg_beta(float(ckpt_beta_current))

    model.image_adapter.load_state_dict(checkpoint["image_adapter"])
    model.text_adapter.load_state_dict(checkpoint["text_adapter"])

    ckpt_prompt_mode = checkpoint.get("prompt_mode", None)
    ckpt_use_hybrid_soft_prompt = bool(checkpoint.get("use_hybrid_soft_prompt", False))
    ckpt_use_soft_prompt = bool(checkpoint.get("use_soft_prompt", False))
    is_phase1_hard = (args.fixed_prompt_config == "phase1_hard") or ("phase1_hard" in args.prompt_configs if hasattr(args, "prompt_configs") else False)

    if is_phase1_hard:
        model.prompt_mode = "hard"
        model.use_soft_prompt = False
        model.use_hybrid_soft_prompt = False
        model.hybrid_alpha_current = 0.0
    elif ckpt_prompt_mode == "hybrid" or ckpt_use_hybrid_soft_prompt:
        if "soft_prompt" not in checkpoint:
            raise ValueError(f"{checkpoint_path} declares hybrid prompt but has no soft_prompt state.")
        ckpt_ctx_len = int(checkpoint.get("soft_prompt_ctx_len", model.soft_prompt_ctx_len))
        if ckpt_ctx_len != model.soft_prompt_ctx_len:
            raise ValueError(f"Checkpoint soft_prompt_ctx_len={ckpt_ctx_len}, expected {model.soft_prompt_ctx_len}.")
        model.soft_prompt.load_state_dict(checkpoint["soft_prompt"])
        model.prompt_mode = "hybrid"
        model.use_soft_prompt = False
        model.use_hybrid_soft_prompt = True
        model.hybrid_alpha_current = float(checkpoint.get("hybrid_alpha_current", 0.2))
    elif ckpt_use_soft_prompt:
        if "soft_prompt" not in checkpoint:
            raise ValueError(f"{checkpoint_path} declares soft prompt but has no soft_prompt state.")
        ckpt_ctx_len = int(checkpoint.get("soft_prompt_ctx_len", model.soft_prompt_ctx_len))
        if ckpt_ctx_len != model.soft_prompt_ctx_len:
            raise ValueError(f"Checkpoint soft_prompt_ctx_len={ckpt_ctx_len}, expected {model.soft_prompt_ctx_len}.")
        model.soft_prompt.load_state_dict(checkpoint["soft_prompt"])
        model.prompt_mode = "soft"
        model.use_soft_prompt = True
        model.use_hybrid_soft_prompt = False
    else:
        model.prompt_mode = "hard"
        model.use_soft_prompt = False
        model.use_hybrid_soft_prompt = False
        model.hybrid_alpha_current = 0.0
    return int(checkpoint["epoch"])


def get_class_text_embedding(model, dataset_name, class_name, device, mode, alpha):
    if mode == "hard":
        return get_hard_phase1_single_class_text_embedding(
            model, dataset_name, class_name, device, adapt_text=True
        )
    if mode == "hybrid":
        old_alpha = float(getattr(model, "hybrid_alpha_current", 0.0))
        model.hybrid_alpha_current = float(alpha)
        try:
            text_features, _, _ = get_hybrid_soft_prompt_single_class_text_embedding(
                model, dataset_name, class_name, device, return_kg=False
            )
        finally:
            model.hybrid_alpha_current = old_alpha
        return text_features
    raise ValueError(f"Unsupported text mode: {mode}")


def build_text_cache(model, dataset_name, class_names, device, prompt_config):
    cfg = PROMPT_CONFIGS[prompt_config]
    cache = {}
    for class_name in class_names:
        cls_text = get_class_text_embedding(
            model, dataset_name, class_name, device, cfg["cls_mode"], cfg["cls_alpha"]
        )
        seg_text = get_class_text_embedding(
            model,
            dataset_name,
            class_name,
            device,
            "hard" if cfg["seg_alpha"] == 0.0 else "hybrid",
            cfg["seg_alpha"],
        )
        cache[class_name] = {"cls": cls_text, "seg": seg_text}
    return cache


def compute_cls_scores(det_features, text_features):
    cls_preds = [
        torch.matmul(
            det_features[i].unsqueeze(dim=1),
            text_features[i],
        ).squeeze(1)
        for i in range(det_features.shape[0])
    ]
    cls_preds = torch.stack(cls_preds, dim=0).mean(dim=0)
    return F.softmax(cls_preds, dim=1)[:, 1]


def top_percent_mean(flat_scores, percent):
    k = max(1, int(math.ceil(flat_scores.shape[1] * percent)))
    return torch.topk(flat_scores, k=k, dim=1).values.mean(dim=1)


def apply_score_rule(row, rule):
    cls_score = row["cls_score"]
    max_pixel = row["max_pixel"]
    top1pct = row["top1pct_pixel"]
    if rule == "cls_only":
        return cls_score
    if rule == "0.9_cls_0.1_max":
        return 0.9 * cls_score + 0.1 * max_pixel
    if rule == "0.8_cls_0.2_max":
        return 0.8 * cls_score + 0.2 * max_pixel
    if rule == "0.9_cls_0.1_top1pct":
        return 0.9 * cls_score + 0.1 * top1pct
    if rule == "0.8_cls_0.2_top1pct":
        return 0.8 * cls_score + 0.2 * top1pct
    if rule == "0.5_cls_0.5_max_current_medical":
        return 0.5 * cls_score + 0.5 * max_pixel
    raise ValueError(f"Unknown score rule: {rule}")


def metric_or_none(scores, labels):
    label_tensor = torch.tensor(labels, dtype=torch.int32)
    if label_tensor.max() == label_tensor.min():
        return None, None
    score_tensor = torch.tensor(scores, dtype=torch.float32)
    return (
        round(auroc(score_tensor, label_tensor, task="binary").item(), 4) * 100,
        round(average_precision(score_tensor, label_tensor, task="binary").item(), 4) * 100,
    )


def prepare_dataset(dataset, args):
    if args.max_samples_per_label is not None:
        dataset = limit_dataset_by_label(dataset, args.max_samples_per_label)
    if args.max_samples is not None:
        dataset = torch.utils.data.Subset(dataset, range(min(args.max_samples, len(dataset))))
    return dataset


def evaluate_dataset(
        model,
        dataset_name,
        class_name,
        dataset,
        text_cache,
        device,
        args,
        epoch,
        prompt_config,
):
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    pixel_auc = BinaryAUROC(thresholds=args.metric_thresholds)
    pixel_ap = BinaryAveragePrecision(thresholds=args.metric_thresholds)
    raw_rows = []
    class_text = text_cache[class_name]

    for input_data in tqdm(dataloader, desc=f"{dataset_name}/{class_name}/e{epoch}/{prompt_config}"):
        image = input_data["image"].to(device)
        mask = input_data["mask"].to(device).to(torch.int32)
        labels = input_data["label"].to(torch.int32)
        file_names = input_data["file_name"]
        batch_class_name = input_data["class_name"]
        assert len(set(batch_class_name)) == 1, "mixed class not supported"

        seg_tokens, det_tokens = model(image)
        seg_features = torch.stack(seg_tokens, dim=0)
        det_features = torch.stack(det_tokens, dim=0)
        batch_size = seg_features.shape[1]

        cls_text = class_text["cls"].unsqueeze(dim=1).repeat(1, batch_size, 1, 1)
        seg_text = class_text["seg"].unsqueeze(dim=1).repeat(1, batch_size, 1, 1)
        cls_scores = compute_cls_scores(det_features, cls_text)
        seg_pred = model.vision_text_fusion_gate_seg(
            seg_features,
            seg_text,
            test_mode=True,
            domain=DOMAINS[dataset_name],
        )

        flat_seg_full = torch.flatten(seg_pred, start_dim=1)
        max_pixel = flat_seg_full.max(dim=1).values
        top1pct_pixel = top_percent_mean(flat_seg_full, 0.01)

        if args.pixel_stride > 1:
            seg_eval = seg_pred[:, ::args.pixel_stride, ::args.pixel_stride]
            mask_eval = mask[:, :, ::args.pixel_stride, ::args.pixel_stride]
        else:
            seg_eval = seg_pred
            mask_eval = mask
        pixel_auc.update(seg_eval.detach().flatten().cpu(), mask_eval.detach().flatten().cpu())
        pixel_ap.update(seg_eval.detach().flatten().cpu(), mask_eval.detach().flatten().cpu())

        for i, file_name in enumerate(file_names):
            raw_rows.append({
                "dataset": dataset_name,
                "epoch": epoch,
                "prompt_config": prompt_config,
                "file_name": file_name,
                "label": int(labels[i].item()),
                "cls_score": float(cls_scores[i].detach().cpu().item()),
                "max_pixel": float(max_pixel[i].detach().cpu().item()),
                "top1pct_pixel": float(top1pct_pixel[i].detach().cpu().item()),
            })

        if device.type == "cuda":
            torch.cuda.empty_cache()

    pixel_row = {
        "dataset": dataset_name,
        "epoch": epoch,
        "prompt_config": prompt_config,
        "pixel_auc": round(pixel_auc.compute().item(), 4) * 100,
        "pixel_ap": round(pixel_ap.compute().item(), 4) * 100,
    }
    return raw_rows, pixel_row


def write_csv(path, rows, fieldnames):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_rows(raw_rows, pixel_rows, image_datasets, score_rules):
    pixel_by_key = defaultdict(list)
    for row in pixel_rows:
        pixel_by_key[(row["epoch"], row["prompt_config"])].append(row)

    image_by_key_dataset = defaultdict(list)
    for row in raw_rows:
        image_by_key_dataset[(row["epoch"], row["prompt_config"], row["dataset"])].append(row)

    detail_rows = []
    aggregate_rows_out = []
    for (epoch, prompt_config), pixels in sorted(pixel_by_key.items()):
        pixel_auc_6 = sum(row["pixel_auc"] for row in pixels) / len(pixels)
        pixel_ap_6 = sum(row["pixel_ap"] for row in pixels) / len(pixels)
        for score_rule in score_rules:
            image_auc_vals = []
            image_ap_vals = []
            for dataset_name in image_datasets:
                rows = image_by_key_dataset.get((epoch, prompt_config, dataset_name), [])
                if not rows:
                    continue
                scores = [apply_score_rule(row, score_rule) for row in rows]
                labels = [row["label"] for row in rows]
                image_auc, image_ap = metric_or_none(scores, labels)
                detail_rows.append({
                    "dataset": dataset_name,
                    "epoch": epoch,
                    "prompt_config": prompt_config,
                    "score_rule": score_rule,
                    "image_auc": "" if image_auc is None else f"{image_auc:.2f}",
                    "image_ap": "" if image_ap is None else f"{image_ap:.2f}",
                })
                if image_auc is not None and image_ap is not None:
                    image_auc_vals.append(image_auc)
                    image_ap_vals.append(image_ap)
            aggregate_rows_out.append({
                "epoch": epoch,
                "prompt_config": prompt_config,
                "score_rule": score_rule,
                "pixel_auc_6": f"{pixel_auc_6:.2f}",
                "pixel_ap_6": f"{pixel_ap_6:.2f}",
                "image_auc_3": "" if not image_auc_vals else f"{sum(image_auc_vals) / len(image_auc_vals):.2f}",
                "image_ap_3": "" if not image_ap_vals else f"{sum(image_ap_vals) / len(image_ap_vals):.2f}",
                "image_n": len(image_ap_vals),
            })
    return aggregate_rows_out, detail_rows


def pick_anchor_config(aggregate_rows):
    valid = []
    for row in aggregate_rows:
        if row["image_ap_3"] == "":
            continue
        valid.append({
            **row,
            "_pixel_ap": float(row["pixel_ap_6"]),
            "_image_ap": float(row["image_ap_3"]),
        })
    if not valid:
        return None
    strict = [row for row in valid if row["_pixel_ap"] >= 40.20 and row["_image_ap"] >= 74.50]
    acceptable = [row for row in valid if row["_pixel_ap"] >= 39.82 and row["_image_ap"] >= 73.80]
    candidates = strict or acceptable or valid
    return max(candidates, key=lambda row: (row["_pixel_ap"], row["_image_ap"]))


def main():
    args = parse_args()
    output_dir = Path(args.output_dir or args.save_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        f"cuda:{args.cuda_device}" if torch.cuda.is_available() and args.cuda_device >= 0 else "cpu"
    )
    model = build_model(args, device)
    ckpt_files = checkpoint_files(args.save_path, args.epochs)

    raw_rows = []
    pixel_rows = []
    for ckpt_path in ckpt_files:
        epoch = load_checkpoint(model, ckpt_path, args, device)
        for prompt_config in args.prompt_configs:
            for dataset_name in args.datasets:
                datasets = get_text_and_image_dataset(dataset_name, args.img_size, "test")
                class_names = list(datasets.keys())
                with torch.no_grad():
                    text_cache = build_text_cache(model, dataset_name, class_names, device, prompt_config)
                    for class_name, dataset in datasets.items():
                        dataset = prepare_dataset(dataset, args)
                        class_raw_rows, pixel_row = evaluate_dataset(
                            model=model,
                            dataset_name=dataset_name,
                            class_name=class_name,
                            dataset=dataset,
                            text_cache=text_cache,
                            device=device,
                            args=args,
                            epoch=epoch,
                            prompt_config=prompt_config,
                        )
                        raw_rows.extend(class_raw_rows)
                        pixel_rows.append(pixel_row)

    raw_path = output_dir / "image_score_raw_predictions.csv"
    write_csv(
        raw_path,
        raw_rows,
        ["dataset", "epoch", "prompt_config", "file_name", "label", "cls_score", "max_pixel", "top1pct_pixel"],
    )
    write_csv(
        output_dir / "pixel_metrics_by_dataset.csv",
        pixel_rows,
        ["dataset", "epoch", "prompt_config", "pixel_auc", "pixel_ap"],
    )

    score_rules = SCORE_RULES if args.mode == "anchor" else [args.fixed_score_rule]
    aggregate_rows_out, detail_rows = aggregate_rows(raw_rows, pixel_rows, args.image_datasets, score_rules)
    if args.mode == "anchor":
        aggregate_name = "image_score_ablation_e10.csv"
    else:
        aggregate_name = "fixed_config_epoch_sweep.csv"
    write_csv(
        output_dir / aggregate_name,
        aggregate_rows_out,
        ["epoch", "prompt_config", "score_rule", "pixel_auc_6", "pixel_ap_6", "image_auc_3", "image_ap_3", "image_n"],
    )
    write_csv(
        output_dir / "image_metrics_by_dataset.csv",
        detail_rows,
        ["dataset", "epoch", "prompt_config", "score_rule", "image_auc", "image_ap"],
    )

    if args.mode == "anchor":
        best = pick_anchor_config(aggregate_rows_out)
        if best is not None:
            with open(output_dir / "anchor_best_config.txt", "w", encoding="utf-8") as handle:
                handle.write("Exploratory test-set ablation, not an unbiased final evaluation.\n")
                handle.write(f"best_prompt_config={best['prompt_config']}\n")
                handle.write(f"best_score_rule={best['score_rule']}\n")
                handle.write(f"pixel_ap_6={best['pixel_ap_6']}\n")
                handle.write(f"image_ap_3={best['image_ap_3']}\n")

    print(f"Wrote raw predictions to {raw_path}")
    print(f"Wrote aggregate results to {output_dir / aggregate_name}")


if __name__ == "__main__":
    main()
