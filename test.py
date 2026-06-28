import argparse
import logging
import os
from glob import glob
from pathlib import Path

import torch
import torch.nn.functional as F
from pandas import DataFrame, Series
from torch.utils.data import DataLoader
from torchmetrics.functional import auroc, average_precision
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from tqdm import tqdm

from dataset import DOMAINS, get_text_and_image_dataset
from utils import (
    get_multiple_adapted_text_embedding, metrics_eval_gpu,
)
from model.adapter import (
    ACDCLIP
)
from model.clip import create_model


def get_epoch_from_checkpoint(path: str) -> int:
    return int(Path(path).stem.split("_")[-1])


def limit_dataset_by_label(dataset, max_samples_per_label: int):
    indices_by_label = {}
    for idx, meta in enumerate(dataset.meta):
        label = int(meta["label"])
        indices_by_label.setdefault(label, [])
        if len(indices_by_label[label]) < max_samples_per_label:
            indices_by_label[label].append(idx)

    selected_indices = []
    for label in sorted(indices_by_label):
        selected_indices.extend(indices_by_label[label])
    return torch.utils.data.Subset(dataset, selected_indices)


def get_predictions(
        model: ACDCLIP,
        class_text_embeddings: torch.Tensor,
        test_loader: DataLoader,
        device,
        dataset: str = "MVTec",
):
    masks = []
    labels = []
    preds = []
    preds_image = []
    file_names = []
    for input_data in tqdm(test_loader):
        image = input_data["image"].to(device)
        mask = input_data["mask"].to(device).to(torch.int32)
        label = input_data["label"].to(device).to(torch.int32)
        file_name = input_data["file_name"]
        # set up class-specific containers
        class_name = input_data["class_name"]
        assert len(set(class_name)) == 1, "mixed class not supported"
        masks.append(mask.cpu())
        labels.append(label.cpu())
        file_names.extend(file_name)
        # get text
        epoch_text_features = class_text_embeddings.unsqueeze(dim=1)  # [n_groups, 1, 768, 2]
        # forward image
        seg_tokens, det_tokens = model(image)  # [bs, patch_size, 768] * n_groups, [bs, 768] * n_groups
        seg_features = torch.stack(seg_tokens, dim=0)  # [n_groups, bs, patch_num, 768]
        det_features = torch.stack(det_tokens, dim=0)  # [n_groups, bs, 768]
        B = seg_features.shape[1]
        epoch_text_features = epoch_text_features.repeat(1, B, 1, 1)  # [n_groups, bs, 768, 2]
        cls_preds = [
            torch.matmul(
                det_features[i].unsqueeze(dim=1),  # [bs, 1, 768]
                epoch_text_features[i],  # [bs, 768, 2]
            ).squeeze(1) for i in range(det_features.shape[0])
        ]  # [bs, 2] * n_groups
        cls_preds = torch.stack(cls_preds, dim=0).mean(dim=0)  # [bs, 2]
        pred = F.softmax(cls_preds, dim=1)[:, 1]
        preds_image.append(pred.cpu())
        # [bs, img_size, img_size]
        seg_pred = model.vision_text_fusion_gate_seg(seg_features, epoch_text_features, test_mode=True,
                                                     domain=DOMAINS[dataset])
        preds.append(seg_pred.cpu())
        if device.type == "cuda":
            torch.cuda.empty_cache()
    masks = torch.concatenate(masks, dim=0)  # [bs, 1, 518, 518]
    labels = torch.concatenate(labels, dim=0)  # [bs]
    preds = torch.concatenate(preds, dim=0)  # [bs, 518, 518]
    preds_image = torch.concatenate(preds_image, dim=0)  # [bs]
    return masks, labels, preds, preds_image, file_names


def get_streaming_metrics(
        model: ACDCLIP,
        class_text_embeddings: torch.Tensor,
        test_loader: DataLoader,
        device,
        class_name: str,
        dataset: str = "MVTec",
        thresholds: int = 1000,
        pixel_stride: int = 1,
):
    pixel_auc = BinaryAUROC(thresholds=thresholds)
    pixel_ap = BinaryAveragePrecision(thresholds=thresholds)
    image_labels = []
    image_preds = []

    for input_data in tqdm(test_loader):
        image = input_data["image"].to(device)
        mask = input_data["mask"].to(device).to(torch.int32)
        label = input_data["label"].to(device).to(torch.int32)
        batch_class_name = input_data["class_name"]
        assert len(set(batch_class_name)) == 1, "mixed class not supported"

        epoch_text_features = class_text_embeddings.unsqueeze(dim=1)
        seg_tokens, det_tokens = model(image)
        seg_features = torch.stack(seg_tokens, dim=0)
        det_features = torch.stack(det_tokens, dim=0)
        B = seg_features.shape[1]
        epoch_text_features = epoch_text_features.repeat(1, B, 1, 1)
        cls_preds = [
            torch.matmul(
                det_features[i].unsqueeze(dim=1),
                epoch_text_features[i],
            ).squeeze(1) for i in range(det_features.shape[0])
        ]
        cls_preds = torch.stack(cls_preds, dim=0).mean(dim=0)
        pred_image = F.softmax(cls_preds, dim=1)[:, 1]
        seg_pred = model.vision_text_fusion_gate_seg(
            seg_features,
            epoch_text_features,
            test_mode=True,
            domain=DOMAINS[dataset],
        )

        flat_seg = torch.flatten(seg_pred, start_dim=1)
        pmax_pred, _ = torch.max(flat_seg, dim=1)
        if DOMAINS[dataset] == "Medical":
            pred_image = pred_image * 0.5 + pmax_pred * 0.5
        else:
            pred_image = pred_image * 0.9 + pmax_pred * 0.1

        if pixel_stride > 1:
            seg_pred_eval = seg_pred[:, ::pixel_stride, ::pixel_stride]
            mask_eval = mask[:, :, ::pixel_stride, ::pixel_stride]
        else:
            seg_pred_eval = seg_pred
            mask_eval = mask
        pixel_auc.update(seg_pred_eval.detach().flatten().cpu(), mask_eval.detach().flatten().cpu())
        pixel_ap.update(seg_pred_eval.detach().flatten().cpu(), mask_eval.detach().flatten().cpu())
        image_labels.append(label.detach().cpu())
        image_preds.append(pred_image.detach().cpu())

        if device.type == "cuda":
            torch.cuda.empty_cache()

    image_label = torch.concatenate(image_labels, dim=0).flatten()
    image_pred = torch.concatenate(image_preds, dim=0).flatten()
    if image_label.max() != image_label.min():
        image_auc = auroc(image_pred, image_label, task="binary")
        image_ap = average_precision(image_pred, image_label, task="binary")
    else:
        image_auc = torch.tensor(0.0)
        image_ap = torch.tensor(0.0)

    return {
        "class name": class_name,
        "pixel AUC": round(pixel_auc.compute().item(), 4) * 100,
        "pixel AP": round(pixel_ap.compute().item(), 4) * 100,
        "image AUC": round(image_auc.item(), 4) * 100,
        "image AP": round(image_ap.item(), 4) * 100,
    }


def main():
    parser = argparse.ArgumentParser(description="Testing")
    # model
    parser.add_argument(
        "--model_name",
        type=str,
        default="ViT-L-14-336",
        help="ViT-L-14-336",
    )
    parser.add_argument("--img_size", type=int, default=518)
    # testing
    parser.add_argument("--n_groups", type=int, default=4, help="number of groups for adapter")

    parser.add_argument("--lora_rank", type=int, default=16, help="rank for LoRA adapters")
    parser.add_argument("--lora_alpha", type=float, default=2.0, help="alpha for LoRA adapters")

    parser.add_argument("--conv_lora_rank", type=int, default=8, help="rank for LoRA adapters")
    parser.add_argument("--conv_lora_alpha", type=float, default=2.0, help="alpha for LoRA adapters")
    parser.add_argument("--conv_kernel_size_list", type=int, nargs="+", default=[3, 5],
                        help="kernel size for convolutional LoRA adapters")
    parser.add_argument(
        "--dfg_mode",
        type=str,
        choices=["mlp", "attn"],
        default="mlp",
        help="DFG fusion mode: original MLP gate or Phase 1A dual-softmax attention",
    )
    parser.add_argument("--dfg_attn_dim", type=int, default=256, help="attention dimension for Phase 1A DFG")
    parser.add_argument("--dfg_attn_tau", type=float, default=4.0, help="fixed attention temperature for Phase 1A DFG")
    parser.add_argument("--use_ss2d_dfg", action="store_true", help="enable Phase 1B SS2D residual query branch")
    parser.add_argument("--dfg_gamma_max", type=float, default=0.2, help="max abs SS2D residual scale for Phase 1B")
    parser.add_argument(
        "--dfg_ss2d_fusion",
        type=str,
        choices=["feature_residual", "weight_residual"],
        default="feature_residual",
        help="SS2D DFG fusion mode: feature residual query shift or post-softmax weight residual",
    )
    parser.add_argument("--dfg_beta", type=float, default=0.10, help="fixed beta for weight_residual SS2D DFG")
    parser.add_argument(
        "--dfg_beta_schedule",
        type=str,
        choices=["fixed", "warmup010"],
        default="fixed",
        help="beta schedule for weight_residual SS2D DFG",
    )
    parser.add_argument("--dfg_beta_target", type=float, default=0.10, help="target beta for beta schedules")

    parser.add_argument("--dataset", type=str, default="MPDD")
    parser.add_argument("--batch_size", type=int, default=84)
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="ckpt/issue")
    parser.add_argument("--num_workers", type=int, default=4 if os.name != "nt" else 0)
    parser.add_argument(
        "--epochs",
        type=int,
        nargs="+",
        default=None,
        help="Only test selected adapter epochs, e.g. --epochs 10 15 20. Default: test all adapter_*.pth files.",
    )
    parser.add_argument(
        "--metric_thresholds",
        type=int,
        default=None,
        help="Use streaming binned AUROC/AP with this many thresholds to avoid storing all pixel maps. Default: exact metrics.",
    )
    parser.add_argument(
        "--pixel_stride",
        type=int,
        default=1,
        help="Evaluate every Nth pixel for memory-safe exact/subsampled metrics, e.g. --pixel_stride 4.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Debug only: limit each class dataset to the first N samples. Do not use for final reporting.",
    )
    parser.add_argument(
        "--max_samples_per_label",
        type=int,
        default=None,
        help="Debug only: limit each class dataset to N samples per label. Prefer this over --max_samples.",
    )

    args = parser.parse_args()
    # ========================================================
    os.makedirs(args.save_path, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(args.save_path, "test.log"),
        encoding="utf-8",
        level=logging.INFO,
        format="%(asctime)s %(filename)s %(lineno)d: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    logger.info("args: %s", vars(args))
    use_cuda = torch.cuda.is_available()
    device = torch.device(f"cuda:{args.cuda_device}" if use_cuda else "cpu")
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
    ).to(device)
    model.eval()
    ckp_files = sorted(glob(args.save_path + "/adapter_*.pth"), key=get_epoch_from_checkpoint)
    if args.epochs is not None:
        selected_epochs = set(args.epochs)
        ckp_files = [file for file in ckp_files if get_epoch_from_checkpoint(file) in selected_epochs]
    assert len(ckp_files) > 0, "adapter checkpoint not found"
    for file in ckp_files:
        checkpoint = torch.load(file, map_location=device)
        if checkpoint.get("dfg_mode", args.dfg_mode) != args.dfg_mode:
            raise ValueError(
                f"Checkpoint DFG mode is {checkpoint['dfg_mode']!r}, "
                f"but --dfg_mode is {args.dfg_mode!r}."
            )
        if checkpoint.get("n_groups", args.n_groups) != args.n_groups:
            raise ValueError(
                f"Checkpoint n_groups is {checkpoint['n_groups']!r}, "
                f"but --n_groups is {args.n_groups!r}."
            )
        if args.dfg_mode == "attn":
            if checkpoint.get("dfg_attn_dim", args.dfg_attn_dim) != args.dfg_attn_dim:
                raise ValueError(
                    f"Checkpoint dfg_attn_dim is {checkpoint['dfg_attn_dim']!r}, "
                    f"but --dfg_attn_dim is {args.dfg_attn_dim!r}."
                )
            ckpt_tau = checkpoint.get("dfg_attn_tau", args.dfg_attn_tau)
            if abs(float(ckpt_tau) - args.dfg_attn_tau) > 1e-8:
                raise ValueError(
                    f"Checkpoint dfg_attn_tau is {ckpt_tau!r}, "
                    f"but --dfg_attn_tau is {args.dfg_attn_tau!r}."
                )
            if bool(checkpoint.get("use_ss2d_dfg", False)) != args.use_ss2d_dfg:
                raise ValueError(
                    f"Checkpoint use_ss2d_dfg is {checkpoint.get('use_ss2d_dfg', False)!r}, "
                    f"but --use_ss2d_dfg is {args.use_ss2d_dfg!r}."
                )
            ckpt_gamma_max = checkpoint.get("dfg_gamma_max", args.dfg_gamma_max)
            if args.use_ss2d_dfg and abs(float(ckpt_gamma_max) - args.dfg_gamma_max) > 1e-8:
                raise ValueError(
                    f"Checkpoint dfg_gamma_max is {ckpt_gamma_max!r}, "
                    f"but --dfg_gamma_max is {args.dfg_gamma_max!r}."
                )
            ckpt_fusion = checkpoint.get("dfg_ss2d_fusion", "feature_residual")
            if ckpt_fusion != args.dfg_ss2d_fusion:
                raise ValueError(
                    f"Checkpoint dfg_ss2d_fusion is {ckpt_fusion!r}, "
                    f"but --dfg_ss2d_fusion is {args.dfg_ss2d_fusion!r}."
                )
            ckpt_beta = checkpoint.get("dfg_beta", 0.10)
            ckpt_beta_schedule = checkpoint.get("dfg_beta_schedule", "fixed")
            ckpt_beta_target = checkpoint.get("dfg_beta_target", ckpt_beta)
            ckpt_beta_current = checkpoint.get("dfg_beta_current", ckpt_beta)
            ckpt_weight_residual_fp32 = checkpoint.get("dfg_weight_residual_fp32", True)
            if ckpt_beta_schedule not in ["fixed", "warmup010"]:
                raise ValueError(f"Checkpoint dfg_beta_schedule is invalid: {ckpt_beta_schedule!r}.")
            if not 0 <= float(ckpt_beta_target) <= 1:
                raise ValueError(f"Checkpoint dfg_beta_target is invalid: {ckpt_beta_target!r}.")
            if not 0 <= float(ckpt_beta_current) <= 1:
                raise ValueError(f"Checkpoint dfg_beta_current is invalid: {ckpt_beta_current!r}.")
            model.dfg_beta_schedule = ckpt_beta_schedule
            model.dfg_beta_target = float(ckpt_beta_target)
            model.dfg_weight_residual_fp32 = bool(ckpt_weight_residual_fp32)
            model.set_dfg_beta(float(ckpt_beta_current))
        model.image_adapter.load_state_dict(checkpoint["image_adapter"])
        model.text_adapter.load_state_dict(checkpoint["text_adapter"])
        test_epoch = checkpoint["epoch"]
        logger.info("-----------------------------------------------")
        logger.info("load model from epoch %d", test_epoch)
        logger.info("-----------------------------------------------")
        image_datasets = get_text_and_image_dataset(
            args.dataset,
            args.img_size,
            "test"
        )
        df = DataFrame(
            columns=[
                "class name",
                "pixel AUC",
                "pixel AP",
                "image AUC",
                "image AP",
            ]
        )
        with torch.no_grad():
            text_embeddings = get_multiple_adapted_text_embedding(model, args.dataset, device)

        for class_name, image_dataset in image_datasets.items():
            if args.max_samples_per_label is not None:
                image_dataset = limit_dataset_by_label(image_dataset, args.max_samples_per_label)
            if args.max_samples is not None:
                image_dataset = torch.utils.data.Subset(
                    image_dataset,
                    range(min(args.max_samples, len(image_dataset))),
                )
            image_dataloader = torch.utils.data.DataLoader(
                image_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=use_cuda,
            )
            with torch.no_grad():
                class_text_embeddings = text_embeddings[class_name]
                if args.metric_thresholds is None:
                    masks, labels, preds, preds_image, file_names = get_predictions(
                        model=model,
                        class_text_embeddings=class_text_embeddings,
                        test_loader=image_dataloader,
                        device=device,
                        dataset=args.dataset,
                    )
                    class_result_dict = metrics_eval_gpu(
                        masks[:, :, ::args.pixel_stride, ::args.pixel_stride],
                        labels,
                        preds[:, ::args.pixel_stride, ::args.pixel_stride],
                        preds_image,
                        class_name,
                        domain=DOMAINS[args.dataset],
                    )
                else:
                    class_result_dict = get_streaming_metrics(
                        model=model,
                        class_text_embeddings=class_text_embeddings,
                        test_loader=image_dataloader,
                        device=device,
                        class_name=class_name,
                        dataset=args.dataset,
                        thresholds=args.metric_thresholds,
                        pixel_stride=args.pixel_stride,
                    )
            df.loc[len(df)] = Series(class_result_dict)
            if use_cuda:
                torch.cuda.empty_cache()
        mean_vals = df[df.columns[1:]].mean()
        df.loc[len(df), df.columns[1:]] = mean_vals
        df.loc[len(df) - 1, "class name"] = "Average"
        logger.info("final results:\n%s", df.to_string(index=False, justify="center"))
        if use_cuda:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
