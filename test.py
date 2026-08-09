import argparse
import heapq
import logging
import os
import tempfile
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pandas import DataFrame, Series
from torch.utils.data import DataLoader
from torchmetrics.functional import auroc, average_precision
from torchmetrics.classification import BinaryAUROC, BinaryAveragePrecision
from tqdm import tqdm

from dataset import DOMAINS, get_text_and_image_dataset
from utils import (
    configure_canonical_fp32, get_multiple_adapted_text_embedding, log_preflight,
    metrics_eval_gpu,
)
from dataset.info import log_data_root
from model.adapter import (
    ACDCLIP
)
from model.checkpoint_utils import h6_config_from_checkpoint, load_adapter_checkpoint
from utils import get_phase2b_global_text_features
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
        if model.h6_enabled:
            visual_output = model(image, return_phase4_features=True)
            h6_batch = model.h6.build_batch(
                model, dataset, list(class_name), visual_output, model.hybrid_alpha_current
            )
            seg_features = torch.stack(visual_output["seg_tokens"], dim=0)
            det_features = torch.stack(visual_output["det_tokens"], dim=0)
            h6_mode = getattr(model, "h6_global_text_mode", "phase2b_hybrid")
            if h6_mode in ("phase2b_hybrid", "hard_anchor"):
                is_hybrid = h6_mode == "phase2b_hybrid" and getattr(model, "use_hybrid_soft_prompt", False)
                epoch_text_features = get_phase2b_global_text_features(
                    model, dataset, list(class_name), device,
                    use_hybrid_soft_prompt=is_hybrid,
                    use_soft_prompt=getattr(model, "use_soft_prompt", False) if is_hybrid else False,
                ).to(dtype=det_features.dtype)
            else:
                epoch_text_features = h6_batch["text_global"].to(dtype=det_features.dtype)
            h6_patch_logits = h6_batch["h6_logits"]
        else:
            epoch_text_embeddings = class_text_embeddings.unsqueeze(dim=1)  # [n_groups, 1, 768, 2]
            seg_tokens, det_tokens = model(image)
            seg_features = torch.stack(seg_tokens, dim=0)
            det_features = torch.stack(det_tokens, dim=0)
            B = seg_features.shape[1]
            epoch_text_features = epoch_text_embeddings.repeat(1, B, 1, 1)
            h6_patch_logits = None
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
        seg_pred = model.vision_text_fusion_gate_seg(
            seg_features, epoch_text_features, test_mode=True, domain=DOMAINS[dataset], h6_patch_logits=h6_patch_logits
        )
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

        if model.h6_enabled:
            visual_output = model(image, return_phase4_features=True)
            h6_batch = model.h6.build_batch(
                model, dataset, list(batch_class_name), visual_output, model.hybrid_alpha_current
            )
            seg_features = torch.stack(visual_output["seg_tokens"], dim=0)
            det_features = torch.stack(visual_output["det_tokens"], dim=0)
            h6_mode = getattr(model, "h6_global_text_mode", "phase2b_hybrid")
            if h6_mode in ("phase2b_hybrid", "hard_anchor"):
                is_hybrid = h6_mode == "phase2b_hybrid" and getattr(model, "use_hybrid_soft_prompt", False)
                epoch_text_features = get_phase2b_global_text_features(
                    model, dataset, list(batch_class_name), device,
                    use_hybrid_soft_prompt=is_hybrid,
                    use_soft_prompt=getattr(model, "use_soft_prompt", False) if is_hybrid else False,
                ).to(dtype=det_features.dtype)
            else:
                epoch_text_features = h6_batch["text_global"].to(dtype=det_features.dtype)
            h6_patch_logits = h6_batch["h6_logits"]
        else:
            epoch_text_features = class_text_embeddings.unsqueeze(dim=1)
            seg_tokens, det_tokens = model(image)
            seg_features = torch.stack(seg_tokens, dim=0)
            det_features = torch.stack(det_tokens, dim=0)
            B = seg_features.shape[1]
            epoch_text_features = epoch_text_features.repeat(1, B, 1, 1)
            h6_patch_logits = None
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
            h6_patch_logits=h6_patch_logits,
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
        image_auc = None
        image_ap = None

    return {
        "class name": class_name,
        "pixel AUC": round(pixel_auc.compute().item(), 4) * 100,
        "pixel AP": round(pixel_ap.compute().item(), 4) * 100,
        "image AUC": "N/A" if image_auc is None else round(image_auc.item(), 4) * 100,
        "image AP": "N/A" if image_ap is None else round(image_ap.item(), 4) * 100,
    }


def _write_sorted_metric_chunk(score_parts, label_parts, temp_dir: str, chunk_index: int):
    scores = np.concatenate(score_parts).astype(np.float32, copy=False)
    labels = np.concatenate(label_parts).astype(np.uint8, copy=False)
    order = np.argsort(-scores, kind="mergesort")
    score_path = os.path.join(temp_dir, f"scores_{chunk_index:05d}.npy")
    label_path = os.path.join(temp_dir, f"labels_{chunk_index:05d}.npy")
    np.save(score_path, scores[order])
    np.save(label_path, labels[order])
    return score_path, label_path


def _exact_auc_ap_from_sorted_chunks(chunks, total_pos: int, total_neg: int):
    if total_pos == 0 or total_neg == 0:
        return 0.0, 0.0
    arrays = [
        (np.load(score_path, mmap_mode="r"), np.load(label_path, mmap_mode="r"))
        for score_path, label_path in chunks
    ]
    heap = []
    for chunk_id, (scores, labels) in enumerate(arrays):
        if len(scores) > 0:
            heapq.heappush(heap, (-float(scores[0]), chunk_id, 0))

    pos_seen = 0
    neg_seen = 0
    auc_pair_sum = 0.0
    ap = 0.0
    while heap:
        score = heap[0][0]
        group_pos = 0
        group_neg = 0
        while heap and heap[0][0] == score:
            _, chunk_id, index = heapq.heappop(heap)
            labels = arrays[chunk_id][1]
            if int(labels[index]) == 1:
                group_pos += 1
            else:
                group_neg += 1
            next_index = index + 1
            scores = arrays[chunk_id][0]
            if next_index < len(scores):
                heapq.heappush(heap, (-float(scores[next_index]), chunk_id, next_index))

        auc_pair_sum += group_neg * (pos_seen + 0.5 * group_pos)
        pos_seen += group_pos
        neg_seen += group_neg
        if group_pos > 0:
            precision = pos_seen / max(pos_seen + neg_seen, 1)
            ap += precision * (group_pos / total_pos)

    return (auc_pair_sum / (total_pos * total_neg)) * 100.0, ap * 100.0


def get_external_exact_metrics(
        model: ACDCLIP,
        class_text_embeddings: torch.Tensor,
        test_loader: DataLoader,
        device,
        class_name: str,
        dataset: str = "MVTec",
        pixel_stride: int = 1,
        chunk_pixels: int = 5_000_000,
):
    image_labels = []
    image_preds = []
    score_parts = []
    label_parts = []
    buffered = 0
    total_pos = 0
    total_neg = 0
    chunks = []
    with tempfile.TemporaryDirectory(prefix="acdclip_exact_pixels_") as temp_dir:
        for input_data in tqdm(test_loader):
            image = input_data["image"].to(device)
            mask = input_data["mask"].to(device).to(torch.int32)
            label = input_data["label"].to(device).to(torch.int32)
            batch_class_name = input_data["class_name"]
            assert len(set(batch_class_name)) == 1, "mixed class not supported"

            if model.h6_enabled:
                visual_output = model(image, return_phase4_features=True)
                h6_batch = model.h6.build_batch(
                    model, dataset, list(batch_class_name), visual_output, model.hybrid_alpha_current
                )
                seg_features = torch.stack(visual_output["seg_tokens"], dim=0)
                det_features = torch.stack(visual_output["det_tokens"], dim=0)
                h6_mode = getattr(model, "h6_global_text_mode", "phase2b_hybrid")
                if h6_mode in ("phase2b_hybrid", "hard_anchor"):
                    is_hybrid = h6_mode == "phase2b_hybrid" and getattr(model, "use_hybrid_soft_prompt", False)
                    epoch_text_features = get_phase2b_global_text_features(
                        model, dataset, list(batch_class_name), device,
                        use_hybrid_soft_prompt=is_hybrid,
                        use_soft_prompt=getattr(model, "use_soft_prompt", False) if is_hybrid else False,
                    ).to(dtype=det_features.dtype)
                else:
                    epoch_text_features = h6_batch["text_global"].to(dtype=det_features.dtype)
                h6_patch_logits = h6_batch["h6_logits"]
            else:
                epoch_text_features = class_text_embeddings.unsqueeze(dim=1)
                seg_tokens, det_tokens = model(image)
                seg_features = torch.stack(seg_tokens, dim=0)
                det_features = torch.stack(det_tokens, dim=0)
                batch_size = seg_features.shape[1]
                epoch_text_features = epoch_text_features.repeat(1, batch_size, 1, 1)
                h6_patch_logits = None

            cls_preds = [
                torch.matmul(det_features[i].unsqueeze(dim=1), epoch_text_features[i]).squeeze(1)
                for i in range(det_features.shape[0])
            ]
            cls_preds = torch.stack(cls_preds, dim=0).mean(dim=0)
            pred_image = F.softmax(cls_preds, dim=1)[:, 1]
            seg_pred = model.vision_text_fusion_gate_seg(
                seg_features,
                epoch_text_features,
                test_mode=True,
                domain=DOMAINS[dataset],
                h6_patch_logits=h6_patch_logits,
            )
            flat_seg = torch.flatten(seg_pred, start_dim=1)
            pmax_pred, _ = torch.max(flat_seg, dim=1)
            if DOMAINS[dataset] == "Medical":
                pred_image = pred_image * 0.5 + pmax_pred * 0.5
            else:
                pred_image = pred_image * 0.9 + pmax_pred * 0.1

            if pixel_stride > 1:
                seg_pred = seg_pred[:, ::pixel_stride, ::pixel_stride]
                mask = mask[:, :, ::pixel_stride, ::pixel_stride]
            scores = seg_pred.detach().float().flatten().cpu().numpy()
            labels = mask.detach().flatten().cpu().numpy().astype(np.uint8, copy=False)
            total_pos += int(labels.sum())
            total_neg += int(labels.size - labels.sum())
            score_parts.append(scores)
            label_parts.append(labels)
            buffered += labels.size
            if buffered >= chunk_pixels:
                chunks.append(_write_sorted_metric_chunk(score_parts, label_parts, temp_dir, len(chunks)))
                score_parts = []
                label_parts = []
                buffered = 0

            image_labels.append(label.detach().cpu())
            image_preds.append(pred_image.detach().cpu())
            if device.type == "cuda":
                torch.cuda.empty_cache()

        if score_parts:
            chunks.append(_write_sorted_metric_chunk(score_parts, label_parts, temp_dir, len(chunks)))

        pixel_auc, pixel_ap = _exact_auc_ap_from_sorted_chunks(chunks, total_pos, total_neg)
        image_label = torch.concatenate(image_labels, dim=0).flatten()
        image_pred = torch.concatenate(image_preds, dim=0).flatten()
        if image_label.max() != image_label.min():
            image_auc = auroc(image_pred, image_label, task="binary")
            image_ap = average_precision(image_pred, image_label, task="binary")
        else:
            image_auc = None
            image_ap = None

    return {
        "class name": class_name,
        "pixel AUC": round(float(pixel_auc), 4),
        "pixel AP": round(float(pixel_ap), 4),
        "image AUC": "N/A" if image_auc is None else round(image_auc.item(), 4) * 100,
        "image AP": "N/A" if image_ap is None else round(image_ap.item(), 4) * 100,
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
    parser.add_argument("--use_soft_prompt", action="store_true", help="force Phase2B soft prompt at test time")
    parser.add_argument("--use_hybrid_soft_prompt", action="store_true", help="force Phase2B hard-soft hybrid prompt")
    parser.add_argument("--hybrid_alpha", type=float, default=None, help="force hybrid alpha when checkpoint lacks it")
    parser.add_argument("--soft_prompt_ctx_len", type=int, default=4)
    parser.add_argument("--soft_prompt_init", type=str, choices=["phrase", "random"], default="phrase")
    parser.add_argument("--soft_prompt_init_phrase", type=str, default="a photo of a")
    parser.add_argument("--h6_progress", type=int, choices=[0, 1], default=None)
    parser.add_argument("--h6_num_factors", type=int, default=4)
    parser.add_argument("--h6_top_k", type=int, default=2)
    parser.add_argument("--h6_bank_dim", type=int, default=256)
    parser.add_argument("--h6_router_dim", type=int, default=128)
    parser.add_argument("--h6_router_temperature", type=float, default=1.0)
    parser.add_argument("--h6_router_soft_epochs", type=int, default=2)
    parser.add_argument("--h6_dense_routing_epochs", type=int, default=None)
    parser.add_argument("--h6_sparse_start_epoch", type=int, default=None)
    parser.add_argument("--h6_sparse_transition_epochs", type=int, default=1)
    parser.add_argument("--h6_global_text_mode", type=str, choices=["hard_anchor", "phase2b_hybrid", "dynamic_legacy"], default="hard_anchor")
    parser.add_argument("--h6_prediction_routing", type=str, choices=["dense", "scheduled_topk", "readiness_topk"], default="dense")
    parser.add_argument("--h6_diagnostics_mode", type=str, choices=["none", "light", "full"], default="light")
    parser.add_argument("--h6_diagnostics_interval", type=int, default=1)
    parser.add_argument("--h6_load_bias_momentum", type=float, default=0.9)
    parser.add_argument("--h6_load_bias_step", type=float, default=0.001)
    parser.add_argument("--h6_load_bias_max", type=float, default=0.03)
    parser.add_argument("--h6_vae_hidden_dim", type=int, default=512)
    parser.add_argument("--h6_vae_latent_dim", type=int, default=256)
    parser.add_argument("--h6_vae_class_ratio", type=float, default=0.25)
    parser.add_argument("--h6_slot_init_enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--h6_slot_init_scale", type=float, default=0.02)
    parser.add_argument("--h6_slot_init_seed_offset", type=int, default=6100)
    parser.add_argument("--h6_factor_grad_diagnostics", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--h6_late_factor_identity_enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--h6_factor_id_scale", type=float, default=0.02)
    parser.add_argument("--h6_factor_id_max_ratio", type=float, default=0.05)
    parser.add_argument("--h6_factor_generator_specialization_enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--h6_factor_head_init_scale", type=float, default=1e-3)
    parser.add_argument("--h6_factor_local_dynamic_mix", type=float, default=0.0)
    parser.add_argument("--h6_cluster_responsibility", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--h6_cluster_centroid_path", type=str, default=None)
    parser.add_argument("--h6_cluster_temperature", type=float, default=0.10)
    parser.add_argument("--h6_lambda_cluster_resp", type=float, default=0.0)
    parser.add_argument(
        "--h6_router_query_mode",
        choices=["raw", "local_residual", "local_global_bypass"],
        default="local_global_bypass",
    )
    parser.add_argument("--h6_router_query_global_weight", type=float, default=0.10)
    parser.add_argument("--h6_router_local_bypass_scale", type=float, default=0.10)
    parser.add_argument("--h6_router_local_bypass_max_ratio", type=float, default=0.20)
    parser.add_argument("--h6_router_local_projection_seed_offset", type=int, default=7200)
    parser.add_argument("--h6_router_key_anchor_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--h6_router_key_anchor_seed_offset", type=int, default=7300)
    parser.add_argument("--h6_router_key_adaptation_initial_ratio", type=float, default=0.10)
    parser.add_argument("--h6_router_key_adaptation_max_ratio", type=float, default=0.25)
    parser.add_argument("--h6_factor_context_anchor_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--h6_factor_context_anchor_seed_offset", type=int, default=7400)
    parser.add_argument("--h6_factor_context_adaptation_initial_ratio", type=float, default=0.10)
    parser.add_argument("--h6_factor_context_adaptation_max_ratio", type=float, default=0.25)
    parser.add_argument("--h6_factor_identity_tangent_projection_enabled", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lambda_h6_dynamic_mean_anchor", type=float, default=0.001)
    parser.add_argument("--h6_dynamic_mean_anchor_min_cosine", type=float, default=0.70)
    parser.add_argument("--h6_dynamic_mean_anchor_start_epoch", type=int, default=4)
    parser.add_argument("--h6_dynamic_mean_anchor_warmup_epochs", type=int, default=3)
    parser.add_argument("--h6_progress_version", choices=["P1-v6", "P1-v7-full", "P1-v8-minimal", "P1-v8.3"], default="P1-v8.3")
    parser.add_argument("--h6_local_factor_mode", type=str, choices=["legacy_mix", "center_spread"], default="center_spread")
    parser.add_argument("--h6_local_center_mix", type=float, default=0.05)
    parser.add_argument("--h6_local_factor_spread", type=float, default=0.10)
    parser.add_argument("--h6_expert_enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--h6_test_rho_override", type=float, default=None, help="Override rho value during inference (e.g., 0.0 to disable local residual)")
    parser.add_argument("--h6_expert_bottleneck", type=int, default=64)
    parser.add_argument("--h6_expert_fofs_seed_offset", type=int, default=7500)
    parser.add_argument("--h6_expert_state_condition_scale", type=float, default=0.25)
    parser.add_argument("--h6_expert_scale_target", type=float, default=0.10)
    parser.add_argument("--h6_expert_scale_start_epoch", type=int, default=1)
    parser.add_argument("--h6_expert_scale_warmup_epochs", type=int, default=6)
    parser.add_argument("--h6_expert_max_relative_ratio", type=float, default=0.10)
    parser.add_argument(
        "--h6_router_teacher_mode",
        choices=["raw_cosine", "state_centered_cosine", "negative_squared_distance"],
        default="raw_cosine",
    )
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
    parser.add_argument(
        "--medical_split",
        choices=["test", "val"],
        default="test",
        help="Phase4 medical evaluation split. Validation is used only for checkpoint selection.",
    )
    parser.add_argument(
        "--medical_manifest_root",
        default=None,
        help="Run-local directory generated by tools/prepare_phase4_medical_splits.py.",
    )
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
        "--external_exact_pixel_metrics",
        action="store_true",
        help="Compute exact pixel AUROC/AP with disk-backed sorted chunks instead of holding all pixels in RAM.",
    )
    parser.add_argument(
        "--external_metric_chunk_pixels",
        type=int,
        default=5_000_000,
        help="Pixels per disk-backed exact metric sort chunk.",
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
    parser.add_argument("--h6_load_bias_enabled", action=argparse.BooleanOptionalAction, default=False)


    args = parser.parse_args()
    configure_canonical_fp32()
    if args.h6_dense_routing_epochs is not None:
        args.h6_router_soft_epochs = int(args.h6_dense_routing_epochs)
    if args.h6_sparse_start_epoch is not None:
        expected_sparse_start = int(args.h6_router_soft_epochs) + 1
        if int(args.h6_sparse_start_epoch) != expected_sparse_start:
            raise ValueError(
                "--h6_sparse_start_epoch must equal --h6_dense_routing_epochs + 1 "
                f"(or --h6_router_soft_epochs + 1); got {args.h6_sparse_start_epoch} vs {expected_sparse_start}"
            )
    if args.use_soft_prompt and args.use_hybrid_soft_prompt:
        raise ValueError("--use_soft_prompt and --use_hybrid_soft_prompt are mutually exclusive")
    preflight_files = sorted(glob(args.save_path + "/adapter_*.pth"), key=get_epoch_from_checkpoint)
    if args.epochs is not None:
        selected_epochs = set(args.epochs)
        preflight_files = [file for file in preflight_files if get_epoch_from_checkpoint(file) in selected_epochs]
    assert len(preflight_files) > 0, "adapter checkpoint not found"
    preflight_checkpoint = torch.load(preflight_files[0], map_location="cpu")
    preflight_h6 = h6_config_from_checkpoint(preflight_checkpoint)
    if preflight_h6 is not None:
        if args.h6_progress is not None and args.h6_progress != int(preflight_h6["progress"]):
            raise ValueError(
                f"checkpoint phase4_progress is {preflight_h6['progress']}, but --h6_progress is {args.h6_progress}"
            )
        args.h6_progress = int(preflight_h6["progress"])
        args.h6_num_factors = int(preflight_h6["num_factors"])
        args.h6_top_k = int(preflight_h6["top_k"])
        args.h6_bank_dim = int(preflight_h6["bank_dim"])
        args.h6_router_dim = int(preflight_h6["router_dim"])
        args.h6_router_temperature = float(preflight_h6["router_temperature"])
        args.h6_router_soft_epochs = int(preflight_h6["router_soft_epochs"])
        args.h6_sparse_transition_epochs = int(preflight_h6.get("sparse_transition_epochs", 1))
        args.h6_load_bias_enabled = bool(preflight_h6.get("load_bias_enabled", False))
        args.h6_load_bias_momentum = float(preflight_h6.get("load_bias_momentum", 0.9))
        args.h6_load_bias_step = float(preflight_h6.get("load_bias_step", 0.001))
        args.h6_load_bias_max = float(preflight_h6.get("load_bias_max", 0.03))
        args.h6_vae_hidden_dim = int(preflight_h6["vae_hidden_dim"])
        args.h6_vae_latent_dim = int(preflight_h6["vae_latent_dim"])
        args.h6_vae_class_ratio = float(preflight_h6.get("vae_class_ratio", 0.25))
        args.h6_slot_init_enabled = bool(preflight_h6.get("slot_init_enabled", False))
        args.h6_slot_init_scale = float(preflight_h6.get("slot_init_scale", 0.02))
        args.h6_slot_init_seed_offset = int(preflight_h6.get("slot_init_seed_offset", 6100))
        args.h6_factor_grad_diagnostics = bool(preflight_h6.get("factor_grad_diagnostics_enabled", False))
        args.h6_late_factor_identity_enabled = bool(preflight_h6.get("late_factor_identity_enabled", False))
        args.h6_factor_id_scale = float(preflight_h6.get("factor_id_scale", 0.02))
        args.h6_factor_id_max_ratio = float(preflight_h6.get("factor_id_max_ratio", 0.05))
        args.h6_factor_generator_specialization_enabled = bool(
            preflight_h6.get("factor_generator_specialization_enabled", False)
        )
        args.h6_factor_head_init_scale = float(preflight_h6.get("factor_head_init_scale", 1e-3))
        args.h6_factor_local_dynamic_mix = float(preflight_h6.get("factor_local_dynamic_mix", 0.0))
        args.h6_cluster_responsibility = bool(preflight_h6.get("cluster_responsibility_enabled", False))
        args.h6_cluster_temperature = float(preflight_h6.get("cluster_temperature", 0.10))
        args.h6_lambda_cluster_resp = float(preflight_h6.get("cluster_loss_weight", 0.0))
        args.h6_router_query_mode = str(preflight_h6.get("router_query_mode", "local_global_bypass"))
        args.h6_router_query_global_weight = float(preflight_h6.get("router_query_global_weight", 0.10))
        args.h6_router_local_bypass_scale = float(preflight_h6.get("router_local_bypass_scale", 0.10))
        args.h6_router_local_bypass_max_ratio = float(preflight_h6.get("router_local_bypass_max_ratio", 0.20))
        args.h6_router_local_projection_seed_offset = int(preflight_h6.get("router_local_projection_seed_offset", 7200))
        args.h6_router_key_anchor_enabled = bool(preflight_h6.get("router_key_anchor_enabled", True))
        args.h6_router_key_anchor_seed_offset = int(preflight_h6.get("router_key_anchor_seed_offset", 7300))
        args.h6_router_key_adaptation_initial_ratio = float(preflight_h6.get("router_key_adaptation_initial_ratio", 0.10))
        args.h6_router_key_adaptation_max_ratio = float(preflight_h6.get("router_key_adaptation_max_ratio", 0.25))
        args.h6_factor_context_anchor_enabled = bool(preflight_h6.get("factor_context_anchor_enabled", True))
        args.h6_factor_context_anchor_seed_offset = int(preflight_h6.get("factor_context_anchor_seed_offset", 7400))
        args.h6_factor_context_adaptation_initial_ratio = float(
            preflight_h6.get("factor_context_adaptation_initial_ratio", 0.10)
        )
        args.h6_factor_context_adaptation_max_ratio = float(
            preflight_h6.get("factor_context_adaptation_max_ratio", 0.25)
        )
        args.h6_factor_identity_tangent_projection_enabled = bool(
            preflight_h6.get("factor_identity_tangent_projection_enabled", True)
        )
        args.lambda_h6_dynamic_mean_anchor = float(preflight_h6.get("lambda_dynamic_mean_anchor", 0.001))
        args.h6_dynamic_mean_anchor_min_cosine = float(preflight_h6.get("dynamic_mean_anchor_min_cosine", 0.70))
        args.h6_dynamic_mean_anchor_start_epoch = int(preflight_h6.get("dynamic_mean_anchor_start_epoch", 4))
        args.h6_dynamic_mean_anchor_warmup_epochs = int(preflight_h6.get("dynamic_mean_anchor_warmup_epochs", 3))
        args.h6_router_teacher_mode = str(preflight_h6.get("router_teacher_mode", "raw_cosine"))
        args.h6_progress_version = str(preflight_h6.get("progress_version", "P1-v6"))
        args.h6_local_factor_mode = str(preflight_h6.get("local_factor_mode", "legacy_mix"))
        args.h6_local_center_mix = float(preflight_h6.get("local_center_mix", 0.05))
        args.h6_local_factor_spread = float(preflight_h6.get("local_factor_spread", 0.10))
        args.h6_expert_enabled = bool(preflight_h6.get("expert_enabled", False))
        args.h6_expert_bottleneck = int(preflight_h6.get("expert_bottleneck", 64))
        args.h6_expert_fofs_seed_offset = int(preflight_h6.get("expert_fofs_seed_offset", 7500))
        args.h6_expert_state_condition_scale = float(preflight_h6.get("expert_state_condition_scale", 0.25))
        args.h6_expert_scale_target = float(preflight_h6.get("expert_scale_target", 0.10))
        args.h6_expert_scale_start_epoch = int(preflight_h6.get("expert_scale_start_epoch", 1))
        args.h6_expert_scale_warmup_epochs = int(preflight_h6.get("expert_scale_warmup_epochs", 6))
        args.h6_expert_max_relative_ratio = float(preflight_h6.get("expert_max_relative_ratio", 0.10))
    elif args.h6_progress is None:
        args.h6_progress = 0
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
    log_data_root(logger)
    log_preflight(logger)
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
        use_soft_prompt=args.use_soft_prompt,
        soft_prompt_ctx_len=args.soft_prompt_ctx_len,
        soft_prompt_init=args.soft_prompt_init,
        soft_prompt_init_phrase=args.soft_prompt_init_phrase,
        h6_progress=args.h6_progress,
        h6_num_factors=args.h6_num_factors,
        h6_top_k=args.h6_top_k,
        h6_bank_dim=args.h6_bank_dim,
        h6_router_dim=args.h6_router_dim,
        h6_router_temperature=args.h6_router_temperature,
        h6_router_soft_epochs=args.h6_router_soft_epochs,
        h6_sparse_transition_epochs=args.h6_sparse_transition_epochs,
        h6_load_bias_enabled=args.h6_load_bias_enabled,
        h6_load_bias_momentum=args.h6_load_bias_momentum,
        h6_load_bias_step=args.h6_load_bias_step,
        h6_load_bias_max=args.h6_load_bias_max,
        h6_vae_hidden_dim=args.h6_vae_hidden_dim,
        h6_vae_latent_dim=args.h6_vae_latent_dim,
        h6_vae_class_ratio=args.h6_vae_class_ratio,
        h6_slot_init_enabled=args.h6_slot_init_enabled,
        h6_slot_init_scale=args.h6_slot_init_scale,
        h6_slot_init_seed_offset=args.h6_slot_init_seed_offset,
        h6_factor_grad_diagnostics=args.h6_factor_grad_diagnostics,
        h6_late_factor_identity_enabled=args.h6_late_factor_identity_enabled,
        h6_factor_id_scale=args.h6_factor_id_scale,
        h6_factor_id_max_ratio=args.h6_factor_id_max_ratio,
        h6_factor_generator_specialization_enabled=args.h6_factor_generator_specialization_enabled,
        h6_factor_head_init_scale=args.h6_factor_head_init_scale,
        h6_factor_local_dynamic_mix=args.h6_factor_local_dynamic_mix,
        h6_cluster_responsibility=args.h6_cluster_responsibility,
        h6_cluster_temperature=args.h6_cluster_temperature,
        h6_router_query_mode=args.h6_router_query_mode,
        h6_router_query_global_weight=args.h6_router_query_global_weight,
        h6_router_local_bypass_scale=args.h6_router_local_bypass_scale,
        h6_router_local_bypass_max_ratio=args.h6_router_local_bypass_max_ratio,
        h6_router_local_projection_seed_offset=args.h6_router_local_projection_seed_offset,
        h6_router_key_anchor_enabled=args.h6_router_key_anchor_enabled,
        h6_router_key_anchor_seed_offset=args.h6_router_key_anchor_seed_offset,
        h6_router_key_adaptation_initial_ratio=args.h6_router_key_adaptation_initial_ratio,
        h6_router_key_adaptation_max_ratio=args.h6_router_key_adaptation_max_ratio,
        h6_factor_context_anchor_enabled=args.h6_factor_context_anchor_enabled,
        h6_factor_context_anchor_seed_offset=args.h6_factor_context_anchor_seed_offset,
        h6_factor_context_adaptation_initial_ratio=args.h6_factor_context_adaptation_initial_ratio,
        h6_factor_context_adaptation_max_ratio=args.h6_factor_context_adaptation_max_ratio,
        h6_factor_identity_tangent_projection_enabled=args.h6_factor_identity_tangent_projection_enabled,
        h6_progress_version=args.h6_progress_version,
        h6_local_factor_mode=args.h6_local_factor_mode,
        h6_local_center_mix=args.h6_local_center_mix,
        h6_local_factor_spread=args.h6_local_factor_spread,
        h6_expert_enabled=args.h6_expert_enabled,
        h6_expert_bottleneck=args.h6_expert_bottleneck,
        h6_expert_fofs_seed_offset=args.h6_expert_fofs_seed_offset,
        h6_expert_state_condition_scale=args.h6_expert_state_condition_scale,
        h6_expert_scale_target=args.h6_expert_scale_target,
        h6_expert_scale_start_epoch=args.h6_expert_scale_start_epoch,
        h6_expert_scale_warmup_epochs=args.h6_expert_scale_warmup_epochs,
        h6_expert_max_relative_ratio=args.h6_expert_max_relative_ratio,
        lambda_h6_dynamic_mean_anchor=args.lambda_h6_dynamic_mean_anchor,
        h6_dynamic_mean_anchor_min_cosine=args.h6_dynamic_mean_anchor_min_cosine,
        h6_dynamic_mean_anchor_start_epoch=args.h6_dynamic_mean_anchor_start_epoch,
        h6_dynamic_mean_anchor_warmup_epochs=args.h6_dynamic_mean_anchor_warmup_epochs,
        h6_router_teacher_mode=args.h6_router_teacher_mode,
        h6_prediction_routing=args.h6_prediction_routing,
        diagnostics_mode=args.h6_diagnostics_mode,
        diagnostics_interval=args.h6_diagnostics_interval,
        test_rho_override=args.h6_test_rho_override,
    ).to(device)
    model.eval()
    model.h6_global_text_mode = args.h6_global_text_mode
    model.prompt_mode = "h6_dynamic" if args.h6_progress == 1 else ("hybrid" if args.use_hybrid_soft_prompt else ("soft" if args.use_soft_prompt else "hard"))
    model.use_hybrid_soft_prompt = bool(args.use_hybrid_soft_prompt or args.h6_progress == 1)
    model.hybrid_alpha_current = 0.0 if args.hybrid_alpha is None else float(args.hybrid_alpha)
    ckp_files = preflight_files
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
        restored_h6 = load_adapter_checkpoint(model, checkpoint)
        ckpt_prompt_mode = checkpoint.get("prompt_mode", None)
        ckpt_use_hybrid_soft_prompt = bool(checkpoint.get("use_hybrid_soft_prompt", False))
        ckpt_use_soft_prompt = bool(checkpoint.get("use_soft_prompt", False))
        if ckpt_prompt_mode == "hybrid" or ckpt_use_hybrid_soft_prompt:
            model.prompt_mode = "hybrid"
            model.use_hybrid_soft_prompt = True
            model.use_soft_prompt = False
            ckpt_ctx_len = int(checkpoint.get("soft_prompt_ctx_len", model.soft_prompt_ctx_len))
            if ckpt_ctx_len != model.soft_prompt_ctx_len:
                raise ValueError(
                    f"Checkpoint soft_prompt_ctx_len is {ckpt_ctx_len}, "
                    f"but model expects {model.soft_prompt_ctx_len}."
                )
            if "soft_prompt" not in checkpoint:
                logger.warning("hybrid checkpoint has no soft_prompt state; using initialized ctx")
            else:
                model.soft_prompt.load_state_dict(checkpoint["soft_prompt"])
            model.hybrid_alpha_current = float(
                checkpoint.get(
                    "hybrid_alpha_current",
                    0.0 if args.hybrid_alpha is None else args.hybrid_alpha,
                )
            )
        elif ckpt_use_soft_prompt:
            model.prompt_mode = "soft"
            model.use_hybrid_soft_prompt = False
            model.use_soft_prompt = True
            ckpt_ctx_len = int(checkpoint.get("soft_prompt_ctx_len", model.soft_prompt_ctx_len))
            if ckpt_ctx_len != model.soft_prompt_ctx_len:
                raise ValueError(
                    f"Checkpoint soft_prompt_ctx_len is {ckpt_ctx_len}, "
                    f"but model expects {model.soft_prompt_ctx_len}."
                )
            if "soft_prompt" not in checkpoint:
                logger.warning("checkpoint declares soft prompt but has no soft_prompt state; using initialized ctx")
            else:
                model.soft_prompt.load_state_dict(checkpoint["soft_prompt"])
        elif args.use_soft_prompt:
            model.prompt_mode = "soft"
            model.use_hybrid_soft_prompt = False
            model.use_soft_prompt = True
            logger.warning(
                "using soft prompt for checkpoint without soft_prompt state; using initialized ctx"
            )
        elif args.use_hybrid_soft_prompt:
            model.prompt_mode = "hybrid"
            model.use_hybrid_soft_prompt = True
            model.use_soft_prompt = False
            model.hybrid_alpha_current = 0.0 if args.hybrid_alpha is None else float(args.hybrid_alpha)
            logger.warning(
                "using hybrid soft prompt for checkpoint without hybrid metadata; using initialized ctx"
            )
        else:
            model.prompt_mode = "hard"
            model.use_hybrid_soft_prompt = False
            model.use_soft_prompt = False
        if restored_h6:
            model.prompt_mode = "h6_dynamic"
            model.use_hybrid_soft_prompt = True
            model.use_soft_prompt = False
            model.hybrid_alpha_current = float(checkpoint.get("hybrid_alpha_current", 0.20))
            model.h6.set_epoch(int(checkpoint.get("router_warmup_epoch", checkpoint["epoch"])))
        test_epoch = checkpoint["epoch"]
        logger.info("-----------------------------------------------")
        logger.info("load model from epoch %d", test_epoch)
        effective_prompt_mode = getattr(model, "prompt_mode", "hard")
        logger.info(
            "effective_prompt_mode=%s effective_alpha=%s hard_branch_lora_used=%s "
            "soft_branch_lora_used=False use_soft_prompt=%s use_hybrid_soft_prompt=%s",
            effective_prompt_mode,
            getattr(model, "hybrid_alpha_current", 0.0),
            effective_prompt_mode in ["hard", "hybrid"],
            getattr(model, "use_soft_prompt", False),
            getattr(model, "use_hybrid_soft_prompt", False),
        )
        logger.info("-----------------------------------------------")
        eval_stage = args.medical_split if DOMAINS[args.dataset] == "Medical" else "test"
        image_datasets = get_text_and_image_dataset(
            args.dataset,
            args.img_size,
            eval_stage,
            medical_manifest_root=args.medical_manifest_root,
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
                if args.metric_thresholds is not None:
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
                elif args.external_exact_pixel_metrics:
                    class_result_dict = get_external_exact_metrics(
                        model=model,
                        class_text_embeddings=class_text_embeddings,
                        test_loader=image_dataloader,
                        device=device,
                        class_name=class_name,
                        dataset=args.dataset,
                        pixel_stride=args.pixel_stride,
                        chunk_pixels=args.external_metric_chunk_pixels,
                    )
                else:
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
            df.loc[len(df)] = Series(class_result_dict)
            if use_cuda:
                torch.cuda.empty_cache()
        mean_vals = df[df.columns[1:]].replace("N/A", np.nan).astype(float).mean()
        df.loc[len(df), df.columns[1:]] = mean_vals
        df.loc[len(df) - 1, "class name"] = "Average"
        df.to_csv(
            os.path.join(args.save_path, f"exact_results_{args.dataset}_{eval_stage}_epoch_{test_epoch}.csv"),
            index=False,
        )
        logger.info("final results:\n%s", df.to_string(index=False, justify="center"))
        if use_cuda:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
