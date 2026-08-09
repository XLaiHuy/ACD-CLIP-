import logging
import os
import platform
import random
import subprocess

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import auroc, average_precision
from torchvision import transforms

from dataset.info import CLASS_NAMES, REAL_NAMES, PROMPTS
from model.tokenizer import tokenize


def seed_worker(worker_id: int) -> None:
    """Seed Python and NumPy RNGs from PyTorch's worker-specific seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_dataloader_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def configure_canonical_fp32() -> None:
    """Make the P1-v8.3 numerical policy explicit instead of relying on defaults."""
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def collect_preflight() -> dict:
    """Collect portable machine/runtime facts for a reproducible run record."""
    payload = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "torchvision": __import__("torchvision").__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "git_sha": None,
    }
    try:
        payload["git_sha"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        pass
    if payload["cuda_available"]:
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)
        payload.update({
            "gpu": torch.cuda.get_device_name(device),
            "gpu_compute_capability": [props.major, props.minor],
            "gpu_vram_bytes": props.total_memory,
            "cuda_device": device,
        })
        try:
            payload["cuda_driver"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=True, capture_output=True, text=True, timeout=5,
            ).stdout.splitlines()[0].strip()
        except Exception:
            payload["cuda_driver"] = None
    else:
        payload.update({"gpu": None, "gpu_compute_capability": None, "gpu_vram_bytes": None, "cuda_driver": None})
    return payload


def log_preflight(logger: logging.Logger | None = None) -> dict:
    payload = collect_preflight()
    (logger or logging.getLogger(__name__)).info("P1-v8.3 preflight=%s", payload)
    return payload


class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(
            self,
            apply_nonlin=None,
            alpha=None,
            gamma=2,
            balance_index=0,
            smooth=1e-5,
            size_average=True,
    ):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError("smooth value should be in [0,1]")

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError("Not support alpha type")

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth
            )
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        return loss


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        N = targets.size()[0]
        smooth = 1
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)
        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (
                input_flat.sum(1) + targets_flat.sum(1) + smooth
        )
        loss = 1 - N_dice_eff.sum() / N
        return loss


prompt = PROMPTS
prompt_normal = prompt["prompt_normal"]
prompt_abnormal = prompt["prompt_abnormal"]
prompt_state = [prompt_normal, prompt_abnormal]
prompt_templates = prompt["prompt_templates"]
EXPECTED_PROMPT_COUNTS = [6, 10]


def get_real_name(dataset_name, class_name):
    if class_name == "object":
        return class_name
    assert class_name in CLASS_NAMES[dataset_name], (
        f"class_name {class_name} not found; available class_names: {CLASS_NAMES[dataset_name]}"
    )
    return REAL_NAMES[dataset_name][class_name]


def get_prompt_sentences(real_name, state_idx):
    prompted_state = [state.format(real_name) for state in prompt_state[state_idx]]
    prompted_sentence = []
    for s in prompted_state:
        for template in prompt_templates:
            prompted_sentence.append(template.format(s))
    expected_count = EXPECTED_PROMPT_COUNTS[state_idx]
    if len(prompted_sentence) != expected_count:
        raise ValueError(
            f"Prompt count mismatch for state {state_idx}: "
            f"got {len(prompted_sentence)}, expected {expected_count}."
        )
    return prompted_sentence


def get_soft_prompt_sentence(real_name, state_idx, ctx_len):
    ctx_prefix = " ".join(["X"] * ctx_len)
    state_word = "normal" if state_idx == 0 else "abnormal"
    return f"{ctx_prefix} {state_word} {real_name}."


def get_structured_prompt_sentence(real_name, state_idx, ctx_len=4):
    """[C1..C4][STATE][CLASS][literal state][REAL_NAME]."""
    placeholders = " ".join(["X"] * (int(ctx_len) + 2))
    state_word = "normal" if state_idx == 0 else "abnormal"
    return f"{placeholders} {state_word} {real_name}."


def aggregate_prompt_features(multi_features):
    aggregated = []
    for layer_feature in multi_features:
        layer_feature = layer_feature / layer_feature.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        layer_feature = layer_feature.mean(dim=0)
        layer_feature = layer_feature / layer_feature.norm().clamp_min(1e-6)
        aggregated.append(layer_feature)
    return aggregated


def stack_state_features(multi_layer_features, device):
    text_features_levels = []
    cnt = len(multi_layer_features) // len(prompt_state)
    for i in range(cnt):
        text_features_levels.append(torch.stack(multi_layer_features[i::cnt], dim=1).to(device))
    return torch.stack(text_features_levels, dim=0)


def flatten_group_state_features(text_features):
    return text_features.permute(0, 2, 1).reshape(-1, text_features.shape[1])


def get_hard_phase1_single_class_text_embedding(
        model,
        dataset_name,
        class_name,
        device,
        adapt_text=True,
):
    real_name = get_real_name(dataset_name, class_name)
    multi_layer_features = []
    for i in range(len(prompt_state)):
        prompted_sentence = get_prompt_sentences(real_name, i)
        prompted_sentence = tokenize(prompted_sentence).to(device)
        multi_features = model.encode_text(prompted_sentence, adapt_text=adapt_text)
        multi_layer_features.extend(aggregate_prompt_features(multi_features))
    return stack_state_features(multi_layer_features, device)


def get_multiple_adapted_single_class_text_embedding(
        model,
        dataset_name, class_name, device
):
    if getattr(model, "use_hybrid_soft_prompt", False):
        text_features, _, _ = get_hybrid_soft_prompt_single_class_text_embedding(
            model, dataset_name, class_name, device, return_kg=False
        )
        return text_features

    if getattr(model, "use_soft_prompt", False):
        text_features, _, _ = get_soft_prompt_single_class_text_embedding(
            model, dataset_name, class_name, device, return_kg=False
        )
        return text_features

    return get_hard_phase1_single_class_text_embedding(
        model, dataset_name, class_name, device, adapt_text=True
    )


def get_hard_anchor_single_class_text_embedding(model, dataset_name, class_name, device):
    cache = getattr(model, "_frozen_anchor_cache", None)
    cache_key = (dataset_name, class_name, str(device))
    if isinstance(cache, dict) and cache_key in cache:
        return cache[cache_key]
    real_name = get_real_name(dataset_name, class_name)
    multi_layer_features = []
    with torch.no_grad():
        for i in range(len(prompt_state)):
            prompted_sentence = get_prompt_sentences(real_name, i)
            prompted_sentence = tokenize(prompted_sentence).to(device)
            if hasattr(model, "encode_frozen_anchor_text"):
                multi_features = model.encode_frozen_anchor_text(prompted_sentence)
            else:
                multi_features = model.encode_text(prompted_sentence, adapt_text=False)
            multi_layer_features.extend(aggregate_prompt_features(multi_features))
        anchor = stack_state_features(multi_layer_features, device)
    anchor = torch.nn.functional.normalize(anchor.float(), dim=1).detach()
    if isinstance(cache, dict):
        cache[cache_key] = anchor
    return anchor


def get_soft_prompt_single_class_text_embedding(
        model,
        dataset_name,
        class_name,
        device,
        return_kg=True,
):
    real_name = get_real_name(dataset_name, class_name)
    multi_layer_features = []
    for state_idx in range(len(prompt_state)):
        soft_sentence = get_soft_prompt_sentence(real_name, state_idx, model.soft_prompt_ctx_len)
        tokenized = tokenize([soft_sentence]).to(device)
        ctx = model.soft_prompt.get_context(state_idx)
        multi_features = model.encode_soft_prompt_text(tokenized, ctx, adapt_text=False)
        multi_layer_features.extend(aggregate_prompt_features(multi_features))
    soft_text = stack_state_features(multi_layer_features, device)
    if not return_kg:
        return soft_text, None, None

    hard_anchor = get_hard_anchor_single_class_text_embedding(model, dataset_name, class_name, device)
    cosine = F.cosine_similarity(
        flatten_group_state_features(soft_text),
        flatten_group_state_features(hard_anchor),
        dim=-1,
    )
    cosine_by_group = cosine.detach().view(soft_text.shape[0], 2)
    kg_by_group = 1.0 - cosine_by_group
    delta_by_group = (soft_text.detach() - hard_anchor.detach()).norm(dim=1)
    soft_state_cos = F.cosine_similarity(
        soft_text.detach()[..., 0],
        soft_text.detach()[..., 1],
        dim=1,
    )
    hard_state_cos = F.cosine_similarity(
        hard_anchor.detach()[..., 0],
        hard_anchor.detach()[..., 1],
        dim=1,
    )
    kg_loss = (1.0 - cosine).mean()
    stats = {
        "soft_hard_cos_mean": float(cosine_by_group.mean().item()),
        "soft_hard_cos_normal": float(cosine_by_group[:, 0].mean().item()),
        "soft_hard_cos_abnormal": float(cosine_by_group[:, 1].mean().item()),
        "kg_loss_normal": float(kg_by_group[:, 0].mean().item()),
        "kg_loss_abnormal": float(kg_by_group[:, 1].mean().item()),
        "delta_soft_hard_normal": float(delta_by_group[:, 0].mean().item()),
        "delta_soft_hard_abnormal": float(delta_by_group[:, 1].mean().item()),
        "soft_normal_abnormal_cos": float(soft_state_cos.mean().item()),
        "hard_normal_abnormal_cos": float(hard_state_cos.mean().item()),
        "soft_normal_norm": float(soft_text.detach()[..., 0].norm(dim=1).mean().item()),
        "soft_abnormal_norm": float(soft_text.detach()[..., 1].norm(dim=1).mean().item()),
        "hard_normal_norm": float(hard_anchor.detach()[..., 0].norm(dim=1).mean().item()),
        "hard_abnormal_norm": float(hard_anchor.detach()[..., 1].norm(dim=1).mean().item()),
        "soft_proto_norm_min": float(soft_text.detach().norm(dim=1).min().item()),
        "soft_proto_norm_max": float(soft_text.detach().norm(dim=1).max().item()),
        "hard_proto_norm_min": float(hard_anchor.detach().norm(dim=1).min().item()),
        "hard_proto_norm_max": float(hard_anchor.detach().norm(dim=1).max().item()),
    }
    for group_idx in range(soft_text.shape[0]):
        prefix = f"g{group_idx + 1}"
        stats[f"{prefix}_soft_hard_cos_normal"] = float(cosine_by_group[group_idx, 0].item())
        stats[f"{prefix}_soft_hard_cos_abnormal"] = float(cosine_by_group[group_idx, 1].item())
        stats[f"{prefix}_delta_soft_hard_normal"] = float(delta_by_group[group_idx, 0].item())
        stats[f"{prefix}_delta_soft_hard_abnormal"] = float(delta_by_group[group_idx, 1].item())
        stats[f"{prefix}_soft_normal_abnormal_cos"] = float(soft_state_cos[group_idx].item())
        stats[f"{prefix}_hard_normal_abnormal_cos"] = float(hard_state_cos[group_idx].item())
    return soft_text, kg_loss, stats


def get_hybrid_soft_prompt_single_class_text_embedding(
        model,
        dataset_name,
        class_name,
        device,
        return_kg=True,
        return_components=False,
):
    hard_text = get_hard_phase1_single_class_text_embedding(
        model, dataset_name, class_name, device, adapt_text=True
    )
    real_name = get_real_name(dataset_name, class_name)
    multi_layer_features = []
    for state_idx in range(len(prompt_state)):
        soft_sentence = get_soft_prompt_sentence(real_name, state_idx, model.soft_prompt_ctx_len)
        tokenized = tokenize([soft_sentence]).to(device)
        ctx = model.soft_prompt.get_context(state_idx)
        multi_features = model.encode_soft_prompt_text(tokenized, ctx, adapt_text=False)
        multi_layer_features.extend(aggregate_prompt_features(multi_features))
    soft_text = stack_state_features(multi_layer_features, device)

    alpha = float(getattr(model, "hybrid_alpha_current", 0.0))
    main_text = F.normalize((1.0 - alpha) * hard_text + alpha * soft_text, dim=1)
    if not return_kg:
        if return_components:
            return main_text, None, None, {"hard_text": hard_text, "soft_text": soft_text}
        return main_text, None, None

    hard_anchor = hard_text.detach()
    soft_flat = flatten_group_state_features(soft_text)
    hard_flat = flatten_group_state_features(hard_anchor)
    main_flat = flatten_group_state_features(main_text.detach())
    cosine = F.cosine_similarity(soft_flat, hard_flat, dim=-1)
    kg_loss = (1.0 - cosine).mean()

    cosine_by_group = cosine.detach().view(soft_text.shape[0], 2)
    kg_by_group = 1.0 - cosine_by_group
    main_hard_cosine = F.cosine_similarity(main_flat, hard_flat, dim=-1).view(soft_text.shape[0], 2)
    main_soft_cosine = F.cosine_similarity(main_flat, soft_text.detach().permute(0, 2, 1).reshape(-1, soft_text.shape[1]), dim=-1).view(soft_text.shape[0], 2)
    delta_soft_hard = (soft_text.detach() - hard_anchor).norm(dim=1)
    delta_main_hard = (main_text.detach() - hard_anchor).norm(dim=1)
    soft_state_cos = F.cosine_similarity(soft_text.detach()[..., 0], soft_text.detach()[..., 1], dim=1)
    hard_state_cos = F.cosine_similarity(hard_anchor[..., 0], hard_anchor[..., 1], dim=1)
    main_state_cos = F.cosine_similarity(main_text.detach()[..., 0], main_text.detach()[..., 1], dim=1)
    stats = {
        "hybrid_alpha": alpha,
        "soft_hard_cos_mean": float(cosine_by_group.mean().item()),
        "soft_hard_cos_normal": float(cosine_by_group[:, 0].mean().item()),
        "soft_hard_cos_abnormal": float(cosine_by_group[:, 1].mean().item()),
        "main_hard_cos_mean": float(main_hard_cosine.mean().item()),
        "main_hard_cos_normal": float(main_hard_cosine[:, 0].mean().item()),
        "main_hard_cos_abnormal": float(main_hard_cosine[:, 1].mean().item()),
        "main_soft_cos_mean": float(main_soft_cosine.mean().item()),
        "main_soft_cos_normal": float(main_soft_cosine[:, 0].mean().item()),
        "main_soft_cos_abnormal": float(main_soft_cosine[:, 1].mean().item()),
        "kg_loss_normal": float(kg_by_group[:, 0].mean().item()),
        "kg_loss_abnormal": float(kg_by_group[:, 1].mean().item()),
        "delta_soft_hard_normal": float(delta_soft_hard[:, 0].mean().item()),
        "delta_soft_hard_abnormal": float(delta_soft_hard[:, 1].mean().item()),
        "delta_main_hard_normal": float(delta_main_hard[:, 0].mean().item()),
        "delta_main_hard_abnormal": float(delta_main_hard[:, 1].mean().item()),
        "soft_normal_abnormal_cos": float(soft_state_cos.mean().item()),
        "hard_normal_abnormal_cos": float(hard_state_cos.mean().item()),
        "main_normal_abnormal_cos": float(main_state_cos.mean().item()),
        "soft_proto_norm_min": float(soft_text.detach().norm(dim=1).min().item()),
        "soft_proto_norm_max": float(soft_text.detach().norm(dim=1).max().item()),
        "hard_proto_norm_min": float(hard_anchor.norm(dim=1).min().item()),
        "hard_proto_norm_max": float(hard_anchor.norm(dim=1).max().item()),
        "main_proto_norm_min": float(main_text.detach().norm(dim=1).min().item()),
        "main_proto_norm_max": float(main_text.detach().norm(dim=1).max().item()),
    }
    for group_idx in range(soft_text.shape[0]):
        prefix = f"g{group_idx + 1}"
        stats[f"{prefix}_soft_hard_cos_normal"] = float(cosine_by_group[group_idx, 0].item())
        stats[f"{prefix}_soft_hard_cos_abnormal"] = float(cosine_by_group[group_idx, 1].item())
        stats[f"{prefix}_main_hard_cos_normal"] = float(main_hard_cosine[group_idx, 0].item())
        stats[f"{prefix}_main_hard_cos_abnormal"] = float(main_hard_cosine[group_idx, 1].item())
        stats[f"{prefix}_main_soft_cos_normal"] = float(main_soft_cosine[group_idx, 0].item())
        stats[f"{prefix}_main_soft_cos_abnormal"] = float(main_soft_cosine[group_idx, 1].item())
        stats[f"{prefix}_delta_soft_hard_normal"] = float(delta_soft_hard[group_idx, 0].item())
        stats[f"{prefix}_delta_soft_hard_abnormal"] = float(delta_soft_hard[group_idx, 1].item())
        stats[f"{prefix}_delta_main_hard_normal"] = float(delta_main_hard[group_idx, 0].item())
        stats[f"{prefix}_delta_main_hard_abnormal"] = float(delta_main_hard[group_idx, 1].item())
        stats[f"{prefix}_soft_normal_abnormal_cos"] = float(soft_state_cos[group_idx].item())
        stats[f"{prefix}_hard_normal_abnormal_cos"] = float(hard_state_cos[group_idx].item())
        stats[f"{prefix}_main_normal_abnormal_cos"] = float(main_state_cos[group_idx].item())
    if return_components:
        return main_text, kg_loss, stats, {"hard_text": hard_text, "soft_text": soft_text}
    return main_text, kg_loss, stats


def get_multiple_adapted_text_embedding(
        model,
        dataset_name, device
):
    ret_dict = {}
    for class_name in CLASS_NAMES[dataset_name]:
        multi_layer_text_features = get_multiple_adapted_single_class_text_embedding(
            model, dataset_name, class_name, device
        )
        ret_dict[class_name] = multi_layer_text_features
    return ret_dict


def get_phase2b_global_text_features(
    model, dataset_name, class_names, device,
    use_hybrid_soft_prompt=False, use_soft_prompt=False
):
    epoch_text_feature_dict = {}
    for class_name in class_names:
        if class_name in epoch_text_feature_dict:
            continue
        if use_hybrid_soft_prompt:
            text_embedding_levels, _, _, _ = get_hybrid_soft_prompt_single_class_text_embedding(
                model, dataset_name, class_name, device, return_kg=True, return_components=True
            )
        elif use_soft_prompt:
            text_embedding_levels, _, _ = get_soft_prompt_single_class_text_embedding(
                model, dataset_name, class_name, device, return_kg=True
            )
        else:
            text_embedding_levels = get_multiple_adapted_single_class_text_embedding(
                model, dataset_name, class_name, device
            )
        epoch_text_feature_dict[class_name] = text_embedding_levels

    epoch_text_features = torch.stack(
        [epoch_text_feature_dict[class_name] for class_name in class_names],
        dim=0,
    )  # [bs, n_groups, 768, 2]
    return epoch_text_features.permute(1, 0, 2, 3)  # [n_groups, bs, 768, 2]

focal_loss = FocalLoss()
dice_loss = BinaryDiceLoss()


def calculate_seg_loss(patch_preds, mask):
    loss = focal_loss(patch_preds, mask)
    loss += dice_loss(patch_preds[:, 0, :, :], 1 - mask)
    loss += dice_loss(patch_preds[:, 1, :, :], mask)
    return loss


def metrics_eval_gpu(
        pixel_label: torch.Tensor,
        image_label: torch.Tensor,
        pixel_preds: torch.Tensor,
        image_preds: torch.Tensor,
        class_names: str,
        domain: str,
):
    pixel_preds = torch.flatten(pixel_preds, start_dim=1)
    pmax_pred, _ = torch.max(pixel_preds, dim=1)
    if domain == "Medical":
        image_preds = image_preds * 0.5 + pmax_pred * 0.5
    else:
        image_preds = image_preds * 0.9 + pmax_pred * 0.1

    pixel_label = pixel_label.flatten()
    pixel_preds = pixel_preds.flatten()

    zero_pixel_auc = auroc(pixel_preds, pixel_label, task="binary")
    zero_pixel_ap = average_precision(pixel_preds, pixel_label, task="binary")

    if image_label.max() != image_label.min():
        image_label = image_label.flatten()
        agg_image_preds = image_preds.flatten()
        agg_image_auc = auroc(agg_image_preds, image_label, task="binary")
        agg_image_ap = average_precision(agg_image_preds, image_label, task="binary")
    else:
        agg_image_auc = None
        agg_image_ap = None
    # ================================================================================================
    result = {
        "class name": class_names,
        "pixel AUC": round(zero_pixel_auc.item(), 4) * 100,
        "pixel AP": round(zero_pixel_ap.item(), 4) * 100,
        "image AUC": "N/A" if agg_image_auc is None else round(agg_image_auc.item(), 4) * 100,
        "image AP": "N/A" if agg_image_ap is None else round(agg_image_ap.item(), 4) * 100,
    }
    return result


class AddGaussianNoise(object):
    def __init__(self, std=1.0, p=0.5):
        self.std = std
        self.p = p

    def __call__(self, x):
        """
        在数据张量上应用噪音
        """
        if random.random() < self.p:
            return x
        if not isinstance(x, torch.Tensor):
            x = transforms.ToTensor()(x)
        noise_mask = (torch.randn(x.shape[-2:]) > 3).int()
        noise = torch.randn_like(x) * self.std  # mean = 0
        noised_img = (1 - noise_mask) * x + noise * x * noise_mask
        noised_img = torch.clamp(noised_img, 0.0, 1.0)
        return transforms.ToPILImage()(noised_img)

    def __repr__(self):
        return self.__class__.__name__ + f"p={self.p}, std={self.std}"
