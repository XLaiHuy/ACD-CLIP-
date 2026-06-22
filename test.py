import argparse
import logging
import os
from glob import glob

import torch
import torch.nn.functional as F
from pandas import DataFrame, Series
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DOMAINS, get_text_and_image_dataset
from utils import (
    get_multiple_adapted_text_embedding, metrics_eval_gpu,
)
from model.adapter import (
    ACDCLIP
)
from model.clip import create_model


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
        masks.append(mask)
        labels.append(label)
        file_names.extend(file_name)
        # get text
        epoch_text_features = class_text_embeddings.unsqueeze(dim=1)  # [n_groups, 1, 768, 2]
        
        # TTA: 4 augmented views (original, horizontal flip, vertical flip, rotation 180)
        # View 1: Original
        seg_tokens, det_tokens = model(image)
        seg_features = torch.stack(seg_tokens, dim=0)
        det_features = torch.stack(det_tokens, dim=0)
        
        B = seg_features.shape[1]
        epoch_text_features_rep = epoch_text_features.repeat(1, B, 1, 1)
        
        logit_scale = model.clipmodel.logit_scale.exp()
        # Class predictions for view 1
        cls_preds_total = [
            torch.matmul(
                det_features[i].unsqueeze(dim=1),
                epoch_text_features_rep[i],
            ).squeeze(1) * logit_scale for i in range(det_features.shape[0])
        ]
        cls_preds_total = torch.stack(cls_preds_total, dim=0).mean(dim=0) # [bs, 2]
        
        # Pixel predictions for view 1
        seg_pred_orig = model.vision_text_fusion_gate_seg(seg_features, epoch_text_features_rep, test_mode=True,
                                                          domain=DOMAINS[dataset])
        
        # View 2: Horizontal Flip
        image_h = torch.flip(image, dims=[-1])
        seg_tokens_h, det_tokens_h = model(image_h)
        seg_features_h = torch.stack(seg_tokens_h, dim=0)
        det_features_h = torch.stack(det_tokens_h, dim=0)
        
        cls_preds_h = [
            torch.matmul(
                det_features_h[i].unsqueeze(dim=1),
                epoch_text_features_rep[i],
            ).squeeze(1) * logit_scale for i in range(det_features_h.shape[0])
        ]
        cls_preds_h = torch.stack(cls_preds_h, dim=0).mean(dim=0)
        cls_preds_total = cls_preds_total + cls_preds_h
        
        seg_pred_h = model.vision_text_fusion_gate_seg(seg_features_h, epoch_text_features_rep, test_mode=True,
                                                        domain=DOMAINS[dataset])
        seg_pred_h = torch.flip(seg_pred_h, dims=[-1])
        
        # View 3: Vertical Flip
        image_v = torch.flip(image, dims=[-2])
        seg_tokens_v, det_tokens_v = model(image_v)
        seg_features_v = torch.stack(seg_tokens_v, dim=0)
        det_features_v = torch.stack(det_tokens_v, dim=0)
        
        cls_preds_v = [
            torch.matmul(
                det_features_v[i].unsqueeze(dim=1),
                epoch_text_features_rep[i],
            ).squeeze(1) * logit_scale for i in range(det_features_v.shape[0])
        ]
        cls_preds_v = torch.stack(cls_preds_v, dim=0).mean(dim=0)
        cls_preds_total = cls_preds_total + cls_preds_v
        
        seg_pred_v = model.vision_text_fusion_gate_seg(seg_features_v, epoch_text_features_rep, test_mode=True,
                                                        domain=DOMAINS[dataset])
        seg_pred_v = torch.flip(seg_pred_v, dims=[-2])
        
        # View 4: Rotation 180 (Flip both H and V)
        image_r = torch.flip(image, dims=[-1, -2])
        seg_tokens_r, det_tokens_r = model(image_r)
        seg_features_r = torch.stack(seg_tokens_r, dim=0)
        det_features_r = torch.stack(det_tokens_r, dim=0)
        
        cls_preds_r = [
            torch.matmul(
                det_features_r[i].unsqueeze(dim=1),
                epoch_text_features_rep[i],
            ).squeeze(1) * logit_scale for i in range(det_features_r.shape[0])
        ]
        cls_preds_r = torch.stack(cls_preds_r, dim=0).mean(dim=0)
        cls_preds_total = cls_preds_total + cls_preds_r
        
        seg_pred_r = model.vision_text_fusion_gate_seg(seg_features_r, epoch_text_features_rep, test_mode=True,
                                                        domain=DOMAINS[dataset])
        seg_pred_r = torch.flip(seg_pred_r, dims=[-1, -2])
        
        # Averages
        cls_preds = cls_preds_total / 4.0
        pred = F.softmax(cls_preds, dim=1)[:, 1]
        preds_image.append(pred)
        
        seg_pred = (seg_pred_orig + seg_pred_h + seg_pred_v + seg_pred_r) / 4.0
        preds.append(seg_pred)
    masks = torch.concatenate(masks, dim=0)  # [bs, 1, 518, 518]
    labels = torch.concatenate(labels, dim=0)  # [bs]
    preds = torch.concatenate(preds, dim=0)  # [bs, 518, 518]
    preds_image = torch.concatenate(preds_image, dim=0)  # [bs]
    return masks, labels, preds, preds_image, file_names


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

    parser.add_argument("--dataset", type=str, default="MPDD")
    parser.add_argument("--batch_size", type=int, default=84)
    parser.add_argument("--cuda_device", type=int, default=0)
    parser.add_argument("--save_path", type=str, default="ckpt/issue")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4 if os.name != 'nt' else 0,
        help="Number of workers for data loading (default: 4 on Linux, 0 on Windows)"
    )

    args = parser.parse_args()
    # ========================================================
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
    ).to(device)
    model.eval()
    ckp_files = glob(args.save_path + "/adapter_*.pth")
    assert len(ckp_files) > 0, "adapter checkpoint not found"
    for file in ckp_files:
        checkpoint = torch.load(file, map_location=device)
        image_state_dict = checkpoint["image_adapter"]
        if "tau" in image_state_dict:
            tau_val = image_state_dict.pop("tau")
            import math
            val = float(tau_val.item() if hasattr(tau_val, "item") else tau_val)
            image_state_dict["log_tau"] = torch.tensor(math.log(max(val, 0.01)))
            
        model.image_adapter.load_state_dict(image_state_dict)
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
            image_dataloader = torch.utils.data.DataLoader(
                image_dataset, 
                batch_size=args.batch_size, 
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True
            )
            with torch.no_grad():
                class_text_embeddings = text_embeddings[class_name]
                masks, labels, preds, preds_image, file_names = get_predictions(
                    model=model,
                    class_text_embeddings=class_text_embeddings,
                    test_loader=image_dataloader,
                    device=device,
                    dataset=args.dataset,
                )
            class_result_dict = metrics_eval_gpu(
                masks,
                labels,
                preds,
                preds_image,
                class_name,
                domain=DOMAINS[args.dataset],
            )
            df.loc[len(df)] = Series(class_result_dict)
        mean_vals = df[df.columns[1:]].mean()
        df.loc[len(df), df.columns[1:]] = mean_vals
        df.loc[len(df) - 1, "class name"] = "Average"
        logger.info("final results:\n%s", df.to_string(index=False, justify="center"))


if __name__ == "__main__":
    main()
