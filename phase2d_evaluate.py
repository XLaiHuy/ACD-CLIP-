"""Validation-only Phase2D checkpoint evaluator using the Phase2C VisA path."""
import argparse
import csv
import json
from pathlib import Path

import torch

from dataset import get_text_and_image_dataset
from phase2c_train import METRIC_FIELDS, build_model, validate_visa
from phase2c_utils import phase2c_config, seed_everything


def parse_checkpoint(value):
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("checkpoint must be NAME=PATH")
    return name, Path(path)


def config_for_payload(payload):
    condition = payload["condition"]
    alpha = float(payload["hybrid_alpha_max"])
    config = phase2c_config(condition, "phase2d-evaluation", alpha)
    for key in (
        "n_groups", "dfg_mode", "dfg_attn_dim", "dfg_attn_tau", "use_ss2d_dfg",
        "dfg_gamma_max", "dfg_ss2d_fusion", "dfg_beta", "dfg_beta_schedule",
        "dfg_beta_target", "soft_prompt_ctx_len", "soft_prompt_freeze_epochs",
        "hybrid_alpha_max",
    ):
        if key in payload:
            config[key] = payload[key]
    return config


def load_model_for_payload(payload, device):
    config = config_for_payload(payload)
    model = build_model(config, device)
    model.text_adapter.load_state_dict(payload["text_adapter"], strict=True)
    model.image_adapter.load_state_dict(payload["image_adapter"], strict=True)
    model.soft_prompt.load_state_dict(payload["soft_prompt"], strict=True)
    model.hybrid_alpha_current = payload["hybrid_alpha_current"]
    model.eval()
    return model, config


def evaluate_checkpoint(name, path, dataset, device, batch_size, num_workers, thresholds):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model, config = load_model_for_payload(payload, device)
    rows = validate_visa(model, dataset, device, int(payload["epoch"]), batch_size, num_workers, thresholds)
    for row in rows:
        row["checkpoint_name"] = name
        row["checkpoint_path"] = str(path)
        row["lambda_b"] = payload.get("phase2d_interpolation", {}).get("lambda_b", "")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows, config


def parse_args():
    parser = argparse.ArgumentParser(description="Validation-only Phase2D evaluator")
    parser.add_argument("--checkpoint", action="append", type=parse_checkpoint, required=True)
    parser.add_argument("--val-manifest", default="splits/visa_val_seed42.csv")
    parser.add_argument("--split-metadata", default="splits/visa_split_seed42_metadata.json")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--metric-thresholds", type=int, default=None)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not Path(args.val_manifest).is_file() or not Path(args.split_metadata).is_file():
        raise FileNotFoundError("fixed VisA split artifact is missing")
    if args.dry_run:
        for _, path in args.checkpoint:
            torch.load(path, map_location="cpu", weights_only=False)
        print(json.dumps({"status": "preflight_pass", "checkpoints": [name for name, _ in args.checkpoint]}))
        return
    seed_everything(42)
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() and args.cuda_device >= 0 else "cpu")
    dataset = get_text_and_image_dataset("VisA", 518, "val", args.val_manifest)
    all_rows = []
    configs = {}
    for name, path in args.checkpoint:
        rows, config = evaluate_checkpoint(name, path, dataset, device, args.batch_size, args.num_workers, args.metric_thresholds)
        all_rows.extend(rows)
        configs[name] = config
    fieldnames = ["checkpoint_name", "checkpoint_path", "lambda_b", *METRIC_FIELDS]
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    Path(args.output_json).write_text(json.dumps({"configs": configs, "rows": all_rows}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
