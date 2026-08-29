#!/usr/bin/env python3
"""CIR/G2-PARITY: end-to-end alpha=0 parity against the native parent path."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch

from model.phase2b_runtime import build_phase2b_trainable, configure_canonical_fp32, forward_phase2b
from tools.cir_rmt.identity import load_cir_config, release_identity_fields
from tools.cir_rmt.runtime import forward_cir


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max())


def _gate_result(config: dict[str, object], payload: dict[str, object], *, real_asset: bool, artifact: dict[str, str] | None = None) -> dict[str, object]:
    return {
        "gate": "G2_REAL",
        "scope": "real",
        "real": True,
        "real_asset": bool(real_asset),
        "identity": release_identity_fields(config),
        "evidence": {
            "kind": "alpha0_parity_real",
            "status": payload.get("status"),
            "real_execution": bool(real_asset and payload.get("status") == "PASS"),
            "artifact": dict(artifact or {}),
            "checks": {key: value for key, value in payload.items() if key.endswith("_max_abs")},
        },
        **payload,
    }


def run_real(args: argparse.Namespace) -> dict[str, object]:
    config = load_cir_config(args.config)
    parent_path = Path(config.get("parent_config_path", "configs/phase2b_canonical_v1.json"))
    if not parent_path.is_absolute():
        parent_path = Path(__file__).resolve().parents[2] / parent_path
    parent_config = json.loads(parent_path.read_text(encoding="utf-8"))
    parent_config["dataset"] = "VisA"
    if not args.clip_asset or not Path(args.clip_asset).is_file():
        return _gate_result(config, {"stage": "CIR/G2-PARITY", "status": "NOT_RUN_NO_ASSETS"}, real_asset=False)
    if args.image is None:
        return _gate_result(config, {"stage": "CIR/G2-PARITY", "status": "NOT_RUN_NO_IMAGE"}, real_asset=False)
    from PIL import Image
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    transform = transforms.Compose([
        transforms.Resize((int(config.get("img_size", 518)), int(config.get("img_size", 518))), InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])
    with Image.open(args.image) as handle:
        image = transform(handle.convert("RGB")).unsqueeze(0)
    device = torch.device(args.device)
    configure_canonical_fp32()
    model = build_phase2b_trainable(parent_config, args.clip_asset, device)
    model.eval()
    names = [str(args.class_name)]
    parent = forward_phase2b(model, image, names, device, parent_config, domain="Industrial", require_grad=False, dataset_name="VisA")
    alpha0 = dict(config)
    alpha0["rmt_transport_alpha"] = 0.0
    cir = forward_cir(model, image, names, device, alpha0, domain="Industrial", require_grad=False, dataset_name="VisA")
    checks = {
        "stage": "CIR/G2-PARITY",
        "native_stage_logits_max_abs": _max_abs(parent.native_logits, cir.cir_logits),
        "native_margins_max_abs": _max_abs(parent.native_margin, cir.native_margin),
        "patch_softmax_max_abs": _max_abs(parent.native_patch_probability, cir.cir_patch_probability),
        "training_probability_max_abs": _max_abs(parent.training_segmentation_probability, cir.cir_training_segmentation_probability),
        "deployed_logits_max_abs": _max_abs(parent.deployed_logits, cir.cir_deployed_logits),
        "deployed_probability_max_abs": _max_abs(parent.deployed_segmentation_probability, cir.cir_segmentation_probability),
        "final_map_max_abs": _max_abs(parent.native_segmentation_probability, cir.cir_segmentation_probability),
        "classification_logits_max_abs": _max_abs(parent.classification_logits, cir.classification_logits),
        "classification_probability_max_abs": _max_abs(parent.classification_probability, cir.classification_probability),
        "threshold": 1e-5,
    }
    checks["status"] = "PASS" if max(value for key, value in checks.items() if key.endswith("_max_abs")) <= checks["threshold"] else "FAIL"
    return _gate_result(config, checks, real_asset=True, artifact={"clip_asset": str(args.clip_asset), "image": str(args.image)})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/cir_dfg_rmt_v1.json"))
    parser.add_argument("--clip-asset", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--class-name", default="candle")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = run_real(args)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["status"] in {"PASS", "NOT_RUN_NO_ASSETS", "NOT_RUN_NO_IMAGE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
