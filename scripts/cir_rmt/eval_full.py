#!/usr/bin/env python3
"""CIR/EVAL-{target}-SOURCE: exact evaluator for one frozen CIR checkpoint."""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path
import threading
import time
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm

from evaluation.datasets import MVTecDatasetAdapter, resolve_mvtec_root
from evaluation.evaluator import evaluate_spool, image_score
from evaluation.spool import EvaluationSpool
from model.phase2b_runtime import build_phase2b_frozen, configure_canonical_fp32
from tools.cir_rmt.identity import config_sha256, load_cir_config, sha256_file, validate_checkpoint_identity
from tools.cir_rmt.runtime import forward_cir


MEDICAL_TARGETS = ("Brain", "Liver", "Retina", "Colon_clinicDB", "Colon_colonDB", "Colon_Kvasir")
IMAGE_SIZE = 518
NORM_MEAN = (0.48145466, 0.4578275, 0.40821073)
NORM_STD = (0.26862954, 0.26130258, 0.27577711)


def _rss_bytes() -> int:
    """Return current process RSS from the kernel, including native NumPy memory."""
    try:
        for line in Path("/proc/self/status").read_text(encoding="ascii").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, OSError, ValueError):
        pass
    return 0


class _RssMonitor:
    """Sample current RSS by evaluation phase without changing model behavior."""

    def __init__(self, interval_seconds: float = 0.10):
        self._interval_seconds = float(interval_seconds)
        self._phase = "startup"
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._last_by_phase: dict[str, int] = {}
        self._peak_by_phase: dict[str, int] = {}

    def sample(self) -> None:
        rss = _rss_bytes()
        if rss <= 0:
            return
        with self._lock:
            phase = self._phase
            self._last_by_phase[phase] = rss
            self._peak_by_phase[phase] = max(rss, self._peak_by_phase.get(phase, 0))

    def set_phase(self, phase: str) -> None:
        with self._lock:
            self._phase = str(phase)
        self.sample()

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            self.sample()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("RSS monitor already started")
        self.sample()
        self._thread = threading.Thread(target=self._run, name="cir-rss-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self.sample()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.sample()

    def report(self, model_loaded_rss: int) -> dict[str, Any]:
        with self._lock:
            last = dict(self._last_by_phase)
            peak = dict(self._peak_by_phase)
        to_mib = lambda value: None if not value else round(value / 2**20, 3)
        return {
            "rss_model_loaded_mib": to_mib(model_loaded_rss),
            "rss_after_inference_mib": to_mib(last.get("post_inference", 0)),
            "rss_after_teardown_mib": to_mib(last.get("after_teardown", 0)),
            "peak_inference_rss_mib": to_mib(peak.get("inference", 0)),
            "peak_metric_rss_mib": to_mib(peak.get("metric", 0)),
            "final_rss_mib": to_mib(last.get("final", 0)),
            "peak_rss_by_phase_mib": {name: to_mib(value) for name, value in peak.items()},
            "last_rss_by_phase_mib": {name: to_mib(value) for name, value in last.items()},
        }


class ManifestDataset(Dataset):
    def __init__(self, root: Path, metadata: Path, image_size: int = IMAGE_SIZE):
        self.root = root.expanduser().resolve()
        self.rows = [json.loads(line) for line in metadata.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.image = transforms.Compose([transforms.Resize((image_size, image_size), InterpolationMode.BICUBIC), transforms.ToTensor(), transforms.Normalize(NORM_MEAN, NORM_STD)])
        self.mask = transforms.Compose([transforms.Resize((image_size, image_size), InterpolationMode.NEAREST), transforms.ToTensor()])
    def __len__(self) -> int:
        return len(self.rows)
    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[int(index)]
        with Image.open((self.root / row["image_path"]).resolve()) as handle:
            image = self.image(handle.convert("RGB")).contiguous()
        if int(row.get("label", 0)):
            with Image.open((self.root / row["mask_path"]).resolve()) as handle:
                mask = self.mask(handle.convert("L")).gt(0).to(torch.float32)
        else:
            mask = torch.zeros((1, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.float32)
        return {"image": image, "mask": mask.contiguous(), "label": torch.tensor(int(row.get("label", 0)), dtype=torch.int64), "class_name": str(row["class_name"]), "image_path": str(row["image_path"])}


class MappingDataset(Dataset):
    def __init__(self, mappings: dict[str, Dataset]):
        self.entries = [(name, index) for name, dataset in mappings.items() for index in range(len(dataset))]
        self.mappings = mappings
    def __len__(self) -> int:
        return len(self.entries)
    def __getitem__(self, index: int) -> dict[str, Any]:
        name, item = self.entries[int(index)]
        row = self.mappings[name][item]
        return {"image": row["image"], "mask": row["mask"], "label": torch.as_tensor(row["label"], dtype=torch.int64), "class_name": name, "image_path": str(row.get("file_name", row.get("image_path", item)))}


def _target_dataset(target: str, root: Path | None) -> Dataset:
    target_lower = target.lower()
    if target_lower == "mvtec":
        if root is None:
            root = resolve_mvtec_root(None)
        if root is None:
            raise ValueError("MVTec target requires --target-root or ACDCLIP_MVTEC_ROOT")
        return MVTecDatasetAdapter(root)
    if target_lower == "visa":
        if root is None:
            raise ValueError("VisA target requires --target-root")
        return ManifestDataset(root, Path(__file__).resolve().parents[2] / "dataset/hub/VisA.jsonl")
    canonical = next((name for name in MEDICAL_TARGETS if name.lower() == target_lower), None)
    if canonical is None:
        raise ValueError(f"unsupported target: {target}")
    if root is None:
        raise ValueError(f"{canonical} target requires --target-root")
    os.environ["ACDCLIP_DATA_ROOT"] = str(root.expanduser().resolve())
    os.environ["MEDICAL_ROOT"] = str(root.expanduser().resolve())
    from dataset import get_text_and_image_dataset
    mappings = get_text_and_image_dataset(canonical, IMAGE_SIZE, stage="test")
    return MappingDataset(mappings)


def _domain(target: str) -> str:
    return "Medical" if any(target.lower() == value.lower() for value in MEDICAL_TARGETS) else "Industrial"


def _write_result(output_dir: Path, result: dict[str, Any], row: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader(); writer.writerow(row)


def _shutdown_loader(loader: DataLoader | None, loader_iter: Any) -> None:
    """Stop persistent workers before metric computation."""
    candidates = []
    if loader_iter is not None:
        candidates.append(loader_iter)
    if loader is not None:
        candidates.append(getattr(loader, "_iterator", None))
    seen: set[int] = set()
    for candidate in candidates:
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))
        shutdown = getattr(candidate, "_shutdown_workers", None)
        if callable(shutdown):
            shutdown()
    if loader is not None and hasattr(loader, "_iterator"):
        loader._iterator = None


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    cir_config = load_cir_config(args.config)
    target_lower = str(args.target).lower()
    if (args.source == "visa" and target_lower == "visa") or (args.source == "mvtec" and target_lower == "mvtec"):
        raise ValueError("source and target datasets must be different")
    stage_name = "CIR/EVAL-" + str(args.source).upper() + "-SOURCE"
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    # The frozen checkpoint records the commit that produced its weights. The
    # evaluator checkout may be a later, evaluator-only commit; all scientific
    # identity fields remain hard-validated by this call.
    validate_checkpoint_identity(checkpoint, cir_config, source_dataset=args.source)
    parent_path = Path(cir_config.get("parent_config_path", "configs/phase2b_canonical_v1.json"))
    if not parent_path.is_absolute():
        parent_path = Path(__file__).resolve().parents[2] / parent_path
    parent_config = dict(checkpoint.get("parent_config") or json.loads(parent_path.read_text(encoding="utf-8")))
    checkpoint_git_sha = str(checkpoint.get("git_sha", ""))
    checkpoint_sha = sha256_file(checkpoint_path)
    epoch = int(checkpoint.get("epoch", 0))
    configure_canonical_fp32()
    device = torch.device(args.device)
    model = build_phase2b_frozen(parent_config, checkpoint, args.clip_asset, device)
    model_loaded_rss = _rss_bytes()
    del checkpoint
    del parent_config
    dataset = _target_dataset(args.target, args.target_root)
    loader_kwargs: dict[str, Any] = {"batch_size": int(args.batch_size), "shuffle": False, "num_workers": int(args.num_workers), "pin_memory": bool(device.type == "cuda")}
    if args.num_workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": int(args.prefetch_factor)})
    loader: DataLoader | None = DataLoader(dataset, **loader_kwargs)
    domain = _domain(args.target)
    source_display = "VisA" if str(args.source).lower() == "visa" else "MVTec"
    target_display = "MVTec" if target_lower == "mvtec" else "VisA" if target_lower == "visa" else str(args.target)
    spool = EvaluationSpool.create(args.output_dir)
    monitor = _RssMonitor()
    monitor_started = False
    loader_iter: Any = None
    eval_progress: Any = None
    batch = image = output = pixels = classifications = masks = labels = paths = pixel = None
    names: list[str] | None = None
    evaluated: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    try:
        loader_iter = iter(loader)
        monitor.start()
        monitor_started = True
        monitor.set_phase("inference")
        eval_progress = tqdm(
            loader_iter,
            desc=f"CIR/EVAL {target_display} | {source_display}→{target_display} | E{epoch:02d}",
            unit="img",
            dynamic_ncols=True,
        )
        eval_started = time.perf_counter()
        try:
            for batch in eval_progress:
                image = batch["image"].to(device, non_blocking=device.type == "cuda").float()
                names = [str(x) for x in batch["class_name"]]
                output = forward_cir(model, image, names, device, cir_config, domain=domain, require_grad=False, dataset_name=args.target)
                pixels = output.cir_segmentation_probability.detach().cpu().numpy()
                classifications = output.classification_probability.detach().cpu().numpy()
                masks = batch["mask"].detach().cpu().numpy()
                labels = batch["label"].detach().cpu().numpy()
                paths = [str(x) for x in batch["image_path"]]
                for index, name in enumerate(names):
                    pixel = pixels[index].reshape(-1)
                    spool.append(
                        name,
                        pixel,
                        masks[index].reshape(-1),
                        image_score(float(classifications[index]), float(pixel.max()), domain),
                        int(labels[index]),
                    )
                eval_elapsed = max(time.perf_counter() - eval_started, 1e-9)
                eval_progress.set_postfix_str(f"{eval_progress.n / eval_elapsed:.1f} img/s")
                batch = image = output = pixels = classifications = masks = labels = paths = pixel = None
                names = None
        finally:
            if eval_progress is not None:
                eval_progress.close()
            eval_progress = None
            batch = image = output = pixels = classifications = masks = labels = paths = pixel = None
            names = None
            monitor.set_phase("post_inference")
            _shutdown_loader(loader, loader_iter)
            loader_iter = None
            loader = None
            dataset = None
            model = None
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            monitor.set_phase("after_teardown")

        monitor.set_phase("metric")
        evaluated = evaluate_spool(
            spool,
            allow_undefined_image_metrics=(domain == "Medical"),
        )
        spool.cleanup()
        monitor.set_phase("serialization")
        checkpoint_sha = sha256_file(checkpoint_path)
        evaluator_hash = sha256_file(Path(__file__).resolve())
        macro = evaluated["macro"]
        identity = {
            "arch_id": cir_config["arch_id"],
            "architecture_version": cir_config["architecture_version"],
            "source": args.source,
            "target": args.target,
            "epoch": epoch,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            "config_sha256": config_sha256(cir_config),
            "git_sha": checkpoint_git_sha,
            "evaluator_protocol": cir_config["evaluator_protocol"],
            "evaluator_hash": evaluator_hash,
        }
        row = {**identity, **{f"macro_{key}": value for key, value in macro.items()}}
        result = {
            "stage": stage_name,
            "status": "PASS",
            **identity,
            "per_class": evaluated["per_class"],
            "macro": macro,
        }
        _write_result(args.output_dir, result, row)
        monitor.set_phase("final")
    finally:
        if monitor_started:
            monitor.stop()
            telemetry = monitor.report(model_loaded_rss)
            telemetry["status"] = "PASS" if result is not None else "FAIL"
            (args.output_dir / "memory_telemetry.json").write_text(
                json.dumps(telemetry, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        spool.cleanup()
    assert result is not None
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["visa", "mvtec"], required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--target-root", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--clip-asset", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/cir_dfg_rmt_v1.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    args = parser.parse_args(argv)
    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
