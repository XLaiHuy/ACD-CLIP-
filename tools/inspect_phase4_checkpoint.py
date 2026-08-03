#!/usr/bin/env python3
"""Print safe metadata from a Phase2B or Phase4 adapter checkpoint."""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.checkpoint_utils import h6_config_from_checkpoint, is_phase4_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Inspect Phase4 checkpoint metadata without loading a model")
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    report = {
        "epoch": checkpoint.get("epoch"),
        "phase4": is_phase4_checkpoint(checkpoint),
        "phase4_progress": checkpoint.get("phase4_progress", 0),
        "n_groups": checkpoint.get("n_groups"),
        "prompt_mode": checkpoint.get("prompt_mode"),
        "precision": checkpoint.get("precision"),
        "h6_config": h6_config_from_checkpoint(checkpoint),
        "has_optimizer_state": "optimizer_state" in checkpoint,
        "has_scheduler_state": "scheduler_state" in checkpoint,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
