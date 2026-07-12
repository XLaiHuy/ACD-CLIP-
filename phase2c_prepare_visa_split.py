#!/usr/bin/env python3
import argparse
from pathlib import Path

from dataset.info import DATA_PATH
from phase2c_split import build_split, verify_split


def parse_args():
    parser = argparse.ArgumentParser(description="Build/verify deterministic group-aware VisA splits")
    parser.add_argument("--source", default="dataset/hub/VisA.jsonl")
    parser.add_argument("--output-dir", default="splits")
    parser.add_argument("--data-root", default=DATA_PATH["VisA"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--phash-distance", type=int, default=4)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output_dir)
    train = output / f"visa_train_seed{args.seed}.csv"
    val = output / f"visa_val_seed{args.seed}.csv"
    metadata = output / f"visa_split_seed{args.seed}_metadata.json"
    if args.verify_only:
        verify_split(args.source, train, val, metadata)
        print(f"Verified {train}, {val}, and {metadata}")
        return
    paths = build_split(
        args.source, output, data_root=args.data_root, seed=args.seed,
        train_ratio=args.train_ratio, phash_distance=args.phash_distance,
    )
    print("Wrote " + ", ".join(map(str, paths)))


if __name__ == "__main__":
    main()
