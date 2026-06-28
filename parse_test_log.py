import argparse
import ast
import re
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Parse ACD-CLIP test.log summaries.")
    parser.add_argument("--log", default="test_train_main_base/test.log")
    parser.add_argument(
        "--paper-summary",
        action="store_true",
        help="Print medical pixel-level and image-level summaries like the paper table.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    lines = Path(args.log).read_text(encoding="utf-8").splitlines()

    current = {
        "dataset": None,
        "epoch": None,
        "metric_thresholds": None,
        "pixel_stride": 1,
        "max_samples": None,
        "max_samples_per_label": None,
    }
    rows = []

    for line in lines:
        if "args:" in line:
            args_text = line.split("args:", 1)[1].strip()
            try:
                parsed = ast.literal_eval(args_text)
            except (SyntaxError, ValueError):
                parsed = {}
            current["dataset"] = parsed.get("dataset")
            current["metric_thresholds"] = parsed.get("metric_thresholds")
            current["pixel_stride"] = parsed.get("pixel_stride", 1)
            current["max_samples"] = parsed.get("max_samples")
            current["max_samples_per_label"] = parsed.get("max_samples_per_label")

        match = re.search(r"load model from epoch (\d+)", line)
        if match:
            current["epoch"] = int(match.group(1))

        if "Average" not in line:
            continue
        nums = re.findall(r"[-+]?\d+(?:\.\d+)?", line)
        if len(nums) < 4 or current["dataset"] is None or current["epoch"] is None:
            continue
        pixel_auc, pixel_ap, image_auc, image_ap = map(float, nums[-4:])
        rows.append({
            **current,
            "pixel_auc": pixel_auc,
            "pixel_ap": pixel_ap,
            "image_auc": image_auc,
            "image_ap": image_ap,
        })

    print("dataset,epoch,metric_thresholds,pixel_stride,max_samples,max_samples_per_label,pixel_auc,pixel_ap,image_auc,image_ap")
    for row in rows:
        print(
            f"{row['dataset']},{row['epoch']},{row['metric_thresholds']},"
            f"{row['pixel_stride']},{row['max_samples']},{row['max_samples_per_label']},"
            f"{row['pixel_auc']:.2f},{row['pixel_ap']:.2f},"
            f"{row['image_auc']:.2f},{row['image_ap']:.2f}"
        )

    grouped = defaultdict(list)
    for row in rows:
        key = (
            row["epoch"],
            row["metric_thresholds"],
            row["pixel_stride"],
            row["max_samples"],
            row["max_samples_per_label"],
        )
        grouped[key].append(row)

    print("\nmeans:")
    print("epoch,metric_thresholds,pixel_stride,max_samples,max_samples_per_label,n,mean_pixel_auc,mean_pixel_ap")
    def sort_key(key):
        return tuple("" if value is None else str(value) for value in key)

    for key in sorted(grouped, key=sort_key):
        vals = grouped[key]
        mean_auc = sum(row["pixel_auc"] for row in vals) / len(vals)
        mean_ap = sum(row["pixel_ap"] for row in vals) / len(vals)
        print(f"{key[0]},{key[1]},{key[2]},{key[3]},{key[4]},{len(vals)},{mean_auc:.2f},{mean_ap:.2f}")

    if args.paper_summary:
        print_paper_summary(rows, sort_key)


def print_paper_summary(rows, sort_key):
    pixel_names = {
        "Colon_colonDB": "ColonDB",
        "Colon_clinicDB": "ClinicDB",
        "Colon_Kvasir": "Kvasir",
        "Brain": "BrainMRI",
        "Liver": "Liver CT",
        "Retina": "Retina OCT",
    }
    image_names = {
        "Brain": "BrainMRI",
        "Liver": "Liver CT",
        "Retina": "Retina OCT",
    }

    grouped = defaultdict(list)
    for row in rows:
        key = (
            row["epoch"],
            row["metric_thresholds"],
            row["pixel_stride"],
            row["max_samples"],
            row["max_samples_per_label"],
        )
        grouped[key].append(row)

    print("\npaper_style_pixel_level:")
    print("epoch,dataset,pixel_auc,pixel_ap")
    for key in sorted(grouped, key=sort_key):
        by_dataset = {row["dataset"]: row for row in grouped[key]}
        for dataset, display_name in pixel_names.items():
            row = by_dataset.get(dataset)
            if row is None:
                continue
            print(f"{key[0]},{display_name},{row['pixel_auc']:.2f},{row['pixel_ap']:.2f}")

    print("\npaper_style_pixel_level_means:")
    print("epoch,n,mean_pixel_auc,mean_pixel_ap")
    for key in sorted(grouped, key=sort_key):
        vals = [row for row in grouped[key] if row["dataset"] in pixel_names]
        if not vals:
            continue
        mean_auc = sum(row["pixel_auc"] for row in vals) / len(vals)
        mean_ap = sum(row["pixel_ap"] for row in vals) / len(vals)
        print(f"{key[0]},{len(vals)},{mean_auc:.2f},{mean_ap:.2f}")

    print("\npaper_style_image_level:")
    print("epoch,dataset,image_auc,image_ap")
    for key in sorted(grouped, key=sort_key):
        by_dataset = {row["dataset"]: row for row in grouped[key]}
        for dataset, display_name in image_names.items():
            row = by_dataset.get(dataset)
            if row is None:
                continue
            print(f"{key[0]},{display_name},{row['image_auc']:.2f},{row['image_ap']:.2f}")

    print("\npaper_style_image_level_means:")
    print("epoch,n,mean_image_auc,mean_image_ap")
    for key in sorted(grouped, key=sort_key):
        vals = [row for row in grouped[key] if row["dataset"] in image_names]
        if not vals:
            continue
        mean_auc = sum(row["image_auc"] for row in vals) / len(vals)
        mean_ap = sum(row["image_ap"] for row in vals) / len(vals)
        print(f"{key[0]},{len(vals)},{mean_auc:.2f},{mean_ap:.2f}")


if __name__ == "__main__":
    main()
