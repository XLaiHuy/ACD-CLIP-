import math
import unittest
from pathlib import Path

from phase2b_anchor_diagnosis import IMAGE_DATASETS, aggregate_rows


class ExactMedicalProtocolTests(unittest.TestCase):
    def test_raw_exact_macro_aggregation_is_not_rounded(self):
        pixel_values = [90.123456789 + index * 1.0e-6 for index in range(6)]
        pixel_rows = [
            {"epoch": 15, "prompt_config": "current_shared", "pixel_auc": value, "pixel_ap": value + 1.0}
            for value in pixel_values
        ]
        raw_rows = []
        for dataset in IMAGE_DATASETS:
            raw_rows.extend(
                {
                    "dataset": dataset,
                    "epoch": 15,
                    "prompt_config": "current_shared",
                    "label": label,
                    "cls_score": score,
                    "max_pixel": score,
                    "top1pct_pixel": score,
                }
                for label, score in ((0, 0.1), (1, 0.9), (1, 0.8), (0, 0.2))
            )
        aggregate, details = aggregate_rows(
            raw_rows,
            pixel_rows,
            IMAGE_DATASETS + ["Colon_clinicDB", "Colon_colonDB", "Colon_Kvasir"],
            ["cls_only"],
            round_result=False,
        )
        row = aggregate[0]
