from __future__ import annotations

from select_phase2b_checkpoint import select_candidate


def _row(epoch, score):
    return {"epoch": epoch, "path": f"adapter_{epoch}.pth", "pixel_auroc": score, "pixel_ap": score, "image_auroc": score, "image_ap": score}


def test_selector_formula_and_earlier_tie():
    selected = select_candidate([_row(10, 0.5), _row(12, 0.5)])
    assert selected["epoch"] == 10
    selected = select_candidate([_row(10, 0.5), _row(12, 0.6)])
    assert selected["epoch"] == 12
