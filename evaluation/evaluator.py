"""One evaluator contract shared by selector, calibration, and test CLI."""
from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

from .metrics import class_metrics, macro_metrics


def image_score(classification_probability: Any, pixel_max: Any, domain: str):
    """Frozen image score: Industrial=.9/.1, Medical=.5/.5."""
    if str(domain) == "Medical":
        return 0.5 * classification_probability + 0.5 * pixel_max
    return 0.9 * classification_probability + 0.1 * pixel_max


def _method_payload(record: Mapping[str, Any], method: str) -> Mapping[str, Any]:
    if method in {"phase2b", "sabra"} and method in record:
        payload = record[method]
        if not isinstance(payload, Mapping):
            raise TypeError(f"record[{method!r}] must be a mapping")
        return payload
    return record


def evaluate_records(records: Iterable[Mapping[str, Any]], method: str = "phase2b") -> dict[str, Any]:
    """Evaluate already-produced arrays; no model or dataset side effects."""
    if method not in {"phase2b", "sabra", "compare"}:
        raise ValueError(f"unknown evaluation method: {method}")
    grouped: dict[str, dict[str, list[np.ndarray]]] = {}
    for record in records:
        class_name = str(record["class_name"])
        grouped.setdefault(class_name, {})
        payloads = ["phase2b", "sabra"] if method == "compare" else [method]
        for payload_name in payloads:
            payload = _method_payload(record, payload_name)
            grouped[class_name].setdefault(f"{payload_name}_pixel_scores", []).append(np.asarray(payload["pixel_scores"]).reshape(-1))
            grouped[class_name].setdefault(f"{payload_name}_pixel_labels", []).append(np.asarray(record["pixel_labels"]).reshape(-1))
            grouped[class_name].setdefault(f"{payload_name}_image_scores", []).append(np.asarray(payload["image_scores"]).reshape(-1))
            grouped[class_name].setdefault(f"{payload_name}_image_labels", []).append(np.asarray(record["image_labels"]).reshape(-1))

    def one(name: str) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
        per_class: dict[str, dict[str, float]] = {}
        for class_name, values in grouped.items():
            per_class[class_name] = class_metrics(
                np.concatenate(values[f"{name}_pixel_scores"]),
                np.concatenate(values[f"{name}_pixel_labels"]),
                np.concatenate(values[f"{name}_image_scores"]),
                np.concatenate(values[f"{name}_image_labels"]),
            )
        return per_class, macro_metrics(per_class)

    if method == "compare":
        phase2b, phase2b_macro = one("phase2b")
        sabra, sabra_macro = one("sabra")
        delta = {key: sabra_macro[key] - phase2b_macro[key] for key in phase2b_macro}
        return {"phase2b": phase2b, "phase2b_macro": phase2b_macro, "sabra": sabra, "sabra_macro": sabra_macro, "delta": delta}
    per_class, macro = one(method)
    return {"per_class": per_class, "macro": macro}
