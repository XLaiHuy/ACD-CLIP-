import numpy as np

from tools.sabra_cure import patch_actionability_r2 as p


def test_frozen_panel_and_numerical_contract():
    assert p.PARENT == "87d3c15b6fe4f62762bc87760960c1f83eda90d3"
    assert p.TARGET_PATCHES_PER_CLASS == 2000
    assert p.CAP_PER_IMAGE == 16
    assert p.STRATA == 5 and p.STRATUM_QUOTA == 80
    assert p.ALPHA == .25 and p.BENEFIT_EPS == 1e-10


def test_panel_hash_is_deterministic_and_gtfree_shape():
    panel = {"image_path": np.array(["a", "b"]), "image_index": np.array([0, 1], dtype=np.int32),
             "patch_index": np.array([1, 2], dtype=np.int32), "rank_stratum": np.array([0, 4], dtype=np.int8),
             "sensitivity_stratum": np.array([1, 3], dtype=np.int8)}
    assert p.panel_hash(panel) == p.panel_hash(panel)


def test_candidate_batch_rejects_invalid_rows():
    native = np.zeros((3, p.PATCHES, 2), dtype=np.float32)
    try:
        p.deployment_batch(native, np.array([], dtype=np.int64), np.array([], dtype=np.float32), __import__("torch").device("cpu"))
    except RuntimeError as error:
        assert "candidate batch" in str(error)
    else:
        raise AssertionError("empty candidate batch accepted")


def test_q1_gate_routing_math():
    folds = {}
    for name in p.r1.CLASSES:
        folds[name] = {"metrics": {"spearman": .25, "sign_auc": .70, "bc20": .40,
                                    "positive_count": 3, "negative_count": 3}}
    result = p.evaluate_q1(folds)
    assert result["pass"] is True and all(result["gates"].values())
    for name in p.r1.CLASSES[:4]:
        folds[name]["metrics"]["spearman"] = -.1
    assert p.evaluate_q1(folds)["pass"] is False
