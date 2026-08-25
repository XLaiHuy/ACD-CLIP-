import json

import numpy as np
import pytest

from tools.sabra_cure import patch_actionability_r3 as p


def _fixture_problem():
    x = np.array(
        [
            [0.0, 2.0, 7.0],
            [1.0, 3.0, 7.0],
            [2.0, 5.0, 7.0],
            [4.0, 8.0, 7.0],
            [0.5, 5.0, 7.0],
            [1.5, 7.0, 7.0],
            [3.0, 11.0, 7.0],
            [6.0, 13.0, 7.0],
        ],
        dtype=np.float64,
    )
    groups = [
        ("a", np.array([-2.0, -0.5, 0.5, 2.0], dtype=np.float64)),
        ("b", np.array([-3.0, -1.0, 1.0, 3.0], dtype=np.float64)),
    ]
    return p.build_pair_problem(x, groups)


def test_transformed_objective_and_gradient_are_original_beta_space_exact():
    problem = _fixture_problem()
    for beta in (
        np.array([0.2, -0.1, 0.0]),
        np.array([-0.35, 0.4, 0.0]),
        np.array([0.01, -0.02, 0.0]),
    ):
        z = p.beta_to_z(problem, beta)
        direct_value, direct_gradient = p.beta_objective_gradient(problem, beta)
        transformed_value, transformed_gradient = p.transformed_objective_gradient(problem, z)
        expected_gradient = direct_gradient[problem.active] / problem.scale[problem.active]
        assert transformed_value == pytest.approx(direct_value, rel=0.0, abs=2e-15)
        assert np.allclose(transformed_gradient, expected_gradient, rtol=0.0, atol=2e-13)
        assert np.allclose(p.z_to_beta(problem, z), beta, rtol=0.0, atol=2e-16)


def test_constant_pair_design_dimension_is_inactive_and_fixed_zero():
    problem = _fixture_problem()
    assert problem.active.tolist() == [True, True, False]
    assert problem.scale[2] == 1.0
    beta = p.z_to_beta(problem, np.array([0.3, -0.2]))
    assert beta[2] == 0.0


def test_known_failure_zero_beta_large_gradient_is_rejected():
    problem = _fixture_problem()
    beta = np.zeros(problem.feature_count, dtype=np.float64)
    _, gradient = p.beta_objective_gradient(problem, beta)
    certificate = p.original_beta_certificate(
        problem,
        beta,
        optimizer_success=True,
        objective_initial=p.beta_objective_gradient(problem, beta)[0],
        objective_final=p.beta_objective_gradient(problem, beta)[0],
    )
    assert np.linalg.norm(gradient) > 0.1
    assert certificate["valid"] is False
    assert certificate["relative_gradient_inf"] > p.RELATIVE_GRADIENT_INF_TOLERANCE


def test_ill_conditioned_problem_recovers_nonzero_valid_solution():
    rng = np.random.default_rng(2503)
    x = rng.normal(size=(80, 4)).astype(np.float64)
    x[:, 0] *= 1e9
    x[:, 1] *= 1e-8
    x[:, 3] = 3.0
    latent = 2e-9 * x[:, 0] - 4e7 * x[:, 1] + 0.2 * x[:, 2]
    groups = [("a", latent[:40]), ("b", latent[40:])]
    model = p.fit_ranker(x, groups)
    score = p.rank_predict(model, x)
    assert model["optimization"]["certificate"]["valid"] is True
    assert np.linalg.norm(model["beta"]) > 0.0
    assert np.std(score) > 0.0
    assert np.unique(score).size > 1
    assert model["optimization"]["solver"] == "deterministic-float64-damped-newton"
    json.dumps(p.json_safe_model(model), allow_nan=False)


def test_exact_transformed_hessian_matches_gradient_finite_difference():
    problem = _fixture_problem()
    z = np.array([0.13, -0.21], dtype=np.float64)
    hessian = p.transformed_hessian(problem, z)
    epsilon = 1e-6
    numerical = np.empty_like(hessian)
    for column in range(len(z)):
        plus = z.copy(); minus = z.copy()
        plus[column] += epsilon; minus[column] -= epsilon
        numerical[:, column] = (
            p.transformed_objective_gradient(problem, plus)[1]
            - p.transformed_objective_gradient(problem, minus)[1]
        ) / (2.0 * epsilon)
    assert np.allclose(hessian, numerical, rtol=0.0, atol=2e-10)


def test_undefined_correlations_remain_null():
    values = np.array([-1.0, 1.0, 2.0], dtype=np.float64)
    metrics = p.q1_metrics(values, np.zeros_like(values))
    assert metrics["spearman"] is None


def test_frozen_p25r2_target_inventory_is_hash_and_alignment_exact():
    audit = p.audit_target_artifacts()
    assert audit["status"] == "PASS"
    assert audit["class_count"] == 12
    assert audit["total_rows"] == 24000
    assert set(audit["classes"]) == set(p.r1.CLASSES)


def test_no_p25r3_attempt_exists_during_rehearsal():
    assert not (p.OUT / "ATTEMPT_STARTED.json").exists()


def test_q1_production_core_rehearsal_uses_certified_model_and_null_semantics():
    rng = np.random.default_rng(2533)
    source = rng.normal(size=(60, 32)).astype(np.float64)
    source[:, -1] = 1.0
    target = source[:, 0] - 0.5 * source[:, 1]
    held = rng.normal(size=(12, 32)).astype(np.float64)
    held[:, -1] = 1.0
    held_target = held[:, 0] - 0.5 * held[:, 1]
    result = p.fit_and_score_q1(
        [source[:30], source[30:]],
        [target[:30], target[30:]],
        ["a", "b"],
        held,
        held_target,
    )
    assert result["model"]["optimization"]["certificate"]["valid"] is True
    assert result["score"].shape == (12,)
    assert result["metrics"]["spearman"] is not None


def test_static_inherited_interface_audit_passes():
    audit = p.static_interface_audit()
    assert audit["status"] == "PASS"
    assert all(row["exists"] for row in audit["references"])


def test_terminal_decision_renderer_preserves_exact_status_and_routing():
    text = p.render_final_decision(
        {
            "status": "P25_PATCH_BENEFIT_NOT_IDENTIFIABLE",
            "scientific_validity": "VALID",
            "q1": {"pass": False, "gates": {"G1": True, "G2": False}},
            "q2": {"status": "NOT_ENTERED_Q1_ROUTING_STOP"},
            "attempt": {"attempt_uuid": "fixture"},
        }
    )
    assert "P25_PATCH_BENEFIT_NOT_IDENTIFIABLE" in text
    assert "NOT_ENTERED_Q1_ROUTING_STOP" in text
    assert "G2" in text
