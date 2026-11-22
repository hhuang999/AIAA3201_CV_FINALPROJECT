import json

import pytest

from depthlab import sota_diagnose, sota_v2_diagnose


@pytest.mark.parametrize(
    ("module", "prior_context", "affected_diagnostics"),
    [
        (
            sota_diagnose,
            "historical_b3_project_baseline",
            {"univariate_residual_r2", "multivariate_residual_r2", "global_spatial_residual_models"},
        ),
        (
            sota_v2_diagnose,
            "round1_test_cases_informed_round2_design",
            {"residual_r2", "residual_ridge_score_discriminators"},
        ),
    ],
)
def test_diagnostic_summary_preserves_train_scope_and_limits_secondary_evidence(
    tmp_path, module, prior_context, affected_diagnostics
):
    output_path = tmp_path / "diagnosis.json"
    baseline_evidence = {"abs_rel": 0.09, "rmse": 0.38, "delta1": 0.93}

    payload = sota_diagnose.write_diagnostic_summary(
        output_path, {"first_stage_baseline": baseline_evidence}, module.DIAGNOSTIC_PROTOCOL
    )
    assert json.loads(output_path.read_text(encoding="utf-8")) == payload
    assert payload["first_stage_baseline"] == baseline_evidence
    assert payload["test_read"] is False

    protocol = payload["diagnostic_protocol"]
    assert protocol["official_test_targets_read"] is False
    assert protocol["prior_official_test_context"] == prior_context
    second_stage = protocol["second_stage_residual_diagnostics"]
    assert second_stage["target_source"] == "reused_baseline_oof_residuals"
    assert second_stage["exploratory"] is True
    assert second_stage["nested_crossfit"] is False
    assert second_stage["validation_labels_independent_of_fit"] is False
    assert set(second_stage["affected_diagnostics"]) == affected_diagnostics
