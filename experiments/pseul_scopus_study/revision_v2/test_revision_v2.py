from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from scripts.pseul import PSEUL, PseulConfig, PseulFeatureProfile

from .nested_evaluation import (
    AUTO_METHOD,
    FrameMedianImputer,
    _delong_auc_covariance,
    build_selections,
    holm_adjust,
    make_model,
    paired_bootstrap,
    signed_rank_biserial,
)


class RevisionV2Tests(unittest.TestCase):
    def test_imputer_uses_training_median_only(self) -> None:
        train = pd.DataFrame({"x": [1.0, 3.0, np.nan]}, index=[10, 11, 12])
        validation = pd.DataFrame({"x": [1000.0, np.nan]}, index=[20, 21])
        imputer = FrameMedianImputer().fit(train)
        transformed = imputer.transform(validation)
        self.assertEqual(imputer.medians_["x"], 2.0)
        self.assertEqual(transformed.loc[21, "x"], 2.0)
        self.assertEqual(imputer.fit_index_hash_, imputer.fit_index_hash_)

    def test_prediction_availability_is_hard_select_gate(self) -> None:
        rng = np.random.default_rng(42)
        y = pd.Series(rng.integers(0, 2, 240))
        X = pd.DataFrame(
            {
                "safe_a": y + rng.normal(0, 0.8, len(y)),
                "safe_b": rng.normal(0, 1, len(y)),
                "future_label": y.astype(float),
            }
        )
        profile = {
            "safe_a": PseulFeatureProfile(available_at_prediction=True),
            "safe_b": PseulFeatureProfile(available_at_prediction=True),
            "future_label": PseulFeatureProfile(
                label_derived_risk=1.0,
                definitional_overlap=1.0,
                target_proxy_strength=1.0,
                available_at_prediction=False,
            ),
        }
        method = PSEUL(
            make_model(42),
            profile,
            PseulConfig(n_splits=3, top_k=2, max_shap_samples_per_fold=100),
        ).fit(X, y)
        self.assertNotIn("future_label", method.selected_features_)
        self.assertTrue(bool(method.summary().set_index("feature").loc["future_label", "available_at_prediction"]) is False)
        self.assertIn("future_label", method.select_features(mode="audit", top_k=3))

    def test_semantic_veto_rejects_available_target_proxy(self) -> None:
        rng = np.random.default_rng(7)
        y = pd.Series(rng.integers(0, 2, 240))
        X = pd.DataFrame(
            {
                "safe_signal": y + rng.normal(0, 0.7, len(y)),
                "safe_noise": rng.normal(0, 1, len(y)),
                "available_proxy": y.astype(float),
            }
        )
        profile = {
            "safe_signal": PseulFeatureProfile(),
            "safe_noise": PseulFeatureProfile(),
            "available_proxy": PseulFeatureProfile(target_proxy_strength=0.95),
        }
        method = PSEUL(
            make_model(7),
            profile,
            PseulConfig(n_splits=3, top_k=3, max_shap_samples_per_fold=100),
        ).fit(X, y)
        row = method.summary().set_index("feature").loc["available_proxy"]
        self.assertTrue(bool(row["semantic_veto"]))
        self.assertEqual(row["rejection_reason"], "semantic-veto")
        self.assertNotIn("available_proxy", method.selected_features_)
        self.assertLess(len(method.selected_features_), 3)

    def test_score_floor_permits_abstention(self) -> None:
        rng = np.random.default_rng(11)
        y = pd.Series(rng.integers(0, 2, 180))
        X = pd.DataFrame({"a": rng.normal(size=len(y)), "b": rng.normal(size=len(y))})
        method = PSEUL(
            make_model(11),
            {},
            PseulConfig(n_splits=3, top_k=2, min_select_score=1.01, max_shap_samples_per_fold=100),
        ).fit(X, y)
        self.assertEqual(method.selected_features_, [])
        self.assertTrue((method.summary()["rejection_reason"] == "below-score-floor").all())

    def test_auto_control_changes_only_leakage_metadata(self) -> None:
        rng = np.random.default_rng(19)
        y = pd.Series(rng.integers(0, 2, 180))
        X = pd.DataFrame(
            {
                "signal": y + rng.normal(0, 0.8, len(y)),
                "noise": rng.normal(size=len(y)),
                "proxy": y + rng.normal(0, 0.05, len(y)),
            }
        )
        profile = {
            "signal": PseulFeatureProfile(evidence=0.8, intervention_availability=0.8),
            "noise": PseulFeatureProfile(evidence=0.2, intervention_availability=0.2),
            "proxy": PseulFeatureProfile(
                target_proxy_strength=0.95,
                evidence=0.7,
                intervention_availability=0.4,
            ),
        }
        selections, _, details = build_selections(
            X,
            y,
            profile,
            top_k=2,
            traps=("proxy",),
            seed=19,
            inner_splits=3,
            include_proxies=False,
            include_auto=True,
            X_pseul_raw=X,
        )
        self.assertIn(AUTO_METHOD, selections)
        self.assertEqual(details["PSEUL-Auto control audit"]["max_abs_signal_difference"], 0.0)

    def test_signed_rank_biserial_preserves_direction(self) -> None:
        self.assertGreater(signed_rank_biserial(np.array([1.0, 2.0, 3.0])), 0)
        self.assertLess(signed_rank_biserial(np.array([-1.0, -2.0, -3.0])), 0)

    def test_holm_adjustment_is_monotone_and_bounded(self) -> None:
        adjusted = holm_adjust(np.array([0.01, 0.04, 0.03]))
        np.testing.assert_allclose(adjusted, np.array([0.03, 0.06, 0.06]))
        self.assertTrue(np.all((adjusted >= 0) & (adjusted <= 1)))

    def test_delong_auc_matches_sklearn(self) -> None:
        truth = np.array([0, 0, 1, 1, 0, 1])
        probability = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9])
        aucs, covariance = _delong_auc_covariance(truth, probability)
        self.assertAlmostEqual(float(aucs[0]), roc_auc_score(truth, probability))
        self.assertGreaterEqual(float(covariance[0, 0]), 0.0)

    def test_paired_delong_identical_predictions_has_zero_delta(self) -> None:
        truth = np.array([0, 0, 1, 1, 0, 1])
        probability = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.9])
        rows = []
        for method in ("PSEUL-Select", "Comparator"):
            for row_id, (target, score) in enumerate(zip(truth, probability)):
                rows.append({"row_id": row_id, "method": method, "y_true": target, "probability": score})
        result = paired_bootstrap(pd.DataFrame(rows), n_bootstrap=100)
        auc = result[result["metric"] == "auc"].iloc[0]
        self.assertAlmostEqual(float(auc["delta_reference_minus_comparator"]), 0.0)
        self.assertAlmostEqual(float(auc["two_sided_bootstrap_p"]), 1.0)


if __name__ == "__main__":
    unittest.main()
