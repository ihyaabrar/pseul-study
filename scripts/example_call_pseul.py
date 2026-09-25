from __future__ import annotations

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from pseul import PSEUL, PseulConfig, PseulFeatureProfile


def profile_for_breast_cancer(features: list[str]) -> dict[str, PseulFeatureProfile]:
    profile: dict[str, PseulFeatureProfile] = {}
    for feature in features:
        upper = feature.upper()
        if any(key in upper for key in ["CONCAVE", "RADIUS", "PERIMETER", "AREA"]):
            profile[feature] = PseulFeatureProfile(
                evidence=0.95,
                intervention_availability=0.75,
                risk_stratification=0.90,
                operational_feasibility=0.90,
                label_derived_risk=0.00,
                definitional_overlap=0.05,
                target_proxy_strength=0.10,
                topic_similarity=0.90,
                interpretation="Morphological feature relevant for malignancy discrimination.",
            )
        else:
            profile[feature] = PseulFeatureProfile(
                evidence=0.65,
                intervention_availability=0.50,
                risk_stratification=0.60,
                operational_feasibility=0.85,
                label_derived_risk=0.00,
                definitional_overlap=0.05,
                target_proxy_strength=0.05,
                topic_similarity=0.60,
            )
    return profile


def main() -> None:
    data = load_breast_cancer(as_frame=True)
    X = data.frame.drop(columns=["target"])
    X.columns = [c.replace(" ", "_").upper() for c in X.columns]
    y = (data.frame["target"] == 0).astype(int)

    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = LGBMClassifier(n_estimators=120, learning_rate=0.05, random_state=42, verbose=-1)

    pseul = PSEUL(
        estimator=model,
        clinical_profile=profile_for_breast_cancer(list(X.columns)),
        config=PseulConfig(n_splits=3, top_k=5, max_shap_samples_per_fold=200),
    )
    pseul.fit(X_train, y_train)

    print("Selected features:")
    print(pseul.selected_features_)
    print("\nScore summary:")
    print(
        pseul.summary()[
            [
                "feature",
                "select_score",
                "audit_score",
                "predictive_utility",
                "shap_stability",
                "clinical_utility",
                "leakage_risk",
                "pseul_category",
                "selected",
            ]
        ].head(10).to_string(index=False)
    )
    print("\nTransformed shape:", pseul.transform(X_test).shape)


if __name__ == "__main__":
    main()
