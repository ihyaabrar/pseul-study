from __future__ import annotations

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from scripts.pseul import PSEUL, PseulConfig

from .nested_evaluation import FrameMedianImputer, RANDOM_STATE, make_model, metric_row
from .study_registry import STUDY_SPECS, load_study


HERE = Path(__file__).resolve().parent
OUTPUT_DIR = HERE / "outputs" / "robustness"


def _without_component(config: PseulConfig, component: str) -> PseulConfig:
    weights = {"alpha": config.alpha, "beta": config.beta, "gamma": config.gamma, "delta": config.delta}
    removed = weights[component]
    scale = 1.0 / (1.0 - removed)
    weights = {name: (0.0 if name == component else value * scale) for name, value in weights.items()}
    return replace(config, **weights)


def _variants(top_k: int) -> dict[str, PseulConfig]:
    base = PseulConfig(
        n_splits=3,
        random_state=RANDOM_STATE,
        top_k=top_k,
        max_shap_samples_per_fold=500,
        permutation_repeats=5,
        semantic_veto_threshold=0.85,
        min_select_score=0.20,
        enforce_prediction_availability=True,
    )
    return {
        "Reference": base,
        "Without predictive utility P": _without_component(base, "alpha"),
        "Without SHAP stability S": _without_component(base, "beta"),
        "Without evidence E": _without_component(base, "gamma"),
        "Without clinical utility U": _without_component(base, "delta"),
        "Without soft leakage penalty": replace(base, use_soft_leakage_penalty=False),
        "Without redundancy penalty": replace(base, redundancy_weight=0.0),
        "Without semantic veto": replace(base, semantic_veto_threshold=None),
        "Semantic veto 0.75": replace(base, semantic_veto_threshold=0.75),
        "Semantic veto 0.95": replace(base, semantic_veto_threshold=0.95),
        "Score floor 0.10": replace(base, min_select_score=0.10),
        "Score floor 0.30": replace(base, min_select_score=0.30),
        "Top-k ceiling 5": replace(base, top_k=5),
    }


def _jaccard(feature_sets: list[set[str]]) -> float:
    values = []
    for left_index, left in enumerate(feature_sets):
        for right in feature_sets[left_index + 1 :]:
            union = left | right
            values.append(len(left & right) / len(union) if union else 1.0)
    return float(np.mean(values)) if values else 1.0


def run() -> Path:
    study = "nhanes"
    spec = STUDY_SPECS[study]
    X, y, metadata, profile = load_study(study)
    X.index = np.arange(len(X))
    y.index = X.index
    development_ids, _ = train_test_split(
        X.index.to_numpy(), test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    X_development = X.loc[development_ids]
    y_development = y.loc[development_ids]
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    variants = _variants(spec.top_k)

    prediction_rows: list[dict] = []
    selection_rows: list[dict] = []
    for fold, (train_position, valid_position) in enumerate(
        folds.split(X_development, y_development), start=1
    ):
        train_ids = X_development.index.to_numpy()[train_position]
        valid_ids = X_development.index.to_numpy()[valid_position]
        X_train_raw = X_development.loc[train_ids]
        X_valid_raw = X_development.loc[valid_ids]
        y_train = y_development.loc[train_ids].reset_index(drop=True)
        y_valid = y_development.loc[valid_ids].reset_index(drop=True)
        imputer = FrameMedianImputer().fit(X_train_raw)
        X_train = imputer.transform(X_train_raw).reset_index(drop=True)
        X_valid = imputer.transform(X_valid_raw).reset_index(drop=True)

        reference = PSEUL(
            estimator=make_model(RANDOM_STATE + fold * 101),
            clinical_profile=profile,
            config=replace(variants["Reference"], random_state=RANDOM_STATE + fold * 101),
        ).fit(X_train_raw.reset_index(drop=True), y_train)

        for variant, config in variants.items():
            print(f"[robustness] fold {fold}/5 · {variant}", flush=True)
            selector = PSEUL(
                estimator=make_model(RANDOM_STATE + fold * 101),
                clinical_profile=profile,
                config=replace(config, random_state=RANDOM_STATE + fold * 101),
            )
            selector.feature_names_ = list(reference.feature_names_)
            selector.fold_data_ = reference.fold_data_
            selector.base_scores_ = reference.base_scores_.copy()
            selector.scores_ = selector._build_score_table(selector.base_scores_, selector.fold_data_)
            selector.selected_features_ = selector._greedy_select(selector.scores_, selector.fold_data_)
            selector.scores_ = selector._attach_selection_info(selector.scores_, selector.fold_data_)
            features = list(selector.selected_features_)
            selection_rows.append(
                {
                    "fold": fold,
                    "variant": variant,
                    "features": json.dumps(features),
                    "n_features": len(features),
                    "designated_traps_selected": len(set(features) & set(spec.canonical_traps)),
                }
            )
            if not features:
                continue

            learners = {
                "LightGBM": make_model(RANDOM_STATE + fold * 101),
                "Logistic regression": make_pipeline(
                    StandardScaler(),
                    LogisticRegression(max_iter=2000, solver="liblinear", random_state=RANDOM_STATE),
                ),
            }
            for learner_name, model in learners.items():
                model.fit(X_train[features], y_train)
                probability = model.predict_proba(X_valid[features])[:, 1]
                prediction_rows.extend(
                    {
                        "fold": fold,
                        "variant": variant,
                        "learner": learner_name,
                        "row_id": int(row_id),
                        "y_true": int(target),
                        "probability": float(score),
                    }
                    for row_id, target, score in zip(valid_ids, y_valid, probability)
                )

    predictions = pd.DataFrame(prediction_rows)
    selections = pd.DataFrame(selection_rows)
    summary_rows = []
    for (variant, learner), frame in predictions.groupby(["variant", "learner"]):
        selected = selections[selections.variant.eq(variant)]
        feature_sets = [set(json.loads(value)) for value in selected.features]
        summary_rows.append(
            {
                "variant": variant,
                "learner": learner,
                **metric_row(frame.y_true, frame.probability.to_numpy()),
                "mean_selected_features": float(selected.n_features.mean()),
                "mean_designated_traps_selected": float(selected.designated_traps_selected.mean()),
                "mean_pairwise_jaccard": _jaccard(feature_sets),
            }
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(OUTPUT_DIR / "nhanes_robustness_oof_predictions.csv", index=False)
    selections.to_csv(OUTPUT_DIR / "nhanes_robustness_selections.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUTPUT_DIR / "nhanes_robustness_summary.csv", index=False)
    audit = {
        "study": study,
        "scope": "development-only exploratory robustness analysis",
        "outer_splits": 5,
        "inner_splits": 3,
        "random_state": RANDOM_STATE,
        "development_rows": int(len(X_development)),
        "metadata": metadata,
        "variants": {name: asdict(config) for name, config in variants.items()},
        "interpretation": (
            "One-factor-at-a-time component and threshold perturbations. Feature selection remains "
            "LightGBM-based; logistic regression tests downstream learner sensitivity on identical subsets."
        ),
    }
    (OUTPUT_DIR / "nhanes_robustness_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(f"robustness -> {OUTPUT_DIR}", flush=True)
    return OUTPUT_DIR


if __name__ == "__main__":
    run()
