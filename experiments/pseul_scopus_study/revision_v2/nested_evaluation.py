from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from scipy import stats
from sklearn.base import clone
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[3]
STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(STUDY_ROOT))

from baseline_knockoff_shapselect import (  # noqa: E402
    gaussian_knockoff_select,
    grasp_select,
    manual_exclusion_shap_select,
    shap_select,
)
from pseul import PSEUL, PseulConfig, PseulFeatureProfile  # noqa: E402

from .study_registry import STUDY_SPECS, load_study, write_registry_artifacts


RANDOM_STATE = 42
PRIMARY_METHOD = "PSEUL-Select"
AUTO_METHOD = "PSEUL-Auto (leakage metadata null)"
AVAILABILITY_BASELINE = "Availability Gate + SHAP"
SEMANTIC_BASELINE = "Semantic Veto + SHAP"


class FrameMedianImputer:
    """A deterministic train-only median imputer that preserves DataFrame names."""

    def fit(self, X: pd.DataFrame) -> "FrameMedianImputer":
        self.columns_ = list(X.columns)
        self.medians_ = X[self.columns_].median(numeric_only=True).reindex(self.columns_).fillna(0.0)
        self.fit_index_hash_ = _index_hash(X.index.to_numpy())
        self.fit_rows_ = int(len(X))
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "medians_"):
            raise RuntimeError("Call fit before transform.")
        return X[self.columns_].apply(pd.to_numeric, errors="coerce").fillna(self.medians_).fillna(0.0)

    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return self.fit(X).transform(X)


def _index_hash(values: np.ndarray) -> str:
    normalized = ",".join(map(str, sorted(map(int, values.tolist()))))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def make_model(seed: int) -> LGBMClassifier:
    return LGBMClassifier(
        random_state=seed,
        verbose=-1,
        n_jobs=1,
        n_estimators=250,
        learning_rate=0.04,
        num_leaves=31,
        subsample=0.90,
        colsample_bytree=0.90,
    )


def _shap_values(model, X: pd.DataFrame) -> np.ndarray:
    values = shap.TreeExplainer(model).shap_values(X)
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, -1]
    return values


def _top_k(series: pd.Series, k: int) -> list[str]:
    return list(series.sort_values(ascending=False).head(min(k, len(series))).index)


def build_selections(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    profile: dict,
    top_k: int,
    traps: tuple[str, ...],
    seed: int,
    inner_splits: int,
    include_proxies: bool,
    include_auto: bool,
    X_pseul_raw: pd.DataFrame | None = None,
) -> tuple[dict[str, list[str]], pd.DataFrame, dict]:
    features = list(X_train.columns)
    pseul_input = X_train if X_pseul_raw is None else X_pseul_raw.reset_index(drop=True)
    pseul = PSEUL(
        estimator=make_model(seed),
        clinical_profile=profile,
        config=PseulConfig(
            n_splits=inner_splits,
            random_state=seed,
            top_k=top_k,
            max_shap_samples_per_fold=500,
            permutation_repeats=5,
            utility_method="permutation",
            target_association="partial_corr",
            semantic_veto_threshold=0.85,
            min_select_score=0.20,
            enforce_prediction_availability=True,
        ),
    )
    pseul.fit(pseul_input, y_train)
    scores = pseul.summary()
    proxy_details: dict[str, dict] = {}

    selections: dict[str, list[str]] = {"All Features": features}
    mi = mutual_info_classif(X_train, y_train, random_state=seed)
    selections["Mutual Information"] = _top_k(pd.Series(mi, index=features), top_k)

    gain_model = make_model(seed)
    gain_model.fit(X_train, y_train)
    selections["LightGBM Gain"] = _top_k(pd.Series(gain_model.feature_importances_, index=features), top_k)

    selector_train, selector_valid, y_selector_train, y_selector_valid = train_test_split(
        X_train,
        y_train,
        test_size=0.25,
        stratify=y_train,
        random_state=seed,
    )
    selector_model = make_model(seed + 1)
    selector_model.fit(selector_train, y_selector_train)
    perm = permutation_importance(
        selector_model,
        selector_valid,
        y_selector_valid,
        scoring="roc_auc",
        n_repeats=5,
        random_state=seed,
        n_jobs=1,
    )
    selections["Permutation Importance"] = _top_k(pd.Series(perm.importances_mean, index=features), top_k)

    shap_sample = X_train.sample(min(1000, len(X_train)), random_state=seed)
    shap_importance = pd.Series(np.abs(_shap_values(gain_model, shap_sample)).mean(axis=0), index=features)
    selections["SHAP Ranking"] = _top_k(shap_importance, top_k)
    available_features = [
        feature
        for feature in features
        if bool(getattr(profile.get(feature, PseulFeatureProfile()), "available_at_prediction", True))
    ]
    semantic_features = [
        feature
        for feature in available_features
        if max(
            profile.get(feature, PseulFeatureProfile()).label_derived_risk,
            profile.get(feature, PseulFeatureProfile()).definitional_overlap,
            profile.get(feature, PseulFeatureProfile()).target_proxy_strength,
        )
        < 0.85
    ]
    selections[AVAILABILITY_BASELINE] = _top_k(shap_importance.loc[available_features], top_k)
    selections[SEMANTIC_BASELINE] = _top_k(shap_importance.loc[semantic_features], top_k)
    selections[PRIMARY_METHOD] = list(pseul.selected_features_)
    selections["PSEUL-Audit"] = pseul.select_features(mode="audit", top_k=top_k)

    if include_auto:
        neutral_profile = {
            feature: replace(
                profile.get(feature, PseulFeatureProfile()),
                label_derived_risk=0.0,
                definitional_overlap=0.0,
                target_proxy_strength=0.0,
                interpretation="Leakage-metadata-null control; evidence, utility, availability, and stochastic path are retained.",
            )
            for feature in features
        }
        auto = PSEUL(
            estimator=make_model(seed),
            clinical_profile=neutral_profile,
            config=PseulConfig(
                n_splits=inner_splits,
                random_state=seed,
                top_k=top_k,
                max_shap_samples_per_fold=500,
                permutation_repeats=5,
                utility_method="permutation",
                target_association="partial_corr",
                semantic_veto_threshold=0.85,
                min_select_score=0.20,
                enforce_prediction_availability=True,
            ),
        ).fit(pseul_input, y_train)
        selections[AUTO_METHOD] = list(auto.selected_features_)
        auto_scores = auto.summary().set_index("feature")
        primary_scores = scores.set_index("feature")
        signal_columns = ["raw_predictive_drop", "predictive_utility", "raw_shap_mean_abs", "shap_stability"]
        signal_delta = (primary_scores[signal_columns] - auto_scores[signal_columns]).abs().to_numpy()
        proxy_details["PSEUL-Auto control audit"] = {
            "same_seed_and_inner_folds": True,
            "retained_metadata": ["evidence", "topic_similarity", "utility", "availability"],
            "nulled_metadata": ["label_derived_risk", "definitional_overlap", "target_proxy_strength"],
            "max_abs_signal_difference": float(np.nanmax(signal_delta)) if signal_delta.size else 0.0,
        }

    present_traps = [feature for feature in traps if feature in X_train.columns]
    manual = manual_exclusion_shap_select(
        make_model(seed),
        X_train,
        y_train,
        excluded_features=present_traps,
        top_k=top_k,
        random_state=seed,
    )
    selections["Oracle Trap Exclusion + SHAP"] = manual["selected_features"]

    if include_proxies:
        knockoff = gaussian_knockoff_select(X_train, y_train, fdr=0.20, random_state=seed, plus=True)
        proxy_details["Gaussian Knockoff Proxy"] = knockoff
        selections["Gaussian Knockoff Proxy"] = knockoff["selected_features"]

        regression = shap_select(
            make_model(seed),
            selector_train,
            y_selector_train,
            selector_valid,
            y_selector_valid,
            alpha=0.05,
            random_state=seed,
        )
        proxy_details["SHAP Regression Proxy"] = regression
        selections["SHAP Regression Proxy"] = regression["selected_features"]

        grasp = grasp_select(make_model(seed), X_train, y_train, top_k=top_k, lam=0.05, random_state=seed)
        proxy_details["GRASP-inspired Proxy"] = grasp
        selections["GRASP-inspired Proxy"] = grasp["selected_features"]

    selections = {name: list(dict.fromkeys(cols)) for name, cols in selections.items()}
    return selections, scores, proxy_details


def metric_row(y_true: pd.Series, probability: np.ndarray) -> dict[str, float]:
    prediction = (np.asarray(probability) >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    return {
        "auc": float(roc_auc_score(y_true, probability)),
        "average_precision": float(average_precision_score(y_true, probability)),
        "mcc": float(matthews_corrcoef(y_true, prediction)),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, prediction)),
        "f1_weighted": float(f1_score(y_true, prediction, average="weighted")),
        "brier": float(brier_score_loss(y_true, probability)),
        "sensitivity": float(tp / max(1, tp + fn)),
        "specificity": float(tn / max(1, tn + fp)),
        "ppv": float(tp / max(1, tp + fp)),
        "npv": float(tn / max(1, tn + fn)),
    }


def calibration_parameters(y_true: pd.Series, probability: np.ndarray) -> tuple[float, float]:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    logits = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
    from sklearn.linear_model import LogisticRegression

    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    calibrator.fit(logits, np.asarray(y_true, dtype=int))
    return float(calibrator.intercept_[0]), float(calibrator.coef_[0, 0])


def signed_rank_biserial(differences: np.ndarray) -> float:
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values) & (values != 0)]
    if len(values) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(values))
    positive = float(ranks[values > 0].sum())
    negative = float(ranks[values < 0].sum())
    return (positive - negative) / (positive + negative)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm family-wise adjustment while preserving the original order."""
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(len(values), np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    if len(finite) == 0:
        return adjusted
    order = finite[np.argsort(values[finite])]
    running = 0.0
    m = len(order)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted


def _midranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start
        while end + 1 < len(values) and sorted_values[end + 1] == sorted_values[start]:
            end += 1
        ranks[start : end + 1] = 0.5 * (start + end) + 1.0
        start = end + 1
    result = np.empty(len(values), dtype=float)
    result[order] = ranks
    return result


def _delong_auc_covariance(y_true: np.ndarray, predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """AUC estimates and DeLong covariance for one or more paired prediction vectors."""
    truth = np.asarray(y_true, dtype=int)
    predictions = np.atleast_2d(np.asarray(predictions, dtype=float))
    order = np.argsort(-truth, kind="stable")
    truth = truth[order]
    predictions = predictions[:, order]
    positives = int(truth.sum())
    negatives = int(len(truth) - positives)
    if positives == 0 or negatives == 0:
        raise ValueError("DeLong inference requires both outcome classes.")
    positive_scores = predictions[:, :positives]
    negative_scores = predictions[:, positives:]
    k = predictions.shape[0]
    tx = np.empty((k, positives), dtype=float)
    ty = np.empty((k, negatives), dtype=float)
    tz = np.empty((k, positives + negatives), dtype=float)
    for row in range(k):
        tx[row] = _midranks(positive_scores[row])
        ty[row] = _midranks(negative_scores[row])
        tz[row] = _midranks(predictions[row])
    aucs = tz[:, :positives].sum(axis=1) / (positives * negatives) - (positives + 1.0) / (2.0 * negatives)
    v01 = (tz[:, :positives] - tx) / negatives
    v10 = 1.0 - (tz[:, positives:] - ty) / positives
    covariance = np.atleast_2d(np.cov(v01, bias=False)) / positives + np.atleast_2d(
        np.cov(v10, bias=False)
    ) / negatives
    return aucs, covariance


def _mcc_from_counts(tp: np.ndarray, tn: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    numerator = tp * tn - fp * fn
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator > 0)


def _paired_mcc_bootstrap(
    y_true: np.ndarray,
    reference_probability: np.ndarray,
    comparator_probability: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    truth = np.asarray(y_true, dtype=int)
    reference = np.asarray(reference_probability) >= 0.5
    comparator = np.asarray(comparator_probability) >= 0.5
    draws: dict[int, np.ndarray] = {}
    for outcome in (0, 1):
        mask = truth == outcome
        patterns = reference[mask].astype(int) * 2 + comparator[mask].astype(int)
        counts = np.bincount(patterns, minlength=4)
        draws[outcome] = rng.multinomial(int(mask.sum()), counts / counts.sum(), size=n_bootstrap)
    negative, positive = draws[0], draws[1]
    reference_tp = positive[:, 2] + positive[:, 3]
    comparator_tp = positive[:, 1] + positive[:, 3]
    reference_fp = negative[:, 2] + negative[:, 3]
    comparator_fp = negative[:, 1] + negative[:, 3]
    n_positive = int((truth == 1).sum())
    n_negative = int((truth == 0).sum())
    reference_mcc = _mcc_from_counts(
        reference_tp, n_negative - reference_fp, reference_fp, n_positive - reference_tp
    )
    comparator_mcc = _mcc_from_counts(
        comparator_tp, n_negative - comparator_fp, comparator_fp, n_positive - comparator_tp
    )
    return reference_mcc - comparator_mcc


def paired_bootstrap(
    predictions: pd.DataFrame,
    reference: str = PRIMARY_METHOD,
    n_bootstrap: int = 2000,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    wide_prob = predictions.pivot(index="row_id", columns="method", values="probability")
    truth = predictions.drop_duplicates("row_id").set_index("row_id")["y_true"].reindex(wide_prob.index)
    if reference not in wide_prob:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    negative = np.flatnonzero(truth.to_numpy() == 0)
    positive = np.flatnonzero(truth.to_numpy() == 1)
    rows = []
    for method in wide_prob.columns:
        if method == reference:
            continue
        complete = wide_prob[[reference, method]].notna().all(axis=1).to_numpy()
        valid_neg = negative[complete[negative]]
        valid_pos = positive[complete[positive]]
        if len(valid_neg) == 0 or len(valid_pos) == 0:
            continue
        valid = complete & truth.notna().to_numpy()
        y_values = truth.to_numpy(dtype=int)[valid]
        ref_values = wide_prob[reference].to_numpy(dtype=float)[valid]
        other_values = wide_prob[method].to_numpy(dtype=float)[valid]

        aucs, covariance = _delong_auc_covariance(y_values, np.vstack([ref_values, other_values]))
        contrast = np.array([1.0, -1.0])
        auc_delta = float(aucs[0] - aucs[1])
        auc_se = float(np.sqrt(max(0.0, contrast @ covariance @ contrast)))
        if auc_se > 0:
            auc_z = auc_delta / auc_se
            auc_p = float(2.0 * stats.norm.sf(abs(auc_z)))
            auc_probability = float(stats.norm.cdf(auc_z))
        else:
            auc_p = 1.0 if auc_delta == 0 else 0.0
            auc_probability = 0.5 if auc_delta == 0 else float(auc_delta > 0)
        rows.append(
            {
                "reference": reference,
                "comparator": method,
                "metric": "auc",
                "delta_reference_minus_comparator": auc_delta,
                "ci_low": auc_delta - 1.96 * auc_se,
                "ci_high": auc_delta + 1.96 * auc_se,
                "probability_delta_gt_zero": auc_probability,
                "two_sided_bootstrap_p": auc_p,
                "bootstrap_replicates": 0,
                "inference_method": "paired DeLong",
            }
        )

        arr = _paired_mcc_bootstrap(y_values, ref_values, other_values, n_bootstrap, rng)
        lower_tail = (np.sum(arr <= 0) + 1) / (len(arr) + 1)
        upper_tail = (np.sum(arr >= 0) + 1) / (len(arr) + 1)
        rows.append(
            {
                "reference": reference,
                "comparator": method,
                "metric": "mcc",
                "delta_reference_minus_comparator": float(np.mean(arr)),
                "ci_low": float(np.quantile(arr, 0.025)),
                "ci_high": float(np.quantile(arr, 0.975)),
                "probability_delta_gt_zero": float(np.mean(arr > 0)),
                "two_sided_bootstrap_p": float(min(1.0, 2.0 * min(lower_tail, upper_tail))),
                "bootstrap_replicates": int(n_bootstrap),
                "inference_method": "paired stratified multinomial bootstrap",
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["holm_adjusted_p"] = np.nan
        for metric, positions in result.groupby("metric").groups.items():
            indices = np.asarray(list(positions), dtype=int)
            result.loc[indices, "holm_adjusted_p"] = holm_adjust(
                result.loc[indices, "two_sided_bootstrap_p"].to_numpy()
            )
    return result


def oof_metric_intervals(
    predictions: pd.DataFrame,
    n_bootstrap: int = 2000,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Stratified patient-level bootstrap intervals for pooled out-of-fold metrics."""
    rng = np.random.default_rng(seed + 17)
    rows = []
    for method, frame in predictions.groupby("method"):
        frame = frame.sort_values("row_id").reset_index(drop=True)
        truth = frame["y_true"].to_numpy(dtype=int)
        probability = frame["probability"].to_numpy(dtype=float)
        aucs, covariance = _delong_auc_covariance(truth, probability)
        auc_se = float(np.sqrt(max(0.0, covariance[0, 0])))
        rows.append(
            {
                "method": method,
                "metric": "auc",
                "estimate": float(aucs[0]),
                "ci_low": float(max(0.0, aucs[0] - 1.96 * auc_se)),
                "ci_high": float(min(1.0, aucs[0] + 1.96 * auc_se)),
                "bootstrap_replicates": 0,
                "inference_method": "DeLong",
            }
        )
        predicted = probability >= 0.5
        positive = truth == 1
        negative = ~positive
        tp = int(np.sum(predicted & positive))
        fp = int(np.sum(predicted & negative))
        n_positive = int(positive.sum())
        n_negative = int(negative.sum())
        tp_draw = rng.binomial(n_positive, tp / n_positive, size=n_bootstrap)
        fp_draw = rng.binomial(n_negative, fp / n_negative, size=n_bootstrap)
        mcc_draw = _mcc_from_counts(tp_draw, n_negative - fp_draw, fp_draw, n_positive - tp_draw)
        point_mcc = float(matthews_corrcoef(truth, predicted))
        rows.append(
            {
                "method": method,
                "metric": "mcc",
                "estimate": point_mcc,
                "ci_low": float(np.quantile(mcc_draw, 0.025)),
                "ci_high": float(np.quantile(mcc_draw, 0.975)),
                "bootstrap_replicates": int(n_bootstrap),
                "inference_method": "stratified binomial bootstrap",
            }
        )
    return pd.DataFrame(rows)


def selection_stability(selections: pd.DataFrame, traps: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise Jaccard stability and feature/trap selection frequency across outer folds."""
    stability_rows = []
    frequency_rows = []
    for method, frame in selections.groupby("method"):
        fold_sets = [set(json.loads(value)) for value in frame.sort_values("fold")["features"]]
        similarities = []
        for left in range(len(fold_sets)):
            for right in range(left + 1, len(fold_sets)):
                union = fold_sets[left] | fold_sets[right]
                similarities.append(len(fold_sets[left] & fold_sets[right]) / len(union) if union else 1.0)
        stability_rows.append(
            {
                "method": method,
                "mean_pairwise_jaccard": float(np.mean(similarities)) if similarities else np.nan,
                "min_pairwise_jaccard": float(np.min(similarities)) if similarities else np.nan,
                "n_outer_folds": int(len(fold_sets)),
            }
        )
        all_features = sorted(set().union(*fold_sets)) if fold_sets else []
        for feature in all_features:
            frequency_rows.append(
                {
                    "method": method,
                    "feature": feature,
                    "selection_frequency": float(np.mean([feature in selected for selected in fold_sets])),
                    "canonical_trap": bool(feature in traps),
                }
            )
    return pd.DataFrame(stability_rows), pd.DataFrame(frequency_rows)


def _selection_summary(
    features: list[str], score_table: pd.DataFrame, traps: tuple[str, ...] = ()
) -> dict[str, float | int]:
    if not features:
        return {
            "n_features": 0,
            "avg_leakage_risk": np.nan,
            "avg_clinical_utility": np.nan,
            "designated_traps_selected": 0,
            "designated_trap_fraction": 0.0 if traps else np.nan,
            "prediction_unavailable_selected": 0,
        }
    indexed = score_table.set_index("feature")
    available = [feature for feature in features if feature in indexed.index]
    if not available:
        return {
            "n_features": len(features),
            "avg_leakage_risk": np.nan,
            "avg_clinical_utility": np.nan,
            "designated_traps_selected": int(len(set(features) & set(traps))),
            "designated_trap_fraction": float(len(set(features) & set(traps)) / len(traps)) if traps else np.nan,
            "prediction_unavailable_selected": np.nan,
        }
    selected = indexed.loc[available]
    trap_count = len(set(features) & set(traps))
    return {
        "n_features": len(features),
        "avg_leakage_risk": float(selected["leakage_risk"].mean()),
        "avg_clinical_utility": float(selected["clinical_utility"].mean()),
        "designated_traps_selected": int(trap_count),
        "designated_trap_fraction": float(trap_count / len(traps)) if traps else np.nan,
        "prediction_unavailable_selected": int((~selected["available_at_prediction"].astype(bool)).sum()),
    }


def _evaluate_fold(
    study: str,
    fold: int,
    X_train_raw: pd.DataFrame,
    X_valid_raw: pd.DataFrame,
    y_train: pd.Series,
    y_valid: pd.Series,
    profile: dict,
    top_k: int,
    traps: tuple[str, ...],
    inner_splits: int,
    include_proxies: bool,
    include_auto: bool,
) -> tuple[list[dict], list[dict], list[dict], pd.DataFrame, dict]:
    imputer = FrameMedianImputer().fit(X_train_raw)
    X_train = imputer.transform(X_train_raw).reset_index(drop=True)
    X_valid = imputer.transform(X_valid_raw).reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_valid = y_valid.reset_index(drop=True)
    seed = RANDOM_STATE + fold * 101
    selections, scores, proxy_details = build_selections(
        X_train,
        y_train,
        profile,
        top_k,
        traps,
        seed,
        inner_splits,
        include_proxies,
        include_auto,
        X_pseul_raw=X_train_raw,
    )

    metrics = []
    predictions = []
    selection_rows = []
    for method, features in selections.items():
        summary = _selection_summary(features, scores, traps)
        selection_rows.append(
            {
                "study": study,
                "fold": fold,
                "method": method,
                "features": json.dumps(features),
                **summary,
            }
        )
        if not features:
            continue
        model = make_model(seed + 1)
        model.fit(X_train[features], y_train)
        probability = model.predict_proba(X_valid[features])[:, 1]
        row = {"study": study, "fold": fold, "method": method, **metric_row(y_valid, probability), **summary}
        intercept, slope = calibration_parameters(y_valid, probability)
        row.update(calibration_intercept=intercept, calibration_slope=slope)
        metrics.append(row)
        for local_id, (row_id, target, prob) in enumerate(zip(X_valid_raw.index, y_valid, probability)):
            predictions.append(
                {
                    "study": study,
                    "fold": fold,
                    "method": method,
                    "row_id": int(row_id),
                    "fold_row": local_id,
                    "y_true": int(target),
                    "probability": float(prob),
                    "prediction_0_5": int(prob >= 0.5),
                }
            )
    score_copy = scores.copy()
    score_copy.insert(0, "fold", fold)
    score_copy.insert(0, "study", study)
    audit = {
        "fold": fold,
        "imputer_fit_rows": imputer.fit_rows_,
        "imputer_fit_index_hash": imputer.fit_index_hash_,
        "outer_train_index_hash": _index_hash(X_train_raw.index.to_numpy()),
        "outer_valid_index_hash": _index_hash(X_valid_raw.index.to_numpy()),
        "train_validation_overlap": int(len(set(X_train_raw.index) & set(X_valid_raw.index))),
        "selector_scope": "outer-training only",
        "proxy_details": proxy_details,
    }
    return metrics, predictions, selection_rows, score_copy, audit


def run_study(
    study: str,
    outer_splits: int,
    inner_splits: int,
    test_size: float,
    evaluate_test: bool,
    include_proxies: bool,
    include_auto: bool,
    n_bootstrap: int,
    max_rows: int | None,
    run_label: str = "full",
) -> Path:
    started = time.time()
    spec = STUDY_SPECS[study]
    X, y, metadata, profile = load_study(study)
    X.index = np.arange(len(X))
    y.index = X.index
    if max_rows and len(X) > max_rows:
        selected, _ = train_test_split(
            np.arange(len(X)), train_size=max_rows, stratify=y, random_state=RANDOM_STATE
        )
        X = X.iloc[selected].copy()
        y = y.iloc[selected].copy()
        metadata["analysis_subsample"] = int(max_rows)
    development_ids, test_ids = train_test_split(
        X.index.to_numpy(), test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )
    X_development = X.loc[development_ids].copy()
    y_development = y.loc[development_ids].copy()
    X_test = X.loc[test_ids].copy()
    y_test = y.loc[test_ids].copy()

    if not run_label.replace("-", "").replace("_", "").isalnum():
        raise ValueError("run_label must contain only letters, numbers, hyphens, or underscores")
    output_dir = Path(__file__).resolve().parent / "outputs" / study / run_label
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "locked_split_indices.npz", development=development_ids, test=test_ids)

    folds = StratifiedKFold(n_splits=outer_splits, shuffle=True, random_state=RANDOM_STATE)
    metric_rows: list[dict] = []
    prediction_rows: list[dict] = []
    selection_rows: list[dict] = []
    score_tables: list[pd.DataFrame] = []
    audits: list[dict] = []
    for fold, (train_position, valid_position) in enumerate(folds.split(X_development, y_development), start=1):
        train_ids = X_development.index.to_numpy()[train_position]
        valid_ids = X_development.index.to_numpy()[valid_position]
        print(f"[{study}] outer fold {fold}/{outer_splits}: train={len(train_ids):,}, valid={len(valid_ids):,}", flush=True)
        fold_result = _evaluate_fold(
            study,
            fold,
            X_development.loc[train_ids],
            X_development.loc[valid_ids],
            y_development.loc[train_ids],
            y_development.loc[valid_ids],
            profile,
            spec.top_k,
            spec.canonical_traps,
            inner_splits,
            include_proxies,
            include_auto,
        )
        fold_metrics, fold_predictions, fold_selections, fold_scores, fold_audit = fold_result
        metric_rows.extend(fold_metrics)
        prediction_rows.extend(fold_predictions)
        selection_rows.extend(fold_selections)
        score_tables.append(fold_scores)
        audits.append(fold_audit)

    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    selections = pd.DataFrame(selection_rows)
    metrics.to_csv(output_dir / "nested_fold_metrics.csv", index=False)
    predictions.to_csv(output_dir / "oof_predictions.csv", index=False)
    selections.to_csv(output_dir / "selected_features_by_fold.csv", index=False)
    pd.concat(score_tables, ignore_index=True).to_csv(output_dir / "pseul_scores_by_fold.csv", index=False)

    paired = paired_bootstrap(predictions, n_bootstrap=n_bootstrap, seed=RANDOM_STATE)
    paired.to_csv(output_dir / "paired_oof_bootstrap.csv", index=False)
    oof_metric_intervals(predictions, n_bootstrap=n_bootstrap, seed=RANDOM_STATE).to_csv(
        output_dir / "oof_metric_intervals.csv", index=False
    )
    stability, frequency = selection_stability(selections, spec.canonical_traps)
    stability.to_csv(output_dir / "selection_stability.csv", index=False)
    frequency.to_csv(output_dir / "feature_selection_frequency.csv", index=False)

    fold_effects = []
    for metric in ("auc", "mcc"):
        wide = metrics.pivot(index="fold", columns="method", values=metric)
        if PRIMARY_METHOD not in wide:
            continue
        for method in wide.columns:
            if method == PRIMARY_METHOD:
                continue
            diff = (wide[PRIMARY_METHOD] - wide[method]).dropna().to_numpy()
            fold_effects.append(
                {
                    "metric": metric,
                    "reference": PRIMARY_METHOD,
                    "comparator": method,
                    "mean_fold_delta": float(np.mean(diff)) if len(diff) else np.nan,
                    "signed_rank_biserial": signed_rank_biserial(diff),
                    "n_outer_folds": int(len(diff)),
                    "inference_status": "descriptive; folds are not independent replications",
                }
            )
    pd.DataFrame(fold_effects).to_csv(output_dir / "descriptive_fold_effects.csv", index=False)

    final_test_rows = []
    final_selection_rows = []
    if evaluate_test:
        print(f"[{study}] locked internal test evaluation", flush=True)
        imputer = FrameMedianImputer().fit(X_development)
        X_dev_imputed = imputer.transform(X_development).reset_index(drop=True)
        X_test_imputed = imputer.transform(X_test).reset_index(drop=True)
        y_dev_reset = y_development.reset_index(drop=True)
        y_test_reset = y_test.reset_index(drop=True)
        final_selections, final_scores, proxy_details = build_selections(
            X_dev_imputed,
            y_dev_reset,
            profile,
            spec.top_k,
            spec.canonical_traps,
            RANDOM_STATE + 999,
            inner_splits,
            include_proxies,
            include_auto,
            X_pseul_raw=X_development,
        )
        final_scores.to_csv(output_dir / "final_pseul_scores.csv", index=False)
        (output_dir / "final_proxy_details.json").write_text(json.dumps(proxy_details, indent=2), encoding="utf-8")
        for method, features in final_selections.items():
            summary = _selection_summary(features, final_scores, spec.canonical_traps)
            final_selection_rows.append({"method": method, "features": json.dumps(features), **summary})
            if not features:
                continue
            model = make_model(RANDOM_STATE + 1000)
            model.fit(X_dev_imputed[features], y_dev_reset)
            probability = model.predict_proba(X_test_imputed[features])[:, 1]
            row = {"study": study, "method": method, **metric_row(y_test_reset, probability), **summary}
            intercept, slope = calibration_parameters(y_test_reset, probability)
            row.update(calibration_intercept=intercept, calibration_slope=slope)
            final_test_rows.append(row)
        pd.DataFrame(final_test_rows).to_csv(output_dir / "locked_test_metrics.csv", index=False)
        pd.DataFrame(final_selection_rows).to_csv(output_dir / "final_selected_features.csv", index=False)

    audit_payload = {
        "study": study,
        "run_label": run_label,
        "spec": asdict(spec),
        "metadata": metadata,
        "outer_splits": outer_splits,
        "inner_splits": inner_splits,
        "test_size": test_size,
        "test_evaluated": evaluate_test,
        "include_proxy_baselines": include_proxies,
        "include_auto_ablation": include_auto,
        "random_state": RANDOM_STATE,
        "development_rows": int(len(X_development)),
        "test_rows": int(len(X_test)),
        "development_test_overlap": int(len(set(development_ids) & set(test_ids))),
        "fold_audits": audits,
        "elapsed_seconds": time.time() - started,
    }
    (output_dir / "pipeline_audit.json").write_text(json.dumps(audit_payload, indent=2), encoding="utf-8")
    print(f"[{study}] complete -> {output_dir}", flush=True)
    return output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PSEUL revision-v2 nested evaluation.")
    parser.add_argument("study", choices=sorted(STUDY_SPECS))
    parser.add_argument("--outer-splits", type=int, default=10)
    parser.add_argument("--inner-splits", type=int, default=5)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--include-proxies", action="store_true")
    parser.add_argument("--include-auto", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--run-label", default="full")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_registry_artifacts(Path(__file__).resolve().parent / "outputs")
    run_study(
        args.study,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        test_size=args.test_size,
        evaluate_test=args.evaluate_test,
        include_proxies=args.include_proxies,
        include_auto=args.include_auto,
        n_bootstrap=args.bootstrap,
        max_rows=args.max_rows,
        run_label=args.run_label,
    )


if __name__ == "__main__":
    main()
