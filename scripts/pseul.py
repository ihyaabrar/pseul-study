from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Literal

import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


@dataclass(frozen=True)
class PseulFeatureProfile:
    """Clinical metadata for one feature.

    All scores are expected in [0, 1].
    """

    evidence: float = 0.50
    intervention_availability: float = 0.50
    risk_stratification: float = 0.50
    operational_feasibility: float = 0.50
    label_derived_risk: float = 0.00
    definitional_overlap: float = 0.00
    target_proxy_strength: float = 0.00
    topic_similarity: float = 0.50
    interpretation: str = ""
    available_at_prediction: bool = True


@dataclass(frozen=True)
class PseulConfig:
    """Locked v10 defaults from the PSEUL draft."""

    n_splits: int = 5
    random_state: int = 42
    alpha: float = 0.30
    beta: float = 0.20
    gamma: float = 0.25
    delta: float = 0.25
    theta_u: float = 0.10
    leakage_z: float = 1.00
    theta_l_min: float = 0.50
    theta_l_max: float = 0.90
    tau_l: float = 10.00
    tau_m_ratio: float = 0.01
    redundancy_weight: float = 0.10
    uncertainty_weight: float = 0.00
    eta: float = 0.50
    epsilon: float = 1e-6
    max_shap_samples_per_fold: int | None = 500
    permutation_repeats: int = 5
    top_k: int | None = None
    semantic_veto_threshold: float | None = 0.85
    min_select_score: float | None = 0.20
    use_soft_leakage_penalty: bool = True
    utility_method: Literal["permutation", "lofo"] = "permutation"
    leakage_estimator: Literal["mean_std", "median_mad"] = "mean_std"
    target_association: Literal["partial_corr", "partial_corr_and_mi"] = "partial_corr"
    mi_random_state: int = 42
    enforce_prediction_availability: bool = True


def _clip01(value: float | np.ndarray) -> float | np.ndarray:
    return np.clip(value, 0.0, 1.0)


def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _minmax_positive(values: pd.Series, epsilon: float) -> pd.Series:
    values = values.clip(lower=0)
    lo = float(values.min())
    hi = float(values.max())
    if hi - lo <= epsilon:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return (values - lo) / (hi - lo + epsilon)


def _safe_corr(a: np.ndarray, b: np.ndarray, epsilon: float = 1e-12) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if np.nanstd(a) <= epsilon or np.nanstd(b) <= epsilon:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _partial_corr_abs(x: pd.Series, y: pd.Series, z: pd.DataFrame | None) -> float:
    """Absolute partial correlation |rho(x, y | Z)|.

    If Z is empty, this falls back to absolute Pearson correlation.
    """

    x_arr = pd.Series(x).astype(float).to_numpy()
    y_arr = pd.Series(y).astype(float).to_numpy()
    if z is None or z.empty:
        return abs(_safe_corr(x_arr, y_arr))

    z_arr = z.astype(float).to_numpy()
    x_res = x_arr - LinearRegression().fit(z_arr, x_arr).predict(z_arr)
    y_res = y_arr - LinearRegression().fit(z_arr, y_arr).predict(z_arr)
    return abs(_safe_corr(x_res, y_res))


def _mutual_info_leakage_signal(
    X_train: pd.DataFrame, y_train: pd.Series, feature_names: list[str], random_state: int
) -> dict[str, float]:
    """Marginal mutual information I(f_j ; y), a nonlinear complement to
    partial correlation. NOT adjusted for covariates Z (see PseulConfig
    docstring / target_association option for the documented trade-off).
    """
    X_arr = X_train[feature_names].astype(float).to_numpy()
    y_arr = y_train.astype(float).to_numpy()
    mi_values = mutual_info_regression(X_arr, y_arr, random_state=random_state)
    return dict(zip(feature_names, mi_values.tolist()))


def _category(row: pd.Series) -> str:
    if row["leakage_risk"] >= 0.80:
        return "Label-Proximal Feature"
    if row["predictive_utility"] >= 0.50 and row["shap_stability"] >= 0.50 and row["leakage_risk"] >= 0.60:
        return "Predictive Proxy Feature"
    if row["evidence_relevance"] >= 0.70 and row["clinical_utility"] >= 0.70 and row["leakage_risk"] <= 0.30:
        return "Clinically Actionable Feature"
    if row["evidence_relevance"] >= 0.50 and row["shap_stability"] >= 0.50 and row["leakage_risk"] <= 0.50:
        return "Domain-Supportive Feature"
    if row["evidence_relevance"] >= 0.70 and row["clinical_utility"] >= 0.60 and row["predictive_utility"] < 0.20:
        return "Clinically Relevant but Weak Predictor"
    if row["shap_stability"] < 0.40:
        return "Unstable Predictor"
    return "Low-Value Feature"


class PSEUL:
    """PSEUL feature selection.

    Usage:

    ```python
    pseul = PSEUL(estimator=base_lgbm, clinical_profile=profile, top_k=8)
    pseul.fit(X_train, y_train)
    X_selected = pseul.transform(X_train)
    selected = pseul.selected_features_
    scores = pseul.summary()
    ```

    IMPORTANT -- candidate pool protocol (see docs/research/
    PSEUL_secondary_findings_D_E_F_B.md, item B): the adaptive leakage
    threshold theta_L is calibrated from the leakage-score distribution of
    whatever features are passed in `X`. It is a RELATIVE threshold, not
    an absolute clinical safety bound. Pre-filtering out the most
    leakage-prone features from `X` before calling `fit()` (e.g. during
    manual data cleaning) will make theta_L recalibrate downward and can
    make the remaining features appear falsely "safe" relative to the
    smaller pool. Always pass the FULL set of available candidate
    features to `fit()`, and let PSEUL's own leakage_risk scoring and
    selection decide which ones to exclude.
    """

    def __init__(
        self,
        estimator,
        clinical_profile: dict[str, PseulFeatureProfile] | None = None,
        config: PseulConfig | None = None,
        top_k: int | None = None,
        adjustment_features: Iterable[str] | None = None,
        metric_fn: Callable[[pd.Series, np.ndarray], float] = roc_auc_score,
    ) -> None:
        self.estimator = estimator
        self.clinical_profile = clinical_profile or {}
        self.config = config or PseulConfig()
        if top_k is not None:
            self.config = PseulConfig(**{**self.config.__dict__, "top_k": top_k})
        self.adjustment_features = list(adjustment_features or [])
        self.metric_fn = metric_fn

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "PSEUL":
        X = self._as_frame(X)
        y = pd.Series(y).reset_index(drop=True)
        X = X.reset_index(drop=True)
        self.feature_names_ = list(X.columns)

        fold_data = self._compute_fold_data(X, y)
        base_scores = self._compute_base_scores(fold_data)
        # Persist the fold-level sufficient statistics so prespecified
        # one-factor rescoring analyses can reuse identical fitted signals
        # rather than confounding an ablation with a second stochastic fit.
        self.fold_data_ = fold_data
        self.base_scores_ = base_scores.copy()
        self.scores_ = self._build_score_table(base_scores, fold_data)
        self.selected_features_ = self._greedy_select(self.scores_, fold_data)
        self.scores_ = self._attach_selection_info(self.scores_, fold_data)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._check_is_fit()
        X = self._as_frame(X)
        return X.loc[:, self.selected_features_].copy()

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def select_features(
        self,
        mode: Literal["select", "audit"] = "select",
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[str]:
        self._check_is_fit()
        score_col = "audit_score" if mode == "audit" else "select_score"
        ranked = self.scores_.sort_values([score_col, "predictive_utility", "feature"], ascending=[False, False, True])
        if mode == "select":
            ranked = ranked[ranked["admissible_for_selection"]]
            if min_score is None:
                min_score = self.config.min_select_score
        if min_score is not None:
            ranked = ranked[ranked[score_col] >= min_score]
        if top_k is not None:
            ranked = ranked.head(top_k)
        return list(ranked["feature"])

    def summary(self) -> pd.DataFrame:
        self._check_is_fit()
        return self.scores_.copy()

    def _compute_fold_data(self, X: pd.DataFrame, y: pd.Series) -> dict:
        cfg = self.config
        folds = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=cfg.random_state)
        drops = {feature: [] for feature in self.feature_names_}
        shap_means = {feature: [] for feature in self.feature_names_}
        leakage_by_fold: list[pd.Series] = []
        redundancy_corrs: list[pd.DataFrame] = []

        for fold_id, (train_idx, valid_idx) in enumerate(folds.split(X, y), start=1):
            X_train_raw = X.iloc[train_idx]
            X_valid_raw = X.iloc[valid_idx]
            y_train = y.iloc[train_idx]
            y_valid = y.iloc[valid_idx]

            # Fit preprocessing inside every inner-training fold. This keeps
            # utility, SHAP, target-association, and redundancy estimates from
            # using inner-validation distributional information.
            medians = X_train_raw.median(numeric_only=True).reindex(self.feature_names_).fillna(0.0)
            X_train = X_train_raw.apply(pd.to_numeric, errors="coerce").fillna(medians).fillna(0.0)
            X_valid = X_valid_raw.apply(pd.to_numeric, errors="coerce").fillna(medians).fillna(0.0)

            model = clone(self.estimator)
            if hasattr(model, "set_params"):
                params = model.get_params()
                if "random_state" in params:
                    model.set_params(random_state=cfg.random_state + fold_id)
            model.fit(X_train, y_train)

            base_score = self.metric_fn(y_valid, model.predict_proba(X_valid)[:, 1])
            if cfg.utility_method == "lofo":
                self._lofo_drops(X_train, X_valid, y_train, y_valid, base_score, drops, fold_id)
            else:
                self._permutation_drops(model, X_valid, y_valid, base_score, drops, fold_id)

            shap_sample = X_valid
            if cfg.max_shap_samples_per_fold and len(X_valid) > cfg.max_shap_samples_per_fold:
                shap_sample = X_valid.sample(cfg.max_shap_samples_per_fold, random_state=cfg.random_state + fold_id)
            explainer = shap.TreeExplainer(model)
            values = explainer.shap_values(shap_sample)
            # SHAP output shape is version/estimator dependent for binary
            # classifiers: older APIs return a list [class0, class1]; newer
            # ones may return a 3-D array (n, features, n_classes). Handle both
            # so mean(|SHAP|) is always computed on the positive-class matrix.
            if isinstance(values, list):
                values = values[1]
            elif getattr(values, "ndim", 2) == 3:
                values = values[:, :, -1]
            for feature, value in zip(self.feature_names_, np.abs(values).mean(axis=0)):
                shap_means[feature].append(float(value))

            leakage_by_fold.append(self._fold_leakage_scores(X_train, y_train))
            redundancy_corrs.append(X_train.corr(numeric_only=True).fillna(0.0).abs())

        return {
            "drops": drops,
            "shap_means": shap_means,
            "leakage_by_fold": leakage_by_fold,
            "redundancy_corrs": redundancy_corrs,
        }

    def _permutation_drops(
        self,
        model,
        X_valid: pd.DataFrame,
        y_valid: pd.Series,
        base_score: float,
        drops: dict[str, list[float]],
        fold_id: int,
    ) -> None:
        rng = np.random.default_rng(self.config.random_state + fold_id)
        for feature in self.feature_names_:
            repeated_drops = []
            for _ in range(max(1, self.config.permutation_repeats)):
                X_perm = X_valid.copy()
                X_perm[feature] = rng.permutation(X_perm[feature].to_numpy())
                score = self.metric_fn(y_valid, model.predict_proba(X_perm)[:, 1])
                repeated_drops.append(max(0.0, base_score - score))
            drops[feature].append(float(np.mean(repeated_drops)))

    def _lofo_drops(
        self,
        X_train: pd.DataFrame,
        X_valid: pd.DataFrame,
        y_train: pd.Series,
        y_valid: pd.Series,
        base_score: float,
        drops: dict[str, list[float]],
        fold_id: int,
    ) -> None:
        for feature in self.feature_names_:
            model = clone(self.estimator)
            if hasattr(model, "set_params") and "random_state" in model.get_params():
                model.set_params(random_state=self.config.random_state + fold_id)
            keep = [f for f in self.feature_names_ if f != feature]
            model.fit(X_train[keep], y_train)
            score = self.metric_fn(y_valid, model.predict_proba(X_valid[keep])[:, 1])
            drops[feature].append(max(0.0, base_score - score))

    def _fold_leakage_scores(self, X_train: pd.DataFrame, y_train: pd.Series) -> pd.Series:
        cfg = self.config
        ta_raw = {}
        for feature in self.feature_names_:
            z_cols = [c for c in self.adjustment_features if c in X_train.columns and c != feature]
            z = X_train[z_cols] if z_cols else None
            ta_raw[feature] = _partial_corr_abs(X_train[feature], y_train, z)
        ta = _minmax_positive(pd.Series(ta_raw), cfg.epsilon)

        if cfg.target_association == "partial_corr_and_mi":
            # Partial correlation only captures LINEAR association after
            # removing Z. A feature with a strong but nonlinear relationship
            # to the label (e.g. a threshold or U-shaped effect) can pass
            # through partial correlation with a low score while still
            # being a real leakage risk. Mutual information (MI) captures
            # nonlinear dependence, at the cost of NOT adjusting for Z
            # (conditional MI estimation is substantially harder and is not
            # implemented here) -- this is a deliberate, documented
            # trade-off, not an oversight. We combine both by taking the
            # elementwise maximum after independent normalization, so a
            # feature is only judged "low target-association" if BOTH the
            # linear (partialled) and nonlinear (marginal) signals are low.
            mi_raw = _mutual_info_leakage_signal(X_train, y_train, self.feature_names_, cfg.mi_random_state)
            mi = _minmax_positive(pd.Series(mi_raw), cfg.epsilon)
            ta = pd.Series(
                {f: max(float(ta[f]), float(mi[f])) for f in self.feature_names_}
            )

        leakage = {}
        for feature in self.feature_names_:
            profile = self._profile(feature)
            leakage[feature] = np.mean(
                [
                    profile.label_derived_risk,
                    profile.definitional_overlap,
                    float(ta[feature]),
                    profile.target_proxy_strength,
                ]
            )
        return pd.Series(leakage).clip(0.0, 1.0)

    def _compute_base_scores(self, fold_data: dict) -> pd.DataFrame:
        cfg = self.config
        raw_drop = pd.Series({f: float(np.mean(v)) for f, v in fold_data["drops"].items()})
        predictive = _minmax_positive(raw_drop, cfg.epsilon)

        shap_mean = pd.Series({f: float(np.mean(v)) for f, v in fold_data["shap_means"].items()})
        max_mean = float(shap_mean.max()) if len(shap_mean) else 0.0
        tau_m = cfg.tau_m_ratio * max_mean
        shap_stability = {}
        shap_cv = {}
        for feature, values in fold_data["shap_means"].items():
            arr = np.asarray(values, dtype=float)
            mean = float(arr.mean())
            cv = float(arr.std(ddof=0) / (mean + cfg.epsilon))
            shap_cv[feature] = cv
            shap_stability[feature] = float(_clip01(1.0 - cv)) if mean >= tau_m else 0.5

        rows = []
        for feature in self.feature_names_:
            profile = self._profile(feature)
            utility = np.mean(
                [
                    profile.intervention_availability,
                    profile.risk_stratification,
                    profile.operational_feasibility,
                ]
            )
            evidence = cfg.eta * profile.evidence + (1.0 - cfg.eta) * profile.topic_similarity
            rows.append(
                {
                    "feature": feature,
                    "raw_predictive_drop": raw_drop[feature],
                    "predictive_utility": predictive[feature],
                    "raw_shap_mean_abs": shap_mean[feature],
                    "raw_shap_cv": shap_cv[feature],
                    "shap_stability": shap_stability[feature],
                    "evidence_relevance": float(_clip01(evidence)),
                    "clinical_utility": float(_clip01(utility)),
                    "label_derived_risk": float(_clip01(profile.label_derived_risk)),
                    "definitional_overlap": float(_clip01(profile.definitional_overlap)),
                    "target_proxy_strength": float(_clip01(profile.target_proxy_strength)),
                    "semantic_risk_max": float(
                        max(
                            profile.label_derived_risk,
                            profile.definitional_overlap,
                            profile.target_proxy_strength,
                        )
                    ),
                    "available_at_prediction": bool(profile.available_at_prediction),
                    "interpretation": profile.interpretation,
                }
            )
        return pd.DataFrame(rows).set_index("feature")

    def _build_score_table(self, base: pd.DataFrame, fold_data: dict) -> pd.DataFrame:
        cfg = self.config
        leakage_matrix = pd.concat(fold_data["leakage_by_fold"], axis=1)
        leakage_matrix.columns = [f"fold_{i + 1}" for i in range(leakage_matrix.shape[1])]
        mean_leakage = leakage_matrix.mean(axis=1)

        theta_values = []
        pre_scores_by_fold = []
        for col in leakage_matrix.columns:
            leakage = leakage_matrix[col]
            if cfg.leakage_estimator == "median_mad":
                central = float(leakage.median())
                mad = float((leakage - central).abs().median())
                spread = 1.4826 * mad
            else:
                central = float(leakage.mean())
                spread = float(leakage.std(ddof=0))
            theta = float(_clip01(central + cfg.leakage_z * spread))
            theta = float(np.clip(theta, cfg.theta_l_min, cfg.theta_l_max))
            theta_values.append(theta)
            leakage_penalty = (
                1.0 - _sigmoid(cfg.tau_l * (leakage - theta))
                if cfg.use_soft_leakage_penalty
                else pd.Series(1.0, index=leakage.index)
            )
            gate_u = np.minimum(1.0, base["clinical_utility"] / max(cfg.theta_u, cfg.epsilon))
            linear = (
                cfg.alpha * base["predictive_utility"]
                + cfg.beta * base["shap_stability"]
                + cfg.gamma * base["evidence_relevance"]
                + cfg.delta * base["clinical_utility"]
            )
            availability_gate = base["available_at_prediction"].astype(float)
            pre_scores_by_fold.append((availability_gate * gate_u * linear * leakage_penalty).rename(col))

        pre_matrix = pd.concat(pre_scores_by_fold, axis=1)
        scores = base.copy()
        scores["leakage_risk"] = mean_leakage
        scores["theta_l_mean"] = float(np.mean(theta_values))
        scores["audit_score"] = (
            cfg.alpha * scores["predictive_utility"]
            + cfg.beta * scores["shap_stability"]
            + cfg.gamma * scores["evidence_relevance"]
            + cfg.delta * scores["clinical_utility"]
        )
        scores["pre_score"] = pre_matrix.mean(axis=1)
        scores["uncertainty_width"] = (
            1.96 * pre_matrix.std(axis=1, ddof=0) / np.sqrt(max(1, cfg.n_splits))
        ).fillna(0.0)
        scores["pseul_category"] = scores.apply(_category, axis=1)
        if cfg.semantic_veto_threshold is None:
            scores["semantic_veto"] = False
        else:
            scores["semantic_veto"] = scores["semantic_risk_max"] >= cfg.semantic_veto_threshold
        scores["admissible_for_selection"] = (
            scores["available_at_prediction"].astype(bool) & ~scores["semantic_veto"].astype(bool)
        )
        self._pre_score_matrix_ = pre_matrix
        return scores.reset_index()

    def _greedy_select(self, scores: pd.DataFrame, fold_data: dict) -> list[str]:
        cfg = self.config
        top_k = cfg.top_k or len(self.feature_names_)
        selected: list[str] = []
        remaining = set(scores.loc[scores["admissible_for_selection"], "feature"])

        while remaining and len(selected) < top_k:
            candidates = []
            for feature in remaining:
                score = self._select_score_for_feature(feature, selected, scores, fold_data)
                p_value = float(scores.loc[scores["feature"] == feature, "predictive_utility"].iloc[0])
                candidates.append((score, p_value, feature))
            candidates.sort(key=lambda x: (-x[0], -x[1], x[2]))
            if cfg.min_select_score is not None and candidates[0][0] < cfg.min_select_score:
                break
            selected_feature = candidates[0][2]
            selected.append(selected_feature)
            remaining.remove(selected_feature)
        return selected

    def _select_score_for_feature(
        self,
        feature: str,
        selected: list[str],
        scores: pd.DataFrame,
        fold_data: dict,
    ) -> float:
        cfg = self.config
        pre = self._pre_score_matrix_.loc[feature]
        if not selected:
            redundancy = pd.Series(np.zeros(len(pre)), index=pre.index)
        else:
            red_values = []
            for corr in fold_data["redundancy_corrs"]:
                available = [f for f in selected if f in corr.columns and feature in corr.index]
                red_values.append(float(corr.loc[feature, available].max()) if available else 0.0)
            redundancy = pd.Series(red_values, index=pre.index)
        uncertainty = float(scores.loc[scores["feature"] == feature, "uncertainty_width"].iloc[0])
        select = (pre - cfg.redundancy_weight * redundancy - cfg.uncertainty_weight * uncertainty).clip(0.0, 1.0)
        return float(select.mean())

    def _attach_selection_info(self, scores: pd.DataFrame, fold_data: dict) -> pd.DataFrame:
        final_scores = []
        for feature in scores["feature"]:
            prior = [f for f in self.selected_features_ if self.selected_features_.index(f) < self.selected_features_.index(feature)] if feature in self.selected_features_ else self.selected_features_
            final_scores.append(self._select_score_for_feature(feature, prior, scores, fold_data))
        scores = scores.copy()
        scores["select_score"] = final_scores
        order = {feature: i + 1 for i, feature in enumerate(self.selected_features_)}
        scores["selection_order"] = scores["feature"].map(order)
        scores["selected"] = scores["feature"].isin(self.selected_features_)
        reasons = []
        for _, row in scores.iterrows():
            if bool(row["selected"]):
                reasons.append("")
            elif not bool(row["available_at_prediction"]):
                reasons.append("post-landmark")
            elif bool(row["semantic_veto"]):
                reasons.append("semantic-veto")
            elif self.config.min_select_score is not None and float(row["select_score"]) < self.config.min_select_score:
                reasons.append("below-score-floor")
            else:
                reasons.append("not-in-final-subset")
        scores["rejection_reason"] = reasons
        return scores.sort_values(["selected", "selection_order", "select_score"], ascending=[False, True, False]).reset_index(drop=True)

    def _profile(self, feature: str) -> PseulFeatureProfile:
        return self.clinical_profile.get(feature, PseulFeatureProfile())

    @staticmethod
    def _as_frame(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return pd.DataFrame(X)

    def _check_is_fit(self) -> None:
        if not hasattr(self, "selected_features_"):
            raise RuntimeError("PSEUL is not fitted yet. Call fit(X, y) first.")


__all__ = ["PSEUL", "PseulConfig", "PseulFeatureProfile"]
