"""Lightweight reimplementations of two comparison baselines for PSEUL.

These are NOT the original published implementations. They reimplement the
core mechanism described in the respective papers, using only dependencies
already pinned in this workspace (numpy, scipy, scikit-learn) to avoid
numpy>=2.0 upgrades that would break reproducibility of existing PSEUL
results (see docs/research/PSEUL_related_work_positioning.md for the
rationale and the numpy 2.x conflict encountered with the official
`knockpy` package).

1) gaussian_knockoff_select
   Core mechanism of Model-X Gaussian knockoffs (Candes et al., 2018) as
   used by Knockoff-ML (Wang et al., 2025, npj Digital Medicine,
   10.1038/s41746-025-02102-2): construct second-order Gaussian knockoff
   features that preserve the covariance structure of X but are
   conditionally independent of y, run a feature-importance statistic on
   [X, X_knockoff], and select features whose importance clears a
   knockoff+ threshold controlling the false discovery rate (FDR).

   This is a simplified second-order (semidefinite-program-free) knockoff
   construction using an equicorrelated construction, which is weaker than
   the SDP-optimized knockoffs in `knockpy` but does not require cvxpy.

2) shap_select
   Core mechanism of shap-select (Kraev et al., 2024, arXiv:2410.06815) /
   GRASP-style SHAP-regression selection (Luo et al., 2026,
   10.1109/icassp55912.2026.11462660): compute SHAP values on a validation
   fold, regress the target on the SHAP values, and keep features whose
   regression coefficient is positive and statistically significant.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression


def _equicorrelated_knockoffs(X: np.ndarray, random_state: int = 42) -> np.ndarray:
    """Second-order Gaussian knockoffs via the equicorrelated construction.

    Reference mechanism: Candes, Fan, Janson, Lv (2018), "Panning for Gold:
    Model-X Knockoffs for High-dimensional Controlled Variable Selection",
    JRSS-B. Equicorrelated s_j = min(1, 2*lambda_min(Sigma)) avoids the SDP
    solve used by knockpy's default optimizer.
    """
    rng = np.random.default_rng(random_state)
    n, p = X.shape
    mu = X.mean(axis=0)
    Xc = X - mu
    sigma = np.cov(Xc, rowvar=False)
    sigma = sigma + np.eye(p) * 1e-8  # numerical stability
    eigvals = np.linalg.eigvalsh(sigma)
    lambda_min = max(eigvals.min(), 1e-8)
    s = min(1.0, 2.0 * lambda_min)
    s_vec = np.full(p, s)

    sigma_inv = np.linalg.pinv(sigma)
    diag_s = np.diag(s_vec)
    # Conditional mean: mu_tilde = X - X Sigma^-1 diag(s)
    mu_tilde = Xc - Xc @ sigma_inv @ diag_s
    # Conditional covariance: 2*diag(s) - diag(s) Sigma^-1 diag(s)
    cov_tilde = 2 * diag_s - diag_s @ sigma_inv @ diag_s
    cov_tilde = (cov_tilde + cov_tilde.T) / 2
    eigvals_t, eigvecs_t = np.linalg.eigh(cov_tilde)
    eigvals_t = np.clip(eigvals_t, 1e-10, None)
    sqrt_cov = eigvecs_t @ np.diag(np.sqrt(eigvals_t)) @ eigvecs_t.T

    noise = rng.standard_normal((n, p))
    X_knockoff = mu_tilde + noise @ sqrt_cov.T + mu
    return X_knockoff


def gaussian_knockoff_select(
    X: pd.DataFrame,
    y: pd.Series,
    fdr: float = 0.2,
    random_state: int = 42,
    plus: bool = True,
) -> dict:
    """Model-X Gaussian knockoff filter with knockoff(+) threshold.

    Returns a dict with selected features, W-statistics, and the FDR
    threshold actually used.
    """
    feature_names = list(X.columns)
    X_arr = X.to_numpy(dtype=float)
    p = X_arr.shape[1]

    X_tilde = _equicorrelated_knockoffs(X_arr, random_state=random_state)
    X_aug = np.hstack([X_arr, X_tilde])

    # Lasso coefficient magnitude as the importance statistic Z_j.
    from sklearn.linear_model import LassoCV

    y_arr = y.to_numpy(dtype=float)
    lasso = LassoCV(cv=5, random_state=random_state, n_jobs=1, max_iter=5000)
    lasso.fit(X_aug, y_arr)
    z = np.abs(lasso.coef_)
    z_orig = z[:p]
    z_knock = z[p:]

    w = z_orig - z_knock  # antisymmetric statistic

    def _threshold(w_stats: np.ndarray, q: float, use_plus: bool) -> float:
        abs_vals = np.sort(np.unique(np.abs(w_stats[w_stats != 0])))
        offset = 1 if use_plus else 0
        candidates = []
        for t in abs_vals:
            num = offset + np.sum(w_stats <= -t)
            den = max(1, np.sum(w_stats >= t))
            if num / den <= q:
                candidates.append(t)
        return float(min(candidates)) if candidates else float("inf")

    tau = _threshold(w, fdr, plus)
    selected = [f for f, w_j in zip(feature_names, w) if w_j >= tau]

    return {
        "selected_features": selected,
        "w_statistics": dict(zip(feature_names, w.tolist())),
        "threshold": tau,
        "fdr_target": fdr,
    }


def shap_select(
    estimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    alpha: float = 0.05,
    max_shap_samples: int = 2000,
    random_state: int = 42,
) -> dict:
    """SHAP-regression feature selection (shap-select / GRASP-style core).

    Fits `estimator` on training data, computes SHAP values on a held-out
    validation fold, regresses y_valid on the per-feature SHAP values, and
    keeps features with a positive, statistically significant (p < alpha)
    coefficient -- mirroring the shap-select heuristic (Kraev et al., 2024)
    and the SHAP-attribution-then-filter logic used by GRASP
    (Luo et al., 2026).
    """
    feature_names = list(X_train.columns)
    model = clone(estimator)
    model.fit(X_train, y_train)

    sample = X_valid
    if len(X_valid) > max_shap_samples:
        sample = X_valid.sample(max_shap_samples, random_state=random_state)
    y_sample = y_valid.loc[sample.index]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_df = pd.DataFrame(shap_values, columns=feature_names, index=sample.index)
    X_design = sm.add_constant(shap_df)
    logit = sm.Logit(y_sample.to_numpy(dtype=float), X_design.to_numpy(dtype=float))
    try:
        result = logit.fit(disp=0, maxiter=200)
        coefs = result.params[1:]  # drop intercept
        pvalues = result.pvalues[1:]
    except Exception:
        # Fallback to OLS on shap values if logit fails to converge
        ols = sm.OLS(y_sample.to_numpy(dtype=float), X_design.to_numpy(dtype=float)).fit()
        coefs = ols.params[1:]
        pvalues = ols.pvalues[1:]

    selected = [
        f
        for f, c, pv in zip(feature_names, coefs, pvalues)
        if c > 0 and pv < alpha
    ]

    return {
        "selected_features": selected,
        "coefficients": dict(zip(feature_names, coefs.tolist())),
        "p_values": dict(zip(feature_names, pvalues.tolist())),
        "alpha": alpha,
    }


__all__ = ["gaussian_knockoff_select", "shap_select"]


def manual_exclusion_shap_select(
    estimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    excluded_features: list,
    top_k: int,
    max_shap_samples: int = 2000,
    random_state: int = 42,
) -> dict:
    """Manual leakage-exclusion followed by SHAP ranking.

    Reimplements the core mechanism of Shetty et al. (2026, BMC Medical
    Informatics and Decision Making, 10.1186/s12911-026-03493-2): a domain
    expert manually flags features suspected of label leakage (a per-feature
    binary decision, not a continuous score), those features are dropped
    from the candidate pool BEFORE model training, and the remaining
    features are ranked by mean(|SHAP|) on the reduced pool.

    This is the natural contrast case for PSEUL: leakage handling here is
    a discrete, manual, all-or-nothing exclusion decided a priori, versus
    PSEUL's continuous, per-fold, automatically computed leakage_risk
    score applied to the full candidate pool.
    """
    remaining = [f for f in X_train.columns if f not in excluded_features]
    model = clone(estimator)
    model.fit(X_train[remaining], y_train)

    sample = X_train[remaining]
    if len(sample) > max_shap_samples:
        sample = sample.sample(max_shap_samples, random_state=random_state)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    importance = pd.Series(np.abs(shap_values).mean(axis=0), index=remaining)
    ranked = importance.sort_values(ascending=False)
    selected = list(ranked.head(top_k).index)

    return {
        "selected_features": selected,
        "excluded_features": list(excluded_features),
        "shap_importance": importance.to_dict(),
    }


def _soft_threshold_group(w: np.ndarray, threshold: float) -> np.ndarray:
    """Group soft-thresholding (proximal operator of the L21 group penalty).

    For a single scalar "group" (one feature = one group here, since PSEUL
    baselines operate on individual tabular features rather than predefined
    feature groups), this reduces to standard soft-thresholding of the
    absolute coefficient magnitude.
    """
    norm = np.abs(w)
    scale = np.maximum(0.0, 1.0 - threshold / (norm + 1e-12))
    return w * scale


def grasp_select(
    estimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    top_k: int,
    lam: float = 0.05,
    max_iter: int = 300,
    lr: float = 0.5,
    max_shap_samples: int = 2000,
    random_state: int = 42,
) -> dict:
    """GRASP: group-Shapley feature selection (full reimplementation).

    Reimplements the two-stage mechanism of Luo, Li, and Cao (2026,
    ICASSP, 10.1109/icassp55912.2026.11462660):
      1. Distill per-feature importance scores from a pretrained tree
         model via SHAP (mean(|SHAP|) per feature).
      2. Fit a logistic regression of y on the SHAP-value matrix, with an
         L21 group-sparsity penalty enforced via proximal gradient descent
         (ISTA), inducing feature-level sparsity. Each feature's SHAP
         column is treated as its own group (single-feature groups),
         since PSEUL's candidate pools are individual tabular features,
         not predefined feature groups as in GRASP's original genomic/
         multi-omics setting.

    This differs from PSEUL by construction: GRASP's sparsity mechanism
    optimizes for compact, non-redundant, *predictive* feature sets via
    SHAP-regression, with no leakage-risk or clinical-utility term at all.
    """
    feature_names = list(X_train.columns)
    model = clone(estimator)
    model.fit(X_train, y_train)

    sample = X_train
    if len(sample) > max_shap_samples:
        sample = sample.sample(max_shap_samples, random_state=random_state)
    y_sample = y_train.loc[sample.index]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_matrix = np.asarray(shap_values, dtype=float)
    # Standardize SHAP columns for a well-conditioned proximal-gradient fit.
    shap_std = shap_matrix.std(axis=0) + 1e-12
    shap_norm = shap_matrix / shap_std
    y_arr = y_sample.to_numpy(dtype=float)

    n, p = shap_norm.shape
    w = np.zeros(p)
    b = 0.0
    for _ in range(max_iter):
        z = shap_norm @ w + b
        pred = 1.0 / (1.0 + np.exp(-z))
        grad_w = shap_norm.T @ (pred - y_arr) / n
        grad_b = float(np.mean(pred - y_arr))
        w = w - lr * grad_w
        b = b - lr * grad_b
        w = _soft_threshold_group(w, lr * lam)

    coef = pd.Series(w, index=feature_names)
    nonzero = coef[coef.abs() > 1e-8]
    ranked = nonzero.abs().sort_values(ascending=False)
    selected = list(ranked.head(top_k).index)
    if len(selected) < top_k:
        # Group-lasso path produced fewer than top_k nonzero features;
        # back-fill by raw SHAP importance among the remaining zeroed-out
        # features so the comparison always has a well-defined Top-K set.
        shap_importance = pd.Series(np.abs(shap_matrix).mean(axis=0), index=feature_names)
        remaining_ranked = shap_importance.drop(index=selected, errors="ignore").sort_values(ascending=False)
        selected += list(remaining_ranked.head(top_k - len(selected)).index)

    return {
        "selected_features": selected,
        "group_lasso_coef": coef.to_dict(),
        "n_nonzero": int(len(nonzero)),
        "lambda": lam,
    }
