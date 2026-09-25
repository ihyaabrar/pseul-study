from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))

from pseul import PSEUL, PseulConfig, PseulFeatureProfile

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
EVAL_SPLITS = 10
TOP_K = 5
DATASET = ROOT / "data" / "Final Prepared Dataset - Diabetes and Hypertension Data.xlsx"
OUTDIR = Path(__file__).resolve().parent / "outputs"
FIGDIR = OUTDIR / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)

C_BG = "#FAFBFC"
C_DARK = "#2C3E4F"
C_TEAL = "#3E7C8F"
C_GREEN = "#5E8C61"
C_RED = "#B65C5A"
C_GOLD = "#C19A4A"
C_SLATE = "#6F7D8A"
C_GRID = "#E4E9F0"
PALETTE = [C_TEAL, C_GREEN, C_GOLD, C_RED, C_SLATE, "#7E6AA8", "#4F6F52"]

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
        "font.size": 9,
        "axes.facecolor": C_BG,
        "figure.facecolor": "white",
        "axes.edgecolor": C_GRID,
        "grid.color": C_GRID,
        "grid.alpha": 0.45,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def make_model(seed: int = RANDOM_STATE) -> LGBMClassifier:
    return LGBMClassifier(
        random_state=seed,
        verbose=-1,
        n_jobs=1,
        n_estimators=300,
        learning_rate=0.035,
        num_leaves=31,
        subsample=0.90,
        colsample_bytree=0.90,
    )


def clinical_profile() -> dict[str, PseulFeatureProfile]:
    return {
        "AGE": PseulFeatureProfile(
            evidence=0.90,
            intervention_availability=0.40,
            risk_stratification=0.95,
            operational_feasibility=0.90,
            label_derived_risk=0.00,
            definitional_overlap=0.00,
            target_proxy_strength=0.05,
            topic_similarity=0.90,
            interpretation="Demographic risk stratifier available before prediction.",
        ),
        "ANNUALCONTRIBUTION": PseulFeatureProfile(
            evidence=0.55,
            intervention_availability=0.45,
            risk_stratification=0.55,
            operational_feasibility=0.65,
            label_derived_risk=0.05,
            definitional_overlap=0.05,
            target_proxy_strength=0.30,
            topic_similarity=0.55,
            interpretation="Affordability and insurance context; not direct adherence evidence.",
        ),
        "ANNUALCLAIMAMOUNT": PseulFeatureProfile(
            evidence=0.70,
            intervention_availability=0.35,
            risk_stratification=0.70,
            operational_feasibility=0.55,
            label_derived_risk=0.15,
            definitional_overlap=0.30,
            target_proxy_strength=0.75,
            topic_similarity=0.75,
            interpretation="Claims-utilization proxy with strong leakage risk.",
        ),
        "UNITSTOTAL": PseulFeatureProfile(
            evidence=0.80,
            intervention_availability=0.40,
            risk_stratification=0.80,
            operational_feasibility=0.60,
            label_derived_risk=0.20,
            definitional_overlap=0.40,
            target_proxy_strength=0.90,
            topic_similarity=0.85,
            interpretation="Dispensed-unit proxy close to refill/PDC behavior.",
        ),
        "GENDER_M": PseulFeatureProfile(
            evidence=0.55,
            intervention_availability=0.20,
            risk_stratification=0.55,
            operational_feasibility=0.85,
            label_derived_risk=0.00,
            definitional_overlap=0.00,
            target_proxy_strength=0.05,
            topic_similarity=0.45,
            interpretation="Subgroup descriptor; useful for stratification, not direct intervention.",
        ),
        "SCHEMETYPE_MEDIUM": PseulFeatureProfile(
            evidence=0.45,
            intervention_availability=0.35,
            risk_stratification=0.45,
            operational_feasibility=0.65,
            label_derived_risk=0.05,
            definitional_overlap=0.05,
            target_proxy_strength=0.25,
            topic_similarity=0.45,
            interpretation="Insurance plan proxy.",
        ),
        "SCHEMETYPE_PREMIUM": PseulFeatureProfile(
            evidence=0.45,
            intervention_availability=0.35,
            risk_stratification=0.45,
            operational_feasibility=0.65,
            label_derived_risk=0.05,
            definitional_overlap=0.05,
            target_proxy_strength=0.25,
            topic_similarity=0.45,
            interpretation="Insurance plan proxy.",
        ),
        "DIAGNOSIS_HYPERTENSION": PseulFeatureProfile(
            evidence=0.85,
            intervention_availability=0.75,
            risk_stratification=0.85,
            operational_feasibility=0.80,
            label_derived_risk=0.00,
            definitional_overlap=0.00,
            target_proxy_strength=0.10,
            topic_similarity=0.85,
            interpretation="Clinically meaningful diagnosis for adherence support.",
        ),
        "COVERTYPE_STANDARD": PseulFeatureProfile(
            evidence=0.45,
            intervention_availability=0.40,
            risk_stratification=0.45,
            operational_feasibility=0.65,
            label_derived_risk=0.05,
            definitional_overlap=0.05,
            target_proxy_strength=0.25,
            topic_similarity=0.45,
            interpretation="Administrative coverage context.",
        ),
        "COMORBIDITY_NO_COMORBIDITY": PseulFeatureProfile(
            evidence=0.85,
            intervention_availability=0.70,
            risk_stratification=0.85,
            operational_feasibility=0.85,
            label_derived_risk=0.00,
            definitional_overlap=0.00,
            target_proxy_strength=0.10,
            topic_similarity=0.85,
            interpretation="Comorbidity status supports clinical risk stratification.",
        ),
        "COMPLICATIONDEVELOPMENT_NO_COMPLICATION": PseulFeatureProfile(
            evidence=0.80,
            intervention_availability=0.65,
            risk_stratification=0.80,
            operational_feasibility=0.75,
            label_derived_risk=0.05,
            definitional_overlap=0.15,
            target_proxy_strength=0.25,
            topic_similarity=0.80,
            interpretation="Clinical status, but temporality must be verified.",
        ),
    }


def load_adherence() -> tuple[pd.DataFrame, pd.Series, dict]:
    df_raw = pd.read_excel(DATASET)
    df = df_raw.drop_duplicates().reset_index(drop=True)
    y = (df["ADHERENCE"].astype(str).str.upper() == "ADHERENT").astype(int)
    X = df.drop(columns=["ADHERENCE"]).copy()
    for col in X.columns:
        if X[col].dtype == "bool":
            X[col] = X[col].astype(int)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    metadata = {
        "raw_rows": int(df_raw.shape[0]),
        "raw_columns": int(df_raw.shape[1]),
        "duplicates_removed": int(df_raw.duplicated().sum()),
        "clean_rows": int(df.shape[0]),
        "clean_features": int(X.shape[1]),
        "positive_class": "ADHERENT",
        "positive_rate": float(y.mean()),
    }
    return X, y, metadata


def fit_pseul(X_train: pd.DataFrame, y_train: pd.Series) -> tuple[PSEUL, pd.DataFrame]:
    pseul = PSEUL(
        estimator=make_model(),
        clinical_profile=clinical_profile(),
        config=PseulConfig(
            n_splits=5,
            random_state=RANDOM_STATE,
            top_k=TOP_K,
            max_shap_samples_per_fold=500,
            utility_method="permutation",
        ),
    )
    pseul.fit(X_train, y_train)
    scores = pseul.summary()
    return pseul, scores


def build_rankings(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    pseul: PSEUL,
    pseul_scores: pd.DataFrame,
) -> dict[str, list[str]]:
    feature_names = list(X_train.columns)
    rankings: dict[str, list[str]] = {"All Features": feature_names}

    mi = mutual_info_classif(X_train, y_train, random_state=RANDOM_STATE)
    rankings["Mutual Information Top-5"] = list(
        pd.Series(mi, index=feature_names).sort_values(ascending=False).head(TOP_K).index
    )

    model = make_model()
    model.fit(X_train, y_train)
    rankings["LightGBM Gain Top-5"] = list(
        pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(TOP_K).index
    )

    perm = permutation_importance(
        model,
        X_train,
        y_train,
        scoring="roc_auc",
        n_repeats=8,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    rankings["Permutation Top-5"] = list(
        pd.Series(perm.importances_mean, index=feature_names).sort_values(ascending=False).head(TOP_K).index
    )

    shap_sample = X_train.sample(min(2000, len(X_train)), random_state=RANDOM_STATE)
    explainer = shap.TreeExplainer(model)
    values = explainer.shap_values(shap_sample)
    if isinstance(values, list):
        values = values[1]
    rankings["SHAP Top-5"] = list(
        pd.Series(np.abs(values).mean(axis=0), index=feature_names).sort_values(ascending=False).head(TOP_K).index
    )

    rankings["PSEUL Select Top-5"] = pseul.selected_features_
    rankings["PSEUL Audit Top-5"] = pseul.select_features(mode="audit", top_k=TOP_K)
    rankings["Low-Leakage Clinical Top-5"] = list(
        pseul_scores.sort_values(["leakage_risk", "clinical_utility"], ascending=[True, False])
        .head(TOP_K)["feature"]
    )
    return rankings


def evaluate_method(
    name: str,
    features: list[str],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    pseul_scores: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    cv = StratifiedKFold(n_splits=EVAL_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_rows = []
    start = time.time()

    for fold_id, (tr_idx, val_idx) in enumerate(cv.split(X_train[features], y_train), start=1):
        model = make_model(RANDOM_STATE + fold_id)
        model.fit(X_train.iloc[tr_idx][features], y_train.iloc[tr_idx])
        pred = model.predict(X_train.iloc[val_idx][features])
        prob = model.predict_proba(X_train.iloc[val_idx][features])[:, 1]
        y_val = y_train.iloc[val_idx]
        fold_rows.append(
            {
                "method": name,
                "fold": fold_id,
                "accuracy": accuracy_score(y_val, pred),
                "auc": roc_auc_score(y_val, prob),
                "f1_weighted": f1_score(y_val, pred, average="weighted"),
                "mcc": matthews_corrcoef(y_val, pred),
            }
        )

    model = make_model()
    model.fit(X_train[features], y_train)
    pred = model.predict(X_test[features])
    prob = model.predict_proba(X_test[features])[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()

    selected_scores = pseul_scores.set_index("feature").loc[features]
    folds = pd.DataFrame(fold_rows)
    row = {
        "method": name,
        "n_features": len(features),
        "selected_features": ", ".join(features),
        "holdout_accuracy": accuracy_score(y_test, pred),
        "holdout_auc": roc_auc_score(y_test, prob),
        "holdout_f1_weighted": f1_score(y_test, pred, average="weighted"),
        "holdout_mcc": matthews_corrcoef(y_test, pred),
        "holdout_precision_weighted": precision_score(y_test, pred, average="weighted", zero_division=0),
        "holdout_recall_weighted": recall_score(y_test, pred, average="weighted", zero_division=0),
        "holdout_brier": brier_score_loss(y_test, prob),
        "adherent_sensitivity": tp / (tp + fn + 1e-12),
        "nonadherent_specificity": tn / (tn + fp + 1e-12),
        "cv_accuracy_mean": folds["accuracy"].mean(),
        "cv_accuracy_std": folds["accuracy"].std(ddof=0),
        "cv_auc_mean": folds["auc"].mean(),
        "cv_auc_std": folds["auc"].std(ddof=0),
        "cv_f1_mean": folds["f1_weighted"].mean(),
        "cv_mcc_mean": folds["mcc"].mean(),
        "avg_select_score": selected_scores["select_score"].mean(),
        "avg_audit_score": selected_scores["audit_score"].mean(),
        "avg_predictive_utility": selected_scores["predictive_utility"].mean(),
        "avg_shap_stability": selected_scores["shap_stability"].mean(),
        "avg_evidence_relevance": selected_scores["evidence_relevance"].mean(),
        "avg_clinical_utility": selected_scores["clinical_utility"].mean(),
        "avg_leakage_risk": selected_scores["leakage_risk"].mean(),
        "runtime_seconds": time.time() - start,
    }
    return row, folds


def run_statistics(fold_results: pd.DataFrame, methods: list[str]) -> dict:
    wide = fold_results.pivot(index="fold", columns="method", values="accuracy")[methods]
    stat = stats.friedmanchisquare(*[wide[m].to_numpy() for m in methods])
    out = {
        "metric": "cv_accuracy",
        "friedman_chi2": float(stat.statistic),
        "friedman_p": float(stat.pvalue),
        "mean_ranks": wide.rank(axis=1, ascending=False).mean().to_dict(),
        "pairwise_vs_pseul_select": {},
    }
    pseul_col = "PSEUL Select Top-5"
    if pseul_col in wide.columns:
        for method in methods:
            if method == pseul_col:
                continue
            try:
                w = stats.wilcoxon(wide[pseul_col], wide[method], zero_method="wilcox")
                out["pairwise_vs_pseul_select"][method] = {
                    "wilcoxon_stat": float(w.statistic),
                    "p_value": float(w.pvalue),
                    "mean_accuracy_delta_pseul_minus_method": float((wide[pseul_col] - wide[method]).mean()),
                }
            except ValueError:
                out["pairwise_vs_pseul_select"][method] = {
                    "wilcoxon_stat": None,
                    "p_value": None,
                    "mean_accuracy_delta_pseul_minus_method": float((wide[pseul_col] - wide[method]).mean()),
                }
    return out


def make_figures(comparison: pd.DataFrame, pseul_scores: pd.DataFrame) -> None:
    ordered = comparison.sort_values("holdout_auc", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(ordered["method"], ordered["holdout_auc"], color=PALETTE[: len(ordered)])
    ax.set_xlabel("Holdout AUC")
    ax.set_title("PSEUL Study - Holdout AUC by Feature Selection Method", color=C_DARK, fontweight="bold")
    ax.grid(axis="x")
    for y, value in enumerate(ordered["holdout_auc"]):
        ax.text(value + 0.003, y, f"{value:.3f}", va="center", fontsize=8, color=C_DARK)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig01_holdout_auc.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for i, row in comparison.iterrows():
        ax.scatter(row["avg_leakage_risk"], row["holdout_auc"], s=90, color=PALETTE[i % len(PALETTE)], edgecolor="white")
        ax.text(row["avg_leakage_risk"] + 0.005, row["holdout_auc"], row["method"], fontsize=7.5, va="center")
    ax.set_xlabel("Average Leakage Risk")
    ax.set_ylabel("Holdout AUC")
    ax.set_title("Performance vs Leakage-Risk Trade-off", color=C_DARK, fontweight="bold")
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig02_auc_vs_leakage.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    top = pseul_scores.sort_values("select_score", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(top["feature"], top["select_score"], color=C_TEAL, label="Select score")
    ax.scatter(top["leakage_risk"], top["feature"], color=C_RED, label="Leakage risk", zorder=3)
    ax.set_xlabel("Score")
    ax.set_title("PSEUL Feature Ranking and Leakage Risk", color=C_DARK, fontweight="bold")
    ax.legend(frameon=True, facecolor="white", fontsize=8)
    ax.grid(axis="x")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig03_pseul_feature_scores.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_protocol(metadata: dict) -> None:
    text = f"""# Paper 5 PSEUL Research Protocol

## Research question

Does PSEUL provide a better medical feature-selection trade-off than generic feature selection for medication-adherence prediction?

## Dataset

- Source file: `{DATASET.name}`
- Raw shape: {metadata["raw_rows"]:,} rows x {metadata["raw_columns"]} columns
- Clean rows: {metadata["clean_rows"]:,}
- Candidate predictors: {metadata["clean_features"]}
- Target: ADHERENCE, encoded as ADHERENT=1 and NON-ADHERENT=0
- Positive class rate: {metadata["positive_rate"]:.3f}

## Stages adapted from paper4

1. Data acquisition.
2. Minimal cleaning and target encoding.
3. Stratified 80/20 train-holdout split with random_state=42.
4. Parallel feature-selection pipelines: All Features, MI, LightGBM gain, permutation, SHAP, PSEUL Select, PSEUL Audit, and low-leakage clinical baseline.
5. Identical LightGBM classifier for all selected feature sets.
6. Ten-fold stratified CV on the training set.
7. Statistical comparison using Friedman test and Wilcoxon comparison against PSEUL Select.
8. Untouched holdout evaluation.
9. Leakage-performance interpretation.

## Notes

This script runs the adherence dataset first because it is already available locally. NHANES 2021-2023 and MIMIC-IV v3.1 should be added as external validation datasets after the raw files/credentials are available.
"""
    (OUTDIR / "research_protocol.md").write_text(text, encoding="utf-8")


def write_analysis(comparison: pd.DataFrame, stats_out: dict, selected: dict[str, list[str]]) -> None:
    best_auc = comparison.sort_values("holdout_auc", ascending=False).iloc[0]
    pseul_row = comparison[comparison["method"] == "PSEUL Select Top-5"].iloc[0]
    safest = comparison.sort_values("avg_leakage_risk", ascending=True).iloc[0]
    text = f"""# Simple Analysis - PSEUL Adherence Study

## Main result

The best holdout AUC was achieved by **{best_auc["method"]}** with AUC={best_auc["holdout_auc"]:.4f}, accuracy={best_auc["holdout_accuracy"]:.4f}, and MCC={best_auc["holdout_mcc"]:.4f}.

PSEUL Select Top-5 achieved AUC={pseul_row["holdout_auc"]:.4f}, accuracy={pseul_row["holdout_accuracy"]:.4f}, MCC={pseul_row["holdout_mcc"]:.4f}, and average leakage risk={pseul_row["avg_leakage_risk"]:.4f}.

The lowest average leakage risk was produced by **{safest["method"]}** with leakage={safest["avg_leakage_risk"]:.4f} and AUC={safest["holdout_auc"]:.4f}.

## Interpretation

On the adherence dataset, generic feature-selection methods still benefit from high-signal claim-utilization variables such as UNITSTOTAL and ANNUALCLAIMAMOUNT. These variables increase AUC but carry proxy/leakage risk because they are close to refill behavior and adherence calculation.

PSEUL Select is more conservative. Its value is not only absolute AUC, but whether it can reduce selection of label-proximal or proxy variables while retaining clinically defensible predictors.

## Statistical note

Friedman test on 10-fold CV accuracy: chi2={stats_out["friedman_chi2"]:.4f}, p={stats_out["friedman_p"]:.4g}.

## Selected features

"""
    for method, features in selected.items():
        text += f"- {method}: {', '.join(features)}\n"
    (OUTDIR / "simple_analysis.md").write_text(text, encoding="utf-8")


def main() -> None:
    print("Stage 1 - Data acquisition")
    X, y, metadata = load_adherence()
    write_protocol(metadata)

    print("Stage 2 - Minimal cleaning completed")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = pd.Series(y_train).reset_index(drop=True)
    y_test = pd.Series(y_test).reset_index(drop=True)

    print("Stage 3 - Fitting PSEUL")
    pseul, pseul_scores = fit_pseul(X_train, y_train)
    pseul_scores.to_csv(OUTDIR / "pseul_feature_scores_locked_v10.csv", index=False)

    print("Stage 4 - Building comparison methods")
    selected = build_rankings(X_train, y_train, pseul, pseul_scores)
    (OUTDIR / "selected_features.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")

    print("Stage 5-8 - LightGBM CV and holdout evaluation")
    rows = []
    fold_tables = []
    for method, features in selected.items():
        print(f"  - {method}: {features}")
        row, folds = evaluate_method(method, features, X_train, X_test, y_train, y_test, pseul_scores)
        rows.append(row)
        fold_tables.append(folds)

    comparison = pd.DataFrame(rows).sort_values(["holdout_auc", "holdout_mcc"], ascending=False).reset_index(drop=True)
    fold_results = pd.concat(fold_tables, ignore_index=True)
    stats_out = run_statistics(fold_results, list(selected.keys()))

    comparison.to_csv(OUTDIR / "method_comparison.csv", index=False)
    fold_results.to_csv(OUTDIR / "cv_fold_results.csv", index=False)
    (OUTDIR / "statistical_tests.json").write_text(json.dumps(stats_out, indent=2), encoding="utf-8")

    with pd.ExcelWriter(OUTDIR / "pseul_adherence_study_summary.xlsx") as writer:
        comparison.to_excel(writer, sheet_name="method_comparison", index=False)
        fold_results.to_excel(writer, sheet_name="cv_folds", index=False)
        pseul_scores.to_excel(writer, sheet_name="pseul_scores", index=False)
        pd.DataFrame(
            [{"method": method, "feature": feature} for method, features in selected.items() for feature in features]
        ).to_excel(writer, sheet_name="selected_features", index=False)

    make_figures(comparison, pseul_scores)
    write_analysis(comparison, stats_out, selected)

    print("\n=== HOLDOUT SUMMARY ===")
    cols = [
        "method",
        "n_features",
        "holdout_accuracy",
        "holdout_auc",
        "holdout_mcc",
        "avg_leakage_risk",
        "avg_clinical_utility",
    ]
    print(comparison[cols].to_string(index=False))
    print(f"\nOutputs written to: {OUTDIR}")


if __name__ == "__main__":
    main()
