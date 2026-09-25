from __future__ import annotations

import json
import sys
import urllib.request
import warnings
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pseul import PSEUL, PseulConfig, PseulFeatureProfile
from run_pseul_adherence_study import evaluate_method, make_model, run_statistics

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TOP_K = 8
URL = "https://www.cdc.gov/brfss/annual_data/2024/files/LLCP2024XPT.zip"
CACHE_DIR = ROOT / "data" / "brfss_2024"
OUTDIR = Path(__file__).resolve().parent / "outputs_brfss_diabetes"
FIGDIR = OUTDIR / "figures"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTDIR.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)


def download_brfss() -> Path:
    zip_path = CACHE_DIR / "LLCP2024XPT.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        xpt_names = [n for n in zf.namelist() if n.strip().lower().endswith(".xpt")]
        if not xpt_names:
            raise RuntimeError("No XPT file found inside BRFSS zip.")
        xpt_name = xpt_names[0]
        xpt_path = CACHE_DIR / xpt_name.strip()
        if not xpt_path.exists():
            with zf.open(xpt_name) as src, xpt_path.open("wb") as dst:
                dst.write(src.read())
    return xpt_path


def clean_code(series: pd.Series, missing_codes: set[int | float] | None = None) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if missing_codes:
        s = s.mask(s.isin(list(missing_codes)))
    return s


def yes_no(series: pd.Series) -> pd.Series:
    s = clean_code(series, {7, 9})
    return s.eq(1).astype(float).where(s.notna())


def profile() -> dict[str, PseulFeatureProfile]:
    low = dict(label_derived_risk=0.0, definitional_overlap=0.0, target_proxy_strength=0.05)
    moderate = dict(label_derived_risk=0.05, definitional_overlap=0.10, target_proxy_strength=0.25)
    high = dict(label_derived_risk=0.90, definitional_overlap=1.00, target_proxy_strength=0.95)
    return {
        "age": PseulFeatureProfile(0.90, 0.35, 0.95, 0.90, **low, topic_similarity=0.90, interpretation="Age is a stable diabetes risk factor."),
        "sex_male": PseulFeatureProfile(0.55, 0.20, 0.55, 0.90, **low, topic_similarity=0.50, interpretation="Sex subgroup marker."),
        "bmi": PseulFeatureProfile(0.95, 0.85, 0.95, 0.85, **moderate, topic_similarity=0.95, interpretation="BMI is a core actionable diabetes risk factor."),
        "general_health": PseulFeatureProfile(0.70, 0.55, 0.75, 0.80, **moderate, topic_similarity=0.70, interpretation="Self-rated general health."),
        "physical_health_days": PseulFeatureProfile(0.55, 0.55, 0.55, 0.70, **moderate, topic_similarity=0.55, interpretation="Physical health burden."),
        "mental_health_days": PseulFeatureProfile(0.45, 0.60, 0.45, 0.70, **low, topic_similarity=0.45, interpretation="Mental health burden."),
        "checkup_recent": PseulFeatureProfile(0.45, 0.55, 0.45, 0.75, **moderate, topic_similarity=0.45, interpretation="Healthcare utilization context."),
        "exercise_any": PseulFeatureProfile(0.70, 0.85, 0.70, 0.85, **low, topic_similarity=0.70, interpretation="Physical activity is actionable."),
        "hypertension": PseulFeatureProfile(0.80, 0.80, 0.85, 0.85, **moderate, topic_similarity=0.80, interpretation="Cardiometabolic comorbidity."),
        "high_cholesterol": PseulFeatureProfile(0.70, 0.75, 0.70, 0.80, **moderate, topic_similarity=0.70, interpretation="Metabolic risk marker."),
        "heart_attack": PseulFeatureProfile(0.60, 0.60, 0.65, 0.75, **moderate, topic_similarity=0.60, interpretation="Cardiovascular comorbidity."),
        "heart_disease": PseulFeatureProfile(0.60, 0.60, 0.65, 0.75, **moderate, topic_similarity=0.60, interpretation="Cardiovascular comorbidity."),
        "kidney_disease": PseulFeatureProfile(0.65, 0.65, 0.70, 0.75, **moderate, topic_similarity=0.65, interpretation="Diabetes-related comorbidity risk."),
        "current_smoker": PseulFeatureProfile(0.45, 0.80, 0.45, 0.80, **low, topic_similarity=0.45, interpretation="Lifestyle risk factor."),
        "insulin_use": PseulFeatureProfile(0.95, 0.70, 0.95, 0.70, **high, topic_similarity=0.95, interpretation="Insulin use is label-proximal for diabetes."),
        "diabetes_age": PseulFeatureProfile(0.95, 0.40, 0.95, 0.55, **high, topic_similarity=0.95, interpretation="Age at diabetes diagnosis is derived after label."),
    }


def load_dataset() -> tuple[pd.DataFrame, pd.Series, dict]:
    csv_cache = CACHE_DIR / "brfss_2024_diabetes_pseul.csv"
    y_cache = CACHE_DIR / "brfss_2024_diabetes_target.csv"
    if csv_cache.exists() and y_cache.exists():
        X = pd.read_csv(csv_cache).fillna(0)
        y = pd.read_csv(y_cache)["diabetes"].astype(int)
        return X, y, {
            "dataset": "BRFSS 2024",
            "rows": int(X.shape[0]),
            "features": int(X.shape[1]),
            "positive_rate": float(y.mean()),
            "source": URL,
        }

    xpt_path = download_brfss()
    df = pd.read_sas(xpt_path, format="xport")
    df.columns = [c.upper() for c in df.columns]

    target = clean_code(df["DIABETE4"], {2, 7, 9})
    keep = target.isin([1, 3, 4])
    y = target.eq(1).astype(int)[keep].reset_index(drop=True)
    d = df.loc[keep].copy()

    def col(name: str, default=np.nan) -> pd.Series:
        return d[name] if name in d.columns else pd.Series(default, index=d.index)

    X = pd.DataFrame(
        {
            "age": clean_code(col("_AGE80"), {999}),
            "sex_male": clean_code(col("SEXVAR"), {7, 9}).eq(1).astype(float),
            "bmi": clean_code(col("_BMI5"), {9999}) / 100.0,
            "general_health": clean_code(col("GENHLTH"), {7, 9}),
            "physical_health_days": clean_code(col("PHYSHLTH"), {77, 88, 99}).fillna(0),
            "mental_health_days": clean_code(col("MENTHLTH"), {77, 88, 99}).fillna(0),
            "checkup_recent": clean_code(col("CHECKUP1"), {7, 8, 9}).eq(1).astype(float),
            "exercise_any": yes_no(col("EXERANY2")),
            "hypertension": yes_no(col("_RFHYPE6")).rsub(1),
            "high_cholesterol": yes_no(col("_RFCHOL3")).rsub(1),
            "heart_attack": yes_no(col("CVDINFR4")),
            "heart_disease": yes_no(col("CVDCRHD4")),
            "kidney_disease": yes_no(col("CHCKDNY2")),
            "current_smoker": clean_code(col("_SMOKER3"), {9}).isin([1, 2]).astype(float),
            "insulin_use": yes_no(col("INSULIN1")).fillna(0),
            "diabetes_age": clean_code(col("DIABAGE4"), {777, 999}).fillna(0),
        }
    )
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(0).reset_index(drop=True)
    X.to_csv(csv_cache, index=False)
    pd.DataFrame({"diabetes": y}).to_csv(y_cache, index=False)
    return X, y, {
        "dataset": "BRFSS 2024",
        "rows": int(X.shape[0]),
        "features": int(X.shape[1]),
        "positive_rate": float(y.mean()),
        "source": URL,
    }


def build_rankings(X_train: pd.DataFrame, y_train: pd.Series, pseul: PSEUL, scores: pd.DataFrame) -> dict[str, list[str]]:
    feature_names = list(X_train.columns)
    rankings = {"All Features": feature_names}
    mi = mutual_info_classif(X_train, y_train, random_state=RANDOM_STATE)
    rankings["Mutual Information Top-8"] = list(pd.Series(mi, index=feature_names).sort_values(ascending=False).head(TOP_K).index)
    model = make_model()
    model.fit(X_train, y_train)
    rankings["LightGBM Gain Top-8"] = list(pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(TOP_K).index)
    perm = permutation_importance(model, X_train, y_train, scoring="roc_auc", n_repeats=2, random_state=RANDOM_STATE, n_jobs=1)
    rankings["Permutation Top-8"] = list(pd.Series(perm.importances_mean, index=feature_names).sort_values(ascending=False).head(TOP_K).index)
    sample = X_train.sample(min(2000, len(X_train)), random_state=RANDOM_STATE)
    values = shap.TreeExplainer(model).shap_values(sample)
    if isinstance(values, list):
        values = values[1]
    rankings["SHAP Top-8"] = list(pd.Series(np.abs(values).mean(axis=0), index=feature_names).sort_values(ascending=False).head(TOP_K).index)
    rankings["PSEUL Select Top-8"] = pseul.selected_features_
    rankings["PSEUL Audit Top-8"] = pseul.select_features(mode="audit", top_k=TOP_K)
    rankings["Leakage-Trap Diabetes-Care Only"] = ["insulin_use", "diabetes_age"]
    return rankings


def plot_results(comparison: pd.DataFrame, scores: pd.DataFrame) -> None:
    ordered = comparison.sort_values("holdout_auc", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(ordered["method"], ordered["holdout_auc"], color="#3E7C8F")
    ax.set_xlabel("Holdout AUC")
    ax.set_title("BRFSS 2024 Diabetes - Holdout AUC")
    for y_pos, val in enumerate(ordered["holdout_auc"]):
        ax.text(val + 0.003, y_pos, f"{val:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig01_brfss_holdout_auc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(comparison["avg_leakage_risk"], comparison["holdout_auc"], s=80, color="#B65C5A", edgecolor="white")
    for _, row in comparison.iterrows():
        ax.text(row["avg_leakage_risk"] + 0.005, row["holdout_auc"], row["method"], fontsize=7)
    ax.set_xlabel("Average Leakage Risk")
    ax.set_ylabel("Holdout AUC")
    ax.set_title("BRFSS 2024 Diabetes - Leakage vs AUC")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig02_brfss_leakage_auc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    ranked = scores.sort_values("select_score", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(ranked["feature"], ranked["select_score"], color="#3E7C8F")
    ax.scatter(ranked["leakage_risk"], ranked["feature"], color="#B65C5A", label="Leakage risk")
    ax.set_xlabel("Score")
    ax.set_title("BRFSS 2024 Diabetes - PSEUL Scores")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig03_brfss_pseul_scores.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_analysis(meta: dict, comparison: pd.DataFrame, stats_out: dict, selected: dict[str, list[str]]) -> None:
    best = comparison.sort_values("holdout_auc", ascending=False).iloc[0]
    pseul = comparison[comparison["method"] == "PSEUL Select Top-8"].iloc[0]
    text = f"""# Simple Analysis - BRFSS 2024 Diabetes

## Dataset

- Source: CDC BRFSS 2024 public-use data
- Rows: {meta["rows"]:,}
- Features: {meta["features"]}
- Positive rate: {meta["positive_rate"]:.3f}
- Task: diabetes classification with diabetes-care leakage traps.

## Main result

Best AUC: **{best["method"]}**, AUC={best["holdout_auc"]:.4f}, accuracy={best["holdout_accuracy"]:.4f}, MCC={best["holdout_mcc"]:.4f}, leakage={best["avg_leakage_risk"]:.4f}.

PSEUL Select Top-8: AUC={pseul["holdout_auc"]:.4f}, accuracy={pseul["holdout_accuracy"]:.4f}, MCC={pseul["holdout_mcc"]:.4f}, leakage={pseul["avg_leakage_risk"]:.4f}.

## Interpretation

BRFSS 2024 is used as the third public validation dataset because it is recent, large, and directly downloadable from CDC. The leakage traps are `insulin_use` and `diabetes_age`, which are not appropriate early predictors because they are only meaningful after a respondent already has diabetes.

Friedman test on CV accuracy: chi2={stats_out["friedman_chi2"]:.4f}, p={stats_out["friedman_p"]:.4g}.

## Selected features

"""
    for method, features in selected.items():
        text += f"- {method}: {', '.join(features)}\n"
    (OUTDIR / "simple_analysis.md").write_text(text, encoding="utf-8")


def main() -> None:
    print("Stage 1 - Download/load BRFSS 2024")
    X, y, meta = load_dataset()
    if len(X) > 50000:
        X, _, y, _ = train_test_split(X, y, train_size=50000, stratify=y, random_state=RANDOM_STATE)
        X = X.reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)
        meta["sampled_rows"] = int(len(X))
    X = X.loc[:, X.nunique(dropna=False) > 1].copy()
    meta["model_features"] = int(X.shape[1])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = pd.Series(y_train).reset_index(drop=True)
    y_test = pd.Series(y_test).reset_index(drop=True)

    print("Stage 2 - Fit PSEUL")
    pseul = PSEUL(
        estimator=make_model(),
        clinical_profile=profile(),
        config=PseulConfig(n_splits=5, random_state=RANDOM_STATE, top_k=TOP_K, max_shap_samples_per_fold=500),
    )
    pseul.fit(X_train, y_train)
    scores = pseul.summary()
    scores.to_csv(OUTDIR / "pseul_feature_scores_locked_v10.csv", index=False)

    print("Stage 3 - Build rankings")
    selected = build_rankings(X_train, y_train, pseul, scores)
    (OUTDIR / "selected_features.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    (OUTDIR / "dataset_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Stage 4 - Evaluate")
    rows, folds = [], []
    for method, features in selected.items():
        print(f"  - {method}: {features}")
        row, fold = evaluate_method(method, features, X_train, X_test, y_train, y_test, scores)
        rows.append(row)
        folds.append(fold)
    comparison = pd.DataFrame(rows).sort_values(["holdout_auc", "holdout_mcc"], ascending=False).reset_index(drop=True)
    fold_results = pd.concat(folds, ignore_index=True)
    stats_out = run_statistics(fold_results, list(selected.keys()))

    comparison.to_csv(OUTDIR / "method_comparison.csv", index=False)
    fold_results.to_csv(OUTDIR / "cv_fold_results.csv", index=False)
    (OUTDIR / "statistical_tests.json").write_text(json.dumps(stats_out, indent=2), encoding="utf-8")
    with pd.ExcelWriter(OUTDIR / "pseul_brfss_diabetes_summary.xlsx") as writer:
        comparison.to_excel(writer, sheet_name="method_comparison", index=False)
        fold_results.to_excel(writer, sheet_name="cv_folds", index=False)
        scores.to_excel(writer, sheet_name="pseul_scores", index=False)
    plot_results(comparison, scores)
    write_analysis(meta, comparison, stats_out, selected)

    print("\n=== BRFSS HOLDOUT SUMMARY ===")
    print(comparison[["method", "n_features", "holdout_accuracy", "holdout_auc", "holdout_mcc", "avg_leakage_risk"]].to_string(index=False))
    print(f"\nOutputs written to: {OUTDIR}")


if __name__ == "__main__":
    main()




