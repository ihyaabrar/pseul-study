from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pseul import PSEUL, PseulConfig, PseulFeatureProfile
from run_pseul_adherence_study import evaluate_method, make_model, run_statistics

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TOP_K = 8
BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/{}.xpt"
CACHE_DIR = ROOT / "data" / "nhanes_2021_2023"
OUTDIR = Path(__file__).resolve().parent / "outputs_nhanes_diabetes"
FIGDIR = OUTDIR / "figures"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTDIR.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)


def read_xpt(name: str) -> pd.DataFrame:
    cached = CACHE_DIR / f"{name}.csv"
    if cached.exists():
        return pd.read_csv(cached)
    df = pd.read_sas(BASE_URL.format(name), format="xport")
    df.to_csv(cached, index=False)
    return df


def profile() -> dict[str, PseulFeatureProfile]:
    base = {
        "age": PseulFeatureProfile(0.90, 0.35, 0.95, 0.90, 0.00, 0.00, 0.05, 0.90, "Age is a stable diabetes risk factor."),
        "sex_male": PseulFeatureProfile(0.65, 0.20, 0.65, 0.90, 0.00, 0.00, 0.05, 0.60, "Sex supports subgroup risk stratification."),
        "race_ethnicity": PseulFeatureProfile(0.65, 0.25, 0.70, 0.80, 0.00, 0.00, 0.05, 0.60, "Race/ethnicity is a population risk descriptor."),
        "education": PseulFeatureProfile(0.45, 0.35, 0.45, 0.70, 0.00, 0.00, 0.05, 0.45, "Socioeconomic context."),
        "income_ratio": PseulFeatureProfile(0.50, 0.45, 0.55, 0.70, 0.00, 0.00, 0.10, 0.50, "Income-to-poverty ratio as socioeconomic context."),
        "bmi": PseulFeatureProfile(0.90, 0.85, 0.90, 0.85, 0.00, 0.00, 0.15, 0.90, "BMI is clinically actionable for diabetes prevention."),
        "waist": PseulFeatureProfile(0.85, 0.80, 0.85, 0.80, 0.00, 0.00, 0.15, 0.85, "Central adiposity risk marker."),
        "weight": PseulFeatureProfile(0.70, 0.75, 0.70, 0.85, 0.00, 0.00, 0.10, 0.70, "Body weight is clinically interpretable."),
        "systolic_bp": PseulFeatureProfile(0.65, 0.70, 0.65, 0.80, 0.00, 0.00, 0.10, 0.65, "Cardiometabolic risk marker."),
        "diastolic_bp": PseulFeatureProfile(0.60, 0.65, 0.60, 0.80, 0.00, 0.00, 0.10, 0.60, "Cardiometabolic risk marker."),
        "total_cholesterol": PseulFeatureProfile(0.55, 0.65, 0.55, 0.75, 0.00, 0.00, 0.15, 0.55, "Metabolic risk marker."),
        "hdl_cholesterol": PseulFeatureProfile(0.60, 0.65, 0.60, 0.75, 0.00, 0.00, 0.15, 0.60, "Metabolic risk marker."),
        "sedentary_minutes": PseulFeatureProfile(0.65, 0.80, 0.65, 0.70, 0.00, 0.00, 0.10, 0.65, "Behavioral intervention target."),
        "ever_smoked": PseulFeatureProfile(0.45, 0.65, 0.45, 0.75, 0.00, 0.00, 0.10, 0.45, "Lifestyle risk context."),
        "prediabetes_told": PseulFeatureProfile(0.80, 0.75, 0.85, 0.75, 0.10, 0.30, 0.65, 0.85, "Prediabetes self-report is close to diabetes label."),
        "hba1c": PseulFeatureProfile(0.95, 0.70, 0.95, 0.85, 0.80, 1.00, 0.95, 0.95, "HbA1c is part of diabetes label definition."),
        "fasting_glucose": PseulFeatureProfile(0.95, 0.70, 0.95, 0.85, 0.80, 1.00, 0.95, 0.95, "Fasting glucose is part of diabetes label definition."),
    }
    return base


def load_dataset() -> tuple[pd.DataFrame, pd.Series, dict]:
    files = {
        "demo": read_xpt("DEMO_L"),
        "bmx": read_xpt("BMX_L"),
        "bpx": read_xpt("BPXO_L"),
        "diq": read_xpt("DIQ_L"),
        "ghb": read_xpt("GHB_L"),
        "glu": read_xpt("GLU_L"),
        "tchol": read_xpt("TCHOL_L"),
        "hdl": read_xpt("HDL_L"),
        "paq": read_xpt("PAQ_L"),
        "smq": read_xpt("SMQ_L"),
    }
    df = files["demo"]
    for key, part in files.items():
        if key == "demo":
            continue
        df = df.merge(part, on="SEQN", how="left")

    df = df[df["RIDAGEYR"] >= 20].copy()
    systolic_cols = [c for c in ["BPXOSY1", "BPXOSY2", "BPXOSY3"] if c in df]
    diastolic_cols = [c for c in ["BPXODI1", "BPXODI2", "BPXODI3"] if c in df]
    df["systolic_bp"] = df[systolic_cols].mean(axis=1)
    df["diastolic_bp"] = df[diastolic_cols].mean(axis=1)

    diabetes_self_report = df["DIQ010"].eq(1)
    diabetes_lab = df["LBXGH"].ge(6.5) | df["LBXGLU"].ge(126)
    y = (diabetes_self_report | diabetes_lab).astype(int)

    features = pd.DataFrame(
        {
            "age": df["RIDAGEYR"],
            "sex_male": df["RIAGENDR"].eq(1).astype(int),
            "race_ethnicity": df["RIDRETH3"],
            "education": df.get("DMDEDUC2"),
            "income_ratio": df.get("INDFMPIR"),
            "bmi": df.get("BMXBMI"),
            "waist": df.get("BMXWAIST"),
            "weight": df.get("BMXWT"),
            "systolic_bp": df["systolic_bp"],
            "diastolic_bp": df["diastolic_bp"],
            "total_cholesterol": df.get("LBXTC"),
            "hdl_cholesterol": df.get("LBDHDD"),
            "sedentary_minutes": df.get("PAD680"),
            "ever_smoked": df.get("SMQ020").eq(1).astype(float),
            "prediabetes_told": df.get("DIQ160").eq(1).astype(float),
            "hba1c": df.get("LBXGH"),
            "fasting_glucose": df.get("LBXGLU"),
        }
    )
    valid = y.notna()
    X = features.loc[valid].copy()
    y = y.loc[valid].astype(int).reset_index(drop=True)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).reset_index(drop=True)
    meta = {
        "dataset": "NHANES August 2021-August 2023",
        "task": "Diabetes classification with label-defining lab leakage traps",
        "rows": int(X.shape[0]),
        "features": int(X.shape[1]),
        "positive_rate": float(y.mean()),
        "source": "CDC/NCHS public XPT files, September 2024 releases for core files",
    }
    return X, y, meta


def build_rankings(X_train: pd.DataFrame, y_train: pd.Series, pseul: PSEUL, scores: pd.DataFrame) -> dict[str, list[str]]:
    feature_names = list(X_train.columns)
    rankings = {"All Features": feature_names}
    mi = mutual_info_classif(X_train, y_train, random_state=RANDOM_STATE)
    rankings["Mutual Information Top-8"] = list(pd.Series(mi, index=feature_names).sort_values(ascending=False).head(TOP_K).index)
    model = make_model()
    model.fit(X_train, y_train)
    rankings["LightGBM Gain Top-8"] = list(pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(TOP_K).index)
    perm = permutation_importance(model, X_train, y_train, scoring="roc_auc", n_repeats=5, random_state=RANDOM_STATE, n_jobs=1)
    rankings["Permutation Top-8"] = list(pd.Series(perm.importances_mean, index=feature_names).sort_values(ascending=False).head(TOP_K).index)
    sample = X_train.sample(min(2000, len(X_train)), random_state=RANDOM_STATE)
    values = shap.TreeExplainer(model).shap_values(sample)
    if isinstance(values, list):
        values = values[1]
    rankings["SHAP Top-8"] = list(pd.Series(np.abs(values).mean(axis=0), index=feature_names).sort_values(ascending=False).head(TOP_K).index)
    rankings["PSEUL Select Top-8"] = pseul.selected_features_
    rankings["PSEUL Audit Top-8"] = pseul.select_features(mode="audit", top_k=TOP_K)
    rankings["Leakage-Trap Labs Only"] = ["hba1c", "fasting_glucose"]
    return rankings


def plot_results(comparison: pd.DataFrame, scores: pd.DataFrame) -> None:
    ordered = comparison.sort_values("holdout_auc", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(ordered["method"], ordered["holdout_auc"], color="#3E7C8F")
    ax.set_xlabel("Holdout AUC")
    ax.set_title("NHANES Diabetes - Holdout AUC")
    for y_pos, val in enumerate(ordered["holdout_auc"]):
        ax.text(val + 0.003, y_pos, f"{val:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig01_nhanes_holdout_auc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(comparison["avg_leakage_risk"], comparison["holdout_auc"], s=80, color="#B65C5A", edgecolor="white")
    for _, row in comparison.iterrows():
        ax.text(row["avg_leakage_risk"] + 0.005, row["holdout_auc"], row["method"], fontsize=7)
    ax.set_xlabel("Average Leakage Risk")
    ax.set_ylabel("Holdout AUC")
    ax.set_title("NHANES Diabetes - Leakage vs AUC")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig02_nhanes_leakage_auc.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    ranked = scores.sort_values("select_score", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(ranked["feature"], ranked["select_score"], color="#3E7C8F")
    ax.scatter(ranked["leakage_risk"], ranked["feature"], color="#B65C5A", label="Leakage risk")
    ax.set_xlabel("Score")
    ax.set_title("NHANES Diabetes - PSEUL Scores")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig03_nhanes_pseul_scores.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_analysis(meta: dict, comparison: pd.DataFrame, stats_out: dict, selected: dict[str, list[str]]) -> None:
    best = comparison.sort_values("holdout_auc", ascending=False).iloc[0]
    pseul = comparison[comparison["method"] == "PSEUL Select Top-8"].iloc[0]
    text = f"""# Simple Analysis - NHANES Diabetes

## Dataset

- Source: {meta["dataset"]}
- Rows: {meta["rows"]:,}
- Features: {meta["features"]}
- Positive rate: {meta["positive_rate"]:.3f}
- Task: {meta["task"]}

## Main result

Best AUC: **{best["method"]}**, AUC={best["holdout_auc"]:.4f}, accuracy={best["holdout_accuracy"]:.4f}, MCC={best["holdout_mcc"]:.4f}, leakage={best["avg_leakage_risk"]:.4f}.

PSEUL Select Top-8: AUC={pseul["holdout_auc"]:.4f}, accuracy={pseul["holdout_accuracy"]:.4f}, MCC={pseul["holdout_mcc"]:.4f}, leakage={pseul["avg_leakage_risk"]:.4f}.

## Interpretation

This dataset intentionally includes HbA1c and fasting glucose as leakage-trap features because the diabetes label is partly defined using these measurements. Generic methods are expected to rank these variables highly. PSEUL is expected to penalize them when selection mode is used.

Friedman test on CV accuracy: chi2={stats_out["friedman_chi2"]:.4f}, p={stats_out["friedman_p"]:.4g}.

## Selected features

"""
    for method, features in selected.items():
        text += f"- {method}: {', '.join(features)}\n"
    (OUTDIR / "simple_analysis.md").write_text(text, encoding="utf-8")


def main() -> None:
    print("Stage 1 - Download/load NHANES")
    X, y, meta = load_dataset()
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
    with pd.ExcelWriter(OUTDIR / "pseul_nhanes_diabetes_summary.xlsx") as writer:
        comparison.to_excel(writer, sheet_name="method_comparison", index=False)
        fold_results.to_excel(writer, sheet_name="cv_folds", index=False)
        scores.to_excel(writer, sheet_name="pseul_scores", index=False)
    plot_results(comparison, scores)
    write_analysis(meta, comparison, stats_out, selected)

    print("\n=== NHANES HOLDOUT SUMMARY ===")
    print(comparison[["method", "n_features", "holdout_accuracy", "holdout_auc", "holdout_mcc", "avg_leakage_risk"]].to_string(index=False))
    print(f"\nOutputs written to: {OUTDIR}")


if __name__ == "__main__":
    main()
