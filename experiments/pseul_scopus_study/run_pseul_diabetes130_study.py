"""PSEUL study on the UCI Diabetes 130-US Hospitals dataset (1999-2008):
30-day hospital readmission — a real EHR benchmark and a fourth validation
dataset for PSEUL, chosen as an openly-licensed (CC BY 4.0) alternative to the
credentialed MIMIC-IV.

Leakage-trap design: `number_inpatient` (prior-year inpatient visits),
`number_diagnoses`, and `discharge_disposition_id` are strong operational
proxies for readmission (e.g. some discharge dispositions encode death/hospice,
which deterministically preclude readmission). Generic methods are expected to
lean on them; PSEUL-Select is expected to down-weight them via leakage risk.

Mirrors run_pseul_nhanes_diabetes_study.py; reuses evaluate_method /
make_model / run_statistics from the adherence study for identical evaluation.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "analysis"))

from pseul import PSEUL, PseulConfig, PseulFeatureProfile  # noqa: E402
from run_pseul_adherence_study import evaluate_method, make_model  # noqa: E402
from recompute_statistics import compute_stats  # noqa: E402

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
TOP_K = 8
SUBSAMPLE = 40_000  # stratified, for runtime parity with BRFSS
DATA = ROOT / "data" / "diabetes130" / "diabetic_data.csv"
OUTDIR = Path(__file__).resolve().parent / "outputs_diabetes130"
FIGDIR = OUTDIR / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)
FIGDIR.mkdir(parents=True, exist_ok=True)

LEAKAGE_TRAPS = ["number_inpatient", "number_diagnoses", "discharge_disposition_id"]

_AGE_MAP = {f"[{i}-{i+10})": i // 10 for i in range(0, 100, 10)}
_A1C_MAP = {"None": 0, "Norm": 1, ">7": 2, ">8": 3}
_GLU_MAP = {"None": 0, "Norm": 1, ">200": 2, ">300": 3}
_INS_MAP = {"No": 0, "Down": 1, "Steady": 2, "Up": 3}


def profile() -> dict[str, PseulFeatureProfile]:
    P = PseulFeatureProfile
    return {
        # numeric utilization / clinical
        "time_in_hospital":   P(0.70, 0.55, 0.70, 0.85, 0.05, 0.10, 0.35, 0.70, "Length of stay; utilization signal."),
        "num_lab_procedures": P(0.55, 0.40, 0.55, 0.75, 0.05, 0.05, 0.25, 0.55, "Care-intensity proxy."),
        "num_procedures":     P(0.55, 0.45, 0.55, 0.75, 0.05, 0.05, 0.25, 0.55, "Care-intensity proxy."),
        "num_medications":    P(0.70, 0.70, 0.70, 0.80, 0.05, 0.10, 0.30, 0.70, "Treatment burden; clinically meaningful."),
        "number_outpatient":  P(0.55, 0.45, 0.60, 0.70, 0.10, 0.15, 0.45, 0.55, "Prior-year outpatient visits; utilization proxy."),
        "number_emergency":   P(0.60, 0.45, 0.65, 0.70, 0.10, 0.20, 0.55, 0.60, "Prior-year ED visits; utilization proxy."),
        # TRAPS
        "number_inpatient":   P(0.65, 0.35, 0.75, 0.65, 0.30, 0.45, 0.90, 0.80, "Prior inpatient visits: strong operational proxy for readmission."),
        "number_diagnoses":   P(0.60, 0.45, 0.70, 0.70, 0.20, 0.40, 0.70, 0.70, "Diagnosis count: proxy for case complexity / readmission."),
        "discharge_disposition_id": P(0.55, 0.30, 0.65, 0.60, 0.40, 0.55, 0.85, 0.75, "Discharge disposition: encodes death/hospice; partly determines readmission."),
        # demographics / labs / meds
        "age_ordinal":        P(0.85, 0.35, 0.90, 0.90, 0.00, 0.00, 0.05, 0.85, "Age is a stable readmission risk factor."),
        "gender_male":        P(0.55, 0.20, 0.55, 0.90, 0.00, 0.00, 0.05, 0.50, "Demographic stratifier."),
        "a1c_ordinal":        P(0.85, 0.80, 0.80, 0.75, 0.05, 0.05, 0.15, 0.85, "HbA1c: actionable glycemic control marker."),
        "max_glu_ordinal":    P(0.70, 0.65, 0.70, 0.70, 0.05, 0.05, 0.20, 0.70, "Serum glucose: clinical marker."),
        "insulin_ordinal":    P(0.70, 0.70, 0.65, 0.75, 0.05, 0.10, 0.25, 0.70, "Insulin regimen change; actionable."),
        "change_flag":        P(0.60, 0.55, 0.55, 0.75, 0.05, 0.10, 0.25, 0.60, "Medication change; actionable."),
        "diabetesMed_flag":   P(0.65, 0.55, 0.60, 0.80, 0.05, 0.10, 0.25, 0.65, "On diabetes medication; actionable."),
        "admission_source_id":P(0.45, 0.30, 0.50, 0.70, 0.10, 0.15, 0.35, 0.45, "Admission source; administrative context."),
    }


def load_dataset() -> tuple[pd.DataFrame, pd.Series, dict]:
    df = pd.read_csv(DATA, na_values="?")
    # one encounter per patient (avoid within-patient train/test leakage)
    df = df.sort_values("encounter_id").drop_duplicates("patient_nbr", keep="first")
    y = (df["readmitted"].astype(str) == "<30").astype(int)

    X = pd.DataFrame(index=df.index)
    for c in ["time_in_hospital", "num_lab_procedures", "num_procedures", "num_medications",
              "number_outpatient", "number_emergency", "number_inpatient", "number_diagnoses",
              "discharge_disposition_id", "admission_source_id"]:
        X[c] = pd.to_numeric(df[c], errors="coerce")
    X["age_ordinal"] = df["age"].map(_AGE_MAP)
    X["gender_male"] = (df["gender"] == "Male").astype(int)
    X["a1c_ordinal"] = df["A1Cresult"].fillna("None").map(_A1C_MAP)
    X["max_glu_ordinal"] = df["max_glu_serum"].fillna("None").map(_GLU_MAP)
    X["insulin_ordinal"] = df["insulin"].map(_INS_MAP)
    X["change_flag"] = (df["change"] == "Ch").astype(int)
    X["diabetesMed_flag"] = (df["diabetesMed"] == "Yes").astype(int)

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).reset_index(drop=True)
    y = y.reset_index(drop=True)

    meta = {
        "dataset": "UCI Diabetes 130-US Hospitals (1999-2008)",
        "task": "30-day hospital readmission with operational leakage traps",
        "rows": int(X.shape[0]), "features": int(X.shape[1]),
        "positive_rate": float(y.mean()),
        "source": "UCI ML Repository #296, CC BY 4.0",
        "leakage_traps": LEAKAGE_TRAPS,
    }
    return X, y, meta


def build_rankings(X_train, y_train, pseul, scores) -> dict[str, list[str]]:
    feats = list(X_train.columns)
    r = {"All Features": feats}
    mi = mutual_info_classif(X_train, y_train, random_state=RANDOM_STATE)
    r["Mutual Information Top-8"] = list(pd.Series(mi, index=feats).sort_values(ascending=False).head(TOP_K).index)
    m = make_model(); m.fit(X_train, y_train)
    r["LightGBM Gain Top-8"] = list(pd.Series(m.feature_importances_, index=feats).sort_values(ascending=False).head(TOP_K).index)
    perm = permutation_importance(m, X_train, y_train, scoring="roc_auc", n_repeats=5, random_state=RANDOM_STATE, n_jobs=1)
    r["Permutation Top-8"] = list(pd.Series(perm.importances_mean, index=feats).sort_values(ascending=False).head(TOP_K).index)
    samp = X_train.sample(min(2000, len(X_train)), random_state=RANDOM_STATE)
    v = shap.TreeExplainer(m).shap_values(samp)
    if isinstance(v, list): v = v[1]
    elif getattr(v, "ndim", 2) == 3: v = v[:, :, -1]
    r["SHAP Top-8"] = list(pd.Series(np.abs(v).mean(axis=0), index=feats).sort_values(ascending=False).head(TOP_K).index)
    r["PSEUL Select Top-8"] = pseul.selected_features_
    r["PSEUL Audit Top-8"] = pseul.select_features(mode="audit", top_k=TOP_K)
    r["Leakage-Trap Readmission-Proxy Only"] = ["number_inpatient", "discharge_disposition_id"]
    return r


def main() -> None:
    print("Stage 1 - load Diabetes 130")
    X, y, meta = load_dataset()
    if len(X) > SUBSAMPLE:
        X, _, y, _ = train_test_split(X, y, train_size=SUBSAMPLE, stratify=y, random_state=RANDOM_STATE)
        X = X.reset_index(drop=True); y = pd.Series(y).reset_index(drop=True)
        meta["subsample"] = SUBSAMPLE
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)
    Xtr, Xte = Xtr.reset_index(drop=True), Xte.reset_index(drop=True)
    ytr, yte = pd.Series(ytr).reset_index(drop=True), pd.Series(yte).reset_index(drop=True)
    print(f"  rows={meta['rows']}, pos_rate={meta['positive_rate']:.3f}, features={meta['features']}")

    print("Stage 2 - fit PSEUL")
    pseul = PSEUL(estimator=make_model(), clinical_profile=profile(),
                  config=PseulConfig(n_splits=5, random_state=RANDOM_STATE, top_k=TOP_K, max_shap_samples_per_fold=500))
    pseul.fit(Xtr, ytr)
    scores = pseul.summary()
    scores.to_csv(OUTDIR / "pseul_feature_scores_locked_v10.csv", index=False)

    print("Stage 3 - rankings")
    selected = build_rankings(Xtr, ytr, pseul, scores)
    (OUTDIR / "selected_features.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    (OUTDIR / "dataset_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Stage 4 - evaluate")
    rows, folds = [], []
    for method, feats in selected.items():
        row, fold = evaluate_method(method, feats, Xtr, Xte, ytr, yte, scores)
        rows.append(row); folds.append(fold)
        print(f"  {method:38s} AUC={row['holdout_auc']:.4f} leak={row['avg_leakage_risk']:.4f}")
    comparison = pd.DataFrame(rows).sort_values(["holdout_auc", "holdout_mcc"], ascending=False).reset_index(drop=True)
    fold_results = pd.concat(folds, ignore_index=True)
    stats_out = compute_stats(fold_results, "accuracy")

    comparison.to_csv(OUTDIR / "method_comparison.csv", index=False)
    fold_results.to_csv(OUTDIR / "cv_fold_results.csv", index=False)
    (OUTDIR / "statistical_tests.json").write_text(json.dumps(stats_out, indent=2), encoding="utf-8")
    with pd.ExcelWriter(OUTDIR / "pseul_diabetes130_summary.xlsx") as w:
        comparison.to_excel(w, sheet_name="method_comparison", index=False)
        fold_results.to_excel(w, sheet_name="cv_folds", index=False)
        scores.to_excel(w, sheet_name="pseul_scores", index=False)

    print("\n=== DIABETES 130 HOLDOUT ===")
    print(comparison[["method", "n_features", "holdout_accuracy", "holdout_auc", "holdout_mcc", "avg_leakage_risk"]].to_string(index=False))
    print(f"Friedman chi2={stats_out['friedman_chi2']:.3f}, p={stats_out['friedman_p']:.3g}")
    print(f"Outputs -> {OUTDIR}")


if __name__ == "__main__":
    main()
