from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
STUDY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(STUDY_ROOT))

from pseul import PseulFeatureProfile  # noqa: E402
from run_pseul_adherence_study import clinical_profile as adherence_profile  # noqa: E402
from run_pseul_brfss_diabetes_study import (  # noqa: E402
    clean_code,
    download_brfss,
    profile as brfss_profile,
    yes_no,
)
from run_pseul_diabetes130_study import (  # noqa: E402
    _A1C_MAP,
    _AGE_MAP,
    _GLU_MAP,
    _INS_MAP,
    profile as diabetes130_profile,
)
from run_pseul_nhanes_diabetes_study import profile as nhanes_profile  # noqa: E402


@dataclass(frozen=True)
class StudySpec:
    key: str
    dataset: str
    task_type: str
    intended_use: str
    prediction_landmark: str
    observation_window: str
    prediction_horizon: str
    eligible_population: str
    top_k: int
    canonical_traps: tuple[str, ...]
    leakage_mechanism: str
    evidence_status: str
    interpretation: str


STUDY_SPECS: dict[str, StudySpec] = {
    "adherence": StudySpec(
        key="adherence",
        dataset="Medication adherence cohort",
        task_type="Retrospective classification",
        intended_use="Audit whether concurrent utilization proxies dominate adherence classification",
        prediction_landmark="End of the recorded adherence assessment period",
        observation_window="Same recorded period as the adherence label; no prospective window available",
        prediction_horizon="None (retrospective classification)",
        eligible_population="Deduplicated adults represented in the prepared cohort",
        top_k=5,
        canonical_traps=("UNITSTOTAL", "ANNUALCLAIMAMOUNT"),
        leakage_mechanism="Concurrent outcome-window proxy",
        evidence_status="Provisional until a temporal data dictionary is supplied",
        interpretation="Not evidence of prospective early-warning performance.",
    ),
    "nhanes": StudySpec(
        key="nhanes",
        dataset="NHANES August 2021-August 2023",
        task_type="Cross-sectional classification stress test",
        intended_use="Detect variables used directly in the operational diabetes definition",
        prediction_landmark="Survey/examination assessment",
        observation_window="Contemporaneous interview, examination, and laboratory data",
        prediction_horizon="None (cross-sectional diabetes status)",
        eligible_population="Participants aged at least 20 with self-report or laboratory label evidence",
        top_k=8,
        canonical_traps=("hba1c", "fasting_glucose"),
        leakage_mechanism="Definitional overlap",
        evidence_status="Operationally defined from the target construction",
        interpretation="A label-definition leakage stress test, not prospective diabetes prediction.",
    ),
    "brfss": StudySpec(
        key="brfss",
        dataset="BRFSS 2024",
        task_type="Cross-sectional structural-leakage stress test",
        intended_use="Detect diabetes-contingent survey items and their skip-pattern indicators",
        prediction_landmark="Survey interview",
        observation_window="Contemporaneous BRFSS responses",
        prediction_horizon="None (self-reported diabetes status)",
        eligible_population="Respondents with valid DIABETE4 status in categories 1, 3, or 4",
        top_k=8,
        canonical_traps=(
            "insulin_use",
            "diabetes_age",
            "insulin_use_observed",
            "diabetes_age_observed",
        ),
        leakage_mechanism="Post-diagnosis information and target-dependent questionnaire skip pattern",
        evidence_status="Structural stress test; not a clinical-performance benchmark",
        interpretation="AUC near 1.0 indicates structural leakage rather than model skill.",
    ),
    "diabetes130_admission": StudySpec(
        key="diabetes130_admission",
        dataset="UCI Diabetes 130-US Hospitals",
        task_type="Admission-time 30-day readmission risk stress test",
        intended_use="Identify patients for discharge-planning attention using information available at admission",
        prediction_landmark="Hospital admission",
        observation_window="History and administrative information available no later than admission",
        prediction_horizon="Readmission within 30 days after discharge",
        eligible_population=(
            "First encounter per patient; death/hospice encounters are retained because discharge status is unknown "
            "at admission and are acknowledged as competing events"
        ),
        top_k=5,
        canonical_traps=(
            "time_in_hospital",
            "num_lab_procedures",
            "num_procedures",
            "num_medications",
            "number_diagnoses",
            "discharge_disposition_id",
            "a1c_ordinal",
            "max_glu_ordinal",
            "insulin_ordinal",
            "change_flag",
            "diabetesMed_flag",
        ),
        leakage_mechanism="Information recorded after the admission landmark",
        evidence_status="Landmark-defined temporal availability stress test",
        interpretation="Prior-year utilization counts remain valid admission-time predictors.",
    ),
    "diabetes130_discharge": StudySpec(
        key="diabetes130_discharge",
        dataset="UCI Diabetes 130-US Hospitals",
        task_type="Discharge-time 30-day readmission risk prediction",
        intended_use="Prioritize post-discharge follow-up using information available by discharge",
        prediction_landmark="Immediately before discharge",
        observation_window="History plus index-encounter data available by discharge",
        prediction_horizon="Readmission within 30 days after discharge",
        eligible_population="First eligible encounter per patient, excluding death/hospice dispositions",
        # Match the admission scenario's cardinality ceiling so the two
        # landmark-specific audits are not confounded by different quotas.
        top_k=5,
        canonical_traps=(),
        leakage_mechanism="Negative-control scenario: no designated temporal trap after eligibility filtering",
        evidence_status="Retrospective internal evaluation; external/temporal validation absent",
        interpretation="Tests whether PSEUL avoids inventing leakage when features are landmark-valid.",
    ),
}


def _binary_equal(series: pd.Series, positive: int | float, valid: set[int | float]) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.eq(positive).astype(float).where(numeric.isin(valid))


def _load_adherence() -> tuple[pd.DataFrame, pd.Series, dict]:
    path = ROOT / "data" / "Final Prepared Dataset - Diabetes and Hypertension Data.xlsx"
    raw = pd.read_excel(path)
    data = raw.drop_duplicates().reset_index(drop=True)
    y = data["ADHERENCE"].astype(str).str.upper().eq("ADHERENT").astype(int)
    X = data.drop(columns=["ADHERENCE"]).copy()
    for column in X.columns:
        if X[column].dtype == "bool":
            X[column] = X[column].astype(int)
    X = X.apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    return X, y.reset_index(drop=True), {
        "source_file": str(path),
        "raw_rows": int(len(raw)),
        "duplicates_removed": int(raw.duplicated().sum()),
    }


def _read_nhanes(name: str) -> pd.DataFrame:
    path = ROOT / "data" / "nhanes_2021_2023" / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing NHANES cache: {path}")
    return pd.read_csv(path)


def _load_nhanes() -> tuple[pd.DataFrame, pd.Series, dict]:
    files = {
        "demo": _read_nhanes("DEMO_L"),
        "bmx": _read_nhanes("BMX_L"),
        "bpx": _read_nhanes("BPXO_L"),
        "diq": _read_nhanes("DIQ_L"),
        "ghb": _read_nhanes("GHB_L"),
        "glu": _read_nhanes("GLU_L"),
        "tchol": _read_nhanes("TCHOL_L"),
        "hdl": _read_nhanes("HDL_L"),
        "paq": _read_nhanes("PAQ_L"),
        "smq": _read_nhanes("SMQ_L"),
    }
    data = files["demo"]
    for name, part in files.items():
        if name != "demo":
            data = data.merge(part, on="SEQN", how="left")
    data = data[pd.to_numeric(data["RIDAGEYR"], errors="coerce").ge(20)].copy()

    systolic = [column for column in ("BPXOSY1", "BPXOSY2", "BPXOSY3") if column in data]
    diastolic = [column for column in ("BPXODI1", "BPXODI2", "BPXODI3") if column in data]
    data["systolic_bp"] = data[systolic].mean(axis=1)
    data["diastolic_bp"] = data[diastolic].mean(axis=1)

    self_report_raw = pd.to_numeric(data["DIQ010"], errors="coerce")
    hba1c = pd.to_numeric(data["LBXGH"], errors="coerce")
    glucose = pd.to_numeric(data["LBXGLU"], errors="coerce")
    eligible = self_report_raw.isin([1, 2, 3]) | hba1c.notna() | glucose.notna()
    positive = self_report_raw.eq(1) | hba1c.ge(6.5) | glucose.ge(126)

    features = pd.DataFrame(
        {
            "age": data["RIDAGEYR"],
            "sex_male": _binary_equal(data["RIAGENDR"], 1, {1, 2}),
            "race_ethnicity": data["RIDRETH3"],
            "education": data.get("DMDEDUC2"),
            "income_ratio": data.get("INDFMPIR"),
            "bmi": data.get("BMXBMI"),
            "waist": data.get("BMXWAIST"),
            "weight": data.get("BMXWT"),
            "systolic_bp": data["systolic_bp"],
            "diastolic_bp": data["diastolic_bp"],
            "total_cholesterol": data.get("LBXTC"),
            "hdl_cholesterol": data.get("LBDHDD"),
            "sedentary_minutes": data.get("PAD680"),
            "ever_smoked": _binary_equal(data.get("SMQ020"), 1, {1, 2}),
            "prediabetes_told": _binary_equal(data.get("DIQ160"), 1, {1, 2}),
            "hba1c": hba1c,
            "fasting_glucose": glucose,
        }
    )
    X = features.loc[eligible].apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    y = positive.loc[eligible].astype(int).reset_index(drop=True)
    return X, y, {"source_dir": str(ROOT / "data" / "nhanes_2021_2023")}


def _brfss_day_count(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    numeric = numeric.mask(numeric.isin([77, 99]))
    return numeric.mask(numeric.eq(88), 0)


def _load_brfss() -> tuple[pd.DataFrame, pd.Series, dict]:
    cache_dir = Path(__file__).resolve().parent / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    x_cache = cache_dir / "brfss_features_with_missingness.csv"
    y_cache = cache_dir / "brfss_target.csv"
    if x_cache.exists() and y_cache.exists():
        return (
            pd.read_csv(x_cache),
            pd.read_csv(y_cache)["diabetes"].astype(int),
            {"source_file": str(ROOT / "data" / "brfss_2024" / "LLCP2024.XPT"), "cache": str(x_cache)},
        )

    xpt = download_brfss()
    data = pd.read_sas(xpt, format="xport")
    data.columns = [column.upper() for column in data.columns]
    target = clean_code(data["DIABETE4"], {2, 7, 9})
    keep = target.isin([1, 3, 4])
    y = target.eq(1).astype(int)[keep].reset_index(drop=True)
    d = data.loc[keep].copy()

    def col(name: str) -> pd.Series:
        return d[name] if name in d.columns else pd.Series(np.nan, index=d.index)

    insulin_raw = clean_code(col("INSULIN1"), {7, 9})
    diabetes_age_raw = clean_code(col("DIABAGE4"), {777, 999})
    smoker = clean_code(col("_SMOKER3"), {9})
    checkup = clean_code(col("CHECKUP1"), {7, 8, 9})
    X = pd.DataFrame(
        {
            "age": clean_code(col("_AGE80"), {999}),
            "sex_male": _binary_equal(col("SEXVAR"), 1, {1, 2}),
            "bmi": clean_code(col("_BMI5"), {9999}) / 100.0,
            "general_health": clean_code(col("GENHLTH"), {7, 9}),
            "physical_health_days": _brfss_day_count(col("PHYSHLTH")),
            "mental_health_days": _brfss_day_count(col("MENTHLTH")),
            "checkup_recent": checkup.eq(1).astype(float).where(checkup.notna()),
            "exercise_any": yes_no(col("EXERANY2")),
            "hypertension": yes_no(col("_RFHYPE6")).rsub(1),
            "high_cholesterol": yes_no(col("_RFCHOL3")).rsub(1),
            "heart_attack": yes_no(col("CVDINFR4")),
            "heart_disease": yes_no(col("CVDCRHD4")),
            "kidney_disease": yes_no(col("CHCKDNY2")),
            "current_smoker": smoker.isin([1, 2]).astype(float).where(smoker.notna()),
            "insulin_use": insulin_raw.eq(1).astype(float).where(insulin_raw.notna()),
            "diabetes_age": diabetes_age_raw,
            "insulin_use_observed": insulin_raw.notna().astype(float),
            "diabetes_age_observed": diabetes_age_raw.notna().astype(float),
        }
    ).apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    X.to_csv(x_cache, index=False)
    pd.DataFrame({"diabetes": y}).to_csv(y_cache, index=False)
    return X, y, {"source_file": str(xpt), "cache": str(x_cache)}


def _load_diabetes130(exclude_death_hospice: bool) -> tuple[pd.DataFrame, pd.Series, dict]:
    path = ROOT / "data" / "diabetes130" / "diabetic_data.csv"
    raw = pd.read_csv(path, na_values="?").sort_values("encounter_id")
    data = raw.drop_duplicates("patient_nbr", keep="first").copy()
    excluded_dispositions = {11, 13, 14, 19, 20, 21}
    disposition = pd.to_numeric(data["discharge_disposition_id"], errors="coerce")
    competing_event_rows = int(disposition.isin(excluded_dispositions).sum())
    if exclude_death_hospice:
        data = data.loc[~disposition.isin(excluded_dispositions)].copy()
    y = data["readmitted"].astype(str).eq("<30").astype(int)

    X = pd.DataFrame(index=data.index)
    numeric_columns = (
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
        "discharge_disposition_id",
        "admission_source_id",
        "admission_type_id",
    )
    for column in numeric_columns:
        X[column] = pd.to_numeric(data[column], errors="coerce")
    X["age_ordinal"] = data["age"].map(_AGE_MAP)
    X["gender_male"] = data["gender"].eq("Male").astype(int)
    X["a1c_ordinal"] = data["A1Cresult"].fillna("None").map(_A1C_MAP)
    X["max_glu_ordinal"] = data["max_glu_serum"].fillna("None").map(_GLU_MAP)
    X["insulin_ordinal"] = data["insulin"].map(_INS_MAP)
    X["change_flag"] = data["change"].eq("Ch").astype(int)
    X["diabetesMed_flag"] = data["diabetesMed"].eq("Yes").astype(int)
    X = X.apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    y = y.reset_index(drop=True)
    return X, y, {
        "source_file": str(path),
        "raw_rows": int(len(raw)),
        "unique_patient_rows": int(len(data)),
        "death_hospice_rows_before_filter": competing_event_rows,
        "death_hospice_excluded": bool(exclude_death_hospice),
        "death_hospice_dispositions": sorted(excluded_dispositions),
    }


def _profile_for(key: str) -> dict[str, PseulFeatureProfile]:
    if key == "adherence":
        return adherence_profile()
    if key == "nhanes":
        return nhanes_profile()
    if key == "brfss":
        profile = brfss_profile()
        high = PseulFeatureProfile(
            evidence=0.95,
            intervention_availability=0.05,
            risk_stratification=0.95,
            operational_feasibility=0.95,
            label_derived_risk=1.0,
            definitional_overlap=1.0,
            target_proxy_strength=1.0,
            topic_similarity=0.95,
            interpretation="Question-observation indicator is determined by the diabetes-contingent skip pattern.",
        )
        profile["insulin_use_observed"] = high
        profile["diabetes_age_observed"] = high
        return profile

    profile = diabetes130_profile()
    profile["admission_type_id"] = PseulFeatureProfile(
        0.55, 0.30, 0.55, 0.80, 0.0, 0.0, 0.10, 0.50,
        "Admission type is available at the admission landmark.",
    )
    if key == "diabetes130_admission":
        for feature in STUDY_SPECS[key].canonical_traps:
            base = profile[feature]
            profile[feature] = replace(
                base,
                label_derived_risk=max(0.85, base.label_derived_risk),
                target_proxy_strength=max(0.85, base.target_proxy_strength),
                intervention_availability=0.0,
                interpretation=f"Unavailable at admission landmark. {base.interpretation}",
                available_at_prediction=False,
            )
        for feature in ("number_outpatient", "number_emergency", "number_inpatient"):
            base = profile[feature]
            profile[feature] = replace(
                base,
                label_derived_risk=0.0,
                definitional_overlap=0.0,
                target_proxy_strength=min(base.target_proxy_strength, 0.35),
                interpretation=f"Prior-year history available at admission. {base.interpretation}",
                available_at_prediction=True,
            )
        return profile

    for feature, base in list(profile.items()):
        profile[feature] = replace(base, available_at_prediction=True)
    for feature in ("number_inpatient", "number_diagnoses", "discharge_disposition_id"):
        base = profile[feature]
        profile[feature] = replace(
            base,
            label_derived_risk=0.0,
            definitional_overlap=0.0,
            target_proxy_strength=min(base.target_proxy_strength, 0.35),
            interpretation=f"Available by discharge after death/hospice exclusions. {base.interpretation}",
        )
    return profile


def load_study(key: str) -> tuple[pd.DataFrame, pd.Series, dict, dict[str, PseulFeatureProfile]]:
    if key not in STUDY_SPECS:
        raise KeyError(f"Unknown study key: {key}")
    if key == "adherence":
        X, y, metadata = _load_adherence()
    elif key == "nhanes":
        X, y, metadata = _load_nhanes()
    elif key == "brfss":
        X, y, metadata = _load_brfss()
    else:
        X, y, metadata = _load_diabetes130(exclude_death_hospice=(key == "diabetes130_discharge"))
    metadata.update(asdict(STUDY_SPECS[key]))
    metadata.update(
        rows=int(len(X)),
        features=int(X.shape[1]),
        positive_rate=float(y.mean()),
        missing_cells=int(X.isna().sum().sum()),
    )
    return X, y, metadata, _profile_for(key)


def write_registry_artifacts(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [asdict(spec) for spec in STUDY_SPECS.values()]
    (output_dir / "study_registry.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    trap_rows = []
    for spec in STUDY_SPECS.values():
        if spec.canonical_traps:
            for feature in spec.canonical_traps:
                trap_rows.append(
                    {
                        "study": spec.key,
                        "feature": feature,
                        "mechanism": spec.leakage_mechanism,
                        "landmark": spec.prediction_landmark,
                        "status": spec.evidence_status,
                    }
                )
        else:
            trap_rows.append(
                {
                    "study": spec.key,
                    "feature": "[none designated]",
                    "mechanism": spec.leakage_mechanism,
                    "landmark": spec.prediction_landmark,
                    "status": spec.evidence_status,
                }
            )
    pd.DataFrame(trap_rows).to_csv(output_dir / "canonical_trap_registry.csv", index=False)


__all__ = ["STUDY_SPECS", "StudySpec", "load_study", "write_registry_artifacts"]
