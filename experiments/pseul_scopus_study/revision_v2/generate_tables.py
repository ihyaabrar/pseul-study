from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .study_registry import STUDY_SPECS


HERE = Path(__file__).resolve().parent
OUTPUT_ROOT = HERE / "outputs"
TABLE_DIR = OUTPUT_ROOT / "tables"

DISPLAY_METHODS = [
    "All Features",
    "Mutual Information",
    "LightGBM Gain",
    "Permutation Importance",
    "SHAP Ranking",
    "PSEUL-Audit",
    "PSEUL-Select",
    "Availability Gate + SHAP",
    "Semantic Veto + SHAP",
    "PSEUL-Auto (leakage metadata null)",
    "Oracle Trap Exclusion + SHAP",
]


def _round(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.select_dtypes(include="number"):
        result[column] = result[column].round(3)
    return result


def main() -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(OUTPUT_ROOT / "combined_pipeline_audit.csv")
    pooled = pd.read_csv(OUTPUT_ROOT / "combined_pooled_oof_metrics.csv")
    test = pd.read_csv(OUTPUT_ROOT / "combined_locked_test_metrics.csv")
    stability = pd.read_csv(OUTPUT_ROOT / "combined_selection_stability.csv")
    traps = pd.read_csv(OUTPUT_ROOT / "combined_trap_performance.csv")

    study_rows = []
    for key, spec in STUDY_SPECS.items():
        record = audit[audit.study.eq(key)].iloc[0]
        study_rows.append(
            {
                "Scenario": key,
                "Task": spec.task_type,
                "Prediction landmark": spec.prediction_landmark,
                "Analysis n": int(record.analysis_subsample),
                "Positive rate": record.positive_rate,
                "Candidate features": int(json.loads((OUTPUT_ROOT / key / "full" / "pipeline_audit.json").read_text())["metadata"]["features"]),
                "Designated traps": len(spec.canonical_traps),
            }
        )
    table1 = _round(pd.DataFrame(study_rows))
    table1.to_csv(TABLE_DIR / "table01_study_scenarios.csv", index=False)

    table2 = pd.DataFrame(
        [
            ("All Features", "Unfiltered comparator", "No availability or trap control"),
            ("Mutual Information", "Training-only filter ranking", "Top-k marginal mutual information"),
            ("LightGBM Gain", "Training-only embedded ranking", "Top-k split gain"),
            ("Permutation Importance", "Training-only wrapper ranking", "Internal selector split; top-k AUC drop"),
            ("SHAP Ranking", "Training-only explanation ranking", "Top-k mean absolute TreeSHAP"),
            ("PSEUL-Audit", "Diagnostic output", "Ranks value without deployment leakage gate"),
            ("Availability Gate + SHAP", "Availability ablation", "Top-k SHAP after the hard prediction-time gate"),
            ("Semantic Veto + SHAP", "Gate-plus-veto ablation", "Top-k SHAP after availability and semantic-veto gates"),
            ("PSEUL-Select", "Primary leakage-aware selector", "Availability and semantic vetoes + soft leakage penalty + score floor"),
            ("PSEUL-Auto", "Leakage-metadata-null control", "Same seed/folds and non-leakage metadata; only leakage metadata is nulled"),
            ("Oracle exclusion + SHAP", "Registry-informed ceiling", "Pre-excludes known traps; not autonomous"),
        ],
        columns=["Method", "Role", "Definition"],
    )
    table2.to_csv(TABLE_DIR / "table02_method_definitions.csv", index=False)

    performance_columns = [
        "study",
        "method",
        "auc",
        "average_precision",
        "mcc",
        "brier",
        "calibration_intercept",
        "calibration_slope",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
    ]
    table3 = _round(pooled[pooled.method.isin(DISPLAY_METHODS)][performance_columns])
    table3.to_csv(TABLE_DIR / "table03_pooled_oof_performance.csv", index=False)

    key_methods = [
        "All Features",
        "PSEUL-Select",
        "PSEUL-Auto (leakage metadata null)",
        "Oracle Trap Exclusion + SHAP",
    ]
    table4 = traps[traps.method.isin(key_methods)].merge(
        stability[stability.method.isin(key_methods)], on=["study", "method"], how="left"
    )
    table4 = _round(
        table4[
            [
                "study",
                "method",
                "n_designated_traps",
                "mean_trap_selection_frequency",
                "mean_pairwise_jaccard",
                "min_pairwise_jaccard",
                "traps_ever_selected",
            ]
        ]
    )
    table4.to_csv(TABLE_DIR / "table04_traps_and_stability.csv", index=False)

    inference_frames = []
    for key in STUDY_SPECS:
        frame = pd.read_csv(OUTPUT_ROOT / key / "full" / "paired_oof_inference.csv")
        frame.insert(0, "study", key)
        inference_frames.append(frame)
    all_inference = pd.concat(inference_frames, ignore_index=True)
    all_inference.to_csv(TABLE_DIR / "supplementary_all_paired_inference.csv", index=False)
    table5 = _round(
        all_inference[
            all_inference.comparator.isin(
                ["All Features", "PSEUL-Auto (leakage metadata null)", "Oracle Trap Exclusion + SHAP"]
            )
        ][
            [
                "study",
                "metric",
                "comparator",
                "delta_reference_minus_comparator",
                "ci_low",
                "ci_high",
                "holm_adjusted_p",
                "inference_method",
            ]
        ]
    )
    table5.to_csv(TABLE_DIR / "table05_primary_paired_inference.csv", index=False)

    selected_rows = []
    for key in STUDY_SPECS:
        frame = pd.read_csv(OUTPUT_ROOT / key / "full" / "final_selected_features.csv")
        for method in ("PSEUL-Select", "PSEUL-Auto (leakage metadata null)"):
            row = frame[frame.method.eq(method)].iloc[0]
            selected_rows.append(
                {
                    "study": key,
                    "method": method,
                    "selected_features": ", ".join(json.loads(row.features)),
                    "designated_traps_selected": int(row.designated_traps_selected),
                    "prediction_unavailable_selected": int(row.prediction_unavailable_selected),
                }
            )
    pd.DataFrame(selected_rows).to_csv(TABLE_DIR / "table06_final_selected_features.csv", index=False)

    locked_columns = [
        "study",
        "method",
        "auc",
        "mcc",
        "brier",
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "designated_traps_selected",
        "prediction_unavailable_selected",
    ]
    _round(test[test.method.isin(key_methods)][locked_columns]).to_csv(
        TABLE_DIR / "supplementary_locked_test_metrics.csv", index=False
    )
    print(f"tables -> {TABLE_DIR}")


if __name__ == "__main__":
    main()
