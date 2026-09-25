from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .study_registry import STUDY_SPECS


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUTPUT_ROOT = HERE / "outputs"
RESEARCH_REPORT = ROOT / "docs" / "research" / "PSEUL_Revision_v2_Results_Audit.md"


def _markdown_table(frame: pd.DataFrame) -> str:
    def render(value) -> str:
        if pd.isna(value):
            return "—"
        if isinstance(value, float):
            return f"{value:.3f}"
        return str(value).replace("|", "\\|")

    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    divider = "| " + " | ".join("---" for _ in frame.columns) + " |"
    rows = ["| " + " | ".join(render(value) for value in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, divider, *rows])


def summarize() -> Path:
    pooled_frames = []
    test_frames = []
    stability_frames = []
    trap_rows = []
    audit_rows = []
    for study, spec in STUDY_SPECS.items():
        output_dir = OUTPUT_ROOT / study / "full"
        required = [
            "pooled_oof_metrics.csv",
            "locked_test_metrics.csv",
            "selection_stability.csv",
            "feature_selection_frequency.csv",
            "pipeline_audit.json",
            "paired_oof_inference.csv",
        ]
        missing = [name for name in required if not (output_dir / name).exists()]
        if missing:
            raise FileNotFoundError(f"{study}: missing {missing}")

        pooled = pd.read_csv(output_dir / "pooled_oof_metrics.csv")
        pooled.insert(1, "scenario", spec.task_type)
        pooled_frames.append(pooled)
        test = pd.read_csv(output_dir / "locked_test_metrics.csv")
        test.insert(1, "scenario", spec.task_type)
        test_frames.append(test)
        stability = pd.read_csv(output_dir / "selection_stability.csv")
        stability.insert(0, "study", study)
        stability_frames.append(stability)

        frequency = pd.read_csv(output_dir / "feature_selection_frequency.csv")
        for method, frame in frequency.groupby("method"):
            trap_frame = frame[frame["canonical_trap"].astype(bool)]
            trap_rows.append(
                {
                    "study": study,
                    "method": method,
                    "n_designated_traps": len(spec.canonical_traps),
                    "mean_trap_selection_frequency": (
                        float(trap_frame["selection_frequency"].sum() / len(spec.canonical_traps))
                        if spec.canonical_traps
                        else float("nan")
                    ),
                    "traps_ever_selected": ", ".join(trap_frame["feature"].tolist()) or "none",
                }
            )

        audit = json.loads((output_dir / "pipeline_audit.json").read_text(encoding="utf-8"))
        pass_overlap = audit["development_test_overlap"] == 0 and all(
            fold["train_validation_overlap"] == 0 for fold in audit["fold_audits"]
        )
        pass_imputer = all(
            fold["imputer_fit_index_hash"] == fold["outer_train_index_hash"] for fold in audit["fold_audits"]
        )
        audit_rows.append(
            {
                "study": study,
                "rows": audit["metadata"]["rows"],
                "analysis_subsample": audit["metadata"].get("analysis_subsample", audit["metadata"]["rows"]),
                "positive_rate": audit["metadata"]["positive_rate"],
                "outer_splits": audit["outer_splits"],
                "train_validation_isolation": pass_overlap,
                "train_only_imputation": pass_imputer,
                "locked_test_evaluated": audit["test_evaluated"],
            }
        )

    pooled_all = pd.concat(pooled_frames, ignore_index=True)
    test_all = pd.concat(test_frames, ignore_index=True)
    stability_all = pd.concat(stability_frames, ignore_index=True)
    traps_all = pd.DataFrame(trap_rows)
    audit_all = pd.DataFrame(audit_rows)
    pooled_all.to_csv(OUTPUT_ROOT / "combined_pooled_oof_metrics.csv", index=False)
    test_all.to_csv(OUTPUT_ROOT / "combined_locked_test_metrics.csv", index=False)
    stability_all.to_csv(OUTPUT_ROOT / "combined_selection_stability.csv", index=False)
    traps_all.to_csv(OUTPUT_ROOT / "combined_trap_performance.csv", index=False)
    audit_all.to_csv(OUTPUT_ROOT / "combined_pipeline_audit.csv", index=False)

    key_methods = [
        "All Features",
        "PSEUL-Select",
        "PSEUL-Auto (leakage metadata null)",
        "Oracle Trap Exclusion + SHAP",
    ]
    metric_view = pooled_all[pooled_all["method"].isin(key_methods)][
        ["study", "method", "auc", "average_precision", "mcc", "brier", "sensitivity", "specificity"]
    ].copy()
    trap_view = traps_all[traps_all["method"].isin(key_methods)][
        ["study", "method", "n_designated_traps", "mean_trap_selection_frequency", "traps_ever_selected"]
    ].copy()
    stability_view = stability_all[stability_all["method"].isin(key_methods)][
        ["study", "method", "mean_pairwise_jaccard", "min_pairwise_jaccard"]
    ].copy()
    robustness = pd.read_csv(OUTPUT_ROOT / "robustness" / "nhanes_robustness_summary.csv")
    robustness_view = robustness[
        robustness.variant.isin(
            [
                "Reference",
                "Without predictive utility P",
                "Without SHAP stability S",
                "Without evidence E",
                "Without clinical utility U",
                "Without soft leakage penalty",
                "Without redundancy penalty",
                "Without semantic veto",
                "Top-k ceiling 5",
            ]
        )
    ][
        [
            "variant",
            "learner",
            "auc",
            "mcc",
            "brier",
            "mean_selected_features",
            "mean_designated_traps_selected",
            "mean_pairwise_jaccard",
        ]
    ].copy()

    report = [
        "# PSEUL revision-v2 results and pipeline audit",
        "",
        "This report is generated from the corrected nested evaluation. It is the numerical source of truth for the revision-v2 manuscript.",
        "",
        "## Pipeline integrity",
        "",
        _markdown_table(audit_all),
        "",
        "## Pooled out-of-fold performance",
        "",
        _markdown_table(metric_view),
        "",
        "## Designated-trap behaviour",
        "",
        _markdown_table(trap_view),
        "",
        "For `PSEUL-Select`, lower trap frequency is desirable. For `PSEUL-Audit`, higher trap frequency means the diagnostic ranking correctly surfaces suspicious drivers. The oracle comparator knows the trap registry in advance and is not an autonomous baseline.",
        "",
        "## Selection stability",
        "",
        _markdown_table(stability_view),
        "",
        "## Development-only NHANES robustness",
        "",
        _markdown_table(robustness_view),
        "",
        "The one-factor variants reuse identical fitted fold-level signals. Logistic regression is a downstream learner sensitivity check; selection signals remain LightGBM-derived.",
        "",
        "## Locked interpretation constraints",
        "",
        "- NHANES and BRFSS are cross-sectional leakage stress tests, not prospective clinical prediction studies.",
        "- Medication-adherence traps are provisional until a temporal data dictionary is available.",
        "- Diabetes 130 admission and discharge are separately sampled landmark-specific estimands with a common top-k ceiling; their numerical difference must not be attributed to landmark alone because eligibility differs.",
        "- PSEUL-Auto is a leakage-metadata-null control. It uses the identical seed/folds and retains evidence, utility, topic, and availability metadata; only label-derived, definitional-overlap, and target-proxy ratings are nulled.",
        "- Manual PSEUL scores remain author-provisional until at least two independent blinded domain raters complete the v2 scoring protocol.",
        "- Fold-level Wilcoxon/Friedman tests are descriptive only. Primary paired inference uses patient-level paired DeLong tests for AUC and paired stratified bootstrap for MCC, with Holm correction.",
        "- A method with inadequate discrimination or sensitivity must be reported as 'no safe useful model' rather than reframed as deployable.",
        "",
    ]
    RESEARCH_REPORT.write_text("\n".join(report), encoding="utf-8")
    print(f"summary -> {RESEARCH_REPORT}")
    return RESEARCH_REPORT


if __name__ == "__main__":
    summarize()
