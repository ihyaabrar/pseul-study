from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .nested_evaluation import (
    _selection_summary,
    calibration_parameters,
    metric_row,
    oof_metric_intervals,
    paired_bootstrap,
    selection_stability,
)
from .study_registry import STUDY_SPECS


ROOT = Path(__file__).resolve().parent
METHOD_RENAMES = {"Manual Exclusion + SHAP": "Oracle Trap Exclusion + SHAP"}


def _canonicalize_method_names(output_dir: Path) -> None:
    for path in output_dir.glob("*.csv"):
        frame = pd.read_csv(path)
        changed = False
        for column in ("method", "reference", "comparator"):
            if column in frame.columns:
                updated = frame[column].replace(METHOD_RENAMES)
                changed = changed or not updated.equals(frame[column])
                frame[column] = updated
        if changed:
            frame.to_csv(path, index=False)


def _enrich_nested(output_dir: Path, traps: tuple[str, ...]) -> None:
    selections = pd.read_csv(output_dir / "selected_features_by_fold.csv")
    scores = pd.read_csv(output_dir / "pseul_scores_by_fold.csv")
    summaries = []
    for row in selections.itertuples(index=False):
        fold_scores = scores[scores["fold"] == row.fold]
        summary = _selection_summary(json.loads(row.features), fold_scores, traps)
        summaries.append({"study": row.study, "fold": row.fold, "method": row.method, **summary})
    summary_frame = pd.DataFrame(summaries)
    base_columns = [
        column
        for column in selections.columns
        if column not in summary_frame.columns or column in {"study", "fold", "method"}
    ]
    selections = selections[base_columns].merge(summary_frame, on=["study", "fold", "method"], how="left")
    selections.to_csv(output_dir / "selected_features_by_fold.csv", index=False)

    metrics = pd.read_csv(output_dir / "nested_fold_metrics.csv")
    drop = [column for column in summary_frame.columns if column not in {"study", "fold", "method"}]
    metrics = metrics.drop(columns=[column for column in drop if column in metrics.columns]).merge(
        summary_frame, on=["study", "fold", "method"], how="left"
    )
    metrics.to_csv(output_dir / "nested_fold_metrics.csv", index=False)

    stability, frequency = selection_stability(selections, traps)
    stability.to_csv(output_dir / "selection_stability.csv", index=False)
    frequency.to_csv(output_dir / "feature_selection_frequency.csv", index=False)


def _enrich_final(output_dir: Path, traps: tuple[str, ...]) -> None:
    selection_path = output_dir / "final_selected_features.csv"
    score_path = output_dir / "final_pseul_scores.csv"
    metric_path = output_dir / "locked_test_metrics.csv"
    if not (selection_path.exists() and score_path.exists()):
        return
    selections = pd.read_csv(selection_path)
    scores = pd.read_csv(score_path)
    summaries = []
    for row in selections.itertuples(index=False):
        summaries.append({"method": row.method, **_selection_summary(json.loads(row.features), scores, traps)})
    summary_frame = pd.DataFrame(summaries)
    drop = [column for column in summary_frame.columns if column != "method"]
    selections = selections.drop(columns=[column for column in drop if column in selections.columns]).merge(
        summary_frame, on="method", how="left"
    )
    selections.to_csv(selection_path, index=False)
    if metric_path.exists():
        metrics = pd.read_csv(metric_path)
        metrics = metrics.drop(columns=[column for column in drop if column in metrics.columns]).merge(
            summary_frame, on="method", how="left"
        )
        metrics.to_csv(metric_path, index=False)


def postprocess(study: str, run_label: str, n_bootstrap: int) -> Path:
    output_dir = ROOT / "outputs" / study / run_label
    if not output_dir.exists():
        raise FileNotFoundError(output_dir)
    _canonicalize_method_names(output_dir)
    traps = STUDY_SPECS[study].canonical_traps
    predictions = pd.read_csv(output_dir / "oof_predictions.csv")
    pooled_rows = []
    for method, frame in predictions.groupby("method"):
        row = {"study": study, "method": method, **metric_row(frame["y_true"], frame["probability"].to_numpy())}
        intercept, slope = calibration_parameters(frame["y_true"], frame["probability"].to_numpy())
        row.update(calibration_intercept=intercept, calibration_slope=slope)
        pooled_rows.append(row)
    pd.DataFrame(pooled_rows).to_csv(output_dir / "pooled_oof_metrics.csv", index=False)
    paired_bootstrap(predictions, n_bootstrap=n_bootstrap).to_csv(
        output_dir / "paired_oof_inference.csv", index=False
    )
    oof_metric_intervals(predictions, n_bootstrap=n_bootstrap).to_csv(
        output_dir / "oof_metric_intervals.csv", index=False
    )
    _enrich_nested(output_dir, traps)
    _enrich_final(output_dir, traps)
    payload = {
        "study": study,
        "run_label": run_label,
        "auc_inference": "paired DeLong",
        "mcc_inference": "paired stratified multinomial bootstrap",
        "mcc_bootstrap_replicates": n_bootstrap,
        "multiplicity": "Holm correction within metric across PSEUL-Select comparisons",
    }
    (output_dir / "postprocessing_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"postprocessed -> {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh revision-v2 inference and selection audit files.")
    parser.add_argument("study", choices=sorted(STUDY_SPECS))
    parser.add_argument("--run-label", default="full")
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    postprocess(args.study, args.run_label, args.bootstrap)


if __name__ == "__main__":
    main()
