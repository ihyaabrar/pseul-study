# -*- coding: utf-8 -*-
"""Tables requested by the October 2026 review council, from existing outputs.

  supplementary_internal_test.csv       K5: the reserved internal test per scenario
  supplementary_component_activity.csv  K7: which PSEUL-Select components acted
  supplementary_council_d130.csv        K3/K4: gate x veto factorial and common cohort
                                        (written once council_revision_runs.py has run)

Nothing is re-estimated here; every value is read from revision_v2/outputs.

    python -m experiments.pseul_scopus_study.revision_v2.council_tables
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.pseul import PseulConfig

from .study_registry import STUDY_SPECS

OUT = Path(__file__).resolve().parent / "outputs"
STUDIES = ["adherence", "nhanes", "brfss", "diabetes130_admission", "diabetes130_discharge"]
METHODS = ["All Features", "Availability Gate + SHAP", "Semantic Veto + SHAP",
           "Oracle Trap Exclusion + SHAP", "PSEUL-Select"]
# "Semantic Veto + SHAP" ranks only features that pass the availability gate first
RENAME = {"Semantic Veto + SHAP": "Gate + Veto + SHAP"}


def internal_test() -> pd.DataFrame:
    t = pd.read_csv(OUT / "combined_locked_test_metrics.csv")
    t = t[t.method.isin(METHODS)].copy()
    t["method"] = t.method.replace(RENAME)
    order = {m: i for i, m in enumerate(RENAME.get(m, m) for m in METHODS)}
    t = t.sort_values(["study", "method"], key=lambda s: s.map(order) if s.name == "method" else s.map(STUDIES.index))
    cols = ["study", "method", "auc", "average_precision", "mcc", "n_features", "designated_traps_selected"]
    return t[cols].round(3)


def component_activity() -> pd.DataFrame:
    cfg = PseulConfig()
    rows = []
    for study in STUDIES:
        s = pd.read_csv(OUT / study / "full" / "pseul_scores_by_fold.csv")
        top_k = STUDY_SPECS[study].top_k
        folds = sorted(s.fold.unique())
        gate_q = 1.0 - 1.0 / (1.0 + np.exp(-cfg.tau_l * (s.leakage_risk - s.theta_l_mean)))
        g = np.minimum(1.0, s.clinical_utility / cfg.theta_u)
        adm = s.admissible_for_selection.astype(bool)
        redundancy_changed = 0
        for f in folds:
            fold = s[s.fold == f]
            by_pre = set(fold[fold.admissible_for_selection.astype(bool)].nlargest(top_k, "pre_score").feature)
            redundancy_changed += by_pre != set(fold[fold.selected.astype(bool)].feature)
        per_fold = len(folds)
        rows.append({
            "study": study,
            "candidates": s.feature.nunique(),
            "excluded_by_landmark_gate": int((s.rejection_reason == "post-landmark").sum() / per_fold),
            "excluded_by_semantic_veto": int((s.rejection_reason == "semantic-veto").sum() / per_fold),
            "vetoable_but_already_gated": int(((s.rejection_reason == "post-landmark") & s.semantic_veto.astype(bool)).sum() / per_fold),
            "utility_gate_below_1": int((g[adm] < 1).sum()),
            "admissible_with_penalty_below_0_5": int((gate_q[adm] < 0.5).sum()),
            "min_penalty_admissible": round(float(gate_q[adm].min()), 3),
            "score_floor_stopped_selection_folds": int(sum(s[s.fold == f].selected.sum() < top_k for f in folds)),
            "redundancy_changed_subset_folds": int(redundancy_changed),
            "uncertainty_weight": cfg.uncertainty_weight,
        })
    return pd.DataFrame(rows)


def council_d130() -> pd.DataFrame | None:
    runs = {
        "both controls (primary)": OUT / "diabetes130_admission" / "full",
        "landmark gate only": OUT / "diabetes130_admission" / "factorial_gate_only",
        "semantic veto only": OUT / "diabetes130_admission" / "factorial_veto_only",
        "neither control": OUT / "diabetes130_admission" / "factorial_neither",
        "admission registry, discharge cohort": OUT / "diabetes130_admission" / "common_cohort",
        "discharge registry, discharge cohort (primary)": OUT / "diabetes130_discharge" / "full",
    }
    if not all((p / "pooled_oof_metrics.csv").exists() for p in runs.values()):
        return None
    rows = []
    for label, path in runs.items():
        pooled = pd.read_csv(path / "pooled_oof_metrics.csv").set_index("method")
        folds = pd.read_csv(path / "nested_fold_metrics.csv").groupby("method")
        for method in ("All Features", "Availability Gate + SHAP", "Semantic Veto + SHAP", "PSEUL-Select"):
            if method not in pooled.index:
                continue
            rows.append({
                "run": label, "method": RENAME.get(method, method),
                "oof_auc": round(float(pooled.loc[method, "auc"]), 3),
                "oof_mcc": round(float(pooled.loc[method, "mcc"]), 3),
                "traps_selected_per_fold": round(float(folds.get_group(method)["designated_traps_selected"].mean()), 2),
            })
    return pd.DataFrame(rows)


def main() -> None:
    tables = OUT / "tables"
    internal_test().to_csv(tables / "supplementary_internal_test.csv", index=False)
    component_activity().to_csv(tables / "supplementary_component_activity.csv", index=False)
    d130 = council_d130()
    if d130 is not None:
        d130.to_csv(tables / "supplementary_council_d130.csv", index=False)
    print("written to", tables, "| D130 council table:", "yes" if d130 is not None else "waiting for runs")


if __name__ == "__main__":
    main()
