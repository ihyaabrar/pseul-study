# -*- coding: utf-8 -*-
"""Refit PSEUL-Select at the veto thresholds that change the admissible set.

tau_v_sensitivity.py establishes analytically where the threshold matters: in
three of the five scenarios no value between 0.60 and 0.95 changes which features
are admissible, so no refit can change the result there. This runs the outer-fold
evaluation only at the (study, tau_V) pairs where the admissible set does move,
which is four distinct configurations rather than the thirty-five a blind sweep
would need.

It reproduces the main pipeline's protocol -- the same 80/20 split at seed 42,
five stratified outer folds, fold-local median imputation, three inner folds,
the same locked LightGBM -- so the numbers are comparable with Table 2.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from scripts.pseul import PSEUL, PseulConfig  # noqa: E402
from experiments.pseul_scopus_study.revision_v2.nested_evaluation import (  # noqa: E402
    FrameMedianImputer, RANDOM_STATE, make_model,
)
from experiments.pseul_scopus_study.revision_v2.study_registry import (  # noqa: E402
    STUDY_SPECS, load_study,
)

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"

# from the analytic pass: the pairs whose admissible set differs from the locked run
CASES = [
    ("adherence", 0.75),   # the retained proxy becomes inadmissible
    ("adherence", 0.95),   # the vetoed proxy becomes admissible
    ("nhanes", 0.60),      # prediabetes_told becomes inadmissible
    ("diabetes130_admission", 0.90),  # the veto switches off entirely
]


def run_one(study: str, tau: float) -> dict:
    spec = STUDY_SPECS[study]
    X, y, _, profile = load_study(study)
    X.index = np.arange(len(X))
    y.index = X.index
    dev_ids, _ = train_test_split(
        X.index.to_numpy(), test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    Xd, yd = X.loc[dev_ids], y.loc[dev_ids]
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    oof = np.full(len(Xd), np.nan)
    traps_seen: list[set[str]] = []
    for fold, (tr, va) in enumerate(folds.split(Xd, yd), start=1):
        tr_ids, va_ids = Xd.index.to_numpy()[tr], Xd.index.to_numpy()[va]
        imp = FrameMedianImputer().fit(Xd.loc[tr_ids])
        Xtr = imp.transform(Xd.loc[tr_ids]).reset_index(drop=True)
        Xva = imp.transform(Xd.loc[va_ids]).reset_index(drop=True)
        ytr = yd.loc[tr_ids].reset_index(drop=True)

        cfg = PseulConfig(
            n_splits=3, random_state=RANDOM_STATE + fold * 101, top_k=spec.top_k,
            max_shap_samples_per_fold=500, permutation_repeats=5,
            utility_method="permutation", semantic_veto_threshold=tau,
        )
        sel = PSEUL(estimator=make_model(RANDOM_STATE + fold * 101),
                    clinical_profile=profile, config=cfg)
        sel.fit(Xtr, ytr)
        chosen = list(sel.selected_features_)
        traps_seen.append(set(chosen) & set(spec.canonical_traps))

        model = make_model(RANDOM_STATE + fold * 101)
        model.fit(Xtr[chosen], ytr)
        oof[va] = model.predict_proba(Xva[chosen])[:, 1]

    n_traps = len(spec.canonical_traps)
    freq = (np.mean([len(t) for t in traps_seen]) / n_traps) if n_traps else float("nan")
    return {
        "study": study, "tau_v": tau,
        "auc": roc_auc_score(yd.to_numpy(), oof),
        "mean_trap_selection_frequency": freq,
        "traps_ever_selected": ", ".join(sorted(set().union(*traps_seen))) or "none",
    }


def main() -> None:
    locked = pd.read_csv(OUT / "combined_pooled_oof_metrics.csv").set_index(["study", "method"])
    traps = pd.read_csv(OUT / "combined_trap_performance.csv").set_index(["study", "method"])

    # write after every case: an earlier run finished all four fits and then lost
    # the summary, so nothing survived a forty-minute computation
    out = OUT / "tau_v_refit.csv"
    rows = []
    for study, tau in CASES:
        print(f"[tau_V] {study} at {tau:.2f} ...", flush=True)
        r = run_one(study, tau)
        r["locked_auc"] = float(locked.auc[(study, "PSEUL-Select")])
        r["locked_trap_freq"] = float(traps.mean_trap_selection_frequency[(study, "PSEUL-Select")])
        rows.append(r)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"    AUC {r['auc']:.4f} vs locked {r['locked_auc']:.4f}, "
              f"traps {r['mean_trap_selection_frequency']:.2f}", flush=True)

    df = pd.DataFrame(rows)
    df["delta_auc"] = df.auc - df.locked_auc
    df.to_csv(out, index=False)

    print("\n=== PSEUL-Select refitted where the veto threshold changes the admissible set ===")
    for _, r in df.iterrows():
        print(f"  {r.study:24s} tau_V={r.tau_v:.2f}  AUC {r.auc:.4f} "
              f"(locked {r.locked_auc:.4f}, delta {r.delta_auc:+.4f})  "
              f"traps {r.mean_trap_selection_frequency:.2f} (locked {r.locked_trap_freq:.2f})")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
