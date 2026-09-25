"""Recompute the main 8-method statistical tests for every dataset.

Why this exists: the original `run_statistics` in run_pseul_adherence_study.py
hard-codes the PSEUL column as "PSEUL Select Top-5". NHANES and BRFSS use
Top-8, so the `pairwise_vs_pseul_select` block silently came out empty for
those two datasets (Friedman ran, but no Wilcoxon pairwise). This script
reads each dataset's already-saved `cv_fold_results.csv` (no model retraining
needed) and rebuilds a complete `statistical_tests.json`:

  - Friedman omnibus on 10-fold CV accuracy across all 8 methods
  - mean ranks per method
  - Wilcoxon signed-rank pairwise vs the auto-detected PSEUL-Select column
  - rank-biserial effect size + Holm-corrected p-values for the pairwise set

It writes back into each dataset's output folder and also emits a combined
`analysis/cross_dataset_statistics.json` for the manuscript stats table.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

STUDY = Path(__file__).resolve().parents[1]
ANALYSIS = Path(__file__).resolve().parent

DATASETS = {
    "adherence": STUDY / "outputs",
    "nhanes": STUDY / "outputs_nhanes_diabetes",
    "brfss": STUDY / "outputs_brfss_diabetes",
    "diabetes130": STUDY / "outputs_diabetes130",
}


def detect_pseul_select(methods: list[str]) -> str:
    for m in methods:
        if m.replace("-", " ").startswith("PSEUL Select"):
            return m
    raise ValueError(f"No PSEUL Select column found in {methods}")


def compute_stats(fold_df: pd.DataFrame, metric: str = "accuracy") -> dict:
    methods = list(dict.fromkeys(fold_df["method"]))  # preserve order
    wide = fold_df.pivot(index="fold", columns="method", values=metric)[methods]

    friedman = stats.friedmanchisquare(*[wide[m].to_numpy() for m in methods])
    mean_ranks = wide.rank(axis=1, ascending=False).mean().to_dict()

    pseul_col = detect_pseul_select(methods)
    others = [m for m in methods if m != pseul_col]
    pairwise: dict[str, dict] = {}
    p_values: list[float] = []
    n = len(wide)

    for method in others:
        diff = (wide[pseul_col] - wide[method])
        try:
            w = stats.wilcoxon(wide[pseul_col], wide[method], zero_method="wilcox")
            wstat, pval = float(w.statistic), float(w.pvalue)
            # rank-biserial effect size for Wilcoxon signed-rank
            effect = 1.0 - (2.0 * wstat) / (n * (n + 1) / 2.0) if n > 0 else float("nan")
        except ValueError:
            wstat, pval, effect = float("nan"), 1.0, float("nan")
        pairwise[method] = {
            "wilcoxon_stat": wstat,
            "p_value": pval,
            "rank_biserial_effect": effect,
            "mean_accuracy_delta_pseul_minus_method": float(diff.mean()),
        }
        p_values.append(pval if not np.isnan(pval) else 1.0)

    if p_values:
        reject, p_adj, _, _ = multipletests(p_values, alpha=0.05, method="holm")
        for method, adj, rej in zip(others, p_adj, reject):
            pairwise[method]["p_value_holm"] = float(adj)
            pairwise[method]["significant_after_holm"] = bool(rej)

    return {
        "metric": f"cv_{metric}",
        "n_folds": int(n),
        "methods": methods,
        "pseul_select_column": pseul_col,
        "friedman_chi2": float(friedman.statistic),
        "friedman_p": float(friedman.pvalue),
        "mean_ranks": mean_ranks,
        "pairwise_vs_pseul_select": pairwise,
    }


def main() -> None:
    combined = {}
    for name, outdir in DATASETS.items():
        fold_path = outdir / "cv_fold_results.csv"
        if not fold_path.exists():
            print(f"[skip] {name}: {fold_path} not found")
            continue
        fold_df = pd.read_csv(fold_path)
        result = compute_stats(fold_df, metric="accuracy")
        (outdir / "statistical_tests.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        combined[name] = result

        print(f"\n=== {name} ===")
        print(f"Friedman chi2={result['friedman_chi2']:.4f}, p={result['friedman_p']:.4g}")
        print(f"PSEUL column: {result['pseul_select_column']}")
        for method, res in result["pairwise_vs_pseul_select"].items():
            print(f"  vs {method:<28} p={res['p_value']:.4g}  "
                  f"p_holm={res['p_value_holm']:.4g}  "
                  f"eff={res['rank_biserial_effect']:+.3f}  "
                  f"sig={res['significant_after_holm']}")

    (ANALYSIS / "cross_dataset_statistics.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    print(f"\nSaved combined stats: {ANALYSIS / 'cross_dataset_statistics.json'}")


if __name__ == "__main__":
    main()
