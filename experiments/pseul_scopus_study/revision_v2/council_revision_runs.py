# -*- coding: utf-8 -*-
"""Diabetes 130 runs requested by the October 2026 review council.

The admission registry marks its eleven designated traps both unavailable at the
landmark and semantically vetoed, so the primary run cannot tell which control
removed what. Three registry variants separate them (review item K3):

  factorial_gate_only   traps unavailable, semantic ratings at their base values
  factorial_veto_only   traps available, semantic ratings raised to the veto
  factorial_neither     traps available, semantic ratings at their base values

The primary admission run is the fourth cell (both controls).

The admission and discharge scenarios also differ in eligibility: admission keeps
the 862 death/hospice encounters that discharge excludes. common_cohort applies
the admission registry to the discharge cohort, so with the same seed both
scenarios sample, split and fold identical records (review items K4 and M9).

Every run goes through nested_evaluation.run_study unchanged; only the registry
or the cohort passed to it differs.

    python -m experiments.pseul_scopus_study.revision_v2.council_revision_runs
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from . import nested_evaluation as ne
from . import study_registry as sr

STUDY = "diabetes130_admission"
TRAPS = sr.STUDY_SPECS[STUDY].canonical_traps
OUT = Path(__file__).resolve().parent / "outputs" / STUDY


def _variant_profile(gate: bool, veto: bool) -> dict:
    admission = sr._profile_for(STUDY)
    base = sr.diabetes130_profile()
    profile = dict(admission)
    for feature in TRAPS:
        p = admission[feature]
        if not veto:
            p = replace(p, label_derived_risk=base[feature].label_derived_risk,
                        target_proxy_strength=base[feature].target_proxy_strength)
        if not gate:
            # the admission registry also zeroes intervention availability for these;
            # an available feature gets its base value back
            p = replace(p, available_at_prediction=True,
                        intervention_availability=base[feature].intervention_availability)
        profile[feature] = p
    return profile


def _run(label: str, loader) -> Path:
    if (OUT / label / "pipeline_audit.json").exists():
        print(f"[{label}] already complete, skipped", flush=True)
        return OUT / label
    original = ne.load_study
    ne.load_study = loader
    try:
        return ne.run_study(STUDY, outer_splits=5, inner_splits=3, test_size=0.2, evaluate_test=True,
                            include_proxies=False, include_auto=False, n_bootstrap=2000,
                            max_rows=40000, run_label=label)
    finally:
        ne.load_study = original


def _with_profile(profile: dict):
    def loader(key):
        X, y, metadata, _ = sr.load_study(key)
        metadata["registry_variant"] = "council factorial"
        return X, y, metadata, profile
    return loader


def _common_cohort(key):
    X, y, metadata = sr._load_diabetes130(exclude_death_hospice=True)
    metadata.update(rows=int(len(X)), features=int(X.shape[1]), positive_rate=float(y.mean()),
                    cohort="discharge eligibility (death/hospice excluded)")
    return X, y, metadata, sr._profile_for(STUDY)


def main() -> None:
    for label, gate, veto in (("factorial_gate_only", True, False),
                              ("factorial_veto_only", False, True),
                              ("factorial_neither", False, False)):
        profile = _variant_profile(gate, veto)
        unavailable = sum(not profile[f].available_at_prediction for f in TRAPS)
        vetoed = sum(max(profile[f].label_derived_risk, profile[f].definitional_overlap,
                         profile[f].target_proxy_strength) >= 0.85 for f in TRAPS)
        print(f"[{label}] traps unavailable={unavailable}/{len(TRAPS)}, at or above veto={vetoed}/{len(TRAPS)}", flush=True)
        _run(label, _with_profile(profile))

    _run("common_cohort", _common_cohort)
    # same records check: the admission run on the discharge cohort must sample and
    # split exactly the rows the primary discharge run used
    common = np.load(OUT / "common_cohort" / "locked_split_indices.npz")
    discharge = np.load(OUT.parent / "diabetes130_discharge" / "full" / "locked_split_indices.npz")
    for part in ("development", "test"):
        assert np.array_equal(np.sort(common[part]), np.sort(discharge[part])), part
    print("common cohort: development and test ids identical to the discharge run", flush=True)


if __name__ == "__main__":
    main()
