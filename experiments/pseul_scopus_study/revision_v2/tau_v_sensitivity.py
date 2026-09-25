# -*- coding: utf-8 -*-
"""How much rating error does the semantic veto tolerate before the contract changes?

The paper concedes that exclusion follows deterministically from author-supplied
ratings. A reviewer pointed out that the concession stops one step short: nobody
has measured how far those ratings can move before the selected subset does.

The veto is a threshold test, max(LD, DO, TP) >= tau_V, so which features it
excludes is exact arithmetic on the supplied ratings -- no refitting needed to
answer the first half of the question. This pass sweeps tau_V across every
scenario and reports, for each threshold, which features are admissible and
whether the designated traps are among them. It also reports the margin: the
distance from each rating to the threshold, which is what "tolerates" means here.

Refitting is only needed where the admissible set actually changes, and this
prints exactly which (study, tau_V) pairs those are, so the expensive step can be
scoped rather than run blind across 35 combinations.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from study_registry import STUDY_SPECS, _profile_for  # noqa: E402

STUDIES = ["adherence", "nhanes", "brfss",
           "diabetes130_admission", "diabetes130_discharge"]
THRESHOLDS = [0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
LOCKED = 0.85


def risk(p) -> float:
    return max(p.label_derived_risk, p.definitional_overlap, p.target_proxy_strength)


def main() -> None:
    rows = []
    for study in STUDIES:
        spec = STUDY_SPECS[study]
        profile = _profile_for(study)
        traps = set(spec.canonical_traps)
        risks = {f: risk(p) for f, p in profile.items()}

        for tau in THRESHOLDS:
            vetoed = {f for f, r in risks.items() if r >= tau}
            rows.append({
                "study": study,
                "tau_v": tau,
                "n_vetoed": len(vetoed),
                "traps_vetoed": len(vetoed & traps),
                "n_traps": len(traps),
                "non_traps_vetoed": len(vetoed - traps),
                "vetoed_non_traps": ", ".join(sorted(vetoed - traps)) or "none",
                "traps_still_admissible": ", ".join(sorted(traps - vetoed)) or "none",
            })

    df = pd.DataFrame(rows)
    out = HERE / "outputs" / "tau_v_sensitivity.csv"
    df.to_csv(out, index=False)

    print("=== admissible-set response to the veto threshold ===")
    for study in STUDIES:
        d = df[df.study == study]
        base = d[d.tau_v == LOCKED].iloc[0]
        changes = d[(d.n_vetoed != base.n_vetoed)]
        print(f"\n  {study}  ({base.n_traps} designated trap(s))")
        for _, r in d.iterrows():
            flag = "" if r.n_vetoed == base.n_vetoed else "   <-- differs from the locked run"
            print(f"    tau_V={r.tau_v:.2f}  vetoed={r.n_vetoed:2d} "
                  f"(traps {r.traps_vetoed}/{r.n_traps}, non-traps {r.non_traps_vetoed}){flag}")
        if len(changes):
            lo = changes.tau_v.min()
            hi = changes.tau_v.max()
            print(f"    the selected subset can only change outside "
                  f"[{d[d.n_vetoed == base.n_vetoed].tau_v.min():.2f}, "
                  f"{d[d.n_vetoed == base.n_vetoed].tau_v.max():.2f}]")
        else:
            print("    no threshold in the sweep changes the admissible set")

    print("\n=== margins: how far each rating sits from the locked threshold ===")
    for study in STUDIES:
        profile = _profile_for(study)
        traps = set(STUDY_SPECS[study].canonical_traps)
        rs = {f: risk(p) for f, p in profile.items()}
        trap_min = min((rs[f] for f in traps), default=float("nan"))
        non_trap_max = max((r for f, r in rs.items() if f not in traps), default=0.0)
        print(f"  {study:24s} lowest trap {trap_min:.2f}, highest non-trap {non_trap_max:.2f}, "
              f"gap {trap_min - non_trap_max:+.2f}")

    print("\n=== (study, tau_V) pairs that would need a refit ===")
    need = df[df.apply(lambda r: r.n_vetoed != df[(df.study == r.study)
                                                  & (df.tau_v == LOCKED)].n_vetoed.iloc[0], axis=1)]
    if len(need):
        for _, r in need.iterrows():
            print(f"  {r.study:24s} tau_V={r.tau_v:.2f}  "
                  f"vetoed set differs by {abs(r.n_vetoed - df[(df.study == r.study) & (df.tau_v == LOCKED)].n_vetoed.iloc[0])} feature(s)")
    else:
        print("  none")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
