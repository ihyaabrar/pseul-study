# PSEUL expert-scoring protocol v2

## Purpose

Make expert-assigned PSEUL components reproducible and separate domain judgment from observed model performance.

## Required raters

- at least two independent domain raters;
- blinded to model importance, AUC, selected subsets, and designated trap outcomes;
- conflicts adjudicated only after independent forms are locked.

## Anchored scale

Use only `0.00`, `0.25`, `0.50`, `0.75`, or `1.00` unless a written exception is supplied.

| Score | Interpretation |
|---:|---|
| 0.00 | No evidence for the construct |
| 0.25 | Weak or indirect evidence |
| 0.50 | Plausible/moderate evidence |
| 0.75 | Strong evidence |
| 1.00 | Direct, defining, or near-certain evidence |

## Per-feature fields

Each rater records: study, landmark, feature, evidence relevance, intervention availability, risk-stratification relevance, operational feasibility, label-derived risk, definitional overlap, target-proxy strength, topic similarity, available at prediction time, evidence citation, rationale, and confidence.

## Locking and analysis

1. Freeze target, landmark, horizon, and feature dictionary.
2. Collect independent ratings without performance outputs.
3. Hash and lock completed forms.
4. Compute agreement per component using ICC for scaled ratings and kappa for availability.
5. Adjudicate disagreements and retain both original ratings.
6. Run the primary analysis with adjudicated scores and sensitivity analyses with each rater separately.

PSEUL-Auto is an algorithmic sensitivity analysis and must not be reported as inter-rater validation.
