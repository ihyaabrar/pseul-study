# PSEUL revision-v2 clinical estimands

This document freezes the intended use of every revision-v2 task before model reruns. It prevents a feature from being called leakage without reference to a prediction landmark.

| Study | Task | Prediction landmark | Horizon | Leakage interpretation |
|---|---|---|---|---|
| Medication adherence | Retrospective adherence classification | End of the recorded assessment period | None | `UNITSTOTAL` and `ANNUALCLAIMAMOUNT` are concurrent outcome-window proxies; this dataset is not evidence of prospective early warning unless a prior-period window becomes available. |
| NHANES | Cross-sectional diabetes-status classification | Survey/examination assessment | None | HbA1c and fasting glucose are direct components of the operational label; this is a definitional-leakage stress test. |
| BRFSS 2024 | Cross-sectional diabetes-status classification | Survey interview | None | Insulin use, diabetes diagnosis age, and their observation patterns are post-diagnosis/skip-pattern features; this is a structural-leakage stress test. |
| Diabetes 130 admission | Admission-time readmission risk stress test | Hospital admission | Readmission within 30 days after discharge | Index-encounter and discharge variables are unavailable at admission; prior-year utilization remains valid. First encounters ending in death/hospice are retained because that status is unknown at admission and are treated as a competing-event limitation. |
| Diabetes 130 discharge | Discharge-time readmission risk | Immediately before discharge | Readmission within 30 days after discharge | Encounter information is considered available after excluding death/hospice dispositions; no temporal trap is designated. |

## Rules

1. Feature availability is a hard PSEUL-Select eligibility gate, not a soft clinical-utility score.
2. PSEUL-Audit may display unavailable variables to diagnose where apparent performance originates.
3. A designated trap is specific to a study and landmark; the same variable can be valid in another scenario.
4. Revision-v2 results are retrospective internal evidence. “Validated,” “safe,” and “deployable” require independent temporal or external evaluation.
5. Manual scores remain author-provisional until independent blinded raters complete the scoring protocol.
6. The admission-time Diabetes 130 analysis is a binary benchmark rather than a competing-risk analysis; its discrimination estimates must therefore be interpreted cautiously.
