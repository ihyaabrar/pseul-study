# PSEUL revision-v2 results and pipeline audit

This report is generated from the corrected nested evaluation. It is the numerical source of truth for the revision-v2 manuscript.

## Pipeline integrity

| study | rows | analysis_subsample | positive_rate | outer_splits | train_validation_isolation | train_only_imputation | locked_test_evaluated |
| --- | --- | --- | --- | --- | --- | --- | --- |
| adherence | 24071 | 24071 | 0.402 | 5 | True | True | True |
| nhanes | 7806 | 7806 | 0.167 | 5 | True | True | True |
| brfss | 453241 | 50000 | 0.145 | 5 | True | True | True |
| diabetes130_admission | 71518 | 40000 | 0.088 | 5 | True | True | True |
| diabetes130_discharge | 69973 | 40000 | 0.090 | 5 | True | True | True |

## Pooled out-of-fold performance

| study | method | auc | average_precision | mcc | brier | sensitivity | specificity |
| --- | --- | --- | --- | --- | --- | --- | --- |
| adherence | All Features | 0.901 | 0.817 | 0.639 | 0.122 | 0.853 | 0.798 |
| adherence | Oracle Trap Exclusion + SHAP | 0.654 | 0.529 | 0.193 | 0.224 | 0.396 | 0.783 |
| adherence | PSEUL-Auto (leakage metadata null) | 0.900 | 0.816 | 0.641 | 0.122 | 0.853 | 0.799 |
| adherence | PSEUL-Select | 0.837 | 0.721 | 0.507 | 0.160 | 0.777 | 0.739 |
| nhanes | All Features | 0.947 | 0.868 | 0.761 | 0.050 | 0.677 | 0.990 |
| nhanes | Oracle Trap Exclusion + SHAP | 0.793 | 0.423 | 0.270 | 0.119 | 0.233 | 0.957 |
| nhanes | PSEUL-Auto (leakage metadata null) | 0.947 | 0.866 | 0.765 | 0.050 | 0.670 | 0.992 |
| nhanes | PSEUL-Select | 0.793 | 0.435 | 0.302 | 0.117 | 0.256 | 0.959 |
| brfss | All Features | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 |
| brfss | Oracle Trap Exclusion + SHAP | 0.803 | 0.395 | 0.213 | 0.104 | 0.119 | 0.985 |
| brfss | PSEUL-Auto (leakage metadata null) | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 1.000 |
| brfss | PSEUL-Select | 0.793 | 0.384 | 0.210 | 0.106 | 0.114 | 0.986 |
| diabetes130_admission | All Features | 0.636 | 0.155 | 0.020 | 0.078 | 0.003 | 0.999 |
| diabetes130_admission | Oracle Trap Exclusion + SHAP | 0.566 | 0.118 | 0.015 | 0.080 | 0.003 | 0.999 |
| diabetes130_admission | PSEUL-Auto (leakage metadata null) | 0.572 | 0.120 | 0.026 | 0.080 | 0.005 | 0.999 |
| diabetes130_admission | PSEUL-Select | 0.572 | 0.120 | 0.026 | 0.080 | 0.005 | 0.999 |
| diabetes130_discharge | All Features | 0.642 | 0.165 | 0.048 | 0.079 | 0.007 | 0.999 |
| diabetes130_discharge | Oracle Trap Exclusion + SHAP | 0.628 | 0.155 | 0.039 | 0.080 | 0.007 | 0.999 |
| diabetes130_discharge | PSEUL-Auto (leakage metadata null) | 0.623 | 0.152 | 0.037 | 0.080 | 0.007 | 0.999 |
| diabetes130_discharge | PSEUL-Select | 0.619 | 0.148 | 0.028 | 0.080 | 0.005 | 0.999 |

## Designated-trap behaviour

| study | method | n_designated_traps | mean_trap_selection_frequency | traps_ever_selected |
| --- | --- | --- | --- | --- |
| adherence | All Features | 2 | 1.000 | ANNUALCLAIMAMOUNT, UNITSTOTAL |
| adherence | Oracle Trap Exclusion + SHAP | 2 | 0.000 | none |
| adherence | PSEUL-Auto (leakage metadata null) | 2 | 1.000 | ANNUALCLAIMAMOUNT, UNITSTOTAL |
| adherence | PSEUL-Select | 2 | 0.500 | ANNUALCLAIMAMOUNT |
| nhanes | All Features | 2 | 1.000 | fasting_glucose, hba1c |
| nhanes | Oracle Trap Exclusion + SHAP | 2 | 0.000 | none |
| nhanes | PSEUL-Auto (leakage metadata null) | 2 | 1.000 | fasting_glucose, hba1c |
| nhanes | PSEUL-Select | 2 | 0.000 | none |
| brfss | All Features | 4 | 1.000 | diabetes_age, diabetes_age_observed, insulin_use, insulin_use_observed |
| brfss | Oracle Trap Exclusion + SHAP | 4 | 0.000 | none |
| brfss | PSEUL-Auto (leakage metadata null) | 4 | 0.750 | diabetes_age, diabetes_age_observed, insulin_use |
| brfss | PSEUL-Select | 4 | 0.000 | none |
| diabetes130_admission | All Features | 11 | 1.000 | a1c_ordinal, change_flag, diabetesMed_flag, discharge_disposition_id, insulin_ordinal, max_glu_ordinal, num_lab_procedures, num_medications, num_procedures, number_diagnoses, time_in_hospital |
| diabetes130_admission | Oracle Trap Exclusion + SHAP | 11 | 0.000 | none |
| diabetes130_admission | PSEUL-Auto (leakage metadata null) | 11 | 0.000 | none |
| diabetes130_admission | PSEUL-Select | 11 | 0.000 | none |
| diabetes130_discharge | All Features | 0 | — | none |
| diabetes130_discharge | Oracle Trap Exclusion + SHAP | 0 | — | none |
| diabetes130_discharge | PSEUL-Auto (leakage metadata null) | 0 | — | none |
| diabetes130_discharge | PSEUL-Select | 0 | — | none |

For `PSEUL-Select`, lower trap frequency is desirable. For `PSEUL-Audit`, higher trap frequency means the diagnostic ranking correctly surfaces suspicious drivers. The oracle comparator knows the trap registry in advance and is not an autonomous baseline.

## Selection stability

| study | method | mean_pairwise_jaccard | min_pairwise_jaccard |
| --- | --- | --- | --- |
| adherence | All Features | 1.000 | 1.000 |
| adherence | Oracle Trap Exclusion + SHAP | 1.000 | 1.000 |
| adherence | PSEUL-Auto (leakage metadata null) | 1.000 | 1.000 |
| adherence | PSEUL-Select | 1.000 | 1.000 |
| nhanes | All Features | 1.000 | 1.000 |
| nhanes | Oracle Trap Exclusion + SHAP | 0.782 | 0.600 |
| nhanes | PSEUL-Auto (leakage metadata null) | 0.840 | 0.600 |
| nhanes | PSEUL-Select | 0.751 | 0.600 |
| brfss | All Features | 1.000 | 1.000 |
| brfss | Oracle Trap Exclusion + SHAP | 1.000 | 1.000 |
| brfss | PSEUL-Auto (leakage metadata null) | 1.000 | 1.000 |
| brfss | PSEUL-Select | 1.000 | 1.000 |
| diabetes130_admission | All Features | 1.000 | 1.000 |
| diabetes130_admission | Oracle Trap Exclusion + SHAP | 0.767 | 0.667 |
| diabetes130_admission | PSEUL-Auto (leakage metadata null) | 0.767 | 0.667 |
| diabetes130_admission | PSEUL-Select | 0.767 | 0.667 |
| diabetes130_discharge | All Features | 1.000 | 1.000 |
| diabetes130_discharge | Oracle Trap Exclusion + SHAP | 1.000 | 1.000 |
| diabetes130_discharge | PSEUL-Auto (leakage metadata null) | 0.800 | 0.667 |
| diabetes130_discharge | PSEUL-Select | 0.686 | 0.429 |

## Development-only NHANES robustness

| variant | learner | auc | mcc | brier | mean_selected_features | mean_designated_traps_selected | mean_pairwise_jaccard |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Reference | LightGBM | 0.792 | 0.293 | 0.118 | 8.000 | 0.000 | 0.751 |
| Reference | Logistic regression | 0.782 | 0.237 | 0.118 | 8.000 | 0.000 | 0.751 |
| Top-k ceiling 5 | LightGBM | 0.771 | 0.224 | 0.122 | 5.000 | 0.000 | 0.671 |
| Top-k ceiling 5 | Logistic regression | 0.769 | 0.198 | 0.120 | 5.000 | 0.000 | 0.671 |
| Without SHAP stability S | LightGBM | 0.797 | 0.277 | 0.117 | 8.000 | 0.000 | 1.000 |
| Without SHAP stability S | Logistic regression | 0.780 | 0.226 | 0.118 | 8.000 | 0.000 | 1.000 |
| Without clinical utility U | LightGBM | 0.793 | 0.297 | 0.117 | 8.000 | 0.000 | 0.614 |
| Without clinical utility U | Logistic regression | 0.784 | 0.232 | 0.117 | 8.000 | 0.000 | 0.614 |
| Without evidence E | LightGBM | 0.788 | 0.292 | 0.118 | 8.000 | 0.000 | 0.531 |
| Without evidence E | Logistic regression | 0.783 | 0.231 | 0.118 | 8.000 | 0.000 | 0.531 |
| Without predictive utility P | LightGBM | 0.781 | 0.259 | 0.120 | 8.000 | 0.000 | 0.674 |
| Without predictive utility P | Logistic regression | 0.777 | 0.218 | 0.119 | 8.000 | 0.000 | 0.674 |
| Without redundancy penalty | LightGBM | 0.787 | 0.283 | 0.118 | 8.000 | 0.000 | 0.747 |
| Without redundancy penalty | Logistic regression | 0.777 | 0.232 | 0.118 | 8.000 | 0.000 | 0.747 |
| Without semantic veto | LightGBM | 0.792 | 0.293 | 0.118 | 8.000 | 0.000 | 0.751 |
| Without semantic veto | Logistic regression | 0.782 | 0.237 | 0.118 | 8.000 | 0.000 | 0.751 |
| Without soft leakage penalty | LightGBM | 0.792 | 0.286 | 0.117 | 8.000 | 0.000 | 0.716 |
| Without soft leakage penalty | Logistic regression | 0.782 | 0.237 | 0.118 | 8.000 | 0.000 | 0.716 |

The one-factor variants reuse identical fitted fold-level signals. Logistic regression is a downstream learner sensitivity check; selection signals remain LightGBM-derived.

## Locked interpretation constraints

- NHANES and BRFSS are cross-sectional leakage stress tests, not prospective clinical prediction studies.
- Medication-adherence traps are provisional until a temporal data dictionary is available.
- Diabetes 130 admission and discharge are separately sampled landmark-specific estimands with a common top-k ceiling; their numerical difference must not be attributed to landmark alone because eligibility differs.
- PSEUL-Auto is a leakage-metadata-null control. It uses the identical seed/folds and retains evidence, utility, topic, and availability metadata; only label-derived, definitional-overlap, and target-proxy ratings are nulled.
- Manual PSEUL scores remain author-provisional until at least two independent blinded domain raters complete the v2 scoring protocol.
- Fold-level Wilcoxon/Friedman tests are descriptive only. Primary paired inference uses patient-level paired DeLong tests for AUC and paired stratified bootstrap for MCC, with Holm correction.
- A method with inadequate discrimination or sensitivity must be reported as 'no safe useful model' rather than reframed as deployable.
