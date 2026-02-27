# Causal Overlay for Rule-Based Recommendations (AIPW)

This overlay evaluates rule-as-treatment effects by country.
Primary claims are for `works_here` pairs; secondary pairs are diagnostics-only.

## Primary Pairs

- `R1-Benin` [works_here] -> ATE=-1198.8 kg/ha (95% CI -1935.5, -379.3), sign_match=1.00, overlap=0.84, max|SMD|=1.20, decision=`Pilot-only` (works_here_but_some_causal_diagnostics_not_fully_met).
- `R1-Nigeria` [works_here] -> ATE=-805.9 kg/ha (95% CI -903.8, -732.5), sign_match=1.00, overlap=0.31, max|SMD|=1.07, decision=`Pilot-only` (works_here_but_some_causal_diagnostics_not_fully_met).
- `R2-Nigeria` [works_here] -> ATE=+545.1 kg/ha (95% CI +205.3, +998.2), sign_match=1.00, overlap=0.27, max|SMD|=1.04, decision=`Pilot-only` (works_here_but_some_causal_diagnostics_not_fully_met).
- `R4-Benin` [works_here] -> ATE=+1473.8 kg/ha (95% CI +1182.8, +1988.8), sign_match=1.00, overlap=0.00, max|SMD|=2.15, decision=`Pilot-only` (works_here_but_some_causal_diagnostics_not_fully_met).

## Secondary Pairs

- `R2-Benin` [unstable_or_small_effect] -> ATE=-6340.1 kg/ha (95% CI -13610.3, -173.0), sign_match=1.00, overlap=0.69, max|SMD|=2.05, decision=`Do-not-recommend` (unstable_or_small_effect_and_causal_diagnostics_weak).
- `R3-Benin` [conflicts_here] -> ATE=-267.5 kg/ha (95% CI -696.2, +250.7), sign_match=1.00, overlap=0.00, max|SMD|=2.11, decision=`Do-not-recommend` (predictive_rule_direction_conflicts_in_country).
- `R5-Burkina Faso` [conflicts_here] -> ATE=+877.7 kg/ha (95% CI +404.3, +1679.0), sign_match=1.00, overlap=0.41, max|SMD|=0.84, decision=`Do-not-recommend` (predictive_rule_direction_conflicts_in_country).
