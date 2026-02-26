# The Story Our Interpretability Outputs Tell (End to End)

This narrative is based on the authoritative interpretability run:
- `outputs/carob_interpretability/iter3_defensible_v5`

## Core Story
The model has a real, repeatable signal, but the signal is context-dependent.  
So we can use it for decision support, but not as a single universal rulebook across all countries.

## Act 1: Credibility Check
From `outputs/carob_interpretability/iter3_defensible_v5/interpretability_runlog.txt`:
- Holdout performance is moderate: `R2=0.5219`, `RMSE=960.13`, `MAE=692.92`.
- SHAP and permutation top-10 overlap is `1.00` (full agreement on top features).
- Average rule sign-match across seeds is `0.96` (rule direction is stable).
- Pipeline verdict: `defensible_case=True`.

Meaning:
- These findings are not random artifacts.
- The importance ranking and rule directions are stable enough to use as a defensible interpretability package.

## Act 2: What Drives Predictions Globally
From `outputs/carob_interpretability/iter3_defensible_v5/iteration2_feature_stability.csv`:
- Stable modifiable drivers include: `variety`, `P_fertilizer`, `flooded`.
- Strong context drivers include: `location`, `soil_P`, `soil_pH`, `country`.

Meaning:
- Yield predictions combine action levers and context.
- Recommendations must account for context; otherwise, we risk overgeneralizing.

## Act 3: What Local Explanations Show
From `outputs/carob_interpretability/iter3_defensible_v5/iteration1_local_cases_overview.csv` and local SHAP plots:
- Some records are predicted closely.
- Some extreme cases have large residuals (both over- and under-prediction).

Meaning:
- The model is useful for directional guidance.
- Individual row-level predictions can still be uncertain, especially in edge cases.

## Act 4: Decision Regimes (Rules)
From `outputs/carob_interpretability/iter3_defensible_v5/iteration3_rules_final.csv` and `iteration3_rules_final_english.md`:
- The final package includes 5 interpretable rules.
- Rules are mostly built around irrigation/flooding plus phosphorus/potassium thresholds.
- Rules include both positive and negative lift regimes.

Meaning:
- The rules are practical operating segments (if-then zones), not just abstract feature weights.

## Act 5: Cross-Country Transferability
From:
- `outputs/carob_interpretability/iter3_defensible_v5/iteration3_rule_country_generalization.csv`
- `outputs/carob_interpretability/iter3_defensible_v5/iteration3_rule_country_summary.csv`
- `outputs/carob_interpretability/iter3_defensible_v5/iteration3_action_playbook.md`

Key outcome:
- Rules do not transfer uniformly across countries.
- The playbook now explicitly labels each rule by country as:
  - `works_here`
  - `conflicts_here`
  - `insufficient_evidence`
  - `unstable_or_small_effect`

Meaning:
- Use rules with a country applicability gate.
- Do not deploy any rule globally without country-level validation status.

## What This Means for Decisions
1. Prioritize stable modifiable levers globally (`variety`, `P_fertilizer`, `flooded`), but keep context controls.
2. Apply rule recommendations only where tagged `works_here`.
3. Treat `conflicts_here` and `insufficient_evidence` as no-auto-recommend zones.
4. Use this as the bridge to causal analysis:
   - Start causal tests with high-confidence rules in countries where they work.
   - Keep treatment assignment explicit and avoid proxy/leakage variables.

## Bottom Line
This interpretability package is defensible for decision support and hypothesis generation.  
It is intentionally cautious: strong on transparency and stability, and explicit about where rules do and do not generalize.

