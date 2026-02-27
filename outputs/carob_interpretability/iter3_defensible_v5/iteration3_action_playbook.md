# Actionable Recommendation Playbook (CAROB)

Built from iteration-3 explainability outputs (SHAP + permutation + cross-seed rule stability).

## 1) What to change (modifiable levers)
- `variety`: joint stability=100.0%, mean SHAP rank=1.3, mean permutation rank=1.3.
- `P_fertilizer`: joint stability=100.0%, mean SHAP rank=6.3, mean permutation rank=4.0.
- `flooded`: joint stability=100.0%, mean SHAP rank=8.0, mean permutation rank=5.3.

Exploratory (low-consensus) modifiable features to validate before acting:
- `K_fertilizer`: joint stability=0.0% (keep as hypothesis, not as primary action).
- `N_fertilizer`: joint stability=0.0% (keep as hypothesis, not as primary action).
- `row_spacing`: joint stability=0.0% (keep as hypothesis, not as primary action).
- `irrigated`: joint stability=0.0% (keep as hypothesis, not as primary action).

Rule-backed segments (plain language):
- `R4`: expected lift=+1674.2 kg/ha, support=10.0%, confidence=Medium (caveats: associative_only,requires_causal_validation).
  Where it works: Benin | Conflicts: none | Insufficient evidence: Burkina Faso, Japan, Nigeria, Philippines | Unstable/small effect: none.
- `R5`: expected lift=-778.0 kg/ha, support=8.1%, confidence=Medium (caveats: associative_only,requires_causal_validation).
  Where it works: none | Conflicts: Burkina Faso | Insufficient evidence: Benin, Japan, Nigeria, Philippines | Unstable/small effect: none.
- `R2`: expected lift=+447.4 kg/ha, support=26.9%, confidence=High (caveats: associative_only,requires_causal_validation).
  Where it works: Nigeria | Conflicts: none | Insufficient evidence: Burkina Faso, Philippines | Unstable/small effect: Benin, Japan.
- `R3`: expected lift=+267.6 kg/ha, support=13.8%, confidence=High (caveats: associative_only,requires_causal_validation).
  Where it works: none | Conflicts: Benin | Insufficient evidence: Burkina Faso, Japan, Nigeria, Philippines | Unstable/small effect: none.
- `R1`: expected lift=-236.0 kg/ha, support=28.1%, confidence=Low (caveats: associative_only,requires_causal_validation,stability_risk).
  Where it works: Benin, Nigeria | Conflicts: none | Insufficient evidence: Burkina Faso, Philippines | Unstable/small effect: Japan.

## 2) What to control for (context features)
- `country`
- `latitude`
- `location`
- `longitude`
- `planting_date`
- `plot_area`
- `rep`
- `season`
- `soil_P`
- `soil_pH`

## 3) What to avoid (proxy/leakage features)
- `dataset_id`
- `dmy_residue`
- `grain_P`
- `record_id`
- `residue_P`
- `treatment`

## 4) Confidence and caveats
- Outputs are predictive associations, not causal effects.
- Prioritize high-confidence rules first and validate with trial-level or quasi-experimental analysis.
- Keep treatment assignment logic explicit in the causal step to avoid post-treatment adjustment bias.