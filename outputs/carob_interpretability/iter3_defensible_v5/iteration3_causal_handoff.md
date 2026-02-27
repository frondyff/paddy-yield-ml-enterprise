# Causal Handoff Notes

This package is causal-ready but not causal by itself.

## Recommended transition to causal analysis
1. Use the final rule set as hypothesis generators for treatment contrasts.
2. Define treatment and control cohorts before modeling outcomes.
3. Avoid conditioning on post-treatment variables and known leakage proxies.
4. Use trial-aware or country-aware adjustment sets based on DAG assumptions.

## Candidate modifiable levers to prioritize
- `variety`
- `P_fertilizer`
- `flooded`
- `K_fertilizer`
- `N_fertilizer`

## Final rule IDs for causal follow-up
- `R4`
- `R5`
- `R2`
- `R3`
- `R1`