# Reflective Practice

## Threats to Validity

### 1) Representativeness and Selection Bias After Quality Gates
**What happened:** Modeling population reduced from 1,202 to 830 rows, countries from 8 to 6, and trials from 19 to 13 after constancy and missingness gates.  
**Why this matters:** The cleaned dataset is more reliable internally but less representative of full smallholder conditions across Africa and Asia.  
**What we did:** Documented country/trial exclusions and preserved audit artifacts for transparency.  
**Residual risk:** Recommendations may not transfer well to excluded regions/trials.  
**Next step:** Run sensitivity analyses with partially recovered cohorts and compare effect/ranking stability.

### 2) Group Heterogeneity and Validation Instability
**What happened:** Leave-one-trial-out baseline was unstable (mean R² = -3.2397), and per-trial performance variability remained large in group-aware diagnostics.  
**Why this matters:** Random split metrics can be overly optimistic when trial context leaks across train/test.  
**What we did:** Used trial-aware holdout as primary evaluation and reported group-level behavior.  
**Residual risk:** Future trials/seasons can still shift beyond training distribution.  
**Next step:** Add repeated group-aware resampling with confidence intervals and explicit failure-case profiling by trial.

### 3) Context/Proxy Dominance and Transportability Risk
**What happened:** SHAP ranking shows strong context variables (location, country, soil_P, soil_pH) among top drivers; modifiable features matter but are not always dominant.  
**Why this matters:** The model may partially encode context identity rather than portable agronomic mechanisms; actionability can become site-conditional.  
**What we did:** Performed proxy/leakage audits and separated modifiable vs context vs excluded features.  
**Residual risk:** Residual proxy effects may still exist through correlated context channels.  
**Next step:** Add country-holdout stress tests and perturbation checks to test recommendation robustness under context shifts.

### 4) Measurement and Protocol Heterogeneity Across Multi-location Trials
**What happened:** Source experiments span many locations (2012–2014), designs, and operational procedures; yield/soil measurements may vary in precision and protocol.  
**Why this matters:** Measurement noise increases residual variance and can blur both prediction and causal inference.  
**What we did:** Added quality audits, role-map governance, and exclusion logic for low-reliability segments.  
**Residual risk:** Non-standardized field measurement still introduces non-causal variation.  
**Next step:** Define a minimum data collection contract (units, protocol metadata, QC flags) for future refreshes.

### 5) Causal Identification Limits and Treatment Assignment Ambiguity
**What happened:** Causal layer is observational rule-as-treatment AIPW; recommendation status is Pilot-only = 4, Do-not-recommend = 3, Recommend = 0.  
**Why this matters:** Validity depends on assumptions (no hidden confounding, overlap, positivity, correct nuisance modeling), and some pairs fail overlap/balance gates.  
**What we did:** Enforced diagnostics and decision gates (CI, seed sign stability, overlap, post-weighting SMD, trial diversity).  
**Residual risk:** Effect estimates remain fragile in low-overlap or sparse-arm settings.  
**Next step:** Use controlled pilots with pre-registered criteria and only promote pairs that clear all gates.

### 6) Single Split / Single Seed Evaluation
**What happened:** Model comparison and tuning used one fixed trial-aware 80/20 split (seed 42); multi-seed checks varied algorithm randomness, not trial membership in train/test.  
**Why this matters:** With only 13 trials in test, one split can inflate or deflate performance materially; point estimates may be unstable.  
**What we did:** Used group-aware splitting and reported seed-level variability as a stability proxy.  
**Residual risk:** Seed-level variance is not split-level variance.  
**Next step:** Run repeated GroupShuffleSplit (10–20 splits) and report distributions for R², RMSE, and MAE.

### 7) Non-Random Trial Exclusion
**What happened:** Trials 7, 14, 15, and 19 were excluded because soil_P and/or soil_pH were fully missing at trial level.  
**Why this matters:** Fully missing trials may be systematically different (for example lower-resource environments), so exclusions may reduce generalizability and bias estimates.  
**What we did:** Kept exclusion logic explicit and auditable in feature-preparation outputs.  
**Residual risk:** Exclusion rationale is defensible but impact on conclusions is not fully bounded.  
**Next step:** Re-run predictive and causal analyses under alternative inclusion/imputation strategies and compare stability.

### 8) MCAR Imputation Assumption
**What happened:** Missing values were imputed using trial-level medians with global train medians as fallback, which assumes missingness is non-informative.  
**Why this matters:** If missingness is related to outcome/context quality (MAR/MNAR), imputation can bias fit and feature importance.  
**What we did:** Computed imputations from training data only and prioritized trial-level medians to reduce leakage.  
**Residual risk:** Informative missingness may still distort predictions and interpretation.  
**Next step:** Add missingness indicators and run imputation sensitivity checks for high-missing variables.

## Lessons Learned

### 1) Governance Before Optimization
Leakage/proxy controls and data quality gates influenced project credibility more than small differences between model families.

### 2) Group-aware Evaluation Is Non-negotiable
In heterogeneous multi-trial data, split strategy determines whether metrics are trustworthy. Trial-aware validation exposed risks hidden by random splits.

### 3) Data Preparation Is Core Modeling Work
Missingness audits, variance checks, and role-based feature governance were central to decision-support validity, not preprocessing overhead.

### 4) Interpretability Must Be Operational
SHAP plots alone are insufficient; useful outputs require rule text, confidence labels, caveats, and country transfer status.

### 5) Causal Analysis Bridges Prediction and Action
Rule-level causal diagnostics transformed outputs from forecasting to pilot-level decision guidance, while preventing premature rollout claims.

### 6) Report Performance as a Distribution, Not a Point Estimate
With small group counts, single-split/single-seed reporting is fragile; confidence should be based on metric distributions across repeated group-aware splits.

### 7) Principled Exclusion Still Requires Robustness Bounds
Even justified exclusions can introduce bias; conclusions should be re-tested under alternative inclusion/imputation scenarios.

### 8) Test Missingness as Signal Before Trusting Importance
For high-missing features, verify whether missingness itself predicts yield; if it does, standard imputation assumptions are violated and important narratives can mislead.

## Final Reflective Statement
The project matured from prediction-first experimentation to governance-aware, context-conditional decision support. Predictive performance is meaningful, interpretability is operational, and causal diagnostics are decision-useful. However, deployment-grade causal confidence has not yet been achieved. The appropriate near-term strategy is controlled, country-specific pilots with strict promotion gates, not broad rollout.
