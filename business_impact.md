# Business Actionability

## Executive Summary
CAROB is not a universal yield optimizer. It is a governed decision-support system that helps teams decide:
- which practice patterns to pilot,
- which to avoid,
- and where evidence is still too weak for rollout.

Current causal decision status:
- Recommend: 0
- Pilot-only: 4
- Do-not-recommend: 3

This is still strong business value: the system improves pilot targeting, reduces costly rollout errors, and creates an auditable decision process.

## 1) Business Problem in Operational Terms
Rice outcomes in CAROB geographies are driven by interactions between:
- modifiable levers (variety, fertilizer, water regime),
- context constraints (country, location, soil, season, trial structure).

A prediction-only system is not enough for operations. Decision makers need:
- what action to take,
- where it applies,
- and how confident they should be before investing.

CAROB's business-oriented workflow:
- Data quality and leakage/proxy controls
- Trial-aware predictive modeling
- Interpretable rules with caveats
- Rule-country causal diagnostics for pilot selection

## 2) Stakeholders and Decision Use
### Farm Managers (field execution)
- Decision: what to test or avoid this season
- Output used: what-if deltas + caveats + country status
- Value: lower-regret operational choices

### Agronomy Teams (technical deployment)
- Decision: which rule-country pairs are pilot-worthy
- Output used: CI, overlap, balance, seed stability, trial diversity
- Value: fewer false-positive interventions

### Planning / Portfolio Teams (resource allocation)
- Decision: where to deploy budget and extension support
- Output used: pair-level expected effect + uncertainty gates
- Value: better capital efficiency and risk-adjusted rollout

## 3) Industry Size in Study Areas (Africa + Asia)
For CAROB study geographies (Benin, Burkina Faso, Gambia, Nigeria, Tanzania, Japan, Philippines, Sri Lanka), latest compiled values:
- Africa study countries: 14.248M tonnes, 6.286M ha, 2.267 t/ha
- Asia study countries: 33.928M tonnes, 7.236M ha, 4.689 t/ha
- Combined: 48.176M tonnes, 13.522M ha, 3.563 t/ha
- World production context: 820.223M tonnes; study geographies are ~5.87% of world output

Interpretation: even pilot-scale improvements in this footprint can produce material food and economic impact.

## 4) Link to the 4 Pilot-Level Rule-Country Decisions
From `causal_rule_scorecard.csv`:

| Pilot Pair | Estimated Effect (kg/ha) | 95% CI | Decision Framing |
|---|---:|---|---|
| R1 - Benin | -1,198.8 | [-1,935.5, -379.3] | Avoidance pilot (prevent likely loss) |
| R1 - Nigeria | -805.9 | [-903.8, -732.5] | Avoidance pilot |
| R2 - Nigeria | +545.1 | [205.3, 998.2] | Controlled positive pilot |
| R4 - Benin | +1,473.8 | [1,182.8, 1,988.8] | Controlled positive pilot |

Important caveat: all four remain Pilot-only because at least one diagnostic gate is not fully satisfied (notably overlap/balance/trial diversity in several pairs).

## 5) Translating Lift to Business Value
Use transparent equations:

`Gross impact (USD) = Eligible hectares × Effect (kg/ha) × Farm-gate price (USD/kg) × Adoption rate`

`Avoided loss (USD) = Eligible hectares × |Negative effect| × Farm-gate price × Compliance rate`

Illustrative magnitude at 1,000 ha and USD 0.25-0.35/kg:
- R4-Benin: USD 0.37M-0.52M gross upside
- R2-Nigeria: USD 0.14M-0.19M gross upside
- R1-Benin avoidance: USD 0.30M-0.42M protected value
- R1-Nigeria avoidance: USD 0.20M-0.28M protected value

Combined signal (1,000 ha each pair): ~USD 1.01M-1.41M before implementation/monitoring costs.

## 6) Why Actionable Even with 0 "Recommend"
Value is already created by:
- prioritizing where to test first,
- explicitly flagging where not to scale,
- quantifying uncertainty before budget commitments,
- aligning technical and business teams on explicit go/no-go logic.

So this is not "just prediction"; it is decision-risk management plus targeted experimentation.

## 7) Investment and Promotion Framework
Promote Pilot-only -> Recommend only if all pass:
- CI excludes zero
- Sign stability across seeds >= 0.67
- Overlap >= 0.60
- Post-weighting max |SMD| <= 0.20
- Trial diversity >= 2 trials per arm

Pilot requirements:
- Pre-registered success criteria
- Uplift KPI and avoided-loss KPI
- Country-specific pilots (no pooled global rollout)
- Causal re-evaluation after each data refresh

## 8) 6-12 Month Action Plan
- Pilot wave 1: R4-Benin, R2-Nigeria
- Risk-control pilots: R1-Benin, R1-Nigeria (avoidance validation)
- Measure: yield delta, gross/net value per ha, overlap/balance/diversity refresh
- Governance: promote only gate-passing pairs; keep others as Pilot-only or Do-not-recommend

## Bottom Line
The project supports controlled pilot investment, not broad rollout.
That is still a strong business outcome: model outputs are translated into staged, auditable decisions with quantified upside and bounded downside risk.
