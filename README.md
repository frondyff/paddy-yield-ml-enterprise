# Paddy Yield Risk Early Warning (Enterprise Analytics Project)

## Overview
This project builds an **early warning system** that classifies paddy cultivation cases into **High-Risk vs Low-Risk yield outcomes**.
Instead of predicting yield as a continuous value, we reframe the problem as a **risk detection** task to support actionable decision-making
(e.g., which farms/areas should receive intervention first).

This project is inspired by:
*Muthukumaran et al. (2023), “A Hybrid Machine Learning Model with Combined Wrapper Feature Selection Techniques to Improve the Yield of Paddy”*.

---

## Business Context (Why this matters)
Agricultural decision-makers (farmers, cooperatives, or government agencies) often have limited resources for support (fertilizer guidance,
irrigation planning, pest control). A numeric yield prediction is less useful if it does not tell **where to intervene first**.

**Enterprise goal:** Identify **high-risk cases early** to prioritize interventions that prevent low yield outcomes.

---

## Project Goal
Develop an end-to-end ML pipeline that:
1. Creates a **Yield Risk** label from historical yield outcomes
2. Trains and compares ML models to predict **High-Risk vs Low-Risk**
3. Explains key risk drivers to support decision-making

---

## Hypothesis
**Primary hypothesis:**
> Reframing paddy yield prediction as a yield-risk classification problem enables more actionable and cost-effective decision-making than direct yield estimation, while maintaining strong predictive performance using a reduced and interpretable feature set.

**Supporting hypothesis:**
> Early-season agronomic and climatic variables are sufficient to detect yield risk prior to harvest, enabling proactive intervention.

---

## Data
- Input: agronomic practices, soil conditions, climatic variables (as available in the provided paddy dataset)
- Target (derived): `Yield_Risk` (binary)

### Label Definition (Yield Risk)
We derive a binary risk label from the yield distribution:
- **High-Risk**: bottom *X%* of yield values (e.g., bottom 25%)
- **Low-Risk**: remaining yield values

The threshold is configurable and will be justified based on class balance and business interpretation.

---

## Methods (End-to-End ML Lifecycle)
### 1) Data preparation
- Missing value handling
- Encoding categorical variables (if any)
- Train/test split

### 2) Feature selection (interpretability + generalization)
We evaluate feature selection strategies (inspired by the paper’s emphasis on dimensionality reduction), such as:
- Wrapper or model-based selection (e.g., RF importance, RFE)
- (Optional) comparison of multiple selection methods

### 3) Modeling
Baseline + improved models:
- Logistic Regression (baseline, interpretable)
- Random Forest (robust nonlinear model)
- (Optional) Gradient Boosting / XGBoost

### 4) Evaluation (enterprise-oriented)
We focus on metrics that matter for **risk detection**:
- Recall for the **High-Risk** class (avoid missed interventions)
- Precision for the **High-Risk** class (avoid wasted interventions)
- F1-score, confusion matrix, ROC-AUC

### 5) Explainability
We report:
- Feature importance / coefficients
- Key drivers of high-risk prediction
- Actionable insights (what variables are early warning signals)

---

## Repository Structure
- `data/` raw and processed datasets
- `notebooks/` end-to-end workflow (numbered)
- `src/` reusable functions for preprocessing, labeling, training, evaluation
- `results/` saved figures, tables, and a short summary
- `presentation/` pitch deck / PDF slides

---
