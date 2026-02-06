#  Paddy Yield Decision Support Using Enterprise Machine Learning

McGill MMA – Enterprise Analytics Group Project


## 1. Enterprise Context

Modern agricultural enterprises—such as cooperatives, agribusiness firms, and government agencies—face increasing pressure to:
- Improve crop yield predictability  
- Optimize resource usage (fertilizer, land, seed varieties)  
- Make data-driven decisions under environmental uncertainty  

Despite the growing availability of agronomic data, many farming decisions remain:
- Fragmented across stakeholders  
- Heavily experience-based  
- Poorly supported by interpretable and reproducible machine learning systems  
This project addresses this gap by designing an **enterprise ML decision-support pipeline**, rather than a standalone prediction model.


## 2. Data Context

This project uses the **UCI Paddy Dataset**, a real-world agricultural dataset containing:

- Cultivation practices  
- Fertilizer usage  
- Environmental conditions  
- Paddy varieties  
- Observed yield outcomes  

From an enterprise analytics perspective, the dataset exhibits:
- Tabular and heterogeneous features  
- Observational correlations rather than verified causal relationships  
- Strong need for feature engineering, explainability, and governance  

These characteristics closely resemble real enterprise data environments.
---

## 3. Business Problem Statement

**How can an enterprise leverage historical agronomic data to support better farming decisions by identifying key yield drivers and generating interpretable insights, while remaining reproducible, explainable, and production-ready?**

This framing aligns with:
- The end-to-end machine learning lifecycle  
- Enterprise analytics best practices  
- Explainability and trust requirements emphasized in the course  
---

## 4. Project Objectives

The objectives of this project are to:
1. Explore and understand the distribution and key drivers of paddy yield  
2. Apply feature engineering and selection to improve model stability and generalization  
3. Compare machine learning models with different accuracy–interpretability trade-offs  
4. Extract interpretable insights that can inform practical farming decisions  
---

## 5. Research Hypotheses

### Hypothesis 1 — Feature Engineering & Selection  
Hybrid feature selection methods will outperform models trained on the full raw feature set in terms of predictive performance and generalization.

### Hypothesis 2 — Model Choice vs Interpretability  
Tree-based ensemble models (e.g., Random Forest, Gradient Boosting) provide a better balance between predictive performance and interpretability than linear or distance-based models for paddy yield prediction.

### Hypothesis 3 — Actionability of ML Outputs  
Interpretable machine learning models can extract actionable decision rules that meaningfully relate cultivation practices (e.g., fertilizer usage, variety selection) to yield outcomes.

## 6. Exploratory Data Analysis (EDA)

Initial exploratory data analysis focuses on:

- Distribution of paddy yield and variability across observations  
- Identification of agronomic and environmental features strongly associated with yield  
- Correlation analysis to inform feature engineering and selection  

The EDA provides an empirical foundation for subsequent modeling while avoiding causal over-interpretation.


## 7. Repository Structure
paddy-yield-decision-support/
│
├── README.md                         ← enterprise context & hypotheses
│
├── data/
│   ├── raw/
│   │   └── paddydataset.csv          ← current
│   └── processed/                   ← planned
│
├── notebooks/
│   ├── 01_initial_eda.ipynb          ← current
│   ├── 02_feature_engineering.ipynb  ← planned
│   ├── 03_model_training.ipynb       ← planned
│   └── 04_model_evaluation.ipynb     ← planned
│
├── src/                              ← planned
│   ├── preprocessing.py
│   ├── feature_selection.py
│   ├── models.py
│   └── evaluation.py
│
├── results/                          ← planned
│   ├── figures/
│   ├── tables/
│   └── summary.md
│
└── presentation/
    └── proposal_slides.pdf           

The structure reflects current progress (EDA completed) and planned extension along the full enterprise ML lifecycle.

## 8. Current Status

- Enterprise context and business framing finalized  
- Research hypotheses defined  
- Initial EDA completed  
- Feature engineering and modeling in progress  

## 9. Team

- Abdelaziz Ahmed
- Frondy Ferdianto
- Hazel Guan
- Muhammad Hydarali
- Simmi Agnihotram 

## 10. Expected Outcomes

- A reproducible end-to-end ML pipeline  
- Clear comparison of feature selection and model strategies  
- Interpretable insights linking agronomic practices to yield outcomes  
- Practical guidance suitable for enterprise decision support  
