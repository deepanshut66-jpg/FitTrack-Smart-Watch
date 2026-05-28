# ⌚ FitTrack Smart Watch — Business Data Analytics Dashboard

An end-to-end **Streamlit** business-analytics dashboard for *FitTrack Smart Watch*, an
affordable, beginner-friendly fitness wearable + companion app aimed at Indian users
(students, working professionals, and fitness beginners).

The dashboard turns raw survey data into business decisions across the full analytics
lifecycle — descriptive, diagnostic, predictive, and prescriptive — and every model
output is paired with a plain-language business explanation written for non-technical
readers.

---

## 🚀 Quick start (run locally)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) regenerate the synthetic dataset — already included as fittrack_data.csv
python generate_data.py

# 3. Launch the dashboard
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

---

## ☁️ Deploy on Streamlit Community Cloud

1. Push **all files in this folder** to a public GitHub repository (keep them in the
   repository root — no sub-folder).
2. Go to [share.streamlit.io](https://share.streamlit.io), select the repo, and set the
   main file to **`app.py`**.
3. Streamlit Cloud installs everything from `requirements.txt` automatically.

The pinned dependency versions are mainstream, Python 3.11/3.12-compatible releases, so
the build resolves cleanly with no dependency errors. The cleaned dataset
(`fittrack_data.csv`) ships with the project, so the app runs immediately without a
generation step.

---

## 📁 Project files

| File | Purpose |
|------|---------|
| `app.py` | The complete Streamlit dashboard (16 analytics sections). |
| `generate_data.py` | Reproducible generator for the synthetic survey dataset (seed = 42). |
| `fittrack_data.csv` | Cleaned dataset — 2,000 Indian respondents × 60 columns. |
| `requirements.txt` | Pinned, deployment-tested dependencies. |
| `README.md` | This file. |
| `.gitignore` | Standard Python / Streamlit ignores. |

---

## 🧭 Dashboard sections

1. **Home & Objectives** — business overview, target customers, project goals, and the
   data-driven decision-making flow.
2. **Feature Engineering** — 10 engineered scores (Health Concern, Fitness Readiness,
   Lifestyle Risk, Digital Adoption, Price Sensitivity, Purchase Potential, Engagement,
   etc.) each explained in business terms.
3. **Descriptive Analytics** — KPIs and interactive distributions for demographics,
   lifestyle, and product preferences.
4. **Diagnostic Analytics** — why customers buy: stress vs. purchase, income vs. WTP,
   correlation heatmap, cross-tabs.
5. **Classification** — Logistic Regression, Decision Tree, Random Forest, KNN, and
   Gradient Boosting compared on accuracy / precision / recall / F1 / AUC, with confusion
   matrix, ROC curve, and feature importance.
6. **Decision Tree** — visual tree, readable rules, and business interpretation.
7. **Clustering & Segments** — K-Means with elbow + silhouette diagnostics and named
   customer personas, each with a recommended bundle and marketing strategy.
8. **Association Rules** — Apriori (mlxtend) feature-affinity rules with support,
   confidence, and lift, mapped to product bundles.
9. **Regression (Pricing)** — willingness-to-pay prediction (Linear / Tree / RF / GB),
   actual-vs-predicted, and a tiered pricing recommendation.
10. **Forecasting** — simulated monthly sales with moving-average, linear-trend, and
    exponential-smoothing projections.
11. **Recommender** — rule-based product-bundle, discount, and marketing-message
    recommendations from a customer profile.
12. **Text Mining & Sentiment** — VADER sentiment on simulated feedback, word cloud, and
    positive/negative themes.
13. **Referral Network** — NetworkX referral graph with most-influential nodes and a
    referral-campaign recommendation.
14. **Ethics & Sustainability** — consent, privacy, explainable AI, bias checks, and ESG
    / e-waste considerations.
15. **Business Recommendations** — prescriptive, founder-style "what FitTrack should do
    next" summary.
16. **New Customer Prediction** — score a single customer via a form, or upload a CSV to
    batch-score new leads (purchase likelihood, predicted budget, segment) and download
    the results.

---

## ⚙️ Notes on the data & models

- **The dataset is fully synthetic** and was generated for this academic project. It
  encodes realistic correlations (e.g. higher stress → higher health concern → higher
  purchase intent; higher income / Tier-1 city → higher willingness to pay) plus
  injected noise, missing values, duplicates, and outliers that are then cleaned.
- Models are trained on the full dataset and cached for performance; the sidebar filters
  drive the exploratory (descriptive / diagnostic / text / network) views.
- Columns that would leak the target are excluded from the model feature set.
- Willingness-to-pay predictions are clamped to the realistic observed price range so the
  linear model never reports an implausible value.

> Built as a Business Data Analytics assessment project. All figures are illustrative and
> based on synthetic data.
