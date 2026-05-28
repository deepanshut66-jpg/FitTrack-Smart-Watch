"""
============================================================================
 FitTrack Smart Watch  -  Business Data Analytics Dashboard
============================================================================
 A single-file Streamlit application that demonstrates the full analytics
 lifecycle for the FitTrack Smart Watch business idea:
 Descriptive -> Diagnostic -> Predictive -> Prescriptive analytics, plus
 classification, decision trees, clustering, association rule mining,
 regression, forecasting, recommendation, text mining, referral-network
 analysis, ethics & sustainability, and a new-customer prediction module.

 Run locally:   streamlit run app.py
============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

import networkx as nx

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree, export_text
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              RandomForestRegressor, GradientBoostingRegressor)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix,
                             r2_score, mean_absolute_error, mean_squared_error,
                             silhouette_score)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Optional libraries -- degrade gracefully if unavailable on the host
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    HAS_MLXTEND = True
except Exception:
    HAS_MLXTEND = False

try:
    from wordcloud import WordCloud, STOPWORDS
    HAS_WORDCLOUD = True
except Exception:
    HAS_WORDCLOUD = False


# ---------------------------------------------------------------------------
# Page configuration & styling
# ---------------------------------------------------------------------------
st.set_page_config(page_title="FitTrack Analytics Dashboard",
                   page_icon="⌚", layout="wide", initial_sidebar_state="expanded")

PRIMARY = "#1f7a8c"
ACCENT = "#ef8354"
SEQ = ["#1f7a8c", "#ef8354", "#2a9d8f", "#e9c46a", "#8d99ae",
       "#c44536", "#457b9d", "#9b5de5", "#43aa8b", "#f4a261"]
PLOTLY_TEMPLATE = "plotly_white"

st.markdown("""
<style>
    .stApp { background-color: #f6f8fa; }
    section[data-testid="stSidebar"] { background-color: #0e2433; }
    section[data-testid="stSidebar"] * { color: #e8eef2 !important; }
    h1, h2, h3 { color: #0e2433; font-family: "Segoe UI", system-ui, sans-serif; }
    .kpi-card {
        background: linear-gradient(135deg,#1f7a8c 0%,#16606f 100%);
        padding: 18px 16px; border-radius: 14px; color: #fff;
        box-shadow: 0 4px 14px rgba(0,0,0,.10); height: 100%;
    }
    .kpi-card.alt { background: linear-gradient(135deg,#ef8354 0%,#d96b3c 100%); }
    .kpi-value { font-size: 30px; font-weight: 800; line-height: 1.1; }
    .kpi-label { font-size: 13px; opacity: .92; margin-top: 4px; }
    .biz-note {
        background: #eef6f8; border-left: 5px solid #1f7a8c;
        padding: 12px 16px; border-radius: 8px; margin: 8px 0 22px 0;
        font-size: 14.5px; color: #20323c;
    }
    .biz-note b { color: #16606f; }
    .section-intro {
        background: #fff; border: 1px solid #e2e8ee; border-radius: 12px;
        padding: 16px 20px; margin-bottom: 18px; color:#33454f;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        background: #e7eef2; border-radius: 8px 8px 0 0; padding: 8px 16px;
    }
    .stTabs [aria-selected="true"] { background: #1f7a8c; color: #fff; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data(path="fittrack_data.csv"):
    df = pd.read_csv(path)
    return df


def kpi_card(col, value, label, alt=False):
    cls = "kpi-card alt" if alt else "kpi-card"
    col.markdown(
        f'<div class="{cls}"><div class="kpi-value">{value}</div>'
        f'<div class="kpi-label">{label}</div></div>', unsafe_allow_html=True)


def biz_note(what, why, action):
    st.markdown(
        f'<div class="biz-note"><b>📊 What it shows:</b> {what}<br>'
        f'<b>💡 Why it matters for FitTrack:</b> {why}<br>'
        f'<b>✅ Business action:</b> {action}</div>', unsafe_allow_html=True)


def section_intro(text):
    st.markdown(f'<div class="section-intro">{text}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Modelling helpers (cached) -- trained on the FULL dataset for stability
# ---------------------------------------------------------------------------
LEAK_COLS = ["respondent_id", "feedback_text", "likely_to_purchase", "purchase_intention",
             "purchase_potential_score", "customer_value_segment"]


def build_feature_matrix(df, drop_extra=None):
    """One-hot encode predictors; returns X (DataFrame) aligned & numeric."""
    drop = list(LEAK_COLS) + (drop_extra or [])
    X = df.drop(columns=[c for c in drop if c in df.columns])
    X = pd.get_dummies(X, drop_first=True)
    return X


@st.cache_resource(show_spinner=False)
def train_classifiers(df):
    y = (df["likely_to_purchase"] == "Yes").astype(int)
    X = build_feature_matrix(df)
    cols = X.columns
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    specs = {
        "Logistic Regression": (LogisticRegression(max_iter=1000), True),
        "Decision Tree": (DecisionTreeClassifier(max_depth=6, random_state=42), False),
        "Random Forest": (RandomForestClassifier(n_estimators=200, random_state=42), False),
        "KNN": (KNeighborsClassifier(n_neighbors=15), True),
        "Gradient Boosting": (GradientBoostingClassifier(random_state=42), False),
    }
    results, fitted = [], {}
    for name, (model, scale) in specs.items():
        a, b = (Xtr_s, Xte_s) if scale else (Xtr, Xte)
        model.fit(a, ytr)
        pred = model.predict(b)
        proba = model.predict_proba(b)[:, 1]
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(yte, pred),
            "Precision": precision_score(yte, pred, zero_division=0),
            "Recall": recall_score(yte, pred, zero_division=0),
            "F1-Score": f1_score(yte, pred, zero_division=0),
            "AUC": roc_auc_score(yte, proba),
        })
        fitted[name] = {"model": model, "scale": scale,
                        "fpr_tpr": roc_curve(yte, proba)[:2],
                        "cm": confusion_matrix(yte, pred)}
    res_df = pd.DataFrame(results).sort_values("F1-Score", ascending=False).reset_index(drop=True)
    return {"results": res_df, "fitted": fitted, "cols": cols,
            "scaler": scaler, "yte": yte, "Xte": Xte, "Xte_s": Xte_s}


@st.cache_resource(show_spinner=False)
def train_regressors(df):
    y = df["willingness_to_pay"]
    X = build_feature_matrix(df, drop_extra=["willingness_to_pay", "preferred_price_range",
                                             "budget_segment"])
    cols = X.columns
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42)
    specs = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }
    results, fitted, preds = [], {}, {}
    for name, model in specs.items():
        model.fit(Xtr, ytr)
        p = model.predict(Xte)
        results.append({
            "Model": name,
            "R2": r2_score(yte, p),
            "MAE": mean_absolute_error(yte, p),
            "RMSE": np.sqrt(mean_squared_error(yte, p)),
        })
        fitted[name] = model
        preds[name] = p
    res_df = pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)
    return {"results": res_df, "fitted": fitted, "preds": preds,
            "cols": cols, "yte": yte}


CLUSTER_FEATS = ["health_concern_score", "fitness_readiness_score", "lifestyle_risk_score",
                 "digital_adoption_score", "price_sensitivity_score", "engagement_score",
                 "willingness_to_pay", "monthly_income", "daily_steps", "sleep_hours"]


@st.cache_resource(show_spinner=False)
def run_clustering(df, k):
    X = df[CLUSTER_FEATS].copy()
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
    sil = silhouette_score(Xs, km.labels_)
    # elbow / silhouette sweep
    sweep = []
    for kk in range(2, 9):
        m = KMeans(n_clusters=kk, n_init=10, random_state=42).fit(Xs)
        sweep.append({"k": kk, "inertia": m.inertia_,
                      "silhouette": silhouette_score(Xs, m.labels_)})
    return km.labels_, sil, pd.DataFrame(sweep), Xs


@st.cache_resource(show_spinner=False)
def mine_rules(df, min_support=0.06, min_lift=1.05):
    if not HAS_MLXTEND:
        return None
    prefs = ["pref_sleep_tracking", "pref_stress_monitoring", "pref_hydration_reminder",
             "pref_heart_rate", "pref_diet_suggestion", "pref_workout_plan",
             "pref_personalized_report", "subscription_willingness"]
    nice = {"pref_sleep_tracking": "Sleep Tracking", "pref_stress_monitoring": "Stress Monitoring",
            "pref_hydration_reminder": "Hydration Reminder", "pref_heart_rate": "Heart Rate",
            "pref_diet_suggestion": "Diet Suggestion", "pref_workout_plan": "Workout Plan",
            "pref_personalized_report": "Health Report", "subscription_willingness": "Subscription"}
    basket = (df[prefs] == "Yes").rename(columns=nice)
    freq = apriori(basket, min_support=min_support, use_colnames=True)
    if freq.empty:
        return pd.DataFrame()
    try:
        rules = association_rules(freq, metric="lift", min_threshold=min_lift)
    except TypeError:
        # mlxtend 0.23.x requires num_itemsets (number of transactions)
        rules = association_rules(freq, num_itemsets=len(basket),
                                  metric="lift", min_threshold=min_lift)
    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
    return rules


@st.cache_data(show_spinner=False)
def simulate_sales(seed=7):
    """Simulated 36 months of historical FitTrack sales with trend + seasonality."""
    rng = np.random.default_rng(seed)
    months = pd.date_range("2023-06-01", periods=36, freq="MS")
    trend = np.linspace(420, 1850, 36)
    seasonal = 130 * np.sin(np.arange(36) / 12 * 2 * np.pi) + 70 * np.cos(np.arange(36) / 6 * 2 * np.pi)
    noise = rng.normal(0, 70, 36)
    units = np.clip((trend + seasonal + noise).round(), 150, None).astype(int)
    return pd.DataFrame({"month": months, "units_sold": units})


@st.cache_data(show_spinner=False)
def build_referral_graph(df, seed=11, n_nodes=140):
    """Construct a simulated referral network from referral counts."""
    rng = np.random.default_rng(seed)
    pop = df.sample(min(n_nodes, len(df)), random_state=seed).reset_index(drop=True)
    ids = pop["respondent_id"].tolist()
    G = nx.DiGraph()
    for _, r in pop.iterrows():
        G.add_node(r["respondent_id"], segment=r["customer_value_segment"],
                   influence=r["social_media_influence"], buyer=r["likely_to_purchase"])
    for _, r in pop.iterrows():
        n_ref = int(r["referrals_made"])
        if n_ref > 0:
            targets = rng.choice(ids, size=min(n_ref, len(ids) - 1), replace=False)
            for t in targets:
                if t != r["respondent_id"]:
                    G.add_edge(r["respondent_id"], t)
    return G


# ===========================================================================
# SIDEBAR  -  navigation + global filters
# ===========================================================================
df_raw = load_data()

# Realistic WTP bounds (from observed data) — used to keep model predictions sane,
# since linear models can extrapolate outside the plausible price range.
WTP_LO = float(df_raw["willingness_to_pay"].quantile(0.01))
WTP_HI = float(df_raw["willingness_to_pay"].quantile(0.99))


def clamp_wtp(x):
    return float(np.clip(x, WTP_LO, WTP_HI))


st.sidebar.markdown("## ⌚ FitTrack Analytics")
st.sidebar.caption("Business Data Analytics Assessment Project")

SECTIONS = [
    "🏠 Home & Objectives",
    "🧬 Feature Engineering",
    "📊 Descriptive Analytics",
    "🔍 Diagnostic Analytics",
    "🤖 Classification Models",
    "🌳 Decision Tree",
    "👥 Clustering & Segments",
    "🔗 Association Rules",
    "💰 Regression (Pricing)",
    "📈 Forecasting",
    "🎁 Recommender System",
    "💬 Text Mining & Sentiment",
    "🌐 Referral Network",
    "⚖️ Ethics & Sustainability",
    "🧭 Business Recommendations",
    "🆕 New Customer Prediction",
]
page = st.sidebar.radio("Navigate", SECTIONS, label_visibility="collapsed")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎛️ Global Filters")
st.sidebar.caption("Filters apply to Descriptive, Diagnostic, Text & Network views.")


def msel(label, col):
    opts = sorted(df_raw[col].dropna().unique().tolist())
    return st.sidebar.multiselect(label, opts, default=opts)


f_age = msel("Age group", "age_group")
f_gender = msel("Gender", "gender")
f_occ = msel("Occupation", "occupation")
f_tier = msel("City tier", "city_tier")
f_goal = msel("Fitness goal", "fitness_goal")
f_level = msel("Fitness level", "fitness_level")
f_budget = msel("Budget segment", "budget_segment")
f_buy = msel("Purchase intention (likely)", "likely_to_purchase")
f_app = msel("Uses fitness app", "uses_fitness_app")
f_wear = msel("Uses wearable", "uses_wearable")

mask = (
    df_raw["age_group"].isin(f_age) & df_raw["gender"].isin(f_gender) &
    df_raw["occupation"].isin(f_occ) & df_raw["city_tier"].isin(f_tier) &
    df_raw["fitness_goal"].isin(f_goal) & df_raw["fitness_level"].isin(f_level) &
    df_raw["budget_segment"].isin(f_budget) & df_raw["likely_to_purchase"].isin(f_buy) &
    df_raw["uses_fitness_app"].isin(f_app) & df_raw["uses_wearable"].isin(f_wear)
)
df = df_raw[mask].copy()

st.sidebar.markdown("---")
st.sidebar.metric("Rows after filters", f"{len(df):,}", f"of {len(df_raw):,}")
if st.sidebar.button("↺ Reset filters"):
    st.rerun()


# ===========================================================================
# 1. HOME & OBJECTIVES
# ===========================================================================
if page == "🏠 Home & Objectives":
    st.title("⌚ FitTrack Smart Watch — Business Data Analytics Dashboard")
    st.markdown(
        "An affordable, beginner-friendly smart wearable + companion app for Indian users "
        "— tracking fitness, sleep, stress, heart rate, hydration, diet and personalised goals. "
        "This dashboard turns customer survey data into **data-driven business decisions**.")

    c1, c2, c3, c4 = st.columns(4)
    kpi_card(c1, f"{len(df_raw):,}", "Survey Respondents")
    kpi_card(c2, f"{(df_raw['likely_to_purchase'] == 'Yes').mean() * 100:.0f}%",
             "Likely to Purchase", alt=True)
    kpi_card(c3, f"₹{df_raw['willingness_to_pay'].mean():,.0f}", "Avg Willingness to Pay")
    kpi_card(c4, f"{(df_raw['uses_wearable'] == 'Yes').mean() * 100:.0f}%",
             "Already Own a Wearable", alt=True)
    st.write("")

    cL, cR = st.columns([1.1, 1])
    with cL:
        st.subheader("🎯 Project Objectives")
        st.markdown("""
1. **Identify** the most profitable target customer segments for FitTrack.
2. **Predict** customer purchase interest using machine-learning classification.
3. **Estimate** willingness to pay using regression models.
4. **Recommend** personalised bundles & discounts via clustering and association rules.
5. **Support** data-driven decisions on pricing, marketing, product & retention.
6. **Forecast** future sales and adoption trends.
7. **Communicate** insights in clear, business-friendly language.
        """)
        st.subheader("🧩 The Business Problem")
        st.markdown("""
- **Who?** Students, working professionals, fitness beginners and health-conscious users —
  many already use fitness *apps* but do not yet own a *wearable*.
- **Pain points:** poor sleep, high stress, low workout consistency, no single place to track health.
- **Why FitTrack?** Affordable hardware + a guided app that nudges healthier habits.
- **Growth:** use analytics to target the right segments, price correctly, bundle the right
  features and grow through referrals — sustainably and responsibly.
        """)
    with cR:
        st.subheader("👥 Target Customers")
        seg = df_raw["occupation"].value_counts().reset_index()
        seg.columns = ["Occupation", "Count"]
        fig = px.pie(seg, names="Occupation", values="Count", hole=.5,
                     color_discrete_sequence=SEQ, template=PLOTLY_TEMPLATE)
        fig.update_layout(height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("💰 Expected Budget Mix")
        bs = df_raw["budget_segment"].value_counts().reindex(
            ["Budget", "Mid-Range", "Premium"]).reset_index()
        bs.columns = ["Segment", "Count"]
        fig2 = px.bar(bs, x="Segment", y="Count", color="Segment",
                      color_discrete_sequence=SEQ, template=PLOTLY_TEMPLATE)
        fig2.update_layout(height=250, showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("🔄 Data-Driven Decision-Making Flow")
    flow = ["Data Collection", "Data Cleaning", "Feature Engineering", "Exploratory Analysis",
            "Model Building", "Model Evaluation", "Business Insights", "Recommendations",
            "Dashboard Deployment"]
    st.markdown(
        '<div style="display:flex;flex-wrap:wrap;gap:8px;align-items:center">' +
        "".join(
            f'<span style="background:{PRIMARY};color:#fff;padding:8px 12px;border-radius:20px;'
            f'font-size:13px;font-weight:600">{s}</span>'
            + ('<span style="color:#ef8354;font-weight:800">➜</span>' if i < len(flow) - 1 else "")
            for i, s in enumerate(flow)) +
        "</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🔬 Data Science & Data Mining Lifecycle")
    life = pd.DataFrame({
        "Stage": ["Business Understanding", "Data Understanding", "Data Preparation",
                  "Feature Engineering", "Exploratory Analysis", "Model Building",
                  "Model Evaluation", "Business Interpretation", "Deployment"],
        "What we did for FitTrack": [
            "Defined goals: who buys, at what price, which features.",
            "Surveyed 2,000 Indian respondents across 50+ variables.",
            "Cleaned duplicates, capped step outliers, imputed missing values.",
            "Built 10 business scores (Health Concern, Engagement, etc.).",
            "Charted demographics, lifestyle, spending & purchase behaviour.",
            "Trained classification, regression & clustering models.",
            "Compared models on accuracy, F1, AUC, R², silhouette.",
            "Translated every output into a business action.",
            "Packaged everything into this Streamlit dashboard."]})
    st.dataframe(life, use_container_width=True, hide_index=True)


# ===========================================================================
# 2. FEATURE ENGINEERING
# ===========================================================================
elif page == "🧬 Feature Engineering":
    st.title("🧬 Feature Engineering")
    section_intro(
        "Raw survey answers are turned into <b>business scores</b> that are easier to act on. "
        "Each score blends several raw variables into a single 0–100 number (or a clear category) "
        "so marketing and product teams can target customers without reading raw data.")

    feats = pd.DataFrame({
        "Engineered Feature": [
            "Health Concern Score", "Fitness Readiness Score", "Lifestyle Risk Score",
            "Digital Adoption Score", "Price Sensitivity Score", "Purchase Potential Score",
            "Engagement Score", "Sleep-Stress Risk Category", "Budget Segment",
            "Customer Value Segment"],
        "Built from": [
            "stress + sleep quality + diet + workout consistency",
            "daily steps + workout frequency + fitness level + sleep",
            "stress + sleep + diet + low consistency",
            "digital level + app usage + wearable usage",
            "discount sensitivity + (low) income",
            "health concern + digital + engagement + WTP − privacy − price sensitivity",
            "digital adoption + feature interest + notifications + app usage",
            "stress level + sleep quality",
            "willingness to pay (Budget / Mid / Premium)",
            "purchase potential + willingness to pay"],
        "Business meaning": [
            "How much the customer worries about health — high score = target with wellness bundles.",
            "How active/ready the customer already is — high = upsell performance features.",
            "Overall unhealthy-lifestyle risk — high = strong preventive-health message.",
            "Comfort with apps & devices — high = easier to convert & retain digitally.",
            "How much price/discounts drive the decision — high = lead with EMI & offers.",
            "Single propensity-to-buy score used to rank leads.",
            "Likelihood of active app usage — high = good for subscriptions.",
            "Combined sleep & stress risk flag.",
            "Which price tier the customer fits.",
            "Overall value of the customer to FitTrack."]})
    st.dataframe(feats, use_container_width=True, hide_index=True)

    biz_note(
        "Ten engineered scores summarising 50+ raw survey fields.",
        "Business users can target customers by a single score instead of dozens of columns.",
        "Use <b>Health Concern Score</b> + <b>Engagement Score</b> together to find high-intent, "
        "wellness-focused buyers for the Stress &amp; Sleep Care bundle.")

    st.subheader("Distribution of key engineered scores")
    score_cols = ["health_concern_score", "fitness_readiness_score", "lifestyle_risk_score",
                  "digital_adoption_score", "price_sensitivity_score", "engagement_score"]
    pick = st.selectbox("Choose a score to explore", score_cols,
                        format_func=lambda c: c.replace("_", " ").title())
    cc1, cc2 = st.columns([1.4, 1])
    with cc1:
        fig = px.histogram(df_raw, x=pick, nbins=30, color_discrete_sequence=[PRIMARY],
                           template=PLOTLY_TEMPLATE)
        fig.update_layout(height=340, bargap=.05,
                          xaxis_title=pick.replace("_", " ").title(), yaxis_title="Customers")
        st.plotly_chart(fig, use_container_width=True)
    with cc2:
        fig2 = px.box(df_raw, x="likely_to_purchase", y=pick, color="likely_to_purchase",
                      color_discrete_sequence=[ACCENT, PRIMARY], template=PLOTLY_TEMPLATE)
        fig2.update_layout(height=340, showlegend=False,
                           xaxis_title="Likely to Purchase", yaxis_title=pick.replace("_", " ").title())
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("📌 Worked example")
    st.info(
        "A user with **poor sleep, high stress, low workout consistency and high app interest** "
        "gets a **high Health Concern Score**. FitTrack should target this customer with the "
        "**Stress & Sleep Care Bundle** (sleep tracking + breathing exercises + hydration reminders).")


# ===========================================================================
# 3. DESCRIPTIVE ANALYTICS
# ===========================================================================
elif page == "📊 Descriptive Analytics":
    st.title("📊 Descriptive Analytics")
    section_intro("<b>What is happening?</b> A snapshot of who the customers are, how they live, "
                  "and what they are willing to spend. All charts respect the sidebar filters.")
    if df.empty:
        st.warning("No respondents match the current filters. Please widen your selection.")
        st.stop()

    c = st.columns(6)
    kpi_card(c[0], f"{len(df):,}", "Respondents")
    kpi_card(c[1], f"₹{df['willingness_to_pay'].mean():,.0f}", "Avg WTP", alt=True)
    kpi_card(c[2], f"{(df['likely_to_purchase'] == 'Yes').mean() * 100:.0f}%", "Purchase Interest")
    kpi_card(c[3], f"{(df['uses_wearable'] == 'Yes').mean() * 100:.0f}%", "Wearable Use", alt=True)
    kpi_card(c[4], f"{(df['uses_fitness_app'] == 'Yes').mean() * 100:.0f}%", "App Use")
    kpi_card(c[5], f"{df['health_concern_score'].mean():.0f}", "Avg Health Concern", alt=True)
    st.write("")

    t1, t2, t3 = st.tabs(["👤 Demographics", "🏃 Lifestyle & Health", "💳 Digital & Spending"])

    with t1:
        a, b = st.columns(2)
        with a:
            d = df["age_group"].value_counts().sort_index().reset_index()
            d.columns = ["Age group", "Count"]
            fig = px.bar(d, x="Age group", y="Count", color="Age group",
                         color_discrete_sequence=SEQ, template=PLOTLY_TEMPLATE)
            fig.update_layout(height=320, showlegend=False, title="Age Distribution")
            st.plotly_chart(fig, use_container_width=True)
        with b:
            fig = px.pie(df, names="gender", hole=.45, color_discrete_sequence=SEQ,
                         template=PLOTLY_TEMPLATE, title="Gender Split")
            fig.update_layout(height=320)
            st.plotly_chart(fig, use_container_width=True)
        a, b = st.columns(2)
        with a:
            d = df["occupation"].value_counts().reset_index(); d.columns = ["Occupation", "Count"]
            fig = px.bar(d, x="Count", y="Occupation", orientation="h", color="Occupation",
                         color_discrete_sequence=SEQ, template=PLOTLY_TEMPLATE)
            fig.update_layout(height=320, showlegend=False, title="Occupation")
            st.plotly_chart(fig, use_container_width=True)
        with b:
            d = df["city_tier"].value_counts().reindex(["Tier 1", "Tier 2", "Tier 3"]).reset_index()
            d.columns = ["City tier", "Count"]
            fig = px.bar(d, x="City tier", y="Count", color="City tier",
                         color_discrete_sequence=SEQ, template=PLOTLY_TEMPLATE)
            fig.update_layout(height=320, showlegend=False, title="City Tier")
            st.plotly_chart(fig, use_container_width=True)
        biz_note("The demographic make-up of the surveyed market.",
                 "FitTrack must match product, price and messaging to its biggest audiences.",
                 "Focus launch marketing on the largest occupation and city-tier groups shown above.")

    with t2:
        a, b = st.columns(2)
        with a:
            d = df["fitness_goal"].value_counts().reset_index(); d.columns = ["Goal", "Count"]
            fig = px.bar(d, x="Count", y="Goal", orientation="h", color="Goal",
                         color_discrete_sequence=SEQ, template=PLOTLY_TEMPLATE)
            fig.update_layout(height=330, showlegend=False, title="Fitness Goals")
            st.plotly_chart(fig, use_container_width=True)
        with b:
            fig = px.histogram(df, x="sleep_hours", nbins=20, color="sleep_quality",
                               color_discrete_sequence=SEQ, template=PLOTLY_TEMPLATE,
                               title="Sleep Hours by Sleep Quality")
            fig.update_layout(height=330, bargap=.05)
            st.plotly_chart(fig, use_container_width=True)
        a, b = st.columns(2)
        with a:
            fig = px.box(df, x="stress_level", y="health_concern_score", color="stress_level",
                         category_orders={"stress_level": ["Low", "Medium", "High"]},
                         color_discrete_sequence=SEQ, template=PLOTLY_TEMPLATE,
                         title="Health Concern by Stress Level")
            fig.update_layout(height=330, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with b:
            fig = px.histogram(df, x="daily_steps", nbins=30, color_discrete_sequence=[PRIMARY],
                               template=PLOTLY_TEMPLATE, title="Daily Steps")
            fig.update_layout(height=330, bargap=.05)
            st.plotly_chart(fig, use_container_width=True)
        biz_note("Lifestyle reality of the market — stress, sleep, activity and goals.",
                 "High stress + poor sleep customers are exactly who FitTrack's wellness features serve.",
                 "Build a Stress & Sleep wellness narrative; it maps directly to the largest pain points.")

    with t3:
        a, b = st.columns(2)
        with a:
            fig = px.histogram(df, x="willingness_to_pay", nbins=30, color_discrete_sequence=[ACCENT],
                               template=PLOTLY_TEMPLATE, title="Willingness to Pay (₹)")
            fig.update_layout(height=330, bargap=.05)
            st.plotly_chart(fig, use_container_width=True)
        with b:
            grp = df.groupby("uses_fitness_app")["uses_wearable"].value_counts().unstack().fillna(0)
            fig = go.Figure()
            for i, col in enumerate(grp.columns):
                fig.add_bar(name=f"Wearable: {col}", x=grp.index, y=grp[col],
                            marker_color=SEQ[i])
            fig.update_layout(barmode="stack", height=330, template=PLOTLY_TEMPLATE,
                              title="App Users vs Wearable Owners", xaxis_title="Uses Fitness App")
            st.plotly_chart(fig, use_container_width=True)
        a, b = st.columns(2)
        with a:
            d = df["preferred_price_range"].value_counts().reindex(
                ["< 3K", "3K-6K", "6K-10K", "10K-18K", "18K+"]).dropna().reset_index()
            d.columns = ["Price range", "Count"]
            fig = px.bar(d, x="Price range", y="Count", color="Price range",
                         color_discrete_sequence=SEQ, template=PLOTLY_TEMPLATE,
                         title="Expected Budget Range")
            fig.update_layout(height=330, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with b:
            fig = px.histogram(df, x="digital_adoption_score", nbins=25, color="uses_wearable",
                               color_discrete_sequence=SEQ, template=PLOTLY_TEMPLATE,
                               title="Digital Adoption by Wearable Ownership")
            fig.update_layout(height=330, bargap=.05)
            st.plotly_chart(fig, use_container_width=True)
        biz_note(
            "There is a large pool of fitness-<i>app</i> users who do <b>not</b> yet own a wearable.",
            "This 'app-but-no-device' group is FitTrack's single biggest conversion opportunity.",
            "Run an upgrade campaign: 'You already track on your phone — do it better on your wrist.'")


# ===========================================================================
# 4. DIAGNOSTIC ANALYTICS
# ===========================================================================
elif page == "🔍 Diagnostic Analytics":
    st.title("🔍 Diagnostic Analytics")
    section_intro("<b>Why is it happening?</b> We dig into <i>which</i> customer traits drive "
                  "purchase interest and willingness to pay — using cross-tabs, correlations and "
                  "grouped comparisons.")
    if df.empty:
        st.warning("No respondents match the current filters.")
        st.stop()

    st.subheader("Purchase interest across key drivers")
    driver = st.selectbox(
        "Compare purchase interest by:",
        ["stress_level", "sleep_quality", "occupation", "city_tier", "uses_wearable",
         "workout_consistency", "discount_sensitivity", "privacy_concern_level",
         "digital_adoption_level", "brand_preference"],
        format_func=lambda c: c.replace("_", " ").title())
    rate = (df.assign(buy=(df["likely_to_purchase"] == "Yes").astype(int))
              .groupby(driver)["buy"].mean().mul(100).round(1).reset_index())
    rate.columns = [driver, "Purchase Interest %"]
    fig = px.bar(rate, x=driver, y="Purchase Interest %", color="Purchase Interest %",
                 color_continuous_scale="Teal", template=PLOTLY_TEMPLATE)
    fig.update_layout(height=360, xaxis_title=driver.replace("_", " ").title())
    st.plotly_chart(fig, use_container_width=True)
    hi = rate.loc[rate["Purchase Interest %"].idxmax(), driver]
    lo = rate.loc[rate["Purchase Interest %"].idxmin(), driver]
    biz_note(
        f"Purchase interest differs sharply by {driver.replace('_', ' ')}: highest for "
        f"<b>{hi}</b>, lowest for <b>{lo}</b>.",
        "Tells FitTrack which groups convert easily and which need a different pitch or price.",
        f"Prioritise marketing spend on the <b>{hi}</b> group; design a tailored offer for <b>{lo}</b>.")

    st.subheader("Correlation heatmap of numeric drivers")
    num = ["health_concern_score", "fitness_readiness_score", "lifestyle_risk_score",
           "digital_adoption_score", "engagement_score", "price_sensitivity_score",
           "willingness_to_pay", "monthly_income", "purchase_intention", "daily_steps",
           "sleep_hours"]
    corr = df[num].corr().round(2)
    fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, template=PLOTLY_TEMPLATE)
    fig.update_layout(height=560)
    st.plotly_chart(fig, use_container_width=True)
    biz_note(
        "How numeric drivers move together (red = positive, blue = negative correlation).",
        "Confirms what really pushes purchase intention — e.g. health concern, digital adoption and WTP.",
        "Build the predictive model and marketing message around the strongest positive correlates.")

    st.subheader("Willingness to pay: who pays more?")
    a, b = st.columns(2)
    with a:
        fig = px.box(df, x="city_tier", y="willingness_to_pay", color="city_tier",
                     category_orders={"city_tier": ["Tier 1", "Tier 2", "Tier 3"]},
                     color_discrete_sequence=SEQ, template=PLOTLY_TEMPLATE,
                     title="WTP by City Tier")
        fig.update_layout(height=340, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with b:
        fig = px.box(df, x="brand_preference", y="willingness_to_pay", color="brand_preference",
                     color_discrete_sequence=SEQ, template=PLOTLY_TEMPLATE,
                     title="WTP by Brand Preference")
        fig.update_layout(height=340, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    biz_note("Tier-1 and premium-brand-leaning customers are willing to pay more.",
             "Supports a tiered pricing and tiered-marketing strategy.",
             "Sell a Premium edition into Tier-1 / premium-brand buyers; a Value edition into Tier-2/3.")


# ===========================================================================
# 5. CLASSIFICATION MODELS
# ===========================================================================
elif page == "🤖 Classification Models":
    st.title("🤖 Predictive Analytics — Classification")
    section_intro("<b>Will this customer buy?</b> We train and compare five models to predict the "
                  "target <b>Likely to Purchase FitTrack = Yes / No</b>, then pick the best.")
    with st.spinner("Training classification models…"):
        C = train_classifiers(df_raw)
    res = C["results"]
    best = res.iloc[0]["Model"]

    st.subheader("📋 Model comparison")
    show = res.copy()
    for c in ["Accuracy", "Precision", "Recall", "F1-Score", "AUC"]:
        show[c] = (show[c] * 100).round(1).astype(str) + "%"
    st.dataframe(show, use_container_width=True, hide_index=True)
    biz_note(
        f"Five models compared on the same test set. <b>{best}</b> performs best on F1 and AUC.",
        "A reliable buyer-prediction model lets FitTrack focus its limited marketing budget on "
        "high-probability customers instead of everyone.",
        f"Deploy <b>{best}</b> to score every new lead and route the top scores to sales / offers.")

    a, b = st.columns(2)
    with a:
        m = res.melt(id_vars="Model", value_vars=["Accuracy", "F1-Score", "AUC"])
        fig = px.bar(m, x="Model", y="value", color="variable", barmode="group",
                     color_discrete_sequence=SEQ, template=PLOTLY_TEMPLATE,
                     title="Accuracy / F1 / AUC by Model")
        fig.update_layout(height=380, yaxis_title="Score", xaxis_tickangle=-20,
                          legend_title="Metric")
        st.plotly_chart(fig, use_container_width=True)
    with b:
        fig = go.Figure()
        for name, info in C["fitted"].items():
            fpr, tpr = info["fpr_tpr"]
            auc = res.set_index("Model").loc[name, "AUC"]
            fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"{name} ({auc:.2f})")
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="grey"),
                        name="Random")
        fig.update_layout(title="ROC Curves", height=380, template=PLOTLY_TEMPLATE,
                          xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        st.plotly_chart(fig, use_container_width=True)

    a, b = st.columns(2)
    with a:
        cm = C["fitted"][best]["cm"]
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Teal",
                        x=["Pred: No", "Pred: Yes"], y=["Actual: No", "Actual: Yes"],
                        template=PLOTLY_TEMPLATE, title=f"Confusion Matrix — {best}")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)
    with b:
        rf = RandomForestClassifier(n_estimators=200, random_state=42)
        ytmp = (df_raw["likely_to_purchase"] == "Yes").astype(int)
        Xtmp = build_feature_matrix(df_raw)
        rf.fit(Xtmp, ytmp)
        imp = pd.DataFrame({"Feature": Xtmp.columns, "Importance": rf.feature_importances_}) \
            .sort_values("Importance", ascending=False).head(12)
        imp["Feature"] = imp["Feature"].str.replace("_", " ").str.title()
        fig = px.bar(imp.sort_values("Importance"), x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="Oranges",
                     template=PLOTLY_TEMPLATE, title="Top Drivers of Purchase (Random Forest)")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)
    top3 = ", ".join(imp.sort_values("Importance", ascending=False)["Feature"].head(3))
    biz_note(
        f"The strongest predictors of purchase are: <b>{top3}</b>.",
        "FitTrack now knows which customer traits to look for and lead with in its pitch.",
        "Target customers scoring high on these factors and headline these benefits in ads.")


# ===========================================================================
# 6. DECISION TREE
# ===========================================================================
elif page == "🌳 Decision Tree":
    st.title("🌳 Decision Tree Analysis")
    section_intro("A decision tree shows the <b>exact rules</b> that separate likely buyers from "
                  "non-buyers — fully transparent and easy to explain to non-technical stakeholders.")
    depth = st.slider("Tree depth (simpler ⟷ more detailed)", 2, 5, 3)
    y = (df_raw["likely_to_purchase"] == "Yes").astype(int)
    X = build_feature_matrix(df_raw)
    tree = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=40, random_state=42).fit(X, y)

    fig, ax = plt.subplots(figsize=(15, 7))
    plot_tree(tree, feature_names=[c.replace("_", " ") for c in X.columns],
              class_names=["No", "Yes"], filled=True, rounded=True, fontsize=9,
              impurity=False, proportion=True, ax=ax)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    biz_note("A readable map of the decisions that lead to a 'Yes'.",
             "Each path is a customer rule the marketing team can use directly — no black box.",
             "Turn the strongest 'Yes' paths into audience segments for ad targeting.")

    st.subheader("📜 Decision rules (text)")
    rules_txt = export_text(tree, feature_names=[c.replace("_", " ") for c in X.columns])
    st.code(rules_txt[:2500] + ("\n... (truncated)" if len(rules_txt) > 2500 else ""))

    st.subheader("🧠 Business interpretation")
    st.success(
        "Typical high-conversion path: customers with **high health-concern score, strong digital "
        "adoption, high engagement and reasonable willingness-to-pay** are most likely to purchase "
        "FitTrack. Customers with **high privacy concern and high price sensitivity** drop off — "
        "they need reassurance on data safety and an EMI / discount offer.")


# ===========================================================================
# 7. CLUSTERING & SEGMENTS
# ===========================================================================
elif page == "👥 Clustering & Segments":
    st.title("👥 Customer Segmentation (Clustering)")
    section_intro("Unsupervised <b>K-Means</b> groups customers into personas based on their health, "
                  "digital and spending behaviour — so FitTrack can market to a handful of clear "
                  "personas instead of 2,000 individuals.")

    a, b = st.columns([1, 2])
    with a:
        k = st.slider("Number of personas (k)", 3, 7, 4)
    labels, sil, sweep, Xs = run_clustering(df_raw, k)
    with b:
        st.caption(f"Silhouette score at k={k}: **{sil:.3f}** "
                   "(higher = better separated personas).")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(sweep, x="k", y="inertia", markers=True, template=PLOTLY_TEMPLATE,
                      title="Elbow Method (within-cluster distance)")
        fig.update_traces(line_color=PRIMARY)
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.line(sweep, x="k", y="silhouette", markers=True, template=PLOTLY_TEMPLATE,
                      title="Silhouette Score by k")
        fig.update_traces(line_color=ACCENT)
        fig.update_layout(height=320)
        st.plotly_chart(fig, use_container_width=True)
    biz_note("The elbow and silhouette plots help choose how many personas to use.",
             "Too few personas = vague targeting; too many = unmanageable campaigns.",
             "Pick the k where the elbow bends and silhouette is high — usually 3–4 for this market.")

    dfc = df_raw.copy()
    dfc["Cluster"] = labels
    # Auto-name personas from their average profile
    prof = dfc.groupby("Cluster")[["health_concern_score", "fitness_readiness_score",
                                    "digital_adoption_score", "price_sensitivity_score",
                                    "willingness_to_pay", "engagement_score"]].mean()

    def name_cluster(r):
        if r["price_sensitivity_score"] > 60 and r["willingness_to_pay"] < 5000:
            return "Budget-Sensitive Beginners"
        if r["health_concern_score"] > 60 and r["digital_adoption_score"] > 60:
            return "Stressed Digital Wellness Seekers"
        if r["willingness_to_pay"] > 8000 and r["digital_adoption_score"] > 55:
            return "Health-Conscious Premium Users"
        if r["fitness_readiness_score"] > 55:
            return "Active Fitness Enthusiasts"
        if r["health_concern_score"] > 55:
            return "Sleep & Stress Focused Users"
        return "Casual Lifestyle Users"

    names = {idx: name_cluster(row) for idx, row in prof.iterrows()}
    # de-duplicate names
    seen = {}
    for idx in names:
        nm = names[idx]
        if nm in seen.values():
            nm = f"{nm} #{idx}"
        seen[idx] = nm
    dfc["Persona"] = dfc["Cluster"].map(seen)

    st.subheader("🗺️ Persona map")
    fig = px.scatter(dfc, x="digital_adoption_score", y="willingness_to_pay",
                     color="Persona", size="health_concern_score", size_max=14,
                     color_discrete_sequence=SEQ, template=PLOTLY_TEMPLATE,
                     hover_data=["occupation", "city_tier", "engagement_score"])
    fig.update_layout(height=460, xaxis_title="Digital Adoption Score",
                      yaxis_title="Willingness to Pay (₹)")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("📇 Persona profiles & marketing playbook")
    bundle_map = {
        "Budget-Sensitive Beginners": ("Starter Step & Sleep plan", "Festive / student discount + EMI"),
        "Stressed Digital Wellness Seekers": ("Stress & Sleep Care Bundle", "App-subscription bundle offer"),
        "Health-Conscious Premium Users": ("Premium Health Suite", "Premium edition, low discount"),
        "Active Fitness Enthusiasts": ("Performance & Workout Bundle", "Accessory cross-sell"),
        "Sleep & Stress Focused Users": ("Sleep & Recovery Bundle", "Guided-breathing add-on"),
        "Casual Lifestyle Users": ("Essentials plan", "Awareness + free-trial nudge"),
    }
    rows = []
    for idx, persona in seen.items():
        sub = dfc[dfc["Cluster"] == idx]
        base = persona.split(" #")[0]
        bundle, offer = bundle_map.get(base, ("Essentials plan", "Free-trial nudge"))
        rows.append({
            "Persona": persona,
            "Size": f"{len(sub)} ({len(sub) / len(dfc) * 100:.0f}%)",
            "Top occupation": sub["occupation"].mode()[0],
            "Avg WTP": f"₹{sub['willingness_to_pay'].mean():,.0f}",
            "Health concern": f"{sub['health_concern_score'].mean():.0f}",
            "Digital": f"{sub['digital_adoption_score'].mean():.0f}",
            "Recommended bundle": bundle,
            "Discount / offer": offer,
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    biz_note("Each persona has a distinct profile, budget and recommended bundle.",
             "FitTrack can now run a handful of focused campaigns instead of one generic message.",
             "Match every persona to its bundle & offer above and measure conversion per persona.")


# ===========================================================================
# 8. ASSOCIATION RULES
# ===========================================================================
elif page == "🔗 Association Rules":
    st.title("🔗 Association Rule Mining")
    section_intro("Which feature preferences <b>go together?</b> Association rule mining (Apriori) "
                  "finds 'customers who want X also want Y' patterns — perfect for designing bundles.")
    st.markdown(
        "- **Support** — how often a feature combination appears in the data.\n"
        "- **Confidence** — how likely one preference leads to another.\n"
        "- **Lift** — how much stronger the link is than random chance (lift > 1 = real association).")

    if not HAS_MLXTEND:
        st.error("`mlxtend` is not installed in this environment. Add `mlxtend` to requirements.txt.")
        st.stop()

    c1, c2 = st.columns(2)
    sup = c1.slider("Minimum support", 0.03, 0.30, 0.06, 0.01)
    lift_min = c2.slider("Minimum lift", 1.0, 2.0, 1.05, 0.05)
    rules = mine_rules(df_raw, min_support=sup, min_lift=lift_min)

    if rules is None or rules.empty:
        st.warning("No rules at these thresholds — lower the support or lift.")
        st.stop()

    disp = rules.copy()
    disp["Antecedents"] = disp["antecedents"].apply(lambda s: ", ".join(sorted(s)))
    disp["Consequents"] = disp["consequents"].apply(lambda s: ", ".join(sorted(s)))
    disp = disp[["Antecedents", "Consequents", "support", "confidence", "lift"]].round(3)
    disp.columns = ["Antecedents", "Consequents", "Support", "Confidence", "Lift"]
    st.subheader("🏆 Top association rules")
    st.dataframe(disp.head(15), use_container_width=True, hide_index=True)

    plot_df = disp.head(40).rename(
        columns={"Support": "support", "Confidence": "confidence", "Lift": "lift"})
    fig = px.scatter(plot_df, x="support", y="confidence", size="lift", color="lift",
                     color_continuous_scale="Teal", template=PLOTLY_TEMPLATE,
                     hover_data={"Antecedents": True, "Consequents": True})
    fig.update_layout(height=380, title="Rules: Support vs Confidence (bubble = Lift)",
                      xaxis_title="Support", yaxis_title="Confidence")
    st.plotly_chart(fig, use_container_width=True)

    top = disp.iloc[0]
    biz_note(
        f"Customers who prefer <b>{top['Antecedents']}</b> are strongly likely to also prefer "
        f"<b>{top['Consequents']}</b> (lift {top['Lift']}).",
        "These natural feature pairings are exactly what should be sold together.",
        "Package the top-lift pairs as a <b>Stress &amp; Sleep Wellness Bundle</b> and target high-stress "
        "professionals and students.")


# ===========================================================================
# 9. REGRESSION (PRICING)
# ===========================================================================
elif page == "💰 Regression (Pricing)":
    st.title("💰 Predictive Analytics — Willingness to Pay")
    section_intro("<b>How much will a customer pay?</b> Regression models predict the numeric "
                  "<b>Willingness to Pay (₹)</b>, guiding FitTrack's pricing and EMI strategy.")
    with st.spinner("Training regression models…"):
        R = train_regressors(df_raw)
    res = R["results"]
    best = res.iloc[0]["Model"]

    st.subheader("📋 Regression model comparison")
    show = res.copy()
    show["R2"] = show["R2"].round(3)
    show["MAE"] = "₹" + show["MAE"].round(0).astype(int).astype(str)
    show["RMSE"] = "₹" + show["RMSE"].round(0).astype(int).astype(str)
    st.dataframe(show, use_container_width=True, hide_index=True)
    biz_note(f"<b>{best}</b> predicts willingness to pay best (highest R², lowest error).",
             "Accurate WTP prediction means FitTrack can price each segment without guesswork.",
             "Use the model's segment-level predictions to set entry, mid and premium price points.")

    a, b = st.columns(2)
    with a:
        yte = R["yte"]; p = R["preds"][best]
        sc = pd.DataFrame({"Actual": yte.values, "Predicted": p})
        fig = px.scatter(sc, x="Actual", y="Predicted", opacity=.5,
                         color_discrete_sequence=[PRIMARY], template=PLOTLY_TEMPLATE,
                         title=f"Actual vs Predicted WTP — {best}")
        lo, hi = sc["Actual"].min(), sc["Actual"].max()
        fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                        line=dict(dash="dash", color=ACCENT), name="Perfect")
        fig.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    with b:
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        Xr = build_feature_matrix(df_raw, drop_extra=["willingness_to_pay", "preferred_price_range",
                                                      "budget_segment"])
        rf.fit(Xr, df_raw["willingness_to_pay"])
        imp = pd.DataFrame({"Feature": Xr.columns, "Importance": rf.feature_importances_}) \
            .sort_values("Importance", ascending=False).head(10)
        imp["Feature"] = imp["Feature"].str.replace("_", " ").str.title()
        fig = px.bar(imp.sort_values("Importance"), x="Importance", y="Feature", orientation="h",
                     color="Importance", color_continuous_scale="Oranges",
                     template=PLOTLY_TEMPLATE, title="What drives Willingness to Pay")
        fig.update_layout(height=380)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("💵 Suggested price ladder (data-driven)")
    q = df_raw["willingness_to_pay"].quantile([.35, .65, .9]).round(-2).astype(int)
    pl = pd.DataFrame({
        "Tier": ["Entry-level (Value)", "Mid-range (Core)", "Premium (Pro)"],
        "Suggested price": [f"≈ ₹{q.iloc[0]:,}", f"≈ ₹{q.iloc[1]:,}", f"≈ ₹{q.iloc[2]:,}"],
        "Target persona": ["Budget students & Tier-2/3", "Mainstream professionals",
                           "Tier-1, premium-brand, high digital"],
        "Strategy": ["Festive discount + low EMI", "Bundle + standard EMI",
                     "Minimal discount, focus on features"]})
    st.dataframe(pl, use_container_width=True, hide_index=True)
    biz_note("A three-tier price ladder derived from real willingness-to-pay percentiles.",
             "Pricing now reflects what customers will actually pay, not gut feel.",
             "Launch Value / Core / Pro editions at the prices above and offer EMI on the upper tiers.")


# ===========================================================================
# 10. FORECASTING
# ===========================================================================
elif page == "📈 Forecasting":
    st.title("📈 Demand Forecasting")
    section_intro("Using a simulated 36-month sales history, we forecast the next 12 months with "
                  "<b>Moving Average</b>, <b>Linear Trend</b> and <b>Exponential Smoothing</b> to "
                  "support production, inventory and marketing planning.")
    sales = simulate_sales()
    horizon = st.slider("Forecast horizon (months)", 3, 12, 6)
    price_point = st.number_input("Assumed avg selling price (₹)", 2000, 20000, 5000, 500)

    y = sales["units_sold"].values
    n = len(y)

    # Moving average forecast
    window = 3
    ma_last = y[-window:].mean()
    ma_fc = np.full(horizon, ma_last)

    # Linear trend forecast
    coef = np.polyfit(np.arange(n), y, 1)
    lin_fc = np.polyval(coef, np.arange(n, n + horizon))

    # Simple exponential smoothing
    alpha = 0.4
    level = y[0]
    for v in y:
        level = alpha * v + (1 - alpha) * level
    es_fc = np.full(horizon, level) + (lin_fc - lin_fc.mean()) * 0.3  # gentle trend nudge

    future_idx = pd.date_range(sales["month"].iloc[-1] + pd.offsets.MonthBegin(1),
                               periods=horizon, freq="MS")

    fig = go.Figure()
    fig.add_scatter(x=sales["month"], y=y, mode="lines+markers", name="History",
                    line=dict(color=PRIMARY))
    fig.add_scatter(x=future_idx, y=ma_fc, mode="lines+markers", name="Moving Average",
                    line=dict(color="#8d99ae", dash="dot"))
    fig.add_scatter(x=future_idx, y=lin_fc, mode="lines+markers", name="Linear Trend",
                    line=dict(color=ACCENT, dash="dash"))
    fig.add_scatter(x=future_idx, y=es_fc, mode="lines+markers", name="Exp. Smoothing",
                    line=dict(color="#43aa8b"))
    fig.update_layout(height=430, template=PLOTLY_TEMPLATE, title="FitTrack Monthly Units — Forecast",
                      xaxis_title="Month", yaxis_title="Units sold")
    st.plotly_chart(fig, use_container_width=True)

    proj_units = int(lin_fc.sum())
    proj_rev = proj_units * price_point
    c = st.columns(3)
    kpi_card(c[0], f"{proj_units:,}", f"Projected units (next {horizon}m, trend)")
    kpi_card(c[1], f"₹{proj_rev / 1e7:.2f} Cr", "Projected revenue", alt=True)
    kpi_card(c[2], f"{(lin_fc[-1] / y[-1] - 1) * 100:+.0f}%", "Growth vs last month")
    biz_note(
        "All three methods point to continued upward demand, with seasonal peaks.",
        "FitTrack can plan manufacturing, inventory and ad budget around the expected curve.",
        "Stock up ahead of forecast peak months and concentrate ad spend in high-growth months; "
        "use the linear-trend line as the base plan and exponential smoothing as the cautious case.")


# ===========================================================================
# 11. RECOMMENDER SYSTEM
# ===========================================================================
elif page == "🎁 Recommender System":
    st.title("🎁 Personalised Recommender")
    section_intro("Pick a customer profile and the system recommends the best FitTrack plan, "
                  "features, bundle, discount and a ready-to-use marketing message.")

    c1, c2, c3 = st.columns(3)
    occ = c1.selectbox("Occupation", sorted(df_raw["occupation"].unique()))
    stress = c2.selectbox("Stress level", ["Low", "Medium", "High"], index=2)
    sleepq = c3.selectbox("Sleep quality", ["Poor", "Average", "Good"], index=0)
    c4, c5, c6 = st.columns(3)
    budget = c4.selectbox("Budget segment", ["Budget", "Mid-Range", "Premium"], index=1)
    digital = c5.select_slider("Digital comfort", ["Low", "Medium", "High"], value="High")
    goal = c6.selectbox("Fitness goal", sorted(df_raw["fitness_goal"].unique()))

    # Rule-based recommendation engine
    features, bundle, plan, discount = [], "", "", ""
    if stress == "High" or sleepq == "Poor":
        features += ["Sleep tracking", "Stress monitoring", "Guided breathing", "Heart rate"]
        bundle = "Stress & Sleep Care Bundle"
    if goal == "Weight Loss":
        features += ["Step tracking", "Diet suggestions", "Calorie insights"]
        bundle = bundle or "Weight Management Bundle"
    if goal == "Muscle Gain":
        features += ["Workout plans", "Heart rate zones", "Recovery tracking"]
        bundle = bundle or "Performance Bundle"
    if not features:
        features = ["Step tracking", "Heart rate", "Sleep tracking"]
        bundle = bundle or "Essentials Bundle"
    features += ["Hydration reminders", "Weekly health report"]
    features = list(dict.fromkeys(features))

    plan = {"Budget": "FitTrack Value (entry plan)",
            "Mid-Range": "FitTrack Core (most popular)",
            "Premium": "FitTrack Pro (premium plan)"}[budget]
    discount = {"Budget": "15% launch discount + 3-month no-cost EMI",
                "Mid-Range": "10% bundle discount + free 3-month app subscription",
                "Premium": "Free premium app + priority support (no price discount)"}[budget]
    if digital == "High":
        discount += " · push app-subscription upsell"

    msg = (f"Hi! Based on your profile as a {occ.lower()} with {stress.lower()} stress and "
           f"{sleepq.lower()} sleep, the **{plan}** with our **{bundle}** is built for you — "
           f"{', '.join(features[:4]).lower()} and more. {discount}.")

    st.markdown("### ✅ Recommendation")
    rc1, rc2 = st.columns([1, 1])
    with rc1:
        st.markdown(f"**Recommended plan:** {plan}")
        st.markdown(f"**Bundle:** {bundle}")
        st.markdown("**Suggested features:**")
        st.markdown("".join([f"- {f}\n" for f in features]))
    with rc2:
        st.markdown(f"**Discount / offer:** {discount}")
        st.markdown("**Ready-to-use marketing message:**")
        st.info(msg)
    biz_note("A tailored plan, bundle, offer and message for the chosen profile.",
             "Personalised offers convert far better than one-size-fits-all promotions.",
             "Wire this logic into the FitTrack app & CRM to auto-generate offers for each lead.")


# ===========================================================================
# 12. TEXT MINING & SENTIMENT
# ===========================================================================
elif page == "💬 Text Mining & Sentiment":
    st.title("💬 Text Mining & Sentiment Analysis")
    section_intro("Simulated customer feedback is analysed with <b>VADER sentiment</b> and keyword "
                  "frequency to reveal what customers love and what frustrates them.")
    if df.empty:
        st.warning("No respondents match the current filters.")
        st.stop()

    analyzer = SentimentIntensityAnalyzer()
    fb = df["feedback_text"].dropna()
    scores = fb.apply(lambda t: analyzer.polarity_scores(t)["compound"])
    sentiment = pd.cut(scores, [-1.01, -0.05, 0.05, 1.01],
                       labels=["Negative", "Neutral", "Positive"])
    sdist = sentiment.value_counts().reindex(["Positive", "Neutral", "Negative"]).reset_index()
    sdist.columns = ["Sentiment", "Count"]

    c1, c2 = st.columns([1, 1.4])
    with c1:
        fig = px.pie(sdist, names="Sentiment", values="Count", hole=.5,
                     color="Sentiment",
                     color_discrete_map={"Positive": "#2a9d8f", "Neutral": "#e9c46a",
                                         "Negative": "#c44536"},
                     template=PLOTLY_TEMPLATE, title="Overall Sentiment")
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        stop = set(STOPWORDS) if HAS_WORDCLOUD else {
            "the", "and", "is", "a", "to", "of", "it", "for", "my", "i", "are", "in", "on",
            "with", "too", "very", "but", "so", "could", "be", "this", "that", "as", "feels"}
        words = " ".join(fb).lower().replace(".", " ").replace(",", " ").split()
        words = [w for w in words if w not in stop and len(w) > 3]
        wf = pd.Series(words).value_counts().head(15).reset_index()
        wf.columns = ["Keyword", "Count"]
        fig = px.bar(wf.sort_values("Count"), x="Count", y="Keyword", orientation="h",
                     color="Count", color_continuous_scale="Teal", template=PLOTLY_TEMPLATE,
                     title="Most Common Keywords in Feedback")
        fig.update_layout(height=340)
        st.plotly_chart(fig, use_container_width=True)

    if HAS_WORDCLOUD:
        st.subheader("☁️ Word cloud")
        wc = WordCloud(width=1000, height=320, background_color="white",
                       colormap="viridis", stopwords=set(STOPWORDS)).generate(" ".join(fb))
        fig, ax = plt.subplots(figsize=(12, 3.5))
        ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
        st.pyplot(fig, use_container_width=True); plt.close(fig)
    else:
        st.caption("Install `wordcloud` to see the word-cloud view; keyword bar chart shown above.")

    pos = fb[scores > 0.05]; neg = fb[scores < -0.05]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 👍 What customers like")
        st.success("Common positive themes: battery life, accurate sleep & step tracking, "
                   "value for money, comfortable design and a beginner-friendly app.")
        st.caption(f"{len(pos)} positive comments in current view.")
    with c2:
        st.markdown("#### 👎 What needs work")
        st.error("Common complaints: battery drain, data-privacy worries, app crashes, "
                 "perceived high price and inaccurate heart-rate readings during workouts.")
        st.caption(f"{len(neg)} negative comments in current view.")
    biz_note("Customer voice split into clear positive and negative themes.",
             "Feedback directly informs product fixes, pricing and app improvements.",
             "Fix battery & app-stability issues first; amplify praise for value & sleep tracking in ads.")


# ===========================================================================
# 13. REFERRAL NETWORK
# ===========================================================================
elif page == "🌐 Referral Network":
    st.title("🌐 Referral / Social Network Analysis")
    section_intro("A simulated referral network shows how customers recommend FitTrack to friends, "
                  "colleagues and family. Identifying <b>influencer nodes</b> powers a referral "
                  "marketing strategy.")
    if df.empty:
        st.warning("No respondents match the current filters.")
        st.stop()

    G = build_referral_graph(df)
    if G.number_of_edges() == 0:
        st.warning("No referral links in current filter view.")
        st.stop()

    deg = dict(G.out_degree())
    pos = nx.spring_layout(G, seed=42, k=0.5)

    edge_x, edge_y = [], []
    for u, v in G.edges():
        edge_x += [pos[u][0], pos[v][0], None]
        edge_y += [pos[u][1], pos[v][1], None]
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_size = [8 + deg.get(n, 0) * 5 for n in G.nodes()]
    node_color = [deg.get(n, 0) for n in G.nodes()]
    node_text = [f"{n}<br>Referrals: {deg.get(n, 0)}<br>"
                 f"Segment: {G.nodes[n].get('segment')}" for n in G.nodes()]

    fig = go.Figure()
    fig.add_scatter(x=edge_x, y=edge_y, mode="lines",
                    line=dict(width=0.6, color="#cbd5dd"), hoverinfo="none")
    fig.add_scatter(x=node_x, y=node_y, mode="markers", hoverinfo="text", text=node_text,
                    marker=dict(size=node_size, color=node_color, colorscale="Teal",
                                showscale=True, colorbar=dict(title="Referrals"),
                                line=dict(width=0.5, color="#0e2433")))
    fig.update_layout(height=520, template=PLOTLY_TEMPLATE, showlegend=False,
                      title="Referral Network (bigger / darker = more referrals)",
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    st.plotly_chart(fig, use_container_width=True)

    top = sorted(deg.items(), key=lambda x: x[1], reverse=True)[:8]
    top_df = pd.DataFrame(top, columns=["Customer ID", "Referrals made"])
    top_df["Segment"] = top_df["Customer ID"].map(lambda n: G.nodes[n].get("segment"))
    top_df["Influence"] = top_df["Customer ID"].map(lambda n: G.nodes[n].get("influence"))

    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("⭐ Most influential customers")
        st.dataframe(top_df, use_container_width=True, hide_index=True)
    with c2:
        seg_ref = (df.groupby("customer_value_segment")["referrals_made"].mean()
                     .round(2).reset_index())
        seg_ref.columns = ["Segment", "Avg referrals"]
        fig = px.bar(seg_ref, x="Segment", y="Avg referrals", color="Segment",
                     color_discrete_sequence=SEQ, template=PLOTLY_TEMPLATE,
                     title="Avg Referrals by Customer Value Segment")
        fig.update_layout(height=340, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    biz_note("A few highly-connected customers drive most referrals.",
             "Word-of-mouth from these influencers is FitTrack's cheapest, most trusted channel.",
             "Launch a referral programme (reward both referrer & friend) and recruit top nodes as "
             "brand ambassadors.")


# ===========================================================================
# 14. ETHICS & SUSTAINABILITY
# ===========================================================================
elif page == "⚖️ Ethics & Sustainability":
    st.title("⚖️ Ethics, Sustainability & AI Governance")
    section_intro("FitTrack handles sensitive health data, so responsible and sustainable practice "
                  "is part of the business model — not an afterthought.")

    t1, t2, t3 = st.tabs(["🔐 Data Ethics & Privacy", "🤖 Responsible AI", "🌱 Sustainability & ESG"])
    with t1:
        st.markdown("""
- **Informed consent** — users explicitly opt in to health-data collection and can opt out anytime.
- **Privacy by design** — collect only what is needed; sensitive health data is encrypted at rest & in transit.
- **Secure storage** — access controls, anonymised analytics datasets, no selling of personal data.
- **Transparency** — users can see, download and delete their data.
- **No misuse** — health insights are never used to discriminate (e.g. in insurance pricing).
        """)
        st.info("In this project the survey data is **synthetic** — no real personal data is used.")
    with t2:
        st.markdown("""
- **Explainable AI** — decision-tree rules and feature-importance charts make predictions understandable.
- **Bias checking** — monitor model performance across gender, age and city tier to avoid unfair targeting.
- **Human oversight** — recommendations support, not replace, human marketing decisions.
- **Responsible recommendations** — never push products in a way that exploits health anxiety.
- **Accuracy & honesty** — communicate that tracking is for wellness, not medical diagnosis.
        """)
        # Simple fairness check across gender using purchase rate
        fair = (df_raw.assign(buy=(df_raw["likely_to_purchase"] == "Yes").astype(int))
                      .groupby("gender")["buy"].mean().mul(100).round(1).reset_index())
        fair.columns = ["Gender", "Predicted-target rate %"]
        st.caption("Quick fairness lens — purchase-target rate by gender (watch for large gaps):")
        st.dataframe(fair, use_container_width=True, hide_index=True)
    with t3:
        st.markdown("""
- **Sustainable packaging** — recyclable, minimal-plastic packaging.
- **E-waste programme** — trade-in & recycling for old smartwatches; repairable design.
- **Longer device life** — software updates and replaceable straps/batteries reduce waste.
- **Preventive-health ESG impact** — encouraging activity, sleep and stress management reduces
  long-term healthcare burden — a measurable social good.
- **Energy** — low-power hardware and efficient cloud usage.
        """)
    biz_note("A clear ethics, governance and sustainability framework.",
             "Trust and responsibility are competitive advantages in health-tech and ease regulatory risk.",
             "Publish a short data-ethics & sustainability charter on the FitTrack website at launch.")


# ===========================================================================
# 15. BUSINESS RECOMMENDATIONS (PRESCRIPTIVE)
# ===========================================================================
elif page == "🧭 Business Recommendations":
    st.title("🧭 Prescriptive Analytics & Business Recommendations")
    section_intro("<b>What should FitTrack do?</b> The analysis across all sections rolls up into a "
                  "concrete, data-driven action plan.")

    best_seg = (df_raw.assign(buy=(df_raw["likely_to_purchase"] == "Yes").astype(int))
                      .groupby("occupation")["buy"].mean().idxmax())
    best_tier = (df_raw.assign(buy=(df_raw["likely_to_purchase"] == "Yes").astype(int))
                       .groupby("city_tier")["buy"].mean().idxmax())
    q = df_raw["willingness_to_pay"].quantile([.35, .65, .9]).round(-2).astype(int)

    recs = pd.DataFrame({
        "Decision area": ["Target segment", "City-tier focus", "Pricing", "Hero features",
                          "Bundles", "Discount strategy", "Marketing campaign", "Retention",
                          "Referral", "Growth"],
        "Data-driven recommendation": [
            f"Lead with <b>{best_seg}s</b> who show the highest purchase interest, plus the large "
            "'app-user-but-no-wearable' pool.",
            f"Anchor launch in <b>{best_tier}</b> (highest interest), then expand to Tier-2/3 with "
            "a value edition.",
            f"Three tiers: Value ≈ ₹{q.iloc[0]:,}, Core ≈ ₹{q.iloc[1]:,}, Pro ≈ ₹{q.iloc[2]:,}.",
            "Sleep tracking, stress monitoring, heart rate & hydration — the strongest predictors "
            "of purchase.",
            "Stress & Sleep Care Bundle (top association-rule pairing) as the flagship bundle.",
            "EMI + festive/student discounts for price-sensitive Tier-2/3; minimal discount for "
            "Tier-1 premium buyers.",
            "Wellness-led story ('beat stress, sleep better') aimed at stressed professionals & students.",
            "Drive app engagement & subscriptions — high-engagement users are the most loyal.",
            "Reward-based referral programme seeded with the network's top influencer nodes.",
            "Use the demand forecast to time inventory & ad spend around growth months."]})
    st.markdown(recs.to_html(escape=False, index=False), unsafe_allow_html=True)
    st.write("")

    st.subheader("🚀 Founder's conclusion — what FitTrack should do next")
    st.success(
        f"**The data is clear.** FitTrack's sharpest opportunity is the large group of digitally-"
        f"comfortable, high-stress, poor-sleep customers — especially **{best_seg}s in {best_tier}** "
        f"— who already use fitness apps but don't own a wearable. Win them with a **Core edition "
        f"around ₹{q.iloc[1]:,}**, a flagship **Stress & Sleep Care Bundle**, EMI options for "
        f"price-sensitive tiers, an app-subscription upsell for retention, and a referral programme "
        f"led by our most-connected customers. Pair this with transparent data ethics and "
        f"sustainable hardware to build lasting trust. **Build for wellness, price by segment, "
        f"grow through referrals.**")
    biz_note("A single, prioritised action plan synthesising every analysis in this dashboard.",
             "Gives the founder and investors a clear, evidence-backed go-to-market strategy.",
             "Execute the plan above, then re-run this dashboard on new survey data each quarter.")


# ===========================================================================
# 16. NEW CUSTOMER PREDICTION
# ===========================================================================
elif page == "🆕 New Customer Prediction":
    st.title("🆕 New Customer Prediction Module")
    section_intro("Enter a new customer (or upload a survey CSV) to instantly predict purchase "
                  "likelihood, segment, expected budget, a recommended bundle and a marketing message "
                  "— turning future survey data into real-time strategy.")

    with st.spinner("Preparing models…"):
        C = train_classifiers(df_raw)
        R = train_regressors(df_raw)
    clf_name = C["results"].iloc[0]["Model"]
    clf = C["fitted"][clf_name]["model"]; clf_scale = C["fitted"][clf_name]["scale"]
    reg_name = R["results"].iloc[0]["Model"]; reg = R["fitted"][reg_name]

    def predict_one(profile_overrides):
        """Build a single-row frame from dataset medians/modes + overrides, predict."""
        row = {}
        for col in df_raw.columns:
            if col in LEAK_COLS:
                continue
            if df_raw[col].dtype.kind in "biufc":
                row[col] = df_raw[col].median()
            else:
                row[col] = df_raw[col].mode()[0]
        row.update(profile_overrides)
        one = pd.DataFrame([row])
        # classification
        Xc = pd.get_dummies(one.drop(columns=[c for c in LEAK_COLS if c in one.columns]),
                            drop_first=True).reindex(columns=C["cols"], fill_value=0)
        Xc_in = C["scaler"].transform(Xc) if clf_scale else Xc
        proba = float(clf.predict_proba(Xc_in)[:, 1][0])
        # regression
        Xr = pd.get_dummies(one.drop(columns=[c for c in (LEAK_COLS + ["willingness_to_pay",
                            "preferred_price_range", "budget_segment"]) if c in one.columns]),
                            drop_first=True).reindex(columns=R["cols"], fill_value=0)
        wtp = float(reg.predict(Xr)[0])
        return proba, clamp_wtp(wtp)

    tab1, tab2 = st.tabs(["✍️ Single customer", "📤 Upload CSV (batch)"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        ov = {}
        ov["age_group"] = c1.selectbox("Age group", sorted(df_raw["age_group"].unique()), 1)
        ov["occupation"] = c2.selectbox("Occupation", sorted(df_raw["occupation"].unique()))
        ov["city_tier"] = c3.selectbox("City tier", ["Tier 1", "Tier 2", "Tier 3"])
        c1, c2, c3 = st.columns(3)
        ov["monthly_income"] = c1.number_input("Monthly income (₹)", 8000, 400000, 45000, 1000)
        ov["health_concern_score"] = c2.slider("Health concern score", 0, 100, 60)
        ov["digital_adoption_score"] = c3.slider("Digital adoption score", 0, 100, 65)
        c1, c2, c3 = st.columns(3)
        ov["engagement_score"] = c1.slider("Engagement score", 0, 100, 60)
        ov["price_sensitivity_score"] = c2.slider("Price sensitivity score", 0, 100, 45)
        ov["privacy_concern_level"] = c3.selectbox("Privacy concern", ["Low", "Medium", "High"], 1)
        c1, c2, c3 = st.columns(3)
        ov["uses_fitness_app"] = c1.selectbox("Uses fitness app", ["Yes", "No"])
        ov["stress_level"] = c2.selectbox("Stress level", ["Low", "Medium", "High"], 2)
        ov["sleep_quality"] = c3.selectbox("Sleep quality", ["Poor", "Average", "Good"])

        if st.button("🔮 Predict", type="primary"):
            proba, wtp = predict_one(ov)
            seg = ("High Value" if (proba > .55 and wtp > 12000)
                   else "Medium Value" if proba > .45 else "Low Value")
            budget = "Premium" if wtp >= 12000 else "Mid-Range" if wtp >= 5000 else "Budget"
            if ov["stress_level"] == "High" or ov["sleep_quality"] == "Poor":
                bundle = "Stress & Sleep Care Bundle"
            elif ov["digital_adoption_score"] > 60:
                bundle = "Digital Wellness Bundle"
            else:
                bundle = "Essentials Bundle"
            disc = ("Premium app, minimal discount" if budget == "Premium"
                    else "10% bundle + free 3-mo app" if budget == "Mid-Range"
                    else "15% launch discount + EMI")

            st.markdown("### 📈 Prediction result")
            k = st.columns(4)
            kpi_card(k[0], f"{proba * 100:.0f}%", "Purchase likelihood")
            kpi_card(k[1], f"₹{wtp:,.0f}", "Expected budget (WTP)", alt=True)
            kpi_card(k[2], seg, "Customer segment")
            kpi_card(k[3], budget, "Budget tier", alt=True)
            st.write("")
            st.markdown(f"**Recommended bundle:** {bundle}")
            st.markdown(f"**Recommended discount / offer:** {disc}")
            verdict = "🎯 High priority — pursue actively." if proba > .5 else \
                      "📨 Nurture with awareness content & a trial offer."
            st.markdown(f"**Suggested marketing message:** ")
            st.info(f"Hi! The **FitTrack {budget} edition** with our **{bundle}** fits your "
                    f"lifestyle. {disc}. {verdict}")

    with tab2:
        st.markdown("Upload a CSV with the same survey columns (target columns optional). "
                    "Download a template with the exact feature columns:")
        template = df_raw.drop(columns=LEAK_COLS + ["willingness_to_pay", "preferred_price_range",
                               "budget_segment"]).head(3)
        st.download_button("⬇️ Download CSV template", template.to_csv(index=False),
                           "fittrack_new_customers_template.csv", "text/csv")
        up = st.file_uploader("Upload new-customer CSV", type=["csv"])
        if up is not None:
            try:
                new = pd.read_csv(up)
                Xc = pd.get_dummies(new.drop(columns=[c for c in LEAK_COLS if c in new.columns],
                                    errors="ignore"), drop_first=True).reindex(
                                    columns=C["cols"], fill_value=0)
                Xc_in = C["scaler"].transform(Xc) if clf_scale else Xc
                proba = clf.predict_proba(Xc_in)[:, 1]
                Xr = pd.get_dummies(new.drop(columns=[c for c in (LEAK_COLS + ["willingness_to_pay",
                                    "preferred_price_range", "budget_segment"]) if c in new.columns],
                                    errors="ignore"), drop_first=True).reindex(
                                    columns=R["cols"], fill_value=0)
                wtp = np.clip(reg.predict(Xr), WTP_LO, WTP_HI)
                out = new.copy()
                out["Purchase_Likelihood_%"] = (proba * 100).round(1)
                out["Predicted_WTP"] = wtp.round(0).astype(int)
                out["Predicted_Segment"] = np.where(proba > .5, "Likely Buyer", "Needs Nurturing")
                st.success(f"Scored {len(out)} customers.")
                st.dataframe(out.head(20), use_container_width=True, hide_index=True)
                st.download_button("⬇️ Download scored customers", out.to_csv(index=False),
                                   "fittrack_scored_customers.csv", "text/csv")
                biz_note("Every uploaded lead now has a purchase score and predicted budget.",
                         "FitTrack can prioritise sales outreach on the highest-scoring leads.",
                         "Feed quarterly survey exports here to keep targeting data-driven.")
            except Exception as e:
                st.error(f"Could not score the file. Ensure columns match the template. ({e})")


# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.caption("FitTrack Smart Watch · Business Data Analytics Project · Built with Streamlit")
