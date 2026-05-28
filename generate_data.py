"""
FitTrack Smart Watch - Synthetic Dataset Generator
===================================================
Generates a realistic survey dataset of Indian respondents for the
FitTrack Smart Watch Business Data Analytics project.

Relationships (stress -> health concern -> purchase interest, income/tier ->
willingness to pay, digital adoption -> app/wearable usage, etc.) are baked in
so that the downstream ML models, clustering and association rules find real,
explainable signal rather than noise.

Run:  python generate_data.py
Output: fittrack_data.csv
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 2200  # generate a bit extra, keep 2000+ after cleaning


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def pick(options, p=None, size=N):
    return RNG.choice(options, size=size, p=p)


def clip01(x):
    return np.clip(x, 0, 1)


def to_band(score, low, high, labels=("Low", "Medium", "High")):
    """Convert a numeric score into ordered bands."""
    out = np.where(score < low, labels[0], np.where(score < high, labels[1], labels[2]))
    return out


# ---------------------------------------------------------------------------
# 1. Demographics
# ---------------------------------------------------------------------------
age_group = pick(["18-24", "25-34", "35-44", "45-54", "55+"],
                 p=[0.30, 0.34, 0.20, 0.11, 0.05])
gender = pick(["Male", "Female", "Other"], p=[0.55, 0.43, 0.02])

# Occupation depends loosely on age
occupation = np.empty(N, dtype=object)
for i, ag in enumerate(age_group):
    if ag == "18-24":
        occupation[i] = RNG.choice(["Student", "Working Professional", "Self-Employed"],
                                   p=[0.62, 0.30, 0.08])
    elif ag == "25-34":
        occupation[i] = RNG.choice(["Working Professional", "Self-Employed", "Student", "Homemaker"],
                                   p=[0.66, 0.18, 0.08, 0.08])
    elif ag in ("35-44", "45-54"):
        occupation[i] = RNG.choice(["Working Professional", "Self-Employed", "Homemaker"],
                                   p=[0.58, 0.27, 0.15])
    else:
        occupation[i] = RNG.choice(["Retired", "Self-Employed", "Homemaker", "Working Professional"],
                                   p=[0.45, 0.20, 0.20, 0.15])

city_tier = pick(["Tier 1", "Tier 2", "Tier 3"], p=[0.40, 0.38, 0.22])
education_level = pick(["High School", "Diploma", "Graduate", "Post Graduate"],
                       p=[0.18, 0.17, 0.42, 0.23])

# Monthly income (INR) influenced by city tier, occupation and education
base_income = np.full(N, 35000.0)
tier_boost = np.select([city_tier == "Tier 1", city_tier == "Tier 2", city_tier == "Tier 3"],
                       [1.55, 1.10, 0.78])
occ_boost = np.select(
    [occupation == "Working Professional", occupation == "Self-Employed",
     occupation == "Student", occupation == "Homemaker", occupation == "Retired"],
    [1.40, 1.25, 0.45, 0.55, 0.85])
edu_boost = np.select(
    [education_level == "Post Graduate", education_level == "Graduate",
     education_level == "Diploma", education_level == "High School"],
    [1.45, 1.20, 0.95, 0.75])
monthly_income = (base_income * tier_boost * occ_boost * edu_boost
                  * RNG.lognormal(0, 0.25, N)).round(-2)
monthly_income = np.clip(monthly_income, 8000, 400000)

def income_band(x):
    if x < 25000:
        return "< 25K"
    if x < 50000:
        return "25K-50K"
    if x < 100000:
        return "50K-1L"
    if x < 200000:
        return "1L-2L"
    return "2L+"
income_range = np.array([income_band(x) for x in monthly_income])


# ---------------------------------------------------------------------------
# 2. Lifestyle & Health
# ---------------------------------------------------------------------------
fitness_goal = pick(
    ["Weight Loss", "General Fitness", "Muscle Gain", "Stress Management",
     "Better Sleep", "Medical / Recovery"],
    p=[0.27, 0.26, 0.15, 0.13, 0.12, 0.07])

fitness_level = pick(["Beginner", "Intermediate", "Advanced"], p=[0.52, 0.34, 0.14])

# Workout frequency (days/week) tied to fitness level
wf_mean = np.select([fitness_level == "Advanced", fitness_level == "Intermediate",
                     fitness_level == "Beginner"], [5.0, 3.2, 1.4])
workout_frequency = np.clip(np.round(RNG.normal(wf_mean, 1.3)), 0, 7).astype(int)

workout_consistency = to_band(workout_frequency, 2, 4, ("Low", "Medium", "High"))

# Daily steps tied to workout frequency
daily_steps = np.clip(
    (3000 + workout_frequency * 1100 + RNG.normal(0, 1800, N)).round(-2), 800, 22000
).astype(int)

# Stress: higher for students & working professionals, Tier 1
stress_base = RNG.normal(0.45, 0.18, N)
stress_base += np.where(np.isin(occupation, ["Working Professional", "Student"]), 0.12, 0)
stress_base += np.where(city_tier == "Tier 1", 0.06, 0)
stress_base += np.where(fitness_goal == "Stress Management", 0.18, 0)
stress_score = clip01(stress_base)  # 0-1
stress_level = to_band(stress_score, 0.40, 0.65)

# Sleep hours inversely related to stress
sleep_hours = np.clip(np.round(7.6 - stress_score * 2.4 + RNG.normal(0, 0.6, N), 1), 3.5, 10.0)
sleep_quality_score = clip01((sleep_hours - 4) / 4 - stress_score * 0.4 + RNG.normal(0, 0.1, N))
sleep_quality = to_band(sleep_quality_score, 0.40, 0.65, ("Poor", "Average", "Good"))

diet_quality = pick(["Poor", "Average", "Good"], p=[0.30, 0.45, 0.25])
water_intake = np.clip(np.round(RNG.normal(2.3, 0.8, N), 1), 0.5, 5.5)  # litres/day

medical_fitness_goal = np.where(fitness_goal == "Medical / Recovery", "Yes",
                                pick(["Yes", "No"], p=[0.18, 0.82]))

# Self-reported health concern (raw survey field)
hc_base = clip01(stress_score * 0.5 + (1 - sleep_quality_score) * 0.4
                 + RNG.normal(0, 0.12, N))
health_concern_level = to_band(hc_base, 0.40, 0.62)


# ---------------------------------------------------------------------------
# 3. Digital & Wearable behaviour
# ---------------------------------------------------------------------------
# Younger + higher income + Tier 1 -> more digital
digital_base = RNG.normal(0.45, 0.15, N)
digital_base += np.select([age_group == "18-24", age_group == "25-34",
                           age_group == "35-44", age_group == "45-54", age_group == "55+"],
                          [0.18, 0.12, 0.02, -0.08, -0.16])
digital_base += np.where(city_tier == "Tier 1", 0.08, np.where(city_tier == "Tier 2", 0.02, -0.05))
digital_base += (monthly_income / 400000) * 0.12
digital_score = clip01(digital_base)
digital_adoption_level = to_band(digital_score, 0.40, 0.62)

uses_fitness_app = np.where(RNG.random(N) < clip01(digital_score * 0.8 + 0.1), "Yes", "No")
uses_wearable = np.where(
    RNG.random(N) < clip01(digital_score * 0.55 + (monthly_income / 400000) * 0.2 - 0.05),
    "Yes", "No")
past_smartwatch_owner = np.where(
    (uses_wearable == "Yes") & (RNG.random(N) < 0.7), "Yes",
    np.where(RNG.random(N) < 0.12, "Yes", "No"))

trust_in_health_data = to_band(clip01(digital_score * 0.5 + RNG.normal(0.35, 0.15, N)), 0.40, 0.62)
privacy_concern_level = pick(["Low", "Medium", "High"], p=[0.30, 0.42, 0.28])
app_notification_preference = pick(["Minimal", "Moderate", "Frequent"], p=[0.30, 0.45, 0.25])


# ---------------------------------------------------------------------------
# 4. Product feature preferences (Yes/No interest flags)
# Preferences are driven by the relevant health signal, so association rules
# and recommendations are meaningful.
# ---------------------------------------------------------------------------
def yesno(prob):
    return np.where(RNG.random(N) < clip01(prob), "Yes", "No")

pref_sleep_tracking   = yesno(0.25 + (1 - sleep_quality_score) * 0.6)
pref_stress_monitoring = yesno(0.20 + stress_score * 0.65)
pref_hydration_reminder = yesno(0.30 + (1 - water_intake / 5.5) * 0.4)
pref_heart_rate       = yesno(0.45 + (health_concern_level == "High") * 0.25)
pref_diet_suggestion  = yesno(0.25 + (diet_quality == "Poor") * 0.45 + (fitness_goal == "Weight Loss") * 0.2)
pref_workout_plan     = yesno(0.25 + (fitness_level == "Beginner") * 0.4 + (fitness_goal == "Muscle Gain") * 0.2)
pref_personalized_report = yesno(0.30 + digital_score * 0.4)

battery_life_importance = pick(["Low", "Medium", "High"], p=[0.12, 0.40, 0.48])
design_importance = pick(["Low", "Medium", "High"], p=[0.20, 0.45, 0.35])
brand_preference = pick(["Budget / Value", "Premium / Trusted", "No Preference"],
                        p=[0.40, 0.30, 0.30])


# ---------------------------------------------------------------------------
# 5. Engineered scores (saved into the CSV; explained in the dashboard)
# ---------------------------------------------------------------------------
def band_to_num(arr, mapping):
    return np.array([mapping[v] for v in arr], dtype=float)

stress_num = band_to_num(stress_level, {"Low": 0.2, "Medium": 0.55, "High": 0.9})
sleepq_num = band_to_num(sleep_quality, {"Poor": 0.9, "Average": 0.5, "Good": 0.15})  # poor sleep = high concern
consist_num = band_to_num(workout_consistency, {"Low": 0.85, "Medium": 0.5, "High": 0.15})
diet_num = band_to_num(diet_quality, {"Poor": 0.85, "Average": 0.5, "Good": 0.2})

# Health Concern Score (0-100): poor sleep + high stress + poor diet + low consistency
health_concern_score = (100 * clip01(
    0.34 * stress_num + 0.30 * sleepq_num + 0.20 * diet_num + 0.16 * consist_num
    + RNG.normal(0, 0.03, N))).round(1)

# Fitness Readiness Score (0-100): steps + frequency + level + good sleep
level_num = band_to_num(fitness_level, {"Beginner": 0.25, "Intermediate": 0.6, "Advanced": 0.95})
fitness_readiness_score = (100 * clip01(
    0.35 * (daily_steps / 15000) + 0.30 * (workout_frequency / 7)
    + 0.20 * level_num + 0.15 * (1 - sleepq_num) + RNG.normal(0, 0.03, N))).round(1)

# Lifestyle Risk Score (0-100): combination of risk factors
lifestyle_risk_score = (100 * clip01(
    0.30 * stress_num + 0.25 * sleepq_num + 0.25 * diet_num + 0.20 * consist_num
    + RNG.normal(0, 0.03, N))).round(1)

# Digital Adoption Score (0-100)
digital_adoption_score = (100 * clip01(
    digital_score * 0.6 + (uses_fitness_app == "Yes") * 0.2
    + (uses_wearable == "Yes") * 0.2)).round(1)

# Price Sensitivity Score (0-100) -- high = very price sensitive
ds_seed = pick(["Low", "Medium", "High"], p=[0.28, 0.42, 0.30])
discount_sensitivity = ds_seed
ds_num = band_to_num(discount_sensitivity, {"Low": 0.2, "Medium": 0.55, "High": 0.9})
price_sensitivity_score = (100 * clip01(
    0.55 * ds_num + 0.45 * (1 - monthly_income / 400000) + RNG.normal(0, 0.04, N))).round(1)

# Engagement Score (0-100): app + notifications + digital + feature interest
n_features = (np.array(
    [(pref_sleep_tracking == "Yes").astype(int), (pref_stress_monitoring == "Yes").astype(int),
     (pref_hydration_reminder == "Yes").astype(int), (pref_heart_rate == "Yes").astype(int),
     (pref_diet_suggestion == "Yes").astype(int), (pref_workout_plan == "Yes").astype(int),
     (pref_personalized_report == "Yes").astype(int)]).sum(axis=0))
notif_num = band_to_num(app_notification_preference,
                        {"Minimal": 0.2, "Moderate": 0.55, "Frequent": 0.9})
engagement_score = (100 * clip01(
    0.35 * (digital_adoption_score / 100) + 0.30 * (n_features / 7)
    + 0.20 * notif_num + 0.15 * (uses_fitness_app == "Yes"))).round(1)


# ---------------------------------------------------------------------------
# 6. Spending, willingness to pay & purchase intent
# ---------------------------------------------------------------------------
monthly_health_spend = np.clip(
    (monthly_income * RNG.uniform(0.005, 0.05, N)
     * (1 + (health_concern_score / 100) * 0.5)).round(-1), 0, 25000).astype(int)

# Willingness to pay (INR) - regression target
wtp = (2200
       + (monthly_income / 1000) * 22
       + digital_adoption_score * 14
       + health_concern_score * 9
       + band_to_num(brand_preference,
                     {"Budget / Value": 0, "No Preference": 600, "Premium / Trusted": 2600})
       + np.where(city_tier == "Tier 1", 1500, np.where(city_tier == "Tier 2", 500, -300))
       - price_sensitivity_score * 16
       + RNG.normal(0, 900, N))
willingness_to_pay = np.clip(wtp.round(-2), 1200, 40000).astype(int)

def price_band(x):
    if x < 3000:
        return "< 3K"
    if x < 6000:
        return "3K-6K"
    if x < 10000:
        return "6K-10K"
    if x < 18000:
        return "10K-18K"
    return "18K+"
preferred_price_range = np.array([price_band(x) for x in willingness_to_pay])

payment_preference = np.where(
    (willingness_to_pay > 8000) & (RNG.random(N) < 0.6), "EMI", "Full Payment")
subscription_willingness = np.where(
    RNG.random(N) < clip01(0.15 + engagement_score / 100 * 0.5), "Yes", "No")

family_influence = pick(["Low", "Medium", "High"], p=[0.35, 0.40, 0.25])
social_media_influence = to_band(clip01(digital_score * 0.6 + RNG.normal(0.2, 0.15, N)), 0.40, 0.62)

# Purchase propensity (latent) -> purchase intention (1-5) -> binary target
purchase_latent = clip01(
    0.10
    + 0.26 * (health_concern_score / 100)
    + 0.24 * (digital_adoption_score / 100)
    + 0.16 * (engagement_score / 100)
    + 0.12 * (willingness_to_pay / 20000)
    + 0.08 * band_to_num(social_media_influence, {"Low": 0.2, "Medium": 0.55, "High": 0.9})
    + 0.06 * (uses_fitness_app == "Yes")
    - 0.10 * band_to_num(privacy_concern_level, {"Low": 0.0, "Medium": 0.4, "High": 0.9})
    - 0.08 * (price_sensitivity_score / 100)
    + RNG.normal(0, 0.11, N))
# Likert intention spread across 1-5 from the latent score
purchase_intention = np.clip(np.round(0.7 + purchase_latent * 5.2), 1, 5).astype(int)
# Binary target: top ~38% of latent propensity are treated as likely buyers
_buy_threshold = np.quantile(purchase_latent, 0.62)
likely_to_purchase = np.where(purchase_latent >= _buy_threshold, "Yes", "No")

upgrade_intention = np.where(
    (past_smartwatch_owner == "Yes") & (RNG.random(N) < 0.55), "Yes",
    np.where(RNG.random(N) < 0.18, "Yes", "No"))

# Purchase Potential Score & Customer Value Segment (engineered)
purchase_potential_score = (100 * purchase_latent).round(1)
customer_value_segment = np.select(
    [(willingness_to_pay >= 12000) & (purchase_potential_score >= 55),
     (purchase_potential_score >= 45),
     (purchase_potential_score >= 0)],
    ["High Value", "Medium Value", "Low Value"], default="Low Value")

budget_segment = np.select(
    [willingness_to_pay < 5000, willingness_to_pay < 12000, willingness_to_pay >= 12000],
    ["Budget", "Mid-Range", "Premium"], default="Mid-Range")

# Sleep-Stress Risk Category
ss_risk = clip01(0.5 * stress_num + 0.5 * sleepq_num)
sleep_stress_risk_category = to_band(ss_risk, 0.40, 0.65, ("Low", "Moderate", "High"))


# ---------------------------------------------------------------------------
# 7. Referral network fields
# ---------------------------------------------------------------------------
referrals_made = np.clip(
    np.round(RNG.poisson(0.6, N)
             + (social_media_influence == "High") * RNG.poisson(1.5, N)
             + (likely_to_purchase == "Yes") * RNG.poisson(0.8, N)), 0, 9).astype(int)


# ---------------------------------------------------------------------------
# 8. Simulated customer feedback text (for text mining / sentiment)
# ---------------------------------------------------------------------------
pos_templates = [
    "The battery life is excellent and lasts almost a week on a single charge.",
    "I love the sleep tracking, it is accurate and easy to understand.",
    "Great value for money, the app is smooth and the design looks premium.",
    "Heart rate monitoring feels reliable and the hydration reminders really help.",
    "Comfortable to wear all day and the stress tracking is genuinely useful.",
    "The personalized health reports keep me motivated to stay consistent.",
    "Affordable, lightweight and the companion app is very beginner friendly.",
    "Step counting is accurate and syncing with the app is fast and seamless.",
]
neg_templates = [
    "The battery drains too quickly and charging takes very long.",
    "I am worried about my health data privacy and how it is stored.",
    "The app keeps crashing and the notifications are unreliable.",
    "Too expensive for the features offered, the price could be lower.",
    "Heart rate readings feel inaccurate during workouts and the strap is uncomfortable.",
    "Sleep tracking data seems wrong and the design feels cheap.",
    "Customer support was slow and the device disconnects from the app often.",
    "Hard to set up for beginners and too many confusing notifications.",
]
neu_templates = [
    "The watch is okay, does the basic job of counting steps.",
    "Average experience overall, nothing special but works fine.",
    "Design is decent, battery is average, app could improve a little.",
    "It works as expected, neither impressed nor disappointed so far.",
    "Reasonable device for the price, a few features could be better.",
]
feedback_text = np.empty(N, dtype=object)
for i in range(N):
    score = purchase_latent[i] + RNG.normal(0, 0.12)
    if score > 0.6:
        feedback_text[i] = RNG.choice(pos_templates)
    elif score < 0.4:
        feedback_text[i] = RNG.choice(neg_templates)
    else:
        feedback_text[i] = RNG.choice(neu_templates)


# ---------------------------------------------------------------------------
# 9. Assemble dataframe
# ---------------------------------------------------------------------------
df = pd.DataFrame({
    "respondent_id": [f"FT{100000 + i}" for i in range(N)],
    # Demographics
    "age_group": age_group, "gender": gender, "occupation": occupation,
    "city_tier": city_tier, "monthly_income": monthly_income, "income_range": income_range,
    "education_level": education_level,
    # Lifestyle & health
    "fitness_goal": fitness_goal, "fitness_level": fitness_level,
    "workout_frequency": workout_frequency, "workout_consistency": workout_consistency,
    "daily_steps": daily_steps, "sleep_hours": sleep_hours, "sleep_quality": sleep_quality,
    "stress_level": stress_level, "diet_quality": diet_quality, "water_intake": water_intake,
    "health_concern_level": health_concern_level, "medical_fitness_goal": medical_fitness_goal,
    # Digital & wearable
    "uses_fitness_app": uses_fitness_app, "uses_wearable": uses_wearable,
    "past_smartwatch_owner": past_smartwatch_owner,
    "digital_adoption_level": digital_adoption_level, "trust_in_health_data": trust_in_health_data,
    "privacy_concern_level": privacy_concern_level,
    "app_notification_preference": app_notification_preference,
    # Product preferences
    "pref_sleep_tracking": pref_sleep_tracking, "pref_stress_monitoring": pref_stress_monitoring,
    "pref_hydration_reminder": pref_hydration_reminder, "pref_heart_rate": pref_heart_rate,
    "pref_diet_suggestion": pref_diet_suggestion, "pref_workout_plan": pref_workout_plan,
    "pref_personalized_report": pref_personalized_report,
    "battery_life_importance": battery_life_importance, "design_importance": design_importance,
    "brand_preference": brand_preference,
    # Spending & purchase
    "monthly_health_spend": monthly_health_spend,
    "preferred_price_range": preferred_price_range, "willingness_to_pay": willingness_to_pay,
    "payment_preference": payment_preference, "discount_sensitivity": discount_sensitivity,
    "subscription_willingness": subscription_willingness,
    "purchase_intention": purchase_intention, "likely_to_purchase": likely_to_purchase,
    "upgrade_intention": upgrade_intention, "family_influence": family_influence,
    "social_media_influence": social_media_influence, "referrals_made": referrals_made,
    # Engineered features
    "health_concern_score": health_concern_score,
    "fitness_readiness_score": fitness_readiness_score,
    "lifestyle_risk_score": lifestyle_risk_score,
    "digital_adoption_score": digital_adoption_score,
    "price_sensitivity_score": price_sensitivity_score,
    "purchase_potential_score": purchase_potential_score,
    "engagement_score": engagement_score,
    "sleep_stress_risk_category": sleep_stress_risk_category,
    "budget_segment": budget_segment,
    "customer_value_segment": customer_value_segment,
    # Feedback
    "feedback_text": feedback_text,
})


# ---------------------------------------------------------------------------
# 10. Inject realistic noise / missingness, then clean
# ---------------------------------------------------------------------------
# Introduce a few missing values in a couple of soft columns
for col, frac in [("water_intake", 0.03), ("diet_quality", 0.02), ("monthly_health_spend", 0.02)]:
    idx = RNG.choice(N, size=int(N * frac), replace=False)
    df.loc[idx, col] = np.nan

# A handful of duplicate rows + a couple of outliers (to demonstrate cleaning)
dupes = df.sample(8, random_state=1)
df = pd.concat([df, dupes], ignore_index=True)
df.loc[df.sample(5, random_state=2).index, "daily_steps"] = 99999  # outliers

# --- Cleaning ---
df = df.drop_duplicates(subset="respondent_id").reset_index(drop=True)
# Cap step outliers at a sane physiological max
df.loc[df["daily_steps"] > 30000, "daily_steps"] = 30000
# Impute soft missing values
df["water_intake"] = df["water_intake"].fillna(round(df["water_intake"].median(), 1))
df["diet_quality"] = df["diet_quality"].fillna(df["diet_quality"].mode()[0])
df["monthly_health_spend"] = df["monthly_health_spend"].fillna(
    int(df["monthly_health_spend"].median()))

# Keep at least 2000 clean rows
df = df.head(2000).reset_index(drop=True) if len(df) >= 2000 else df

df.to_csv("fittrack_data.csv", index=False)
print(f"Saved fittrack_data.csv  ->  {df.shape[0]} rows x {df.shape[1]} columns")
print("Purchase rate (Yes):", round((df['likely_to_purchase'] == 'Yes').mean() * 100, 1), "%")
print("Avg willingness to pay: INR", int(df['willingness_to_pay'].mean()))
