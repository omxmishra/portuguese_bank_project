import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

#PAGE CONFIG
st.set_page_config(
    page_title="Bank Term Deposit Predictor",
    page_icon="💰",
    layout="wide"
)

#LOAD MODEL 
model = joblib.load("bank_model.pkl")
expected_columns = model.feature_names_in_

# Extract components
preprocessor = model.named_steps["preprocessor"]
rf_model = model.named_steps["model"]

st.title("💰 Portuguese Bank Term Deposit Predictor")
st.markdown("Predict whether a customer will subscribe to a term deposit.")

st.divider()

# INPUT SECTION 
col1, col2 = st.columns(2)

with col1:
    st.subheader("📌 Personal Details")

    age = st.number_input("Age", 18, 100, 30)

    job = st.selectbox("Job", [
        "admin.", "technician", "services", "management", "retired",
        "blue-collar", "unemployed", "entrepreneur",
        "housemaid", "student", "self-employed", "unknown"
    ])

    marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
    education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])
    default = st.selectbox("Credit in Default?", ["yes", "no"])
    housing = st.selectbox("Housing Loan?", ["yes", "no"])
    loan = st.selectbox("Personal Loan?", ["yes", "no"])

with col2:
    st.subheader("📊 Campaign & Economic Details")

    contact = st.selectbox("Contact Type", ["cellular", "telephone"])
    month = st.selectbox("Month", [
        "jan","feb","mar","apr","may","jun",
        "jul","aug","sep","oct","nov","dec"
    ])
    day_of_week = st.selectbox("Day of Week", ["mon","tue","wed","thu","fri"])

    campaign = st.number_input("Campaign Contacts", 0, 50, 1)
    pdays = st.number_input("Pdays", -1, 999, 999)
    previous = st.number_input("Previous Contacts", 0, 50, 0)

    poutcome = st.selectbox("Previous Outcome", ["failure","nonexistent","success"])

    emp_var_rate = st.number_input("Employment Variation Rate", value=1.0)
    cons_price_idx = st.number_input("Consumer Price Index", value=93.0)
    cons_conf_idx = st.number_input("Consumer Confidence Index", value=-40.0)
    euribor3m = st.number_input("Euribor 3M Rate", value=4.0)
    nr_employed = st.number_input("Number of Employees", value=5000.0)

    previously_contacted = st.number_input("Previously Contacted (0/1)", 0, 1, 0)
    campaign_log = st.number_input("Campaign Log", value=0.0)

    campaign_level = st.selectbox(
        "Campaign Level",
        ["low", "medium", "high"]
    )

    previous_campaign_interaction = st.number_input(
        "Previous Campaign Interaction",
        0, 10, 0
    )

st.divider()

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict Subscription"):

    input_dict = {
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "month": month,
        "day_of_week": day_of_week,
        "campaign": campaign,
        "pdays": pdays,
        "previous": previous,
        "poutcome": poutcome,
        "emp.var.rate": float(emp_var_rate),
        "cons.price.idx": float(cons_price_idx),
        "cons.conf.idx": float(cons_conf_idx),
        "euribor3m": float(euribor3m),
        "nr.employed": float(nr_employed),
        "previously_contacted": previously_contacted,
        "campaign_log": float(campaign_log),
        "campaign_level": campaign_level,
        "previous_campaign_interaction": previous_campaign_interaction
    }

    input_data = pd.DataFrame([input_dict])
    input_data = input_data[expected_columns]

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("📈 Prediction Result")

    st.progress(float(probability))

    if prediction == 1:
        st.success(f"High likelihood of Subscription ({probability:.2%})")
        st.info("💡 Recommendation: Prioritize this customer for follow-up.")
    else:
        st.error(f"Low likelihood of Subscription ({probability:.2%})")
        st.info("💡 Recommendation: Consider deprioritizing this lead.")

    st.divider()

    # ---------------- FEATURE IMPORTANCE ----------------
    st.subheader("🔎 Top 10 Important Features")

    # Get transformed feature names
    ohe = preprocessor.named_transformers_["cat"]
    cat_features = ohe.get_feature_names_out()

    num_features = preprocessor.transformers_[1][2]

    all_features = list(cat_features) + list(num_features)

    importances = rf_model.feature_importances_

    feature_importance_df = pd.DataFrame({
        "Feature": all_features,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(6, 4))  # smaller width & height

    ax.barh(
    feature_importance_df["Feature"],
    feature_importance_df["Importance"]
    )

    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Top 10 Feature Importances")

    plt.tight_layout()

    st.pyplot(fig)

    st.divider()
    st.caption("Model: Random Forest inside Scikit-learn Pipeline")