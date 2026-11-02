
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# CUSTOM CSS - MODERN FRONTEND
# --------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #f4f7ff 0%, #eef2ff 50%, #f8fafc 100%);
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}

.hero {
    background: linear-gradient(120deg, #101b46, #3949ab, #6757d9);
    padding: 38px;
    border-radius: 22px;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 12px 35px rgba(45, 55, 130, 0.18);
}

.hero h1 {
    color: white;
    font-size: 36px;
    font-weight: 800;
    margin-bottom: 8px;
}

.hero p {
    color: #e0e7ff;
    font-size: 15px;
    margin-bottom: 0;
}

.section-title {
    color: #18234b;
    font-size: 21px;
    font-weight: 700;
    margin-top: 20px;
    margin-bottom: 12px;
}

.info-card {
    background: white;
    border: 1px solid #e4e9f5;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 5px 18px rgba(20, 35, 90, 0.05);
    margin-bottom: 15px;
}

.result-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 28px;
    text-align: center;
    box-shadow: 0 8px 25px rgba(20, 35, 90, 0.07);
}

.result-label {
    color: #64748b;
    font-size: 14px;
    font-weight: 600;
}

.result-value {
    color: #172554;
    font-size: 32px;
    font-weight: 800;
    margin: 10px 0;
}

.small-note {
    color: #64748b;
    font-size: 13px;
}

div.stButton > button {
    background: linear-gradient(90deg, #3949ab, #6757d9);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 13px 24px;
    font-weight: 700;
    width: 100%;
    transition: 0.2s ease;
}

div.stButton > button:hover {
    background: linear-gradient(90deg, #283593, #5142c8);
    color: white;
    transform: translateY(-2px);
}

[data-testid="stSidebar"] {
    background: #111b3c;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

[data-testid="stSidebar"] .stCaption {
    color: #cbd5e1 !important;
}

div[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border: 1px solid #e2e8f0;
    border-radius: 15px;
}

div[data-testid="stMetricValue"] {
    color: #26358b;
    font-weight: 800;
}

.stAlert {
    border-radius: 12px;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD TRAINED MODEL AND PREPROCESSOR
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "telco_churn_ann.keras"
PREPROCESSOR_PATH = BASE_DIR / "telco_churn_preprocessor.joblib"
FEATURES_PATH = BASE_DIR / "telco_churn_feature_columns.joblib"


@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    feature_info = joblib.load(FEATURES_PATH)

    return model, preprocessor, feature_info


try:
    if not all(path.exists() for path in [
        MODEL_PATH, PREPROCESSOR_PATH, FEATURES_PATH
    ]):
        st.error(
            "Model files are missing. Please place all three trained "
            "model files in the same folder as app.py."
        )
        st.stop()

    model, preprocessor, feature_info = load_artifacts()

except Exception as e:
    st.error(f"Unable to load the trained model: {e}")
    st.stop()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown("## 📡 Telco Analytics")
    st.caption("AI-powered customer retention")

    st.markdown("---")

    st.markdown("### About this project")
    st.write(
        "This application uses an Artificial Neural Network "
        "to estimate whether a telecommunications customer "
        "is likely to leave the service."
    )

    st.markdown("---")

    st.markdown("### Model information")
    st.write("**Algorithm:** Artificial Neural Network")
    st.write("**Framework:** TensorFlow / Keras")
    st.write("**Preprocessing:** Scikit-learn Pipeline")
    st.write("**Task:** Binary Classification")

    st.markdown("---")
    st.caption("Developed as a Data Science & Deep Learning project.")


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div class="hero">
    <h1>📡 Customer Churn Prediction</h1>
    <p>
        Predict customer retention with Artificial Intelligence.
        Enter customer details below to estimate churn probability
        and support smarter retention decisions.
    </p>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# DASHBOARD INTRODUCTION
# --------------------------------------------------

st.markdown("### Welcome to the Churn Prediction Dashboard")

st.write(
    "Provide the customer's account, service and billing details. "
    "The trained ANN model will analyze the information and generate "
    "a churn prediction."
)

st.markdown("---")


# --------------------------------------------------
# CUSTOMER INPUT FORM
# --------------------------------------------------

st.markdown(
    '<div class="section-title">👤 Customer Information</div>',
    unsafe_allow_html=True
)

with st.form("customer_form"):

    # CUSTOMER PROFILE
    st.markdown("#### 1. Personal Details")

    c1, c2, c3 = st.columns(3)

    with c1:
        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

    with c2:
        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No"
        )

    with c3:
        partner = st.selectbox(
            "Has Partner?",
            ["Yes", "No"]
        )

    c4, c5, c6 = st.columns(3)

    with c4:
        dependents = st.selectbox(
            "Has Dependents?",
            ["Yes", "No"]
        )

    with c5:
        tenure = st.number_input(
            "Tenure (Months)",
            min_value=0,
            max_value=72,
            value=12,
            step=1
        )

    with c6:
        contract = st.selectbox(
            "Contract Type",
            ["Month-to-month", "One year", "Two year"]
        )

    st.markdown("---")

    # PHONE SERVICES
    st.markdown("#### 2. Phone and Internet Services")

    c1, c2, c3 = st.columns(3)

    with c1:
        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

    with c2:
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "Yes", "No phone service"]
        )

    with c3:
        internet_service = st.selectbox(
            "Internet Service",
            ["DSL", "Fiber optic", "No"]
        )

    st.markdown("#### Additional Services")

    c1, c2 = st.columns(2)

    internet_options = ["Yes", "No", "No internet service"]

    with c1:
        online_security = st.selectbox(
            "Online Security", internet_options
        )

        online_backup = st.selectbox(
            "Online Backup", internet_options
        )

        device_protection = st.selectbox(
            "Device Protection", internet_options
        )

    with c2:
        tech_support = st.selectbox(
            "Tech Support", internet_options
        )

        streaming_tv = st.selectbox(
            "Streaming TV", internet_options
        )

        streaming_movies = st.selectbox(
            "Streaming Movies", internet_options
        )

    st.markdown("---")

    # BILLING DETAILS
    st.markdown("#### 3. Billing and Payment Details")

    c1, c2, c3 = st.columns(3)

    with c1:
        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

    with c2:
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    with c3:
        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=1000.0,
            value=70.0,
            step=1.0
        )

    total_charges = st.number_input(
        "Total Charges",
        min_value=0.0,
        max_value=100000.0,
        value=840.0,
        step=10.0
    )

    st.markdown("---")

    predict_button = st.form_submit_button(
        "🔍 Predict Customer Churn"
    )


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if predict_button:

    # Create customer input dictionary
    customer_data = {
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    # Use the exact input feature order from training
    input_columns = feature_info["input_columns"]

    input_df = pd.DataFrame(
        [customer_data],
        columns=input_columns
    )

    try:
        # Apply the saved preprocessing pipeline
        processed_input = preprocessor.transform(input_df).astype(
            "float32"
        )

        # Predict churn probability
        probability = float(
            model.predict(processed_input, verbose=0)[0][0]
        )

        # Use the threshold saved during training
        threshold = float(feature_info["threshold"])

        prediction = (
            "Likely to Churn"
            if probability >= threshold
            else "Likely to Stay"
        )

        st.markdown("---")
        st.markdown("## 📊 Prediction Results")

        # Result cards
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(
                '<div class="result-card">'
                '<div class="result-label">PREDICTED CUSTOMER STATUS</div>'
                f'<div class="result-value">{prediction}</div>'
                '<div class="small-note">Based on the trained ANN model</div>'
                '</div>',
                unsafe_allow_html=True
            )

        with col2:
            st.markdown(
                '<div class="result-card">'
                '<div class="result-label">CHURN PROBABILITY</div>'
                f'<div class="result-value">{probability:.2%}</div>'
                '<div class="small-note">Estimated probability of churn</div>'
                '</div>',
                unsafe_allow_html=True
            )

        st.markdown("### Churn Probability Indicator")

        st.progress(
            min(max(probability, 0.0), 1.0),
            text=f"Estimated churn probability: {probability:.2%}"
        )

        # Business interpretation
        st.markdown("### 💡 Business Recommendation")

        if probability >= threshold:
            st.warning(
                "This customer is predicted to be likely to churn. "
                "Consider prioritizing retention efforts, contacting "
                "the customer, reviewing service concerns, and "
                "offering suitable retention benefits."
            )
        else:
            st.success(
                "This customer is predicted to be likely to stay. "
                "Continue providing good service and maintain "
                "customer engagement."
            )

        st.caption(
            f"Classification threshold: {threshold:.2f}. "
            "The probability is a model estimate, not a guarantee "
            "of future customer behavior."
        )

        with st.expander("View submitted customer details"):
            st.dataframe(
                input_df,
                use_container_width=True,
                hide_index=True
            )

    except Exception as e:
        st.error(f"Prediction failed: {e}")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; color:#64748b; padding:15px;">
        <b>Telco Customer Churn Prediction</b><br>
        Built with Python • Streamlit • TensorFlow • Scikit-learn
        <br><br>
        <span style="font-size:12px;">
        ANN-based customer churn prediction project
        </span>
    </div>
    """,
    unsafe_allow_html=True
)
