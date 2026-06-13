import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main{
    background:#f4f7fc;
}

.hero{
    background:linear-gradient(135deg,#2563eb,#06b6d4);
    padding:35px;
    border-radius:20px;
    text-align:center;
    color:white;
    margin-bottom:20px;
}

.hero h1{
    font-size:50px;
}

.hero p{
    font-size:18px;
}

[data-testid="stMetric"]{
    background:white;
    padding:15px;
    border-radius:15px;
    box-shadow:0 4px 20px rgba(0,0,0,0.08);
}

.stButton>button{
    width:100%;
    height:50px;
    border:none;
    border-radius:10px;
    background:linear-gradient(135deg,#2563eb,#06b6d4);
    color:white;
    font-size:18px;
    font-weight:bold;
}

.prediction-box{
    background:white;
    padding:30px;
    border-radius:20px;
    text-align:center;
    box-shadow:0px 5px 20px rgba(0,0,0,0.1);
}

.footer{
    text-align:center;
    padding:20px;
    color:gray;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv("car data.csv")

# =====================================================
# PREPROCESSING
# =====================================================

data = df.copy()

data["Car_Age"] = 2025 - data["Year"]

data.drop(["Car_Name", "Year"], axis=1, inplace=True)

fuel_encoder = LabelEncoder()
selling_encoder = LabelEncoder()
trans_encoder = LabelEncoder()

data["Fuel_Type"] = fuel_encoder.fit_transform(data["Fuel_Type"])
data["Selling_type"] = selling_encoder.fit_transform(data["Selling_type"])
data["Transmission"] = trans_encoder.fit_transform(data["Transmission"])

X = data.drop("Selling_Price", axis=1)
y = data["Selling_Price"]

# =====================================================
# MODEL
# =====================================================

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

model.fit(X, y)

predictions = model.predict(X)

accuracy = r2_score(y, predictions)

# =====================================================
# HERO SECTION
# =====================================================

st.markdown("""
<div class="hero">
<h1>🚗 Smart Car Price Prediction System</h1>
<p>Predict resale value of used cars using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# =====================================================
# DASHBOARD METRICS
# =====================================================

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric("Records", len(df))

with m2:
    st.metric("Features", X.shape[1])

with m3:
    st.metric("Accuracy", f"{accuracy*100:.2f}%")

with m4:
    st.metric("Cars", df["Car_Name"].nunique())

st.markdown("---")

# =====================================================
# USER INPUT
# =====================================================

st.subheader("🚘 Enter Car Details")

col1, col2 = st.columns(2)

with col1:

    present_price = st.number_input(
        "Present Price (Lakhs)",
        min_value=0.0,
        value=5.0
    )

    driven_kms = st.number_input(
        "Driven Kilometers",
        min_value=0,
        value=50000
    )

    owner = st.selectbox(
        "Previous Owners",
        [0,1,2,3]
    )

with col2:

    car_age = st.slider(
        "Car Age",
        0,
        20,
        5
    )

    fuel_type = st.selectbox(
        "Fuel Type",
        ["Petrol","Diesel","CNG"]
    )

    transmission = st.selectbox(
        "Transmission",
        ["Manual","Automatic"]
    )

selling_type = st.selectbox(
    "Selling Type",
    ["Dealer","Individual"]
)

# =====================================================
# MANUAL ENCODING
# =====================================================

fuel_map = {
    "CNG":0,
    "Diesel":1,
    "Petrol":2
}

selling_map = {
    "Dealer":0,
    "Individual":1
}

trans_map = {
    "Automatic":0,
    "Manual":1
}

# =====================================================
# PREDICTION
# =====================================================

if st.button("Predict Price"):

    input_data = np.array([[
        present_price,
        driven_kms,
        fuel_map[fuel_type],
        selling_map[selling_type],
        trans_map[transmission],
        owner,
        car_age
    ]])

    result = model.predict(input_data)[0]

    st.markdown(f"""
    <div class="prediction-box">
        <h2>Estimated Selling Price</h2>
        <h1 style="color:green;">
            ₹ {result:.2f} Lakhs
        </h1>
    </div>
    """, unsafe_allow_html=True)

    confidence = min((result/20)*100,100)

    st.progress(int(confidence))

    st.success(
        f"Predicted Market Value : ₹ {result:.2f} Lakhs"
    )

    report = pd.DataFrame({
        "Present Price":[present_price],
        "Driven KM":[driven_kms],
        "Fuel Type":[fuel_type],
        "Transmission":[transmission],
        "Owner":[owner],
        "Predicted Price":[round(result,2)]
    })

    csv = report.to_csv(index=False)

    st.download_button(
        "📥 Download Prediction Report",
        csv,
        "car_prediction_report.csv",
        "text/csv"
    )

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

st.markdown("---")

st.subheader("📊 Feature Importance")

importance = model.feature_importances_

fig, ax = plt.subplots(figsize=(8,5))

ax.barh(
    X.columns,
    importance
)

ax.set_title("Feature Importance")
ax.set_xlabel("Importance Score")

st.pyplot(fig)

# =====================================================
# ACTUAL VS PREDICTED
# =====================================================

st.markdown("---")

st.subheader("📈 Actual vs Predicted Prices")

fig2, ax2 = plt.subplots(figsize=(8,5))

ax2.scatter(
    y,
    predictions,
    alpha=0.7
)

ax2.set_xlabel("Actual Price")
ax2.set_ylabel("Predicted Price")
ax2.set_title("Actual vs Predicted")

st.pyplot(fig2)

# =====================================================
# TOP FEATURES
# =====================================================

st.markdown("---")

st.subheader("🏆 Top Important Features")

feature_df = pd.DataFrame({
    "Feature":X.columns,
    "Importance":importance
}).sort_values(
    by="Importance",
    ascending=False
)

st.dataframe(feature_df)

# =====================================================
# DATASET PREVIEW
# =====================================================

st.markdown("---")

st.subheader("📁 Dataset Preview")

st.dataframe(df.head(10))

# =====================================================
# DATASET INFO
# =====================================================

st.markdown("---")

st.subheader("📋 Dataset Statistics")

st.dataframe(df.describe())

# =====================================================
# FOOTER
# =====================================================

st.markdown("""
<div class="footer">
<h4>🚀 Built with Streamlit & Machine Learning</h4>
<p>Random Forest Regression • Car Price Prediction Project</p>
</div>
""", unsafe_allow_html=True)