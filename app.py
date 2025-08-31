import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import xgboost as xgb

# -------------------------------
# Load Data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")  # Make sure the file is in same folder
    return df

df = load_data()

st.title("📊 Customer Churn Prediction App")

st.write("This app predicts whether a customer will churn based on given features.")

# -------------------------------
# Data Preprocessing
# -------------------------------
# Encode categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# -------------------------------
# Train Models (Offline Training Step)
# -------------------------------
X = df.drop("Churn", axis=1)   # Change target column name if needed
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

# Train each model
for name, model in models.items():
    model.fit(X_train, y_train)

# Evaluate models (just once, offline)
st.subheader("📈 Model Performance on Test Set")
for name, model in models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"**{name}** Accuracy: {acc:.2f}")

# -------------------------------
# User Input
# -------------------------------
st.sidebar.header("Provide Customer Details")

def user_input_features():
    data = {}
    for col in X.columns:
        if col in categorical_cols:
            options = label_encoders[col].classes_
            choice = st.sidebar.selectbox(f"{col}", options)
            data[col] = label_encoders[col].transform([choice])[0]
        else:
            val = st.sidebar.number_input(f"{col}", float(X[col].min()), float(X[col].max()), float(X[col].mean()))
            data[col] = val
    return pd.DataFrame([data])

input_df = user_input_features()

st.subheader("🔍 User Input Features")
st.write(input_df)

# -------------------------------
# Predictions
# -------------------------------
if st.button("Predict Churn"):
    predictions = {}
    for name, model in models.items():
        pred = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]
        predictions[name] = (pred, prob)

    st.subheader("🔮 Predictions")
    for model_name, (pred, prob) in predictions.items():
        st.write(f"**{model_name}**: {'Churn' if pred==1 else 'No Churn'} (Probability: {prob:.2f})")




if st.checkbox("Show ROC Curves"):
    st.write("ROC Curves for all models")
    fig, ax = plt.subplots(figsize=(8,6))
    for name, model in models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
    st.pyplot(fig)
