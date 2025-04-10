import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load('model.pkl')

# Load the original dataset for scaling and column reference
data = pd.read_csv('creditcard.csv')

# Prepare features and labels
X = data.drop(columns='Class', axis=1)
y = data['Class']

# Fit scaler on original features
scaler = StandardScaler()
scaler.fit(X)

# Transform the entire dataset for model accuracy display
X_scaled = scaler.transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
y_pred = model.predict(X_scaled_df)
acc = accuracy_score(y, y_pred)

# Set up Streamlit UI
st.set_page_config(page_title="Credit Card Fraud Detector", layout="centered")
st.title("💳 Credit Card Fraud Detection App")
st.markdown("Check if a credit card transaction is **legitimate** or **fraudulent** by entering 30 feature values.")

# Show model accuracy
st.sidebar.header("📊 Model Accuracy")
st.sidebar.write(f"Accuracy on full dataset: `{acc:.2%}`")

# Input section
input_features = st.text_input("📥 Enter 30 feature values (comma-separated):")

if st.button("Predict"):
    try:
        # Convert input to list
        input_list = [float(x.strip()) for x in input_features.split(',')]

        if len(input_list) != 30:
            st.error("⚠️ You must enter exactly 30 feature values.")
        else:
            # Convert to NumPy array and scale
            input_array = np.array(input_list).reshape(1, -1)
            input_scaled = scaler.transform(input_array)

            # Convert to DataFrame to match training format (fixes warning)
            input_df = pd.DataFrame(input_scaled, columns=X.columns)

            # Make prediction
            prediction = model.predict(input_df)
            prob = model.predict_proba(input_df)[0][1]

            if prediction[0] == 0:
                st.success("✅ Legitimate transaction")
                st.info(f"Fraud Probability: `{prob:.2%}`")
            else:
                st.error("🚨 Fraudulent transaction detected!")
                st.info(f"Fraud Probability: `{prob:.2%}`")
    except Exception as e:
        st.error(f"❌ Error: {e}")
