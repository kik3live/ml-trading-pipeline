import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("models/best_model.pkl")

st.set_page_config(
    page_title="Trading ML App",
    page_icon="📈",
    layout="centered"
)

st.title("📈 Trading ML Prediction App")

st.markdown("""
Aplicación de Machine Learning para generar señales predictivas sobre SPY.
""")

st.header("📊 Market Inputs")

aapl_return = st.slider(
    "AAPL Daily Return",
    min_value=-0.1,
    max_value=0.1,
    value=0.01,
    step=0.001
)

nvda_return = st.slider(
    "NVDA Daily Return",
    min_value=-0.1,
    max_value=0.1,
    value=0.01,
    step=0.001
)

spy_return = st.slider(
    "SPY Daily Return",
    min_value=-0.1,
    max_value=0.1,
    value=0.01,
    step=0.001
)

if st.button("Generate Trading Signal"):

    features = np.array([[
        aapl_return,
        nvda_return,
        spy_return
    ]])

    probability = model.predict_proba(features)[0][1]

    prediction = "📈 UP" if probability > 0.5 else "📉 DOWN"

    st.subheader("Prediction")

    st.metric(
        label="SPY Direction",
        value=prediction
    )

    st.metric(
        label="Probability",
        value=f"{probability:.2%}"
    )

    st.success("Prediction generated successfully.")