import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timezone
from train import parse_area

st.set_page_config("Yogyakarta House Price Prediction", layout="centered")

DATA_PATH = Path("data/rumah123_yogya_unfiltered.csv")
MODEL_LR = Path("model/linear_model.pkl")
MODEL_XGB = Path("model/xgb_model.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH).dropna()
    df["surface_area"] = df["surface_area"].apply(parse_area)
    df["building_area"] = df["building_area"].apply(parse_area)
    return df

@st.cache_resource
def load_models():
    return {
        "Linear Regression": joblib.load(MODEL_LR),
        "XGBoost": joblib.load(MODEL_XGB)
    }

df = load_data()
models = load_models()

st.title("Yogyakarta House Price Prediction")
st.write(
    """
    Compare **Linear Regression (baseline)** with **XGBoost (gradient boosting)**
    for predicting house prices in Yogyakarta, Indonesia.
    """
)

lr_metrics = models["Linear Regression"]["metrics"]
xgb_metrics = models["XGBoost"]["metrics"]

delta_metrics = {
    "mae": lr_metrics["mae"] - xgb_metrics["mae"],
    "rmse": lr_metrics["rmse"] - xgb_metrics["rmse"],
    "r2": xgb_metrics["r2"] - lr_metrics["r2"]
}

model_choice = st.selectbox("Choose Model", list(models.keys()))
artifact = models[model_choice]
model = artifact["model"]
features = artifact["features"]

st.sidebar.header("House Feature")

input_data = {}
for col in features["numerical"]:
    input_data[col] = st.sidebar.number_input(
        col.replace("_", " ").title(),
        value=float(df[col].median())
    )

for col in features["categorical"]:
    input_data[col] = st.sidebar.selectbox(
        col.replace("_", " ").title(),
        sorted(df[col].unique())
    )

input_df = pd.DataFrame([input_data])

if st.sidebar.button("Predict Price"):
    pred = model.predict(input_df)[0]

    st.success(f"Predicted Price: **IDR {pred:,.0f}**")

    result_df = input_df.copy()
    result_df["predicted_price"] = pred
    result_df["model_used"] = model_choice
    result_df["timestamp"] = datetime.now(timezone.utc).isoformat()

    st.download_button(
        "⬇️ Download Prediction CSV",
        result_df.to_csv(index=False).encode("utf-8"),
        "house_price_prediction.csv",
        "text/csv"
    )

st.subheader("Model Comparison")
cols = st.columns(2)

for col, (name, art) in zip(cols, models.items()):
    with col:
        st.markdown(f"### {name}")
        st.metric("MAE", f"{art['metrics']['mae']:.2f}")
        st.metric("RMSE", f"{art['metrics']['rmse']:.2f}")
        st.metric("R²", f"{art['metrics']['r2']:.3f}")

st.subheader("Metric Improvement (XGBoost vs Linear Regression)")

c1, c2, c3 = st.columns(3)

c1.metric(
    "MAE Improvement",
    f"{delta_metrics['mae']:.2f}",
    help="Positive means XGBoost reduces error"
)

c2.metric(
    "RMSE Improvement",
    f"{delta_metrics['rmse']:.2f}",
    help="Positive means XGBoost reduced error"
)

c3.metric(
    "R² Improvement",
    f"+{delta_metrics['r2']:.3f}",
    help="Positive means better variance eplained"
)

st.subheader("Residual Analysis")
res = artifact["residuals"]

fig, ax = plt.subplots()
ax.scatter(res["y_pred"], res["residual"], alpha=0.6)
ax.axhline(0, color="red", linestyle="--")
ax.set_xlabel("Predicted Price")
ax.set_ylabel("Residual")
st.pyplot(fig)