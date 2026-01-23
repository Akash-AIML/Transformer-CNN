import streamlit as st
import torch
import numpy as np
import pandas as pd
from joblib import load
import json

from model import TimeSeriesTransformer

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(page_title="Energy Forecasting", layout="wide")

# =========================================================
# DEVICE
# =========================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================================================
# LOAD CONFIG (CACHED)
# =========================================================
@st.cache_resource
def load_config():
    with open("model_config.json") as f:
        return json.load(f)

CONFIG = load_config()

FEATURES = CONFIG["feature_order"]
SEQ_LEN = CONFIG["seq_len"]
PRED_LEN = CONFIG["pred_len"]

# =========================================================
# LOAD MODEL & SCALER
# =========================================================
@st.cache_resource
def load_model():
    model = TimeSeriesTransformer(
        input_dim=CONFIG["input_dim"],
        d_model=CONFIG["d_model"],
        num_heads=CONFIG["num_heads"],
        num_encoder_layers=CONFIG["num_encoder_layers"],
        num_decoder_layers=CONFIG["num_decoder_layers"],
        dropout=CONFIG["dropout"],
        output_dim=CONFIG["output_dim"],
    )
    model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


@st.cache_resource
def load_scaler():
    return load("scaler.pkl")


model = load_model()
scaler = load_scaler()

# =========================================================
# SLIDING WINDOW FORECAST
# =========================================================
def sliding_window_forecast(data):
    """
    data: numpy array (N, features)
    returns: dict {timestep: averaged prediction}
    """
    scaled = scaler.transform(data)
    scaled = torch.tensor(scaled, dtype=torch.float32)

    timeline = {}

    with torch.no_grad():
        for i in range(len(scaled) - SEQ_LEN - PRED_LEN + 1):
            src = scaled[i : i + SEQ_LEN].unsqueeze(0).to(DEVICE)

            last_step = src[:, -1:, :]
            tgt = last_step.repeat(1, PRED_LEN, 1)

            preds = model(src, tgt).cpu().numpy()[0, :, 0]

            for j, val in enumerate(preds):
                t = i + SEQ_LEN + j
                timeline.setdefault(t, []).append(val)

    return {k: float(np.mean(v)) for k, v in timeline.items()}

# =========================================================
# UI
# =========================================================
st.title("🔌 Energy Consumption Forecasting")
st.info(
    "📌 If you upload more than 24 rows, **sliding window forecasting** "
    "is used automatically."
)

input_mode = st.radio(
    "Select Input Method",
    ["Upload CSV", "Manual Entry"],
    horizontal=True
)

data = None

# =========================================================
# CSV INPUT
# =========================================================
if input_mode == "Upload CSV":
    file = st.file_uploader("Upload CSV (≥24 rows)", type=["csv"])

    if file:
        df = pd.read_csv(file)

        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
            st.stop()

        if len(df) < SEQ_LEN:
            st.error("CSV must contain at least 24 rows.")
            st.stop()

        data = df[FEATURES].values
        st.success(f"Loaded {len(df)} rows.")

# =========================================================
# MANUAL INPUT
# =========================================================
else:
    st.write("Enter exactly **24 rows**:")

    df_manual = st.data_editor(
        pd.DataFrame(np.zeros((SEQ_LEN, len(FEATURES))), columns=FEATURES),
        num_rows="fixed"
    )

    if st.button("Use Manual Input"):
        data = df_manual.values

# =========================================================
# PREDICTION & VISUALIZATION
# =========================================================
if data is not None:
    try:
        results = sliding_window_forecast(data)

        forecast_df = (
            pd.DataFrame({
                "Timestep": list(results.keys()),
                "Predicted Energy (normalized)": list(results.values())
            })
            .sort_values("Timestep")
            .reset_index(drop=True)
        )

        st.subheader("📈 Sliding Window Forecast")
        st.line_chart(forecast_df.set_index("Timestep"), height=300)

        # ---------------- Rolling Mean ----------------
        st.subheader("📊 Smoothed Forecast (Rolling Mean)")
        max_window = min(24, len(forecast_df))
        window = st.slider("Rolling window size", 3, max_window, min(6, max_window))

        forecast_df["Rolling Mean"] = (
            forecast_df["Predicted Energy (normalized)"]
            .rolling(window)
            .mean()
        )

        st.line_chart(
            forecast_df.set_index("Timestep")[["Rolling Mean"]],
            height=250
        )

        # ---------------- Histogram (BINNED) ----------------
        st.subheader("📉 Prediction Distribution")
        hist_vals, bin_edges = np.histogram(
            forecast_df["Predicted Energy (normalized)"], bins=30
        )
        hist_df = pd.DataFrame({
            "Energy Bin": bin_edges[:-1],
            "Count": hist_vals
        }).set_index("Energy Bin")

        st.bar_chart(hist_df)

        # ---------------- Zoomed View ----------------
        st.subheader("🔍 Zoomed Forecast (Last N Steps)")
        max_zoom = min(200, len(forecast_df))
        zoom_n = st.slider("Show last N predictions", 20, max_zoom, min(50, max_zoom))

        zoom_df = forecast_df.tail(zoom_n)
        st.line_chart(zoom_df.set_index("Timestep"), height=250)

        # ---------------- Volatility ----------------
        st.subheader("⚡ Prediction Volatility")
        forecast_df["Rolling Std"] = (
            forecast_df["Predicted Energy (normalized)"]
            .rolling(window)
            .std()
        )

        st.line_chart(
            forecast_df.set_index("Timestep")[["Rolling Std"]],
            height=250
        )

        # ---------------- Raw Table ----------------
        with st.expander("📋 View Raw Forecast Table"):
            st.dataframe(forecast_df, use_container_width=True)

        st.caption("Predictions are shown in normalized scale.")

    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)
