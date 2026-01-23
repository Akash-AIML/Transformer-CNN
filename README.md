

# ⚡ Energy Consumption Forecasting Dashboard

An end-to-end **Transformer-based time series forecasting application** for predicting energy consumption using historical and contextual features.
This project demonstrates **time-series modeling, inference design, visualization, and cloud deployment** using Streamlit.

---

## 🚀 Live Demo

👉 **Streamlit App:**
[https://transformer-cnn-energy-forecasting.streamlit.app/](https://transformer-cnn-energy-forecasting.streamlit.app/)

Users can:

* Enter **manual time-series data (minimum 24 rows)**
* Upload a **CSV file**
* Visualize forecasts and analytical insights interactively

---

## 🧠 Model Overview

* **Architecture:** Transformer (sequence-to-sequence)
* **Task:** Multivariate time-series forecasting
* **Prediction target:** Energy Consumption
* **Input features:**

  * Temperature
  * EnergyConsumption (historical)
  * Hour of Day
  * Day of Week

The model is **trained offline** and loaded **only for inference** during application runtime.

---

## 🔮 Prediction Logic (Core Design)

### 1️⃣ Minimum Input Requirement

* The model is trained on **24 consecutive time steps**
* **At least 24 rows are required** for prediction
* This applies to:

  * Manual input
  * CSV upload

If fewer than 24 rows are provided, the app will not generate predictions.

> This design ensures predictions remain consistent with the model’s training assumptions.

---

### 2️⃣ Forecast Horizon

* Given **24 input rows**
* The model predicts the **next 12 time steps**

---

### 3️⃣ Sliding Window Forecast (For Large Inputs)

When more than 24 rows are provided, the app applies a **sliding window strategy**:

* Rows 1–24 → predict row 25
* Rows 2–25 → predict row 26
* Rows 3–26 → predict row 27
* … and so on

This approach:

* Preserves temporal continuity
* Enables long-sequence forecasting
* Mimics real-world deployment behavior

---

## 📊 Visual Analytics & Forecast Interpretation

The dashboard is designed not only to generate predictions, but also to **explain, validate, and interpret model behavior** through multiple complementary visualizations.

---

### 📈 Sliding Window Forecast

**What it shows**

* Forecasted energy consumption generated using a **24-step sliding window**

**Why it is used**

* Mirrors real-world forecasting usage
* Shows how predictions evolve over time

---

### 📊 Smoothed Forecast (Rolling Mean)

**What it shows**

* Rolling mean of predicted energy consumption
* User-adjustable rolling window size

**Why it is used**

* Reduces short-term noise
* Highlights long-term demand trends

---

### 📉 Prediction Distribution

**What it shows**

* Histogram of predicted values (normalized scale)

**Why it is used**

* Analyzes forecast spread and uncertainty
* Helps detect bias or skew

---

### 🔍 Zoomed Forecast (Last N Steps)

**What it shows**

* Focused view of the most recent predictions

**Why it is used**

* Supports short-term monitoring
* Avoids clutter in long time series

---

### ⚡ Prediction Volatility

**What it shows**

* Short-term variability in predictions

**Why it is used**

* Identifies unstable demand periods
* Supports reliability analysis

---

### 📋 Raw Forecast Table

**What it shows**

* Tabular view of predictions aligned with input rows

**Why it is used**

* Transparency and debugging
* Export-ready for downstream analysis

---

## 🧠 Summary

This project combines:

* Transformer-based modeling
* Correct time-series inference design
* Interpretability through analytics
* Production-ready deployment

It demonstrates a **complete ML lifecycle**, from model design to user-facing deployment.
