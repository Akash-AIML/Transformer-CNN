import streamlit as st
import pandas as pd
import numpy as np
from backend.model_backend import load_model, predict
import altair as alt

st.set_page_config(page_title="Energy Forecasting", layout="wide")
st.title("Energy Consumption Forecasting")

# ----------------- Input Options -----------------
input_option = st.radio(
    "Choose input type:",
    ("Single Manual Input", "Multiple Manual Input", "Upload CSV")
)

df_input = None

# ---------- Single Manual Input ----------
if input_option == "Single Manual Input":
    temp = st.number_input("Temperature", value=25)
    energy = st.number_input("EnergyConsumption", value=100)
    hour = st.number_input("HourOfDay", value=12, min_value=0, max_value=23)
    day = st.number_input("DayOfWeek", value=0, min_value=0, max_value=6)
    
    df_input = pd.DataFrame([{
        "Temperature": temp,
        "EnergyConsumption": energy,
        "HourOfDay": hour,
        "DayOfWeek": day
    }])

# ---------- Multiple Manual Input ----------
elif input_option == "Multiple Manual Input":
    st.info("Enter multiple rows manually. Press 'Add Row' for more rows.")
    if 'manual_data' not in st.session_state:
        st.session_state.manual_data = [{"Temperature":25,"EnergyConsumption":100,"HourOfDay":12,"DayOfWeek":0}]
    
    manual_data = st.session_state.manual_data
    rows_to_remove = []

    for i, row in enumerate(manual_data):
        col1, col2, col3, col4, col5 = st.columns([1,1,1,1,0.3])
        with col1:
            row["Temperature"] = st.number_input(f"Temp {i+1}", value=row["Temperature"], key=f"temp_{i}")
        with col2:
            row["EnergyConsumption"] = st.number_input(f"Energy {i+1}", value=row["EnergyConsumption"], key=f"energy_{i}")
        with col3:
            row["HourOfDay"] = st.number_input(f"Hour {i+1}", value=row["HourOfDay"], key=f"hour_{i}", min_value=0, max_value=23)
        with col4:
            row["DayOfWeek"] = st.number_input(f"Day {i+1}", value=row["DayOfWeek"], key=f"day_{i}", min_value=0, max_value=6)
        with col5:
            if st.button("Remove", key=f"remove_{i}"):
                rows_to_remove.append(i)
    
    for idx in reversed(rows_to_remove):
        manual_data.pop(idx)
    
    if st.button("Add Row"):
        manual_data.append({"Temperature":25,"EnergyConsumption":100,"HourOfDay":12,"DayOfWeek":0})
    
    st.session_state.manual_data = manual_data
    df_input = pd.DataFrame(manual_data)

# ---------- Upload CSV ----------
elif input_option == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded_file:
        df_input = pd.read_csv(uploaded_file)
        st.success("CSV loaded successfully.")

# ---------- Prediction ----------
if df_input is not None and not df_input.empty:
    st.subheader("Input Data")
    st.dataframe(df_input)

    try:
        predictions = predict(df_input)
        df_input["PredictedEnergyConsumption"] = predictions

        st.subheader("Predictions")
        st.dataframe(df_input)

        # ---------------- Graphs ----------------
        if len(df_input) > 1:
            df_plot = df_input.copy().reset_index(drop=True)
            # Ensure numeric and fill NaNs
            df_plot[['EnergyConsumption','PredictedEnergyConsumption']] = df_plot[['EnergyConsumption','PredictedEnergyConsumption']].apply(pd.to_numeric, errors='coerce').fillna(0)
            df_plot['index'] = df_plot.index

            # Line chart: Actual vs Predicted
            line_data = df_plot.melt(id_vars=['index'], value_vars=['EnergyConsumption','PredictedEnergyConsumption'], 
                                     var_name='Variable', value_name='Value')
            line_chart = alt.Chart(line_data).mark_line(point=True).encode(
                x='index',
                y='Value',
                color='Variable'
            ).properties(title="Actual vs Predicted Energy Consumption")
            st.altair_chart(line_chart, use_container_width=True)

            # Scatter plot
            scatter = alt.Chart(df_input).mark_circle(size=60).encode(
                x=alt.X('EnergyConsumption', type='quantitative'),
                y=alt.Y('PredictedEnergyConsumption', type='quantitative'),
                tooltip=['Temperature','HourOfDay','DayOfWeek']
            ).properties(title="Predicted vs Actual Scatter Plot")
            st.altair_chart(scatter, use_container_width=True)

            # Residuals histogram
            df_plot['Residual'] = df_plot['EnergyConsumption'] - df_plot['PredictedEnergyConsumption']
            hist = alt.Chart(df_plot).mark_bar().encode(
                x=alt.X('Residual', bin=alt.Bin(maxbins=30)),
                y='count()'
            ).properties(title="Residuals Distribution")
            st.altair_chart(hist, use_container_width=True)

            # Residuals over index
            res_line = alt.Chart(df_plot).mark_line().encode(
                x='index',
                y='Residual'
            ).properties(title="Residuals over Index")
            st.altair_chart(res_line, use_container_width=True)

        else:
            # Single row: show bar chart
            single_row = df_input.iloc[0]
            chart_data = pd.DataFrame({
                'Type': ['Actual', 'Predicted'],
                'EnergyConsumption': [single_row['EnergyConsumption'], single_row['PredictedEnergyConsumption']]
            })
            chart = alt.Chart(chart_data).mark_bar(color='teal').encode(
                x='Type',
                y='EnergyConsumption'
            )
            st.altair_chart(chart, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction error: {e}")
