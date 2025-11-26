import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

# --- CONFIGURATION ---
# Set page to wide mode for a dashboard feel
st.set_page_config(layout="wide", page_title="EcoScale Energy Analytics")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "outputs"

# --- DATA LOADING (Cached for speed) ---
@st.cache_data
def load_data():
    # Load the main dataset (features + raw usage)
    # We only load a subset of columns to keep it fast
    df = pd.read_parquet(
        DATA_DIR / "electricity_features.parquet", 
        columns=['timestamp', 'building_id', 'meter_reading']
    )
    
    # Load the anomalies we detected earlier
    anomalies = pd.read_csv(OUTPUT_DIR / "electricity_anomalies.csv")
    anomalies['timestamp'] = pd.to_datetime(anomalies['timestamp'])
    
    return df, anomalies

# --- SIDEBAR ---
st.sidebar.title("⚡ EcoScale Analytics")
st.sidebar.info("Select a building to view energy performance and detected anomalies.")

try:
    with st.spinner("Loading massive dataset..."):
        df_all, anomalies_all = load_data()
        
    # Get list of buildings that actually have anomalies
    problem_buildings = anomalies_all['building_id'].unique()
    selected_building = st.sidebar.selectbox("Select Building:", problem_buildings)
    
    # Filter data for this building
    df_b = df_all[df_all['building_id'] == selected_building].sort_values('timestamp')
    anom_b = anomalies_all[anomalies_all['building_id'] == selected_building].sort_values('timestamp')

    # --- MAIN PAGE ---
    st.title(f"Building Analysis: {selected_building}")

    # Top Level Metrics
    col1, col2, col3 = st.columns(3)
    
    total_waste = anom_b['wasted_cost'].sum()
    total_kwh_waste = anom_b['wasted_kwh'].sum()
    anomaly_count = len(anom_b)
    
    col1.metric("Est. Wasted Cost", f"${total_waste:,.2f}", delta_color="inverse")
    col2.metric("Wasted Energy", f"{total_kwh_waste:,.0f} kWh", delta_color="inverse")
    col3.metric("Anomalies Detected", f"{anomaly_count} events")

    st.markdown("---")

    # --- CHART: The Virtual Meter ---
    st.subheader("📉 Virtual Meter (Actual vs. Expected)")
    
    # Interactive Date Range Picker
    min_date = df_b['timestamp'].min().date()
    max_date = df_b['timestamp'].max().date()
    
    # Default to viewing the last month of data
    start_date, end_date = st.slider(
        "Select Date Range",
        min_value=min_date,
        max_value=max_date,
        value=(max_date - pd.Timedelta(days=30), max_date)
    )
    
    # Filter by date slider
    mask = (df_b['timestamp'].dt.date >= start_date) & (df_b['timestamp'].dt.date <= end_date)
    df_chart = df_b.loc[mask]
    
    # Filter anomalies by date slider
    mask_anom = (anom_b['timestamp'].dt.date >= start_date) & (anom_b['timestamp'].dt.date <= end_date)
    anom_chart = anom_b.loc[mask_anom]

    # Plotly Chart
    fig = go.Figure()

    # 1. Actual Usage
    fig.add_trace(go.Scatter(
        x=df_chart['timestamp'], 
        y=df_chart['meter_reading'],
        mode='lines',
        name='Actual Usage (kWh)',
        line=dict(color='#1f77b4', width=2)
    ))

    # 2. Expected Usage (we need to merge this back if we want to plot the baseline line, 
    # but for now, let's plot the ANOMALIES as red dots on top of the actuals)
    fig.add_trace(go.Scatter(
        x=anom_chart['timestamp'],
        y=anom_chart['meter_reading'],
        mode='markers',
        name='Anomaly Detected',
        marker=dict(color='red', size=8, symbol='x')
    ))

    fig.update_layout(
        title="Energy Consumption & Anomaly Detection",
        xaxis_title="Time",
        yaxis_title="Energy (kWh)",
        template="plotly_white",
        height=500,
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    # --- DATA TABLE ---
    st.subheader("📋 Recent Anomalies List")
    st.dataframe(
        anom_chart[['timestamp', 'meter_reading', 'expected_reading', 'wasted_kwh', 'wasted_cost']]
        .style.format({
            'meter_reading': '{:.2f}',
            'expected_reading': '{:.2f}', 
            'wasted_kwh': '{:.2f}',
            'wasted_cost': '${:.2f}'
        })
    )

except Exception as e:
    st.error(f"An error occurred: {e}")
    st.write("Make sure you have run the anomaly detection script first!")