import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)

from statsmodels.tsa.arima.model import ARIMA

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="UAC Predictive Forecasting Dashboard",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# CUSTOM STYLING
# =========================================================

st.markdown("""
<style>

.main {
    background-color: #071018;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
    padding-left: 1.5rem;
    padding-right: 1.5rem;
}

h1 {
    color: #00D4FF !important;
    font-weight: 800 !important;
}

h2, h3 {
    color: #E5F3FF !important;
}

[data-testid="stMetric"] {
    background: linear-gradient(145deg,#0E1A25,#122535);
    border: 1px solid #1F3B52;
    padding: 16px;
    border-radius: 14px;
    box-shadow: 0 0 12px rgba(0,212,255,0.15);
}

[data-testid="stMetricLabel"] {
    color: #9BC9E2;
    font-size: 14px;
    font-weight: 600;
}

[data-testid="stMetricValue"] {
    color: #FFFFFF;
    font-size: 28px;
    font-weight: bold;
}

.alert-green {
    background-color: #0F2E1D;
    border-left: 6px solid #00E676;
    padding: 14px;
    border-radius: 10px;
    color: white;
    margin-bottom: 15px;
}

.alert-yellow {
    background-color: #3A2B00;
    border-left: 6px solid #FFC107;
    padding: 14px;
    border-radius: 10px;
    color: white;
    margin-bottom: 15px;
}

.alert-red {
    background-color: #3A0F12;
    border-left: 6px solid #FF5252;
    padding: 14px;
    border-radius: 10px;
    color: white;
    margin-bottom: 15px;
}

.executive-box {
    background: linear-gradient(135deg,#08131d,#102433);
    padding: 20px;
    border-radius: 16px;
    border: 1px solid #23445D;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================

st.title("📈 Predictive Forecasting of Care Load & Placement Demand")

st.markdown("""
<div class='executive-box'>

### Executive Summary

This operational intelligence dashboard enables predictive monitoring of
future HHS care load, discharge demand, operational pressure,
capacity breach risk, and emergency surge forecasting.

The system combines:

- Machine Learning Forecasting
- Time-Series Intelligence
- Capacity Risk Analytics
- Early Warning Indicators
- Scenario Simulation
- Operational Forecast Monitoring

</div>
""", unsafe_allow_html=True)

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv("uac_data.csv")

    df = df.dropna()

    df['Date'] = pd.to_datetime(df['Date'])

    df['Children in HHS Care'] = (
        df['Children in HHS Care']
        .astype(str)
        .str.replace(',', '')
        .astype(float)
    )

    df = df.sort_values('Date')

    df.set_index('Date', inplace=True)

    return df

df = load_data()

# =========================================================
# FEATURE ENGINEERING
# =========================================================

df['lag_1'] = df['Children in HHS Care'].shift(1)

df['lag_7'] = df['Children in HHS Care'].shift(7)

df['lag_14'] = df['Children in HHS Care'].shift(14)

df['rolling_mean_7'] = (
    df['Children in HHS Care']
    .rolling(7)
    .mean()
)

df['rolling_mean_14'] = (
    df['Children in HHS Care']
    .rolling(14)
    .mean()
)

df['rolling_std_7'] = (
    df['Children in HHS Care']
    .rolling(7)
    .std()
)

df['net_pressure'] = (
    df['Children transferred out of CBP custody']
    -
    df['Children discharged from HHS Care']
)

df['dayofweek'] = df.index.dayofweek

df['month'] = df.index.month

df.dropna(inplace=True)

# =========================================================
# FEATURES
# =========================================================

features = [
    'lag_1',
    'lag_7',
    'lag_14',
    'rolling_mean_7',
    'rolling_mean_14',
    'rolling_std_7',
    'net_pressure',
    'dayofweek',
    'month'
]

X = df[features]

y = df['Children in HHS Care']

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

train_size = int(len(df) * 0.8)

X_train = X.iloc[:train_size]
X_test = X.iloc[train_size:]

y_train = y.iloc[:train_size]
y_test = y.iloc[train_size:]

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙ Forecast Controls")

selected_model = st.sidebar.selectbox(
    "Select Forecasting Model",
    [
        "Gradient Boosting",
        "Random Forest",
        "ARIMA"
    ]
)

forecast_horizon = st.sidebar.selectbox(
    "Forecast Horizon",
    [7,14,30,60]
)

# =========================================================
# RANDOM FOREST
# =========================================================

rf = RandomForestRegressor(
    n_estimators=120,
    random_state=42
)

rf.fit(X_train, y_train)

rf_preds = rf.predict(X_test)

# =========================================================
# GRADIENT BOOSTING
# =========================================================

gb = GradientBoostingRegressor()

gb.fit(X_train, y_train)

gb_preds = gb.predict(X_test)

# =========================================================
# ARIMA
# =========================================================

series = df['Children in HHS Care']

train_series = series.iloc[:train_size]

test_series = series.iloc[train_size:]

arima_model = ARIMA(
    train_series,
    order=(5,1,2)
)

arima_fit = arima_model.fit()

arima_forecast = arima_fit.forecast(
    steps=len(test_series)
)

# =========================================================
# METRICS
# =========================================================

rf_mae = mean_absolute_error(y_test, rf_preds)

rf_rmse = np.sqrt(
    mean_squared_error(y_test, rf_preds)
)

gb_mae = mean_absolute_error(y_test, gb_preds)

gb_rmse = np.sqrt(
    mean_squared_error(y_test, gb_preds)
)

mape = mean_absolute_percentage_error(
    y_test,
    gb_preds
)

forecast_accuracy = 100 - (mape * 100)

arima_mae = mean_absolute_error(
    test_series,
    arima_forecast
)

arima_rmse = np.sqrt(
    mean_squared_error(
        test_series,
        arima_forecast
    )
)

# =========================================================
# ALERTS
# =========================================================

latest_pressure = df['net_pressure'].iloc[-1]

if latest_pressure < 0:

    st.markdown("""
    <div class='alert-green'>
    🟢 LOW RISK: Operational pressure remains stable.
    </div>
    """, unsafe_allow_html=True)

elif latest_pressure < 300:

    st.markdown("""
    <div class='alert-yellow'>
    🟡 MODERATE SURGE RISK: Elevated intake pressure detected.
    </div>
    """, unsafe_allow_html=True)

else:

    st.markdown("""
    <div class='alert-red'>
    🔴 HIGH CAPACITY ALERT: Potential operational overload detected.
    </div>
    """, unsafe_allow_html=True)

# =========================================================
# KPI SECTION
# =========================================================

k1,k2,k3,k4,k5 = st.columns(5)

with k1:
    st.metric(
        "Forecast Accuracy",
        f"{forecast_accuracy:.2f}%"
    )

with k2:
    st.metric(
        "GB MAE",
        f"{gb_mae:.2f}"
    )

with k3:
    st.metric(
        "GB RMSE",
        f"{gb_rmse:.2f}"
    )

with k4:
    st.metric(
        "Operational Pressure",
        f"{latest_pressure:.0f}"
    )

with k5:
    st.metric(
        "Dataset Size",
        len(df)
    )

# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Forecasting",
    "⚠ Risk Intelligence",
    "🌊 Scenario Analysis",
    "📈 Model Comparison",
    "🗺 Regional Intelligence",
    "🏥 Operations Center",
    "📂 Dataset"
])

# =========================================================
# TAB 1
# =========================================================

with tab1:

    col1, col2 = st.columns([3,1])

    with col1:

        forecast_df = pd.DataFrame(index=y_test.index)

        forecast_df['Actual'] = y_test.values

        if selected_model == "Random Forest":
            forecast_df['Forecast'] = rf_preds

        elif selected_model == "Gradient Boosting":
            forecast_df['Forecast'] = gb_preds

        else:
            forecast_df['Forecast'] = arima_forecast.values

        upper_band = forecast_df['Forecast'] + 120
        lower_band = forecast_df['Forecast'] - 120

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df['Actual'],
            mode='lines',
            name='Actual'
        ))

        fig.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df['Forecast'],
            mode='lines',
            name='Forecast'
        ))

        fig.add_trace(go.Scatter(
            x=forecast_df.index,
            y=upper_band,
            line=dict(width=0),
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=forecast_df.index,
            y=lower_band,
            fill='tonexty',
            line=dict(width=0),
            name='Confidence Interval'
        ))

        fig.update_layout(
            height=500,
            template='plotly_dark',
            title='Forecast vs Actual'
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:

        utilization = min(
            int((df['Children in HHS Care'].iloc[-1] / threshold) * 100),
            100
        )

        gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = utilization,
            title = {'text': "Capacity Utilization"},
            gauge = {
                'axis': {'range': [0,100]},
                'bar': {'color': "#00D4FF"},
                'steps': [
                    {'range': [0,60], 'color': "#0F2E1D"},
                    {'range': [60,85], 'color': "#3A2B00"},
                    {'range': [85,100], 'color': "#3A0F12"}
                ]
            }
        ))

        gauge.update_layout(
            height=400,
            template='plotly_dark'
        )

        st.plotly_chart(
            gauge,
            use_container_width=True
        )

# =========================================================
# TAB 2
# =========================================================

with tab2:

    fig_pressure = px.line(
        df,
        x=df.index,
        y='net_pressure',
        title='Operational Pressure Monitoring',
        template='plotly_dark'
    )

    st.plotly_chart(
        fig_pressure,
        use_container_width=True
    )

    st.info("""
    Early-warning systems help HHS planners:

    • Prevent overcrowding  
    • Prepare medical staff  
    • Scale shelters proactively  
    • Reduce operational burnout  
    """)

# =========================================================
# TAB 3
# =========================================================

with tab3:

    moderate_X = X_test.copy()

    moderate_X['net_pressure'] *= 1.15

    extreme_X = X_test.copy()

    extreme_X['net_pressure'] *= 1.30

    moderate_preds = gb.predict(moderate_X)

    extreme_preds = gb.predict(extreme_X)

    scenario_df = pd.DataFrame({
        'Normal': gb_preds,
        'Moderate Surge': moderate_preds,
        'Extreme Surge': extreme_preds
    }, index=y_test.index)

    fig_scenario = px.line(
        scenario_df,
        x=scenario_df.index,
        y=scenario_df.columns,
        template='plotly_dark',
        title='Emergency Surge Simulation'
    )

    st.plotly_chart(
        fig_scenario,
        use_container_width=True
    )

# =========================================================
# TAB 4
# =========================================================

with tab4:

    comparison = pd.DataFrame({

        'Model': [
            'Random Forest',
            'Gradient Boosting',
            'ARIMA'
        ],

        'MAE': [
            rf_mae,
            gb_mae,
            arima_mae
        ],

        'RMSE': [
            rf_rmse,
            gb_rmse,
            arima_rmse
        ]
    })

    st.dataframe(
        comparison,
        use_container_width=True
    )

    importance = gb.feature_importances_

    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    })

    feature_importance = feature_importance.sort_values(
        by='Importance',
        ascending=False
    )

    fig_importance = px.bar(
        feature_importance,
        x='Feature',
        y='Importance',
        template='plotly_dark',
        title='Feature Importance'
    )

    st.plotly_chart(
        fig_importance,
        use_container_width=True
    )

# =========================================================
# TAB 5
# =========================================================

with tab5:

    region_df = pd.DataFrame({
        'Region': [
            'Texas',
            'Arizona',
            'California',
            'New Mexico'
        ],
        'Load': [4200,3100,2800,1900]
    })

    fig_map = px.scatter_geo(
        region_df,
        locations='Region',
        locationmode='USA-states',
        size='Load',
        scope='usa',
        template='plotly_dark',
        title='Regional Flow Monitoring'
    )

    st.plotly_chart(
        fig_map,
        use_container_width=True
    )

# =========================================================
# TAB 6
# =========================================================

with tab6:

    st.subheader("National Operations Center")

    c1,c2 = st.columns(2)

    with c1:

        staff_needed = int(
            df['Children in HHS Care'].iloc[-1] / 50
        )

        st.metric(
            "Estimated Staff Requirement",
            staff_needed
        )

    with c2:

        shelters = int(
            df['Children in HHS Care'].iloc[-1] / 120
        )

        st.metric(
            "Estimated Shelter Capacity",
            shelters
        )

    fig_load = px.area(
        df,
        x=df.index,
        y='Children in HHS Care',
        template='plotly_dark',
        title='Shelter Utilization Trend'
    )

    st.plotly_chart(
        fig_load,
        use_container_width=True
    )

# =========================================================
# TAB 7
# =========================================================

with tab7:

    st.dataframe(
        df.head(),
        use_container_width=True
    )

    results_df = pd.DataFrame({
        'Actual': y_test,
        'Random Forest': rf_preds,
        'Gradient Boosting': gb_preds
    })

    csv = results_df.to_csv(index=True).encode('utf-8')

    st.download_button(
        label="Download Forecast Results",
        data=csv,
        file_name='forecast_results.csv',
        mime='text/csv'
    )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown("""
### Predictive Intelligence for Healthcare & Public-Sector Operations

This dashboard demonstrates:

- Time-Series Forecasting
- Early Warning Systems
- Capacity Intelligence
- Healthcare Operations Analytics
- Government Resource Planning
""")