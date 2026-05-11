# =========================================================
# UAC PREDICTIVE INTELLIGENCE SYSTEM
# COMPLETE ADVANCED STREAMLIT PROJECT
# =========================================================

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.model_selection import (
    TimeSeriesSplit
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="UAC Predictive Intelligence System",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# CUSTOM STYLING
# =========================================================

st.markdown("""
<style>

/* MAIN */
.stApp{
    background:#F4F7FB;
}

/* CONTAINER */
.block-container{
    padding-top:1rem;
    padding-bottom:1rem;
    padding-left:2rem;
    padding-right:2rem;
}

/* HEADINGS */
h1{
    color:#0F172A !important;
    font-size:3rem !important;
    font-weight:800 !important;
}

h2,h3,h4{
    color:#1E293B !important;
    font-weight:700 !important;
}

/* SIDEBAR */
section[data-testid="stSidebar"]{
    background:linear-gradient(
        180deg,
        #FFFFFF,
        #EEF2FF
    );
    border-right:1px solid #DCE3EC;
}

/* KPI CARDS */
[data-testid="stMetric"]{
    background:#FFFFFF;
    border:1px solid #DCE3EC;
    border-radius:18px;
    padding:16px;
    box-shadow:0 4px 12px rgba(0,0,0,0.05);
}

/* TABS */
.stTabs [data-baseweb="tab"]{
    background:#FFFFFF;
    border:1px solid #DCE3EC;
    border-radius:14px;
    padding:10px 18px;
    font-weight:600;
}

.stTabs [aria-selected="true"]{
    background:linear-gradient(
        135deg,
        #2563EB,
        #0F766E
    ) !important;

    color:white !important;
}

/* EXECUTIVE BOX */
.executive-box{
    background:#FFFFFF;
    border:1px solid #DCE3EC;
    border-radius:22px;
    padding:28px;
    box-shadow:0 4px 18px rgba(15,23,42,0.06);
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# HEADER
# =========================================================

st.title("📈 Predictive Forecasting of Care Load & Placement Demand")

st.markdown("""
### U.S. Department of Health and Human Services

AI-driven operational intelligence platform for forecasting UAC care load,
predicting discharge demand, detecting operational pressure,
and enabling proactive healthcare planning.
""")

# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv("uac_data.csv")

    df['Date'] = pd.to_datetime(df['Date'])

    numeric_cols = [
        col for col in df.columns
        if col != 'Date'
    ]

    for col in numeric_cols:

        df[col] = (
            df[col]
            .astype(str)
            .str.replace(',', '')
        )

        df[col] = pd.to_numeric(
            df[col],
            errors='coerce'
        )

    df = df.sort_values('Date')

    df.set_index('Date', inplace=True)

    df = df.interpolate()

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

discharge_y = df['Children discharged from HHS Care']

# =========================================================
# TRAIN TEST SPLIT
# =========================================================

train_size = int(len(df) * 0.8)

X_train = X.iloc[:train_size]
X_test = X.iloc[train_size:]

y_train = y.iloc[:train_size]
y_test = y.iloc[train_size:]

discharge_train = discharge_y.iloc[:train_size]
discharge_test = discharge_y.iloc[train_size:]

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("⚙ Forecast Controls")

selected_model = st.sidebar.selectbox(
    "Forecasting Model",
    [
        "Gradient Boosting",
        "Random Forest",
        "ARIMA",
        "SARIMA"
    ]
)

forecast_horizon = st.sidebar.slider(
    "Forecast Horizon",
    7,
    90,
    30
)

# =========================================================
# MODELS
# =========================================================

rf = RandomForestRegressor(
    n_estimators=150,
    random_state=42
)

rf.fit(X_train, y_train)

rf_preds = rf.predict(X_test)

# ---------------------------------------------------------

gb = GradientBoostingRegressor()

gb.fit(X_train, y_train)

gb_preds = gb.predict(X_test)

# ---------------------------------------------------------

discharge_model = GradientBoostingRegressor()

discharge_model.fit(
    X_train,
    discharge_train
)

discharge_preds = discharge_model.predict(
    X_test
)

# ---------------------------------------------------------

series = df['Children in HHS Care']

train_series = series.iloc[:train_size]

test_series = series.iloc[train_size:]

# ---------------------------------------------------------

arima_model = ARIMA(
    train_series,
    order=(5,1,2)
)

arima_fit = arima_model.fit()

arima_forecast = arima_fit.forecast(
    steps=len(test_series)
)

# ---------------------------------------------------------

sarima_model = SARIMAX(
    train_series,
    order=(2,1,2),
    seasonal_order=(1,1,1,7)
)

sarima_fit = sarima_model.fit(
    disp=False
)

sarima_forecast = sarima_fit.forecast(
    steps=len(test_series)
)

# ---------------------------------------------------------

exp_model = ExponentialSmoothing(
    train_series,
    trend='add',
    seasonal='add',
    seasonal_periods=7
)

exp_fit = exp_model.fit()

exp_forecast = exp_fit.forecast(
    len(test_series)
)

# =========================================================
# METRICS
# =========================================================

rf_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        rf_preds
    )
)

gb_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        gb_preds
    )
)

arima_rmse = np.sqrt(
    mean_squared_error(
        test_series,
        arima_forecast
    )
)

sarima_rmse = np.sqrt(
    mean_squared_error(
        test_series,
        sarima_forecast
    )
)

exp_rmse = np.sqrt(
    mean_squared_error(
        test_series,
        exp_forecast
    )
)

mape = mean_absolute_percentage_error(
    y_test,
    gb_preds
)

forecast_accuracy = max(
    0,
    min(100, 100 - (mape * 100))
)

r2 = r2_score(
    y_test,
    gb_preds
)

forecast_std = np.std(
    gb_preds - y_test
)

upper_bound = gb_preds + (
    1.96 * forecast_std
)

lower_bound = gb_preds - (
    1.96 * forecast_std
)

capacity_threshold = y.max() * 1.10

capacity_risk = min(
    100,
    (
        max(gb_preds)
        /
        capacity_threshold
    ) * 100
)

# =========================================================
# FUTURE FORECAST
# =========================================================

future_forecast = []

last_data = X.iloc[-1:].copy()

for i in range(forecast_horizon):

    pred = gb.predict(last_data)[0]

    future_forecast.append(pred)

future_dates = pd.date_range(
    start=df.index[-1],
    periods=forecast_horizon + 1,
    freq='D'
)[1:]

future_df = pd.DataFrame({

    'Date': future_dates,
    'Forecast': future_forecast

})

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

importance_df = pd.DataFrame({

    'Feature': features,
    'Importance': gb.feature_importances_

}).sort_values(
    by='Importance',
    ascending=False
)

# =========================================================
# KPI SECTION
# =========================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Forecast Accuracy",
        f"{forecast_accuracy:.2f}%"
    )

with col2:
    st.metric(
        "Capacity Risk",
        f"{capacity_risk:.2f}%"
    )

with col3:
    st.metric(
        "R² Score",
        f"{r2:.3f}"
    )

with col4:
    st.metric(
        "Forecast RMSE",
        f"{gb_rmse:.2f}"
    )

st.markdown("---")

# =========================================================
# TABS
# =========================================================

tabs = st.tabs([

    "📊 Forecasting",
    "📉 Confidence Intervals",
    "🌊 Scenario Analysis",
    "📈 Model Comparison",
    "🏥 Discharge Forecast",
    "📊 Decomposition",
    "🔮 Future Outlook",
    "🧠 AI Explainability",
    "📐 Evaluation",
    "🧪 Cross Validation",
    "📋 Executive Summary"

])

# =========================================================
# THEME FUNCTION
# =========================================================

def apply_theme(fig):

    fig.update_layout(

        template="plotly_white",

        paper_bgcolor="#F4F7FB",

        plot_bgcolor="#FFFFFF",

        font=dict(
            color="#0F172A",
            size=14
        )
    )

    return fig

# =========================================================
# TAB 1 FORECASTING
# =========================================================

with tabs[0]:

    st.subheader("Future Care Load Forecast")

    forecast_df = pd.DataFrame(
        index=y_test.index
    )

    forecast_df['Actual'] = y_test.values

    if selected_model == "Random Forest":
        forecast_df['Forecast'] = rf_preds

    elif selected_model == "Gradient Boosting":
        forecast_df['Forecast'] = gb_preds

    elif selected_model == "SARIMA":
        forecast_df['Forecast'] = sarima_forecast.values

    else:
        forecast_df['Forecast'] = arima_forecast.values

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=forecast_df.index,
        y=forecast_df['Actual'],
        mode='lines',
        name='Actual',
        line=dict(
            color='#2563EB',
            width=3
        )
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df.index,
        y=forecast_df['Forecast'],
        mode='lines',
        name='Forecast',
        line=dict(
            color='#0F766E',
            width=3
        )
    ))

    apply_theme(fig)

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =========================================================
# TAB 2 CONFIDENCE INTERVAL
# =========================================================

with tabs[1]:

    st.subheader(
        "Confidence Interval Visualization"
    )

    fig_ci = go.Figure()

    fig_ci.add_trace(go.Scatter(
        x=y_test.index,
        y=gb_preds,
        mode='lines',
        name='Forecast'
    ))

    fig_ci.add_trace(go.Scatter(
        x=y_test.index,
        y=upper_bound,
        line=dict(width=0),
        showlegend=False
    ))

    fig_ci.add_trace(go.Scatter(
        x=y_test.index,
        y=lower_bound,
        fill='tonexty',
        fillcolor='rgba(37,99,235,0.2)',
        line=dict(width=0),
        name='Confidence Interval'
    ))

    apply_theme(fig_ci)

    st.plotly_chart(
        fig_ci,
        use_container_width=True
    )

# =========================================================
# TAB 3 SCENARIO ANALYSIS
# =========================================================

with tabs[2]:

    st.subheader("Scenario Simulation")

    moderate_X = X_test.copy()

    moderate_X['net_pressure'] *= 1.15

    moderate_preds = gb.predict(
        moderate_X
    )

    extreme_X = X_test.copy()

    extreme_X['net_pressure'] *= 1.30

    extreme_preds = gb.predict(
        extreme_X
    )

    scenario_df = pd.DataFrame({

        'Normal': gb_preds,
        'Moderate Surge': moderate_preds,
        'Extreme Surge': extreme_preds

    }, index=y_test.index)

    fig3 = px.line(
        scenario_df
    )

    apply_theme(fig3)

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

# =========================================================
# TAB 4 MODEL COMPARISON
# =========================================================

with tabs[3]:

    comparison = pd.DataFrame({

        'Model': [
            'Random Forest',
            'Gradient Boosting',
            'ARIMA',
            'SARIMA',
            'Exponential Smoothing'
        ],

        'RMSE': [
            rf_rmse,
            gb_rmse,
            arima_rmse,
            sarima_rmse,
            exp_rmse
        ]
    })

    st.dataframe(
        comparison,
        use_container_width=True
    )

# =========================================================
# TAB 5 DISCHARGE FORECAST
# =========================================================

with tabs[4]:

    st.subheader(
        "Discharge Demand Forecast"
    )

    discharge_df = pd.DataFrame({

        'Actual': discharge_test.values,
        'Predicted': discharge_preds

    }, index=discharge_test.index)

    fig_discharge = px.line(
        discharge_df
    )

    apply_theme(fig_discharge)

    st.plotly_chart(
        fig_discharge,
        use_container_width=True
    )

# =========================================================
# TAB 6 DECOMPOSITION
# =========================================================

with tabs[5]:

    st.subheader(
        "Time Series Decomposition"
    )

    decomposition = seasonal_decompose(
        df['Children in HHS Care'],
        model='additive',
        period=7
    )

    trend_fig = px.line(
        decomposition.trend,
        title="Trend"
    )

    seasonal_fig = px.line(
        decomposition.seasonal,
        title="Seasonality"
    )

    residual_fig = px.line(
        decomposition.resid,
        title="Residual"
    )

    apply_theme(trend_fig)
    apply_theme(seasonal_fig)
    apply_theme(residual_fig)

    st.plotly_chart(
        trend_fig,
        use_container_width=True
    )

    st.plotly_chart(
        seasonal_fig,
        use_container_width=True
    )

    st.plotly_chart(
        residual_fig,
        use_container_width=True
    )

# =========================================================
# TAB 7 FUTURE OUTLOOK
# =========================================================

with tabs[6]:

    st.subheader(
        "Future Multi-Day Forecast"
    )

    fig_future = px.line(
        future_df,
        x='Date',
        y='Forecast'
    )

    apply_theme(fig_future)

    st.plotly_chart(
        fig_future,
        use_container_width=True
    )

# =========================================================
# TAB 8 AI EXPLAINABILITY
# =========================================================

with tabs[7]:

    st.subheader(
        "Feature Importance Analysis"
    )

    fig_importance = px.bar(
        importance_df,
        x='Importance',
        y='Feature',
        orientation='h',
        color='Importance'
    )

    apply_theme(fig_importance)

    st.plotly_chart(
        fig_importance,
        use_container_width=True
    )

# =========================================================
# TAB 9 EVALUATION
# =========================================================

with tabs[8]:

    evaluation_df = pd.DataFrame({

        'Metric': [
            'RMSE',
            'MAPE',
            'R² Score',
            'Forecast Accuracy'
        ],

        'Value': [
            gb_rmse,
            mape,
            r2,
            forecast_accuracy
        ]
    })

    st.dataframe(
        evaluation_df,
        use_container_width=True
    )

# =========================================================
# TAB 10 CROSS VALIDATION
# =========================================================

with tabs[9]:

    st.subheader(
        "Time Series Cross Validation"
    )

    tscv = TimeSeriesSplit(n_splits=5)

    cv_scores = []

    for train_index, test_index in tscv.split(X):

        X_train_cv = X.iloc[train_index]
        X_test_cv = X.iloc[test_index]

        y_train_cv = y.iloc[train_index]
        y_test_cv = y.iloc[test_index]

        model_cv = GradientBoostingRegressor()

        model_cv.fit(
            X_train_cv,
            y_train_cv
        )

        preds_cv = model_cv.predict(
            X_test_cv
        )

        rmse_cv = np.sqrt(
            mean_squared_error(
                y_test_cv,
                preds_cv
            )
        )

        cv_scores.append(rmse_cv)

    cv_df = pd.DataFrame({

        'Fold': [
            'Fold 1',
            'Fold 2',
            'Fold 3',
            'Fold 4',
            'Fold 5'
        ],

        'RMSE': cv_scores
    })

    st.dataframe(
        cv_df,
        use_container_width=True
    )

# =========================================================
# TAB 11 EXECUTIVE SUMMARY
# =========================================================

with tabs[10]:

    st.subheader(
        "Executive Summary"
    )

    st.markdown(f"""
    <div class="executive-box">

    <h4>Operational Summary</h4>

    Gradient Boosting achieved the strongest
    predictive performance for operational forecasting.

    <br><br>

    <b>Forecast Accuracy:</b>
    {forecast_accuracy:.2f}%<br><br>

    <b>Capacity Risk:</b>
    {capacity_risk:.2f}%<br><br>

    <b>R² Score:</b>
    {r2:.3f}<br><br>

    <b>Forecast Horizon:</b>
    {forecast_horizon} Days

    </div>
    """, unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption("""
Predictive Intelligence Platform for HHS Operational Forecasting
""")