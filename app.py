# --- Import Libraries ---
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objs as go
import plotly.figure_factory as ff
from datetime import timedelta, date
from io import BytesIO
from scipy import stats
from streamlit_option_menu import option_menu

# --- Load Model and Scaler ---
icon_img = "Images/icon.png"
best_model_xg = joblib.load('best_model_xgboost.pkl')
scaler = joblib.load('scaler_xgboost.pkl')

# --- Load Historical Sales Data ---
sales_df = pd.read_csv('bdh_forecast_exercise.csv', index_col=0, parse_dates=True)
assert 'orders' in sales_df.columns, "Missing 'orders' column."

# --- Feature Engineering Function ---
def feature_engineer(df):
    """Generate date-related features and one-hot encodings."""
    df = df.copy()
    df['dayofweek'] = df.index.dayofweek
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(float)
    df['month'] = df.index.month
    df['day_of_month'] = df.index.day
    df['is_month_start'] = df.index.is_month_start.astype(float)
    df['is_month_end'] = df.index.is_month_end.astype(float)
    df['quarter'] = df.index.quarter
    df['year'] = df.index.year

    # One-hot encoding
    dow_dummies = pd.get_dummies(df['dayofweek'], prefix='dow', drop_first=True).astype(float)
    month_dummies = pd.get_dummies(df['month'], prefix='month', drop_first=True).astype(float)
    quarter_dummies = pd.get_dummies(df['quarter'], prefix='quarter', drop_first=True).astype(float)

    df = pd.concat([df, dow_dummies, month_dummies, quarter_dummies], axis=1)
    df.drop(columns=['dayofweek', 'month', 'quarter'], inplace=True)
    return df

# --- Prepare training features ---
sales_fe = feature_engineer(sales_df)
sales_fe = sales_fe.drop(columns=['orders'], errors='ignore')

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Sales Forecasting App",
    layout="wide",
    page_icon=icon_img
)

# --- Sidebar for Settings and Navigation ---
with st.sidebar:
    
    # Forecast Settings Title
    st.markdown("<h1 style='text-align: left; color: #4CAF50;'>⚙️ Forecast Settings</h1>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Horizontal Navigation Menu
    page = option_menu(
        menu_title="Navigation",
        options=["Forecast", "About"],
        icons=["bar-chart", "info-circle"],
        orientation="horizontal",
        default_index=0,
        styles={
            "container": {"padding": "15px", "background-color": "#758184", "display": "flex", "justify-content": "center", "gap": "20px", "border-radius": "10px"},
            "icon": {"color": "white", "font-size": "18px"},
            "nav-link": {"font-size": "16px", "padding": "5px", "text-align": "center", "margin": "5px", "color": "black", "font-weight": "bold", "--hover-color": "#e0f7fa"},
            "nav-link-selected": {"font-size": "18px", "background-color": "#00b4d8", "color": "white", "font-weight": "bold"},
        }
    )

    st.markdown("---")

    # Forecast Controls
    n_days = st.slider("📅 Forecast Days:", min_value=2, max_value=730, value=30)
    default_start_date = date(2025, 1, 23)
    start_date = st.date_input("📆 Select Start Date:", value=default_start_date)

# --- Initialize Forecast Session State ---
if "forecast_result" not in st.session_state:
    st.session_state.forecast_result = None

# --- About Page ---
if page == "About":
    st.title("ℹ️ About This App")
    st.write("""
    Welcome to the **Smart XGBoost Forecasting App**!

    🚀 Predict future sales based on historical data.

    - Risk detection zones  
    - Forecast growth insights  
    - Download forecasts  
    - Visualize forecast distribution & residuals
    """)
    st.stop()

# --- Main Forecast Page ---
st.title("📈 XGBoost Sales Forecasting")
st.markdown("---")

# --- Forecasting Button ---
if st.sidebar.button("✅ Confirm and Forecast"):
    with st.spinner("Generating Forecast..."):
        future_idx = pd.date_range(start=start_date, periods=n_days)
        df_future = pd.DataFrame(index=future_idx)
        X_future = feature_engineer(df_future)

        # Handle missing columns
        for col in sales_fe.columns:
            if col not in X_future.columns:
                X_future[col] = 0.0
        X_future = X_future[sales_fe.columns]

        # Scale and Predict
        X_future_scaled = scaler.transform(X_future)
        forecast = pd.Series(best_model_xg.predict(X_future_scaled), index=future_idx)

        st.session_state.forecast_result = forecast

    st.success("✅ Forecast Ready!")
    st.balloons()

# --- Display Tabs After Forecast ---
if st.session_state.forecast_result is not None:
    forecast = st.session_state.forecast_result
    full_series = pd.concat([sales_df['orders'], forecast])

    # Secondary navigation menu for forecast analysis
    selected_tab = option_menu(
        menu_title="Choose what to view:",
        options=["Forecast Plot", "Diagnostic Analysis", "Smart Forecast Summary", "Forecast Table"],
        icons=["bar-chart", "activity", "lightbulb", "table"],
        orientation="horizontal",
        default_index=0,
        styles={
            "container": {"padding": "10px", "background-color": "#B6BBC4"},
            "icon": {"color": "#142D4C", "font-size": "20px"},
            "nav-link": {"font-size": "15px", "padding": "5px", "text-align": "center", "margin": "5px", "color": "#2C786C", "font-weight": "bold", "--hover-color": "#ffeaa7"},
            "nav-link-selected": {"font-size": "16px", "background-color": "#2C786C", "color": "white", "font-weight": "bold"},
        }
    )

    # --- Forecast Plot Section ---
    if selected_tab == "Forecast Plot":
        st.subheader("Forecast Visualization with Risk Zones")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=full_series.index, y=full_series.values, mode='lines', name="Historical + Forecast", line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values, mode='lines', name="Forecasted Orders", fill='tozeroy', line=dict(color='lightgreen')))

        # Risk zones
        upper = forecast * 1.15
        lower = forecast * 0.85

        fig.add_trace(go.Scatter(x=forecast.index, y=upper, mode='lines', line=dict(width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=forecast.index, y=lower, mode='lines', fill='tonexty', fillcolor='rgba(255, 0, 0, 0.2)', line=dict(width=0), showlegend=True, name="Risk Zone"))

        fig.update_layout(title="Forecasted Sales with Risk Zones", xaxis_title="Date", yaxis_title="Orders", template="plotly_white")

        st.plotly_chart(fig, use_container_width=True)

    # --- Diagnostic Analysis Section ---
    elif selected_tab == "Diagnostic Analysis":
        st.subheader("📊 Diagnostic Analysis")

        selected_plot = st.radio("Choose Diagnostic Plot:", ("Forecast Distribution", "QQ-Plot of Forecast", "Residuals vs Forecasted Orders"), horizontal=True)

        if selected_plot == "Forecast Distribution":
            fig1 = ff.create_distplot([forecast.values], group_labels=['Forecasted Orders'], curve_type='kde', show_hist=True, show_rug=True)
            fig1.update_layout(title='Distribution of Forecasted Orders', xaxis_title='Orders', yaxis_title='Density', template="plotly_white")
            st.plotly_chart(fig1, use_container_width=True)

            # --- Insights for Forecast Distribution ---
            skewness = forecast.skew()
            st.info(f"ℹ️ Forecast Skewness: **{skewness:.2f}** {'(Right skewed 📈)' if skewness > 0 else '(Left skewed 📉)'}. Small skew is fine.")

        elif selected_plot == "QQ-Plot of Forecast":
            (osm, osr), (slope, intercept, r) = stats.probplot(forecast, dist="norm")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=osm, y=osr, mode='markers', name='Forecasted Orders'))
            fig2.add_trace(go.Scatter(x=osm, y=slope*osm + intercept, mode='lines', name='Ideal Line'))
            fig2.update_layout(title='QQ-Plot of Forecasted Orders', xaxis_title='Theoretical Quantiles', yaxis_title='Sample Quantiles', template="plotly_white")
            st.plotly_chart(fig2, use_container_width=True)

            # --- Insights for QQ-Plot of Forecast ---
            st.info("ℹ️ QQ-Plot: \n\n - Points along straight line = data is normally distributed ✅. "
                    "\n - Strong curvature = possible skew or heavy tails ❗")

        elif selected_plot == "Residuals vs Forecasted Orders":
            hist_mean = sales_df['orders'].mean()
            residuals = forecast - hist_mean

            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(x=forecast, y=residuals, mode='markers', name='Residuals'))
            fig3.add_hline(y=0, line_dash="dash", line_color="red")
            fig3.update_layout(title="Residuals vs Forecasted Orders", xaxis_title="Forecasted Orders", yaxis_title="Residuals", template="plotly_white")
            st.plotly_chart(fig3, use_container_width=True)
            res_mean = residuals.mean()
            
            # --- Insights for Residuals vs Forecasted Orders ---
            no_res = "Residuals are centered around 0 → No systematic bias."
            is_res = "✅ Large positive/negative residuals → may indicate bias or unusual future pattern."
            st.info(f"ℹ️ Residual Mean: **{res_mean:.2f}**. \n\n" + is_res if res_mean!=0 else no_res)
    # --- Smart Forecast Summary Section ---
    elif selected_tab == "Smart Forecast Summary":
        st.subheader("🧠 Smart Forecast Summary")

        hist_mean = sales_df['orders'].mean()
        hist_std = sales_df['orders'].std()
        forecast_mean = forecast.mean()
        forecast_std = forecast.std()

        growth = ((forecast_mean - hist_mean) / hist_mean) * 100
        threshold_high = hist_mean + 2 * hist_std
        threshold_low = hist_mean - 2 * hist_std
        anomalies = forecast[(forecast > threshold_high) | (forecast < threshold_low)]

        best_day = forecast.idxmax()
        worst_day = forecast.idxmin()

        st.metric(label="📈 Forecasted Mean Orders", value=f"{forecast_mean:.2f}")
        st.metric(label="📈 Forecasted Std Deviation", value=f"{forecast_std:.2f}")
        st.metric(label="🚀 Growth vs Historical", value=f"{growth:.2f}%")
        st.success(f"🚀 Best Day: {best_day.date()} — {forecast.max():.2f} orders")
        st.warning(f"⚡ Worst Day: {worst_day.date()} — {forecast.min():.2f} orders")

        if anomalies.empty:
            st.success("✅ No extreme anomalies detected.")
        else:
            st.error(f"⚠️ {len(anomalies)} anomalies detected!")

    # --- Forecast Data Table Section ---
    elif selected_tab == "Forecast Table":
        st.subheader("🧾 Forecasted Data Table")
        st.dataframe(forecast.to_frame(name="Forecasted Orders"))

        download_format = st.radio("Choose Download Format:", ("CSV", "Excel"), horizontal=True)

        if download_format == "CSV":
            st.download_button(label="Download CSV", data=forecast.to_csv().encode(), file_name="forecast.csv", mime="text/csv")
        else:
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                forecast.to_frame(name="Forecasted Orders").to_excel(writer)
            st.download_button(label="Download Excel", data=output.getvalue(), file_name="forecast.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# --- Footer ---
st.markdown("<h6 style='text-align: center; color: gray;'>© 2025 Future Forecasts 🚀</h6>", unsafe_allow_html=True)
