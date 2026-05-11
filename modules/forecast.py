import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from prophet import Prophet
from sklearn.linear_model import LinearRegression


# =========================================================
# FORECAST GRAPH
# =========================================================
def forecast_graph(data, forecast):

    forecast['smooth'] = (
        forecast['yhat']
        .rolling(7)
        .mean()
    )

    fig = go.Figure()

    # =====================================================
    # ACTUAL
    # =====================================================
    fig.add_trace(go.Scatter(
        x=data['ds'],
        y=data['y'],
        name="Actual",
        line=dict(width=2)
    ))

    # =====================================================
    # FORECAST
    # =====================================================
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['smooth'],
        name="Forecast",
        line=dict(width=3)
    ))

    # =====================================================
    # UPPER RANGE
    # =====================================================
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_upper'],
        line=dict(width=0),
        showlegend=False
    ))

    # =====================================================
    # LOWER RANGE
    # =====================================================
    fig.add_trace(go.Scatter(
        x=forecast['ds'],
        y=forecast['yhat_lower'],
        fill='tonexty',
        name='Confidence Range',
        opacity=0.2
    ))

    fig.update_layout(
        template="plotly_white",
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="forecast_graph"
    )


# =========================================================
# GROWTH + CONFIDENCE
# =========================================================
def growth_confidence(forecast, days):

    recent = forecast['yhat'].tail(7).mean()

    past = forecast['yhat'].iloc[
           -(days + 7):-days
           ].mean()

    if pd.isna(past) or past == 0:

        growth = 0

    else:

        growth = (
            (recent - past)
            / past
        ) * 100

    future_data = forecast.tail(days)

    avg_pred = future_data['yhat'].mean()

    uncertainty = (
        future_data['yhat_upper']
        -
        future_data['yhat_lower']
    ).mean()

    if avg_pred == 0:

        confidence = 50

    else:

        relative = uncertainty / (
            abs(avg_pred) + uncertainty
        )

        confidence = (1 - relative) * 100

    confidence = round(confidence, 1)

    confidence = max(
        35,
        min(95, confidence)
    )

    if confidence >= 75:

        stability = "📉 Stable"

    elif confidence >= 50:

        stability = "📊 Moderate"

    else:

        stability = "📈 Uncertain"

    return growth, confidence, stability


# =========================================================
# MAIN FUNCTION
# =========================================================
def show_forecast(df):

    st.markdown("## 🔮 Forecast Dashboard")

    # =====================================================
    # FORECAST SETTINGS
    # =====================================================
    st.subheader("⚙️ Forecast Settings")

    col1, col2 = st.columns(2)

    with col1:

        days_input = st.text_input(
            "Enter Forecast Days",
            placeholder="Enter days between 7 - 365",
            key="forecast_days_input"
        )

        st.caption("""
        ✅ Minimum: 7 Days

        ✅ Maximum: 365 Days
        """)

    with col2:

        metric = st.selectbox(
            "Forecast Metric",
            [
                "quantity",
                "total_price"
            ],
            key="forecast_metric_select"
        )

    forecast_btn = st.button(
        "🚀 Confirm Forecast Search",
        use_container_width=True,
        key="forecast_confirm_button"
    )

    # =====================================================
    # SESSION STATE
    # =====================================================
    if forecast_btn:
        st.session_state["forecast_run"] = True

    st.markdown("---")

    # =====================================================
    # FORECAST RESULT
    # =====================================================
    if st.session_state.get(
        "forecast_run",
        False
    ):

        if days_input == "":

            st.warning(
                "Please enter forecast days"
            )

        elif not days_input.isdigit():

            st.error(
                "Forecast days must be numeric"
            )

        else:

            days = int(days_input)

            if days < 7 or days > 365:

                st.error(
                    "Forecast days must be between 7 and 365"
                )

            else:

                # =============================================
                # DATA
                # =============================================
                data = df.groupby(
                    'order_date'
                )[metric].sum().reset_index()

                data.columns = [
                    'ds',
                    'y'
                ]

                # =============================================
                # MODEL
                # =============================================
                model = Prophet()

                model.fit(data)

                future = model.make_future_dataframe(
                    periods=days
                )

                forecast = model.predict(future)

                # =============================================
                # GRAPH
                # =============================================
                st.subheader("📈 Forecast Graph")

                forecast_graph(
                    data,
                    forecast
                )

                st.markdown("---")

                # =============================================
                # GROWTH
                # =============================================
                growth, confidence, stability = (
                    growth_confidence(
                        forecast,
                        days
                    )
                )

                g1, g2, g3 = st.columns(3)

                g1.metric(
                    "Growth %",
                    f"{growth:.2f}%"
                )

                g2.metric(
                    "Confidence %",
                    f"{confidence:.1f}%"
                )

                g3.metric(
                    "Stability",
                    stability
                )

                st.markdown("---")

                # =============================================
                # FUTURE SUMMARY
                # =============================================
                future_data = forecast.tail(days)

                f1, f2, f3 = st.columns(3)

                f1.metric(
                    "📊 Avg Future",
                    f"{future_data['yhat'].mean():.1f}"
                )

                f2.metric(
                    "🚀 Best Case",
                    f"{future_data['yhat_upper'].max():.1f}"
                )

                f3.metric(
                    "⚠️ Worst Case",
                    f"{future_data['yhat_lower'].min():.1f}"
                )

                st.markdown("---")

                # =============================================
                # FINAL FORECAST INSIGHT
                # =============================================
                st.subheader("🧠 Forecast Insight")

                if growth > 15:

                    st.success("""
                    🚀 Strong future growth expected.

                    Recommended Actions:
                    • Increase inventory
                    • Prepare warehouse
                    • Increase marketing budget
                    """)

                elif growth > 0:

                    st.info("""
                    📈 Stable future growth detected.

                    Recommended Actions:
                    • Maintain stock balance
                    • Monitor customer demand
                    """)

                else:

                    st.warning("""
                    ⚠️ Slow growth predicted.

                    Recommended Actions:
                    • Run discount campaigns
                    • Reduce excess inventory
                    • Improve promotions
                    """)

    st.markdown("---")

    # =====================================================
    # REVENUE STRATEGY
    # =====================================================
    st.subheader("💰 Revenue Strategy")

    r1, r2 = st.columns([3, 1])

    with r1:

        revenue_days_input = st.text_input(
            "Enter Revenue Forecast Days",
            placeholder="Enter days between 7 - 365",
            key="forecast_revenue_days_input"
        )

        st.caption("""
        ✅ Minimum: 7 Days

        ✅ Maximum: 365 Days
        """)

    with r2:

        st.write("")
        st.write("")

        revenue_btn = st.button(
            "🚀 Confirm Revenue Forecast",
            use_container_width=True,
            key="forecast_revenue_btn"
        )

    # =====================================================
    # SESSION STATE
    # =====================================================
    if revenue_btn:
        st.session_state["revenue_run"] = True

    st.markdown("---")

    # =====================================================
    # REVENUE RESULT
    # =====================================================
    if st.session_state.get(
        "revenue_run",
        False
    ):

        if revenue_days_input == "":

            st.warning(
                "Please enter forecast days"
            )

        elif not revenue_days_input.isdigit():

            st.error(
                "Forecast days must be numeric"
            )

        else:

            revenue_days = int(
                revenue_days_input
            )

            if revenue_days < 7 or revenue_days > 365:

                st.error(
                    "Forecast days must be between 7 and 365"
                )

            else:

                revenue_df = df.groupby(
                    'order_date'
                ).agg({
                    'total_price': 'sum'
                }).reset_index()

                revenue_df['day'] = range(
                    len(revenue_df)
                )

                X = revenue_df[['day']]
                y = revenue_df['total_price']

                revenue_model = LinearRegression()

                revenue_model.fit(X, y)

                future_revenue = revenue_model.predict(
                    [[
                        len(revenue_df)
                        + revenue_days
                    ]]
                )[0]

                current_avg = revenue_df[
                    'total_price'
                ].mean()

                revenue_growth = (
                    (
                        future_revenue
                        -
                        current_avg
                    )
                    /
                    current_avg
                ) * 100

                # =============================================
                # METRICS
                # =============================================
                rev1, rev2 = st.columns(2)

                rev1.metric(
                    "Predicted Revenue",
                    f"{future_revenue:.0f}"
                )

                rev2.metric(
                    "Growth Forecast",
                    f"{revenue_growth:.2f}%"
                )

                st.markdown("---")

                # =============================================
                # AI
                # =============================================
                if revenue_growth > 15:

                    st.success("""
                    🚀 Strong Growth Expected

                    Strategy:
                    • Increase inventory
                    • Increase marketing budget
                    • Focus on best-selling products
                    """)

                elif revenue_growth > 0:

                    st.info("""
                    📈 Stable Revenue Trend

                    Strategy:
                    • Maintain operations
                    • Improve customer retention
                    """)

                else:

                    st.warning("""
                    📉 Revenue Risk Detected

                    Strategy:
                    • Reduce weak inventory
                    • Run discount campaigns
                    • Optimize spending
                    """)

                st.markdown("---")

                # =============================================
                # GRAPH
                # =============================================
                graph_data = pd.DataFrame({
                    'Type': [
                        'Current Revenue',
                        'Future Revenue'
                    ],
                    'Revenue': [
                        current_avg,
                        future_revenue
                    ]
                })

                fig2 = px.bar(
                    graph_data,
                    x='Type',
                    y='Revenue',
                    title="Revenue Forecast"
                )

                st.plotly_chart(
                    fig2,
                    use_container_width=True,
                    key=f"forecast_revenue_graph_{revenue_days}"
                )