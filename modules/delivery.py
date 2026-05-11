import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor


# =========================================================
# DELIVERY AI
# =========================================================
def show_delivery_ai(df):

    st.markdown("## 🚚 Smart Delivery AI")

    # =====================================================
    # COPY
    # =====================================================
    df = df.copy()

    # =====================================================
    # DATE FORMAT
    # =====================================================
    df['order_date'] = pd.to_datetime(
        df['order_date']
    )

    df['delivery_date'] = pd.to_datetime(
        df['delivery_date']
    )

    # =====================================================
    # DELIVERY DAYS
    # =====================================================
    df['delivery_days'] = (
        df['delivery_date'] -
        df['order_date']
    ).dt.days

    df = df[
        df['delivery_days'] >= 0
    ]

    # =====================================================
    # STATE CHECK
    # =====================================================
    if 'state' not in df.columns:

        df['state'] = "Dhaka"

    # =====================================================
    # FEATURES
    # =====================================================
    df['day'] = (
        df['order_date']
        .dt.dayofweek
    )

    df['month'] = (
        df['order_date']
        .dt.month
    )

    # =====================================================
    # LATE LABEL
    # =====================================================
    df['late'] = np.where(
        df['delivery_days'] > 5,
        1,
        0
    )

    # =====================================================
    # STATE ENCODE
    # =====================================================
    states = df['state'].dropna().unique()

    state_map = {
        state: i
        for i, state in enumerate(states)
    }

    df['state_encoded'] = df[
        'state'
    ].map(state_map)

    # =====================================================
    # KPI
    # =====================================================
    avg_delivery = df[
        'delivery_days'
    ].mean()

    rush_day = df.groupby(
        df['order_date']
        .dt.day_name()
    )['order_id'].count().idxmax()

    late_day = df.groupby(
        df['order_date']
        .dt.day_name()
    )['late'].mean().idxmax()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "🚚 Avg Delivery",
        f"{avg_delivery:.1f} Days"
    )

    col2.metric(
        "🔥 Rush Day",
        rush_day
    )

    col3.metric(
        "⚠️ Most Late Day",
        late_day
    )

    st.markdown("---")

    # =====================================================
    # ORDER RUSH GRAPH
    # =====================================================
    st.subheader("📈 Order Rush Days")

    rush_df = df.groupby(
        df['order_date']
        .dt.day_name()
    )['order_id'].count().reset_index()

    rush_df.columns = [
        'Day',
        'Orders'
    ]

    days_order = [
        'Monday',
        'Tuesday',
        'Wednesday',
        'Thursday',
        'Friday',
        'Saturday',
        'Sunday'
    ]

    rush_df['Day'] = pd.Categorical(
        rush_df['Day'],
        categories=days_order,
        ordered=True
    )

    rush_df = rush_df.sort_values(
        'Day'
    )

    fig1 = px.bar(
        rush_df,
        x='Day',
        y='Orders',
        title="Order Creation Rush"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    st.markdown("---")

    # =====================================================
    # DELIVERY RUSH GRAPH
    # =====================================================
    st.subheader("🚚 Delivery Completion Days")

    delivery_df = df.groupby(
        df['delivery_date']
        .dt.day_name()
    )['order_id'].count().reset_index()

    delivery_df.columns = [
        'Day',
        'Deliveries'
    ]

    delivery_df['Day'] = pd.Categorical(
        delivery_df['Day'],
        categories=days_order,
        ordered=True
    )

    delivery_df = delivery_df.sort_values(
        'Day'
    )

    fig2 = px.bar(
        delivery_df,
        x='Day',
        y='Deliveries',
        title="Delivery Rush Days"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.markdown("---")

    # =====================================================
    # STATE VISUALIZATION
    # =====================================================
    st.subheader("📍 State Order Analysis")

    state_orders = df.groupby(
        'state'
    )['order_id'].count().reset_index()

    state_orders.columns = [
        'State',
        'Orders'
    ]

    fig3 = px.bar(
        state_orders.sort_values(
            by='Orders',
            ascending=False
        ),
        x='State',
        y='Orders',
        title="Orders by State"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

    st.markdown("---")

    # =====================================================
    # ML DATA
    # =====================================================
    X = df[[
        'quantity',
        'day',
        'month',
        'state_encoded'
    ]]

    y_class = df['late']

    y_reg = df['delivery_days']

    # =====================================================
    # MODELS
    # =====================================================
    late_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )

    late_model.fit(
        X,
        y_class
    )

    eta_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )

    eta_model.fit(
        X,
        y_reg
    )

    # =====================================================
    # SESSION STATE
    # =====================================================
    if "delivery_result" not in st.session_state:

        st.session_state.delivery_result = None

    # =====================================================
    # FORM
    # =====================================================
    st.subheader("🤖 Predict Delivery")

    with st.form("delivery_form"):

        col1, col2 = st.columns(2)

        # =================================================
        # LEFT
        # =================================================
        with col1:
            quantity_input = st.text_input(
                "Quantity",
                placeholder="Enter Quantity",
                key="delivery_quantity_input"
            )

            order_day = st.selectbox(
                "Order Day",
                ["Select Order Day"] + days_order,
                index=0,
                key="delivery_order_day"
            )

        # =================================================
        # RIGHT
        # =================================================
        with col2:
            selected_state = st.selectbox(
                "Delivery State",
                ["Select State"] + list(states),
                index=0,
                key="delivery_state_select"
            )

        predict_btn = st.form_submit_button(
            "🚀 Predict Delivery"
        )

    # =====================================================
    # BUTTON ACTION
    # =====================================================
    if predict_btn:

        # =============================================
        # VALIDATION
        # =============================================
        if quantity_input == "":

            st.warning(
                "Please enter quantity"
            )

        elif not quantity_input.isdigit():

            st.error(
                "Quantity must be numeric"
            )

        elif order_day == "Select Order Day":

            st.warning(
                "Please select order day"
            )

        elif selected_state == "Select State":

            st.warning(
                "Please select delivery state"
            )

        else:

            quantity = int(quantity_input)

            day_map = {
                'Monday': 0,
                'Tuesday': 1,
                'Wednesday': 2,
                'Thursday': 3,
                'Friday': 4,
                'Saturday': 5,
                'Sunday': 6
            }

            current_month = pd.Timestamp.now().month

            input_df = pd.DataFrame({
                'quantity': [quantity],
                'day': [day_map[order_day]],
                'month': [current_month],
                'state_encoded': [
                    state_map[selected_state]
                ]
            })

            # =================================================
            # PREDICT
            # =================================================
            late_prob = late_model.predict_proba(
                input_df
            )[0][1] * 100

            eta = eta_model.predict(
                input_df
            )[0]

            eta = max(1, eta)

            st.session_state.delivery_result = {
                "eta": eta,
                "late_prob": late_prob,
                "state": selected_state,
                "quantity": quantity,
                "order_day": order_day
            }

    # =====================================================
    # SHOW RESULT
    # =====================================================
    if st.session_state.delivery_result:

        result = st.session_state.delivery_result

        st.markdown("---")

        st.subheader("📦 Delivery Prediction Result")

        c1, c2 = st.columns(2)

        c1.metric(
            "🚚 Expected Delivery",
            f"{result['eta']:.1f} Days"
        )

        c2.metric(
            "⚠️ Late Probability",
            f"{result['late_prob']:.1f}%"
        )

        st.markdown("---")

        # =================================================
        # STATUS
        # =================================================
        if result['late_prob'] >= 70:

            st.error("""
            ⚠️ High Late Delivery Risk
            """)

        elif result['late_prob'] >= 40:

            st.warning("""
            📈 Moderate Delay Risk
            """)

        else:

            st.success("""
            ✅ Normal Delivery Expected
            """)

        st.markdown("---")

        # =================================================
        # AI SUGGESTION
        # =================================================
        st.subheader("🧠 AI Delivery Suggestion")

        best_day = rush_df.sort_values(
            by='Orders'
        ).iloc[0]['Day']

        if result['late_prob'] >= 70:

            st.warning(f"""
            ⚠️ AI detected high delivery risk.

            Suggested Actions:
            • Avoid placing orders on {result['order_day']}
            • Try creating orders on {best_day}
            • Reduce large quantity orders
            • Use priority delivery for {result['state']}
            """)

        elif result['late_prob'] >= 40:

            st.info(f"""
            📈 Moderate delivery risk detected.

            Suggested Actions:
            • Prefer order creation on {best_day}
            • Monitor warehouse workload
            • Keep quantity moderate
            """)

        else:

            st.success(f"""
            ✅ Delivery condition looks good.

            AI Recommendation:
            • {result['order_day']} is suitable
            • Delivery risk is low
            • Expected smooth delivery flow
            """)

        st.markdown("---")

        # =================================================
        # FINAL INSIGHT
        # =================================================
        st.info(f"""
        🔥 Highest Rush Day:
        {rush_day}

        ⚠️ Highest Delay Day:
        {late_day}

        🚚 Average Delivery:
        {avg_delivery:.1f} Days

        📍 Selected State:
        {result['state']}
        """)