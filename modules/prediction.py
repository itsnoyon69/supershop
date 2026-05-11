import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression


# =========================================================
# PREDICTION SYSTEM
# =========================================================
def show_prediction(df):

    st.markdown("## 🤖 Advanced AI Prediction System")

    # =====================================================
    # COPY
    # =====================================================
    df = df.copy()

    df['order_date'] = pd.to_datetime(
        df['order_date']
    )

    # =====================================================
    # USER PRODUCT MATRIX
    # =====================================================
    user_product = df.pivot_table(
        index='customer_id',
        columns='product_name',
        values='quantity',
        aggfunc='sum',
        fill_value=0
    )

    # =====================================================
    # COSINE SIMILARITY
    # =====================================================
    similarity = cosine_similarity(
        user_product
    )

    similarity_df = pd.DataFrame(
        similarity,
        index=user_product.index,
        columns=user_product.index
    )

    # =====================================================
    # SESSION STATES
    # =====================================================
    session_keys = [
        "similarity_run",
        "churn_run",
        "demand_run",
        "next_purchase_run",
        "repeat_run",
        "recommend_run"
    ]

    for key in session_keys:

        if key not in st.session_state:

            st.session_state[key] = False
    # =====================================================
    # CUSTOMER SEARCH PROFILE
    # =====================================================
    st.subheader("👤 Customer Search")

    search_col1, search_col2 = st.columns([3, 1])

    with search_col1:

        customer_profile_input = st.text_input(
            "Search Customer ID",
            placeholder="Enter Customer ID",
            key="prediction_customer_profile"
        )

        customer_profile_btn = st.button(
            "🔍 Search Customer",
            key="prediction_customer_profile_btn"
        )

    with search_col2:

        st.info("""
        📌 Customer Profile

        Shows:
        • Spending
        • Orders
        • Products
        • Customer Behaviour
        • Purchase Activity
        """)

    # =====================================================
    # SESSION
    # =====================================================
    if customer_profile_btn:
        st.session_state.profile_run = True
        st.session_state.profile_customer = (
            customer_profile_input
        )

    st.markdown("---")

    # =====================================================
    # RESULT
    # =====================================================
    if st.session_state.get("profile_run", False):

        customer_profile_input = (
            st.session_state.profile_customer
        )

        if customer_profile_input == "":

            st.warning(
                "Please enter customer ID"
            )

        elif not customer_profile_input.isdigit():

            st.error(
                "Customer ID must be numeric"
            )

        else:

            customer_id = int(
                customer_profile_input
            )

            customer_data = df[
                df['customer_id']
                ==
                customer_id
                ]

            if len(customer_data) == 0:

                st.error(
                    "Customer not found"
                )

            else:

                # =============================================
                # CUSTOMER SUMMARY
                # =============================================
                total_orders = customer_data[
                    'order_id'
                ].nunique()

                total_quantity = customer_data[
                    'quantity'
                ].sum()

                total_spending = customer_data[
                    'total_price'
                ].sum()

                total_products = customer_data[
                    'product_name'
                ].nunique()

                last_order = customer_data[
                    'order_date'
                ].max()

                inactive_days = (
                        pd.Timestamp.now()
                        -
                        last_order
                ).days

                # =============================================
                # CUSTOMER LEVEL
                # =============================================
                if total_spending >= df[
                    'total_price'
                ].quantile(0.75):

                    level = "👑 VIP Customer"

                elif total_orders >= 5:

                    level = "🔁 Loyal Customer"

                elif inactive_days > 60:

                    level = "⚠️ Inactive Customer"

                else:

                    level = "🙂 Normal Customer"

                # =============================================
                # METRICS
                # =============================================
                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "Orders",
                    total_orders
                )

                c2.metric(
                    "Products",
                    total_products
                )

                c3.metric(
                    "Quantity",
                    int(total_quantity)
                )

                c4.metric(
                    "Spending",
                    f"{total_spending:.0f}"
                )

                st.markdown("---")

                # =============================================
                # STATUS
                # =============================================
                if "VIP" in level:

                    st.success(level)

                elif "Loyal" in level:

                    st.warning(level)

                elif "Inactive" in level:

                    st.error(level)

                else:

                    st.info(level)

                st.markdown("---")

                # =============================================
                # PRODUCT HISTORY
                # =============================================
                st.subheader(
                    "🛒 Purchase History"
                )

                history_df = customer_data.groupby(
                    'product_name',
                    as_index=False
                ).agg({
                    'quantity': 'sum',
                    'total_price': 'sum',
                    'order_id': 'nunique'
                })

                history_df.columns = [
                    'Product',
                    'Quantity',
                    'Total Spending',
                    'Times Purchased'
                ]

                history_df = history_df.sort_values(
                    by='Quantity',
                    ascending=False
                )

                st.dataframe(
                    history_df,
                    use_container_width=True
                )

                # =============================================
                # CHART
                # =============================================
                fig_customer_profile = px.bar(
                    history_df.head(10),
                    x='Product',
                    y='Quantity',
                    color='Quantity',
                    title="Customer Product Activity"
                )

                st.plotly_chart(
                    fig_customer_profile,
                    use_container_width=True,
                    key=f"profile_chart_{customer_id}"
                )

                st.markdown("---")

                # =============================================
                # CUSTOMER REPORT
                # =============================================
                top_product = history_df.iloc[0][
                    'Product'
                ]

                if total_spending >= df[
                    'total_price'
                ].quantile(0.75):

                    st.success(f"""
                    📌 Customer Analysis Report

                    Customer ID:
                    {customer_id}

                    Customer Type:
                    VIP Customer

                    Behaviour:
                    • High spending customer
                    • Frequently purchases products
                    • Strong customer loyalty
                    • Important for revenue growth

                    Most Purchased Product:
                    {top_product}

                    Suggested Business Actions:
                    • Offer premium discounts
                    • Give loyalty rewards
                    • Send early-access offers
                    • Maintain VIP engagement
                    """)

                elif inactive_days > 60:

                    st.error(f"""
                    📌 Customer Analysis Report

                    Customer ID:
                    {customer_id}

                    Customer Type:
                    Inactive Customer

                    Behaviour:
                    • Customer activity is low
                    • Long inactivity detected
                    • Purchase frequency decreasing

                    Most Purchased Product:
                    {top_product}

                    Suggested Business Actions:
                    • Send reactivation campaign
                    • Offer personalized discounts
                    • Push reminder notifications
                    • Recommend trending products
                    """)

                else:

                    st.warning(f"""
                    📌 Customer Analysis Report

                    Customer ID:
                    {customer_id}

                    Customer Type:
                    Normal / Loyal Customer

                    Behaviour:
                    • Stable purchasing activity
                    • Moderate engagement level
                    • Good future potential

                    Most Purchased Product:
                    {top_product}

                    Suggested Business Actions:
                    • Maintain engagement
                    • Send personalized offers
                    • Promote bundle products
                    • Increase customer retention
                    """)

    st.markdown("---")
    # =====================================================
    # SIMILAR CUSTOMER SEARCH
    # =====================================================
    st.subheader("👥 Similar Customer Search")

    col1, col2 = st.columns([3, 1])

    with col1:

        customer_input = st.text_input(
            "Enter Customer ID",
            placeholder="Enter Customer ID",
            key="similar_customer_input"
        )

        similarity_btn = st.button(
            "🔍 Search",
            key="similarity_btn"
        )

    with col2:

        st.info("""
        📌 Reference

        0.90 - 1.00 → Very Similar

        0.70 - 0.89 → Similar

        0.40 - 0.69 → Moderate

        Below 0.40 → Low Similarity
        """)

    if similarity_btn:

        st.session_state.similarity_run = True
        st.session_state.similar_customer = (
            customer_input
        )

    st.markdown("---")

    # =====================================================
    # RESULT
    # =====================================================
    if st.session_state.similarity_run:

        customer_input = (
            st.session_state.similar_customer
        )

        if customer_input == "":

            st.warning("Please enter customer ID")

        elif not customer_input.isdigit():

            st.error("Customer ID must be numeric")

        else:

            customer_id = int(customer_input)

            if customer_id not in similarity_df.columns:

                st.error("Customer not found")

            else:

                similar_users = similarity_df[
                    customer_id
                ].sort_values(
                    ascending=False
                )

                similar_users = similar_users.iloc[1:6]

                similar_df = pd.DataFrame({
                    'Customer ID':
                        similar_users.index,

                    'Similarity Score':
                        similar_users.values
                })

                st.dataframe(
                    similar_df,
                    use_container_width=True
                )

                fig1 = px.bar(
                    similar_df,
                    x='Customer ID',
                    y='Similarity Score',
                    color='Similarity Score',
                    title="Top Similar Customers"
                )

                st.plotly_chart(
                    fig1,
                    use_container_width=True,
                    key=f"similarity_chart_{customer_id}"
                )

                top_score = similar_df.iloc[0][
                    'Similarity Score'
                ]

                if top_score >= 0.80:

                    st.success("""
                    ✅ Strong customer similarity detected
                    """)

                elif top_score >= 0.50:

                    st.warning("""
                    ⚠️ Moderate similarity detected
                    """)

                else:

                    st.error("""
                    ❌ Weak customer similarity detected
                    """)

    st.markdown("---")

    # =====================================================
    # CHURN PREDICTION
    # =====================================================
    st.subheader("📉 Customer Churn Prediction")

    # =====================================================
    # CUSTOMER DATA
    # =====================================================
    customer_df = df.groupby(
        'customer_id'
    ).agg({
        'order_id': 'nunique',
        'quantity': 'sum',
        'total_price': 'sum',
        'order_date': 'max'
    }).reset_index()

    customer_df.columns = [
        'customer_id',
        'orders',
        'quantity',
        'spending',
        'last_order'
    ]

    # =====================================================
    # RECENCY
    # =====================================================
    today = df['order_date'].max()

    customer_df['inactive_days'] = (
            today -
            customer_df['last_order']
    ).dt.days

    # =====================================================
    # NORMALIZED SCORES
    # =====================================================
    customer_df['order_score'] = (
            customer_df['orders']
            /
            customer_df['orders'].max()
    )

    customer_df['spending_score'] = (
            customer_df['spending']
            /
            customer_df['spending'].max()
    )

    customer_df['inactive_score'] = (
            customer_df['inactive_days']
            /
            customer_df['inactive_days'].max()
    )

    # =====================================================
    # SMART CHURN SCORE
    # =====================================================
    customer_df['churn_probability'] = (
                                               (
                                                       customer_df['inactive_score'] * 0.55
                                               )
                                               -
                                               (
                                                       customer_df['order_score'] * 0.25
                                               )
                                               -
                                               (
                                                       customer_df['spending_score'] * 0.20
                                               )
                                       ) * 100

    # =====================================================
    # FIX NEGATIVE
    # =====================================================
    customer_df['churn_probability'] = (
        customer_df['churn_probability']
        .clip(lower=5, upper=95)
    )

    # =====================================================
    # CHURN LABEL
    # =====================================================
    customer_df['churn'] = np.where(
        customer_df['churn_probability'] >= 50,
        1,
        0
    )

    # =====================================================
    # FEATURES
    # =====================================================
    X = customer_df[[
        'orders',
        'quantity',
        'spending',
        'inactive_days'
    ]]

    y = customer_df['churn']

    # =====================================================
    # MODEL
    # =====================================================
    churn_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=42
    )

    churn_model.fit(X, y)

    # =====================================================
    # SEARCH
    # =====================================================
    c1, c2 = st.columns([3, 1])

    with c1:

        churn_input = st.text_input(
            "Enter Customer ID",
            placeholder="Enter Customer ID",
            key="churn_input"
        )

        churn_btn = st.button(
            "🚀 Predict",
            key="churn_btn"
        )

    with c2:

        st.info("""
        📌 Risk Reference

        0% - 20% → Very Low Risk

        21% - 40% → Low Risk

        41% - 60% → Moderate Risk

        61% - 80% → High Risk

        81% - 100% → Critical Risk
        """)

    # =====================================================
    # SESSION
    # =====================================================
    if churn_btn:
        st.session_state.churn_run = True
        st.session_state.churn_customer = (
            churn_input
        )

    st.markdown("---")

    # =====================================================
    # RESULT
    # =====================================================
    if st.session_state.get("churn_run", False):

        churn_input = (
            st.session_state.churn_customer
        )

        if churn_input == "":

            st.warning(
                "Please enter customer ID"
            )

        elif not churn_input.isdigit():

            st.error(
                "Customer ID must be numeric"
            )

        else:

            customer_id = int(churn_input)

            customer_exists = customer_df[
                customer_df['customer_id']
                ==
                customer_id
                ]

            if len(customer_exists) == 0:

                st.error(
                    "Customer not found"
                )

            else:

                row = customer_exists.iloc[0]

                # =============================================
                # INPUT
                # =============================================
                input_df = pd.DataFrame({
                    'orders': [row['orders']],
                    'quantity': [row['quantity']],
                    'spending': [row['spending']],
                    'inactive_days': [
                        row['inactive_days']
                    ]
                })

                # =============================================
                # MODEL PREDICTION
                # =============================================
                # =============================================
                # SAFE MODEL PROBABILITY
                # =============================================
                proba = churn_model.predict_proba(
                    input_df
                )

                # only one class exists
                if proba.shape[1] == 1:

                    if churn_model.classes_[0] == 1:

                        model_probability = 100

                    else:

                        model_probability = 0

                # normal case
                else:

                    model_probability = (
                            proba[0][1] * 100
                    )

                # =============================================
                # FINAL PROBABILITY
                # =============================================
                final_probability = (
                        (
                                model_probability * 0.6
                        )
                        +
                        (
                                row['churn_probability'] * 0.4
                        )
                )

                final_probability = round(
                    final_probability,
                    1
                )

                # =============================================
                # STATUS
                # =============================================
                if final_probability >= 80:

                    status = "🔴 Critical Risk"

                elif final_probability >= 60:

                    status = "🔴 High Risk"

                elif final_probability >= 40:

                    status = "🟡 Moderate Risk"

                elif final_probability >= 20:

                    status = "🟢 Low Risk"

                else:

                    status = "✅ Very Low Risk"

                # =============================================
                # METRICS
                # =============================================
                cc1, cc2, cc3, cc4 = st.columns(4)

                cc1.metric(
                    "Inactive Days",
                    int(row['inactive_days'])
                )

                cc2.metric(
                    "Total Orders",
                    int(row['orders'])
                )

                cc3.metric(
                    "Total Spending",
                    f"{row['spending']:.0f}"
                )

                cc4.metric(
                    "Churn Risk",
                    f"{final_probability:.1f}%"
                )

                st.markdown("---")

                # =============================================
                # CHART
                # =============================================
                churn_chart = pd.DataFrame({
                    'Type': [
                        'Churn Risk',
                        'Retention Chance'
                    ],
                    'Score': [
                        final_probability,
                        100 - final_probability
                    ]
                })

                fig2 = px.bar(
                    churn_chart,
                    x='Type',
                    y='Score',
                    color='Score',
                    title="Customer Churn Analysis"
                )

                st.plotly_chart(
                    fig2,
                    use_container_width=True,
                    key=f"churn_chart_{customer_id}"
                )

                st.markdown("---")

                # =============================================
                # RESULT OUTPUT
                # =============================================
                if final_probability >= 80:

                    st.error(f"""
                    Customer ID:
                    {customer_id}

                    Status:
                    {status}

                    Customer Behaviour:
                    • Customer is inactive for long time
                    • Purchase activity is extremely low
                    • Retention probability is weak
                    • Customer may stop purchasing completely

                    Business Impact:
                    • Revenue loss risk is high
                    • Customer engagement is poor

                    Suggested Actions:
                    • Immediate reactivation campaign
                    • Strong discount offers
                    • Personalized follow-up
                    • Customer retention strategy needed
                    """)

                elif final_probability >= 60:

                    st.error(f"""
                    Customer ID:
                    {customer_id}

                    Status:
                    {status}

                    Customer Behaviour:
                    • Activity level is decreasing
                    • Purchase frequency becoming weak
                    • Engagement risk detected

                    Suggested Actions:
                    • Send promotional offers
                    • Increase engagement campaigns
                    • Push personalized products
                    • Monitor customer activity
                    """)

                elif final_probability >= 40:

                    st.warning(f"""
                    Customer ID:
                    {customer_id}

                    Status:
                    {status}

                    Customer Behaviour:
                    • Customer activity is moderate
                    • Some inactivity risk exists
                    • Purchase trend still stable

                    Suggested Actions:
                    • Send reminder offers
                    • Maintain engagement
                    • Recommend trending products
                    """)

                elif final_probability >= 20:

                    st.success(f"""
                    Customer ID:
                    {customer_id}

                    Status:
                    {status}

                    Customer Behaviour:
                    • Customer is mostly active
                    • Purchase behaviour is healthy
                    • Retention chance is strong

                    Suggested Actions:
                    • Continue loyalty benefits
                    • Maintain customer relationship
                    • Offer personalized recommendations
                    """)

                else:

                    st.success(f"""
                    Customer ID:
                    {customer_id}

                    Status:
                    {status}

                    Customer Behaviour:
                    • Customer is highly loyal
                    • Purchase activity is strong
                    • Spending behaviour is excellent
                    • Retention probability is very high

                    Suggested Actions:
                    • Offer premium services
                    • Maintain VIP benefits
                    • Use customer for loyalty campaigns
                    """)
    # =====================================================
    # PRODUCT DEMAND PREDICTION
    # =====================================================
    st.subheader("📦 Product Demand Prediction")

    product_options = sorted(
        df['product_name']
        .dropna()
        .unique()
    )

    d1, d2 = st.columns([3, 1])

    with d1:

        selected_product = st.selectbox(
            "Select Product",
            ["Select Product"] + list(product_options),
            index=0,
            key="demand_select"
        )

        demand_btn = st.button(
            "🚀 Predict",
            key="demand_btn"
        )

    with d2:

        st.info("""
        📌 Demand Reference

        Above Average → High Demand

        Near Average → Stable

        Below Average → Low Demand
        """)

    if demand_btn:
        st.session_state.demand_run = True
        st.session_state.demand_product = (
            selected_product
        )

    st.markdown("---")

    if st.session_state.demand_run:

        selected_product = (
            st.session_state.demand_product
        )

        if selected_product == "Select Product":

            st.warning("Please select product")

        else:

            product_data = df[
                df['product_name']
                ==
                selected_product
                ]

            demand_df = product_data.groupby(
                'order_date'
            )['quantity'].sum().reset_index()

            demand_df['day'] = range(
                len(demand_df)
            )

            X = demand_df[['day']]

            y = demand_df['quantity']

            model = LinearRegression()

            model.fit(X, y)

            future_day = len(demand_df) + 30

            future_prediction = model.predict(
                [[future_day]]
            )[0]

            avg_qty = y.mean()

            if future_prediction > avg_qty:

                demand_status = "🔥 High Future Demand"

            else:

                demand_status = "📉 Low Future Demand"

            p1, p2, p3 = st.columns(3)

            p1.metric(
                "Current Avg",
                f"{avg_qty:.1f}"
            )

            p2.metric(
                "Future Demand",
                f"{future_prediction:.1f}"
            )

            p3.metric(
                "Demand Status",
                demand_status
            )

            # =============================================
            # GRAPH
            # =============================================
            chart_df = pd.DataFrame({
                'Type': [
                    'Current Average',
                    'Future Prediction'
                ],
                'Value': [
                    avg_qty,
                    future_prediction
                ]
            })

            fig3 = px.bar(
                chart_df,
                x='Type',
                y='Value',
                color='Value',
                title="Demand Forecast"
            )

            st.plotly_chart(
                fig3,
                use_container_width=True,
                key=f"demand_chart_{selected_product}"
            )

            st.markdown("---")

            # =============================================
            # AI RESULT
            # =============================================
            if future_prediction > avg_qty * 1.2:

                st.success(f"""
                🤖 AI Product Demand Report

                Product:
                {selected_product}

                Predicted Demand:
                {future_prediction:.1f}

                Status:
                High Future Demand

                AI Analysis:
                • Future sales trend analyzed
                • Demand growth pattern checked
                • Inventory pressure estimated

                Recommended Actions:
                • Increase stock
                • Run marketing campaigns
                • Prepare warehouse capacity
                • Create combo offers
                """)

            elif future_prediction > avg_qty * 0.8:

                st.warning(f"""
                🤖 AI Product Demand Report

                Product:
                {selected_product}

                Predicted Demand:
                {future_prediction:.1f}

                Status:
                Stable Future Demand

                AI Analysis:
                • Product demand is stable
                • Inventory pressure is moderate
                • Customer demand is balanced

                Recommended Actions:
                • Maintain current stock
                • Monitor demand regularly
                • Use moderate promotions
                """)

            else:

                st.error(f"""
                🤖 AI Product Demand Report

                Product:
                {selected_product}

                Predicted Demand:
                {future_prediction:.1f}

                Status:
                Low Future Demand

                AI Analysis:
                • Demand trend is decreasing
                • Inventory risk detected
                • Sales growth is weak

                Recommended Actions:
                • Reduce stock level
                • Run discount campaigns
                • Bundle with top-selling products
                • Optimize inventory cost
                """)

    # =====================================================
    # NEXT PRODUCT PURCHASE
    # =====================================================
    st.subheader("🛒 Next Product Purchase Prediction Using Behavior")

    np1, np2 = st.columns([3, 1])

    with np1:

        next_input = st.text_input(
            "Enter Customer ID",
            placeholder="Enter Customer ID",
            key="next_product_input"
        )

        next_btn = st.button(
            "🚀 Predict",
            key="next_btn"
        )

    with np2:

        st.info("""
        📌 Buy Probability Reference

        70% - 100% → High Chance

        40% - 69% → Medium Chance

        Below 40% → Low Chance
        """)

    if next_btn:

        st.session_state.next_purchase_run = True
        st.session_state.next_customer = (
            next_input
        )

    st.markdown("---")

    if st.session_state.next_purchase_run:

        next_input = (
            st.session_state.next_customer
        )

        if next_input == "":

            st.warning("Please enter customer ID")

        elif not next_input.isdigit():

            st.error("Customer ID must be numeric")

        else:

            customer_id = int(next_input)

            customer_history = df[
                df['customer_id']
                ==
                customer_id
            ]

            if len(customer_history) == 0:

                st.error("Customer not found")

            else:

                top_products = customer_history.groupby(
                    'product_name'
                ).agg({
                    'quantity': 'sum',
                    'total_price': 'sum'
                }).reset_index()

                top_products = top_products.sort_values(
                    by='quantity',
                    ascending=False
                )

                next_product = top_products.iloc[0][
                    'product_name'
                ]

                expected_spending = (
                    top_products.iloc[0][
                        'total_price'
                    ]
                )

                next_days = np.random.randint(
                    7,
                    30
                )

                buy_probability = min(
                    95,
                    (
                        top_products.iloc[0]['quantity']
                        /
                        top_products['quantity'].sum()
                    ) * 100
                )

                if buy_probability >= 70:

                    probability_status = (
                        "🔥 High Chance"
                    )

                elif buy_probability >= 40:

                    probability_status = (
                        "⚠️ Medium Chance"
                    )

                else:

                    probability_status = (
                        "❌ Low Chance"
                    )

                n1, n2, n3 = st.columns(3)

                n1.metric(
                    "Next Product",
                    next_product
                )

                n2.metric(
                    "Expected Spend",
                    f"{expected_spending:.0f}"
                )

                n3.metric(
                    "Buy Probability",
                    f"{buy_probability:.1f}%"
                )

                st.markdown("---")

                st.info(f"""
                📌 Prediction Result

                Probability Level:
                {probability_status}

                Expected Purchase Time:
                {next_days} Days

                Predicted Product:
                {next_product}

                Predicted Spending:
                {expected_spending:.0f}
                """)

    st.markdown("---")

    # =====================================================
    # BUSINESS GROWTH PRODUCT PREDICTION
    # =====================================================
    st.subheader(
        "🚀 Product Purchase Prediction for Business Growth"
    )

    bg1, bg2 = st.columns([3, 1])

    with bg1:

        business_input = st.text_input(
            "Enter Customer ID",
            placeholder="Enter Customer ID",
            key="business_growth_input"
        )

        business_btn = st.button(
            "🚀 Predict",
            key="business_growth_btn"
        )

    with bg2:

        st.info("""
        📌 Business Growth Reference

        High Revenue Product →
        Better Profit

        Trending Product →
        Better Sales Growth

        High Demand →
        Better Business Opportunity
        """)

    # =====================================================
    # SESSION
    # =====================================================
    if business_btn:
        st.session_state.business_growth_run = True
        st.session_state.business_growth_customer = (
            business_input
        )

    st.markdown("---")

    # =====================================================
    # RESULT
    # =====================================================
    if st.session_state.get(
            "business_growth_run",
            False
    ):

        business_input = (
            st.session_state.business_growth_customer
        )

        if business_input == "":

            st.warning(
                "Please enter customer ID"
            )

        elif not business_input.isdigit():

            st.error(
                "Customer ID must be numeric"
            )

        else:

            customer_id = int(
                business_input
            )

            customer_data = df[
                df['customer_id']
                ==
                customer_id
                ]

            if len(customer_data) == 0:

                st.error(
                    "Customer not found"
                )

            else:

                # =============================================
                # HIGH VALUE PRODUCTS
                # =============================================
                product_profit = df.groupby(
                    'product_name'
                ).agg({
                    'total_price': 'sum',
                    'quantity': 'sum',
                    'customer_id': 'nunique'
                }).reset_index()

                product_profit.columns = [
                    'Product',
                    'Revenue',
                    'Quantity',
                    'Customers'
                ]

                # =============================================
                # BUSINESS SCORE
                # =============================================
                product_profit['business_score'] = (
                                                           (
                                                                   product_profit['Revenue']
                                                                   /
                                                                   product_profit['Revenue'].max()
                                                           ) * 0.5
                                                           +
                                                           (
                                                                   product_profit['Quantity']
                                                                   /
                                                                   product_profit['Quantity'].max()
                                                           ) * 0.3
                                                           +
                                                           (
                                                                   product_profit['Customers']
                                                                   /
                                                                   product_profit['Customers'].max()
                                                           ) * 0.2
                                                   ) * 100

                # =============================================
                # REMOVE ALREADY PURCHASED
                # =============================================
                purchased_products = set(
                    customer_data['product_name']
                    .unique()
                )

                recommended_df = product_profit[
                    ~product_profit['Product']
                    .isin(purchased_products)
                ]

                recommended_df = recommended_df.sort_values(
                    by='business_score',
                    ascending=False
                )

                top_product = recommended_df.iloc[0]

                # =============================================
                # METRICS
                # =============================================
                b1, b2, b3 = st.columns(3)

                b1.metric(
                    "Recommended Product",
                    top_product['Product']
                )

                b2.metric(
                    "Business Score",
                    f"{top_product['business_score']:.1f}"
                )

                b3.metric(
                    "Revenue Potential",
                    f"{top_product['Revenue']:.0f}"
                )

                st.markdown("---")

                # =============================================
                # GRAPH
                # =============================================
                fig_growth = px.bar(
                    recommended_df.head(10),
                    x='Product',
                    y='business_score',
                    color='business_score',
                    title="Business Growth Products"
                )

                st.plotly_chart(
                    fig_growth,
                    use_container_width=True,
                    key=f"business_growth_chart_{customer_id}"
                )

                st.markdown("---")

                # =============================================
                # STATUS
                # =============================================
                score = top_product[
                    'business_score'
                ]

                if score >= 70:

                    st.success(f"""
                    📌 Business Growth Recommendation

                    Customer ID:
                    {customer_id}

                    Recommended Product:
                    {top_product['Product']}

                    Prediction Type:
                    High Business Growth Opportunity

                    Why This Product:
                    • High revenue generating
                    • Strong customer demand
                    • High repeat purchase chance
                    • Better business profit margin

                    Business Benefits:
                    • Increase revenue growth
                    • Improve inventory movement
                    • Better conversion opportunity
                    • Strong sales performance

                    Suggested Actions:
                    • Push targeted promotion
                    • Recommend through ads
                    • Use combo campaigns
                    • Highlight in homepage
                    """)

                elif score >= 40:

                    st.warning(f"""
                    📌 Business Growth Recommendation

                    Customer ID:
                    {customer_id}

                    Recommended Product:
                    {top_product['Product']}

                    Prediction Type:
                    Moderate Growth Opportunity

                    Why This Product:
                    • Stable customer demand
                    • Good revenue potential
                    • Medium growth opportunity

                    Suggested Actions:
                    • Moderate promotion
                    • Personalized recommendation
                    • Bundle strategy
                    """)

                else:

                    st.error(f"""
                    📌 Business Growth Recommendation

                    Customer ID:
                    {customer_id}

                    Recommended Product:
                    {top_product['Product']}

                    Prediction Type:
                    Low Growth Opportunity

                    Why This Product:
                    • Lower demand trend
                    • Weak revenue performance

                    Suggested Actions:
                    • Limited promotion
                    • Monitor performance
                    • Avoid overstock
                    """)

    # =====================================================
    # REPEAT PURCHASE
    # =====================================================
    st.subheader("🔁 Repeat Purchase Prediction")

    repeat_df = df.groupby(
        'customer_id'
    ).agg({
        'order_id': 'nunique',
        'quantity': 'sum',
        'total_price': 'sum'
    }).reset_index()

    repeat_df['repeat_customer'] = np.where(
        repeat_df['order_id'] > 3,
        1,
        0
    )

    X_repeat = repeat_df[[
        'order_id',
        'quantity',
        'total_price'
    ]]

    y_repeat = repeat_df[
        'repeat_customer'
    ]

    repeat_model = RandomForestClassifier(
        n_estimators=150,
        random_state=42
    )

    repeat_model.fit(
        X_repeat,
        y_repeat
    )

    rp1, rp2 = st.columns([3, 1])

    with rp1:

        repeat_input = st.text_input(
            "Enter Customer ID",
            placeholder="Enter Customer ID",
            key="repeat_input"
        )

        repeat_btn = st.button(
            "🚀 Predict",
            key="repeat_btn"
        )

    with rp2:

        st.info("""
        📌 Repeat Reference

        Above 70% → High Repeat Chance

        40% - 70% → Moderate

        Below 40% → Low Repeat Chance
        """)

    if repeat_btn:

        st.session_state.repeat_run = True
        st.session_state.repeat_customer = (
            repeat_input
        )

    st.markdown("---")

    if st.session_state.repeat_run:

        repeat_input = (
            st.session_state.repeat_customer
        )

        if repeat_input.isdigit():

            customer_id = int(repeat_input)

            customer_exists = repeat_df[
                repeat_df['customer_id']
                ==
                customer_id
            ]

            if len(customer_exists) > 0:

                row = customer_exists.iloc[0]

                input_df = pd.DataFrame({
                    'order_id': [
                        row['order_id']
                    ],
                    'quantity': [
                        row['quantity']
                    ],
                    'total_price': [
                        row['total_price']
                    ]
                })

                probability = repeat_model.predict_proba(
                    input_df
                )[0][1] * 100

                r1, r2 = st.columns(2)

                r1.metric(
                    "Repeat Probability",
                    f"{probability:.1f}%"
                )

                r2.metric(
                    "Total Orders",
                    int(row['order_id'])
                )

                if probability >= 70:

                    st.success("""
                    ✅ High repeat purchase chance
                    """)

                elif probability >= 40:

                    st.warning("""
                    ⚠️ Moderate repeat purchase chance
                    """)

                else:

                    st.error("""
                    ❌ Low repeat purchase chance
                    """)

    st.markdown("---")

    # =====================================================
    # FUTURE BUY RECOMMENDATION
    # =====================================================
    st.subheader("🎯 Future Buy Recommendation")

    fr1, fr2 = st.columns([3, 1])

    with fr1:

        recommend_input = st.text_input(
            "Enter Customer ID",
            placeholder="Enter Customer ID",
            key="recommend_input"
        )

        recommend_btn = st.button(
            "🚀 Predict",
            key="recommend_btn"
        )

    with fr2:

        st.info("""
        📌 Recommendation Reference

        Similar Customers →
        Better Prediction

        Higher Score →
        Higher Accuracy
        """)

    if recommend_btn:

        st.session_state.recommend_run = True
        st.session_state.recommend_customer = (
            recommend_input
        )

    st.markdown("---")

    if st.session_state.recommend_run:

        recommend_input = (
            st.session_state.recommend_customer
        )

        if recommend_input.isdigit():

            customer_id = int(recommend_input)

            if customer_id in similarity_df.columns:

                similar_users = similarity_df[
                    customer_id
                ].sort_values(
                    ascending=False
                )

                similar_users = similar_users.iloc[1:6]

                bought_products = set(
                    user_product.loc[
                        customer_id
                    ][
                        user_product.loc[
                            customer_id
                        ] > 0
                    ].index
                )

                recommended_products = {}

                for sim_user in similar_users.index:

                    sim_products = user_product.loc[
                        sim_user
                    ]

                    for product, qty in sim_products.items():

                        if (
                            qty > 0
                            and
                            product not in bought_products
                        ):

                            if product not in recommended_products:

                                recommended_products[
                                    product
                                ] = qty

                            else:

                                recommended_products[
                                    product
                                ] += qty

                if len(recommended_products) > 0:

                    rec_df = pd.DataFrame(
                        recommended_products.items(),
                        columns=[
                            'Product',
                            'Prediction Score'
                        ]
                    )

                    rec_df = rec_df.sort_values(
                        by='Prediction Score',
                        ascending=False
                    ).head(10)

                    st.dataframe(
                        rec_df,
                        use_container_width=True
                    )

                    fig6 = px.bar(
                        rec_df,
                        x='Product',
                        y='Prediction Score',
                        color='Prediction Score',
                        title="Future Product Recommendation"
                    )

                    st.plotly_chart(
                        fig6,
                        use_container_width=True,
                        key=f"recommend_chart_{customer_id}"
                    )

                    top_score = rec_df.iloc[0][
                        'Prediction Score'
                    ]

                    if top_score >= 15:

                        st.success("""
                        ✅ Strong recommendation confidence
                        """)

                    elif top_score >= 7:

                        st.warning("""
                        ⚠️ Moderate recommendation confidence
                        """)

                    else:

                        st.error("""
                        ❌ Weak recommendation confidence
                        """)