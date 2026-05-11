import streamlit as st
import pandas as pd
import plotly.express as px


# =========================================================
# STRATEGY SYSTEM
# =========================================================
def show_strategy(df):

    st.markdown("## 🎯 AI Business Strategy")

    # =====================================================
    # DATA PREP
    # =====================================================
    product_df = df.groupby('product_name').agg({
        'quantity': 'sum',
        'total_price': 'sum',
        'order_id': 'nunique',
        'customer_id': 'nunique'
    }).reset_index()

    product_df.columns = [
        'product_name',
        'sales',
        'revenue',
        'orders',
        'customers'
    ]

    # =====================================================
    # PRODUCT SCORE
    # =====================================================
    product_df['sales_score'] = (
        product_df['sales']
        /
        product_df['sales'].max()
    ) * 100

    product_df['revenue_score'] = (
        product_df['revenue']
        /
        product_df['revenue'].max()
    ) * 100

    product_df['customer_score'] = (
        product_df['customers']
        /
        product_df['customers'].max()
    ) * 100

    product_df['strategy_score'] = (
        product_df['sales_score'] * 0.4 +
        product_df['revenue_score'] * 0.4 +
        product_df['customer_score'] * 0.2
    )

    # =====================================================
    # STATUS
    # =====================================================
    def product_status(score):

        if score >= 75:
            return "🔥 Growth Opportunity"

        elif score >= 45:
            return "📈 Stable"

        else:
            return "⚠️ Risk Product"

    product_df['status'] = product_df[
        'strategy_score'
    ].apply(product_status)

    # =====================================================
    # OVERVIEW
    # =====================================================
    st.subheader("📊 Strategy Overview")

    top_product = product_df.sort_values(
        by='strategy_score',
        ascending=False
    ).iloc[0]

    weak_products = product_df[
        product_df['status']
        ==
        "⚠️ Risk Product"
    ]

    o1, o2, o3 = st.columns(3)

    o1.metric(
        "🔥 Best Product",
        top_product['product_name']
    )

    o2.metric(
        "📈 Top Score",
        f"{top_product['strategy_score']:.1f}"
    )

    o3.metric(
        "⚠️ Risk Products",
        len(weak_products)
    )

    st.markdown("---")

    # =====================================================
    # PRODUCT STRATEGY
    # =====================================================
    st.subheader("📦 Product Strategy")

    product_options = sorted(
        product_df['product_name']
        .dropna()
        .unique()
    )

    p1, p2 = st.columns([3, 1])

    with p1:

        selected_product = st.selectbox(
            "Select Product",
            ["Select Product"] + list(product_options),
            index=0,
            key="strategy_product_select"
        )

    with p2:

        st.write("")
        st.write("")

        product_btn = st.button(
            "🔍 Confirm Product Search",
            use_container_width=True,
            key="strategy_product_btn"
        )

    # =====================================================
    # SESSION STATE FIX
    # =====================================================
    if product_btn:
        st.session_state["strategy_product_run"] = True

    st.markdown("---")

    # =====================================================
    # PRODUCT RESULT
    # =====================================================
    if st.session_state.get(
        "strategy_product_run",
        False
    ):

        if selected_product == "Select Product":

            st.warning(
                "Please select a product"
            )

        else:

            row = product_df[
                product_df['product_name']
                ==
                selected_product
            ].iloc[0]

            # =================================================
            # PRODUCT METRICS
            # =================================================
            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Sales",
                int(row['sales'])
            )

            c2.metric(
                "Revenue",
                f"{row['revenue']:.0f}"
            )

            c3.metric(
                "Strategy",
                row['status']
            )

            st.markdown("---")

            # =================================================
            # AI STRATEGY
            # =================================================
            if row['strategy_score'] >= 75:

                st.success(f"""
                🔥 {selected_product} has strong growth potential.

                Recommended Strategy:
                • Increase stock
                • Push marketing campaigns
                • Highlight in homepage
                • Create combo offers
                """)

            elif row['strategy_score'] >= 45:

                st.info(f"""
                📈 {selected_product} is stable.

                Recommended Strategy:
                • Monitor performance
                • Improve promotions
                • Test bundle strategy
                """)

            else:

                st.warning(f"""
                ⚠️ {selected_product} is risky.

                Recommended Strategy:
                • Apply discounts
                • Reduce inventory
                • Bundle with top products
                """)

            st.markdown("---")

            # =================================================
            # GRAPH
            # =================================================
            graph_df = pd.DataFrame({
                'Metric': [
                    'Sales',
                    'Revenue',
                    'Customers'
                ],
                'Score': [
                    row['sales_score'],
                    row['revenue_score'],
                    row['customer_score']
                ]
            })

            fig = px.bar(
                graph_df,
                x='Metric',
                y='Score',
                title="Product Strategy Analysis"
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                key=f"strategy_graph_{selected_product}"
            )

    st.markdown("---")

    # =====================================================
    # CUSTOMER STRATEGY
    # =====================================================
    st.subheader("👥 Customer Strategy")

    customer_df = df.groupby('customer_id').agg({
        'order_id': 'nunique',
        'quantity': 'sum',
        'total_price': 'sum',
        'order_date': 'max'
    }).reset_index()

    customer_df.columns = [
        'customer_id',
        'orders',
        'items',
        'spending',
        'last_order'
    ]

    # =====================================================
    # INACTIVE DAYS
    # =====================================================
    today = df['order_date'].max()

    customer_df['inactive_days'] = (
        today -
        customer_df['last_order']
    ).dt.days

    # =====================================================
    # SCORE
    # =====================================================
    customer_df['customer_score'] = (
        (
            customer_df['orders']
            /
            customer_df['orders'].max()
        ) * 30
        +
        (
            customer_df['spending']
            /
            customer_df['spending'].max()
        ) * 50
        +
        (
            1 -
            (
                customer_df['inactive_days']
                /
                customer_df['inactive_days'].max()
            )
        ) * 20
    )

    # =====================================================
    # SEARCH
    # =====================================================
    s1, s2 = st.columns([3, 1])

    with s1:

        customer_input = st.text_input(
            "Search Customer ID",
            placeholder="Enter Customer ID",
            key="strategy_customer_search"
        )

    with s2:

        st.write("")
        st.write("")

        customer_btn = st.button(
            "🔍 Confirm Customer Search",
            use_container_width=True,
            key="strategy_customer_btn"
        )

    # =====================================================
    # SESSION STATE FIX
    # =====================================================
    if customer_btn:
        st.session_state["strategy_customer_run"] = True

    st.markdown("---")

    # =====================================================
    # CUSTOMER RESULT
    # =====================================================
    if st.session_state.get(
        "strategy_customer_run",
        False
    ):

        if customer_input == "":

            st.warning(
                "Please enter customer ID"
            )

        elif not customer_input.isdigit():

            st.error(
                "Customer ID must be numeric"
            )

        else:

            customer_id = int(customer_input)

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

                cust = customer_exists.iloc[0]

                # =============================================
                # CUSTOMER TYPE
                # =============================================
                if cust['customer_score'] >= 75:

                    customer_status = "💎 High Value"

                    recommendation = """
                    • Send loyalty rewards
                    • Offer premium products
                    • Give early access offers
                    """

                elif cust['customer_score'] >= 45:

                    customer_status = "📈 Moderate"

                    recommendation = """
                    • Personalized offers
                    • Product recommendations
                    • Engagement campaigns
                    """

                else:

                    customer_status = "⚠️ At Risk"

                    recommendation = """
                    • Reactivation discount
                    • Retarget campaigns
                    • Follow-up promotions
                    """

                # =============================================
                # METRICS
                # =============================================
                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Customer Score",
                    f"{cust['customer_score']:.1f}"
                )

                c2.metric(
                    "Total Spending",
                    f"{cust['spending']:.0f}"
                )

                c3.metric(
                    "Status",
                    customer_status
                )

                st.markdown("---")

                # =============================================
                # AI STRATEGY
                # =============================================
                st.success(f"""
                👤 Customer:
                {customer_id}

                Recommended Strategy:
                {recommendation}
                """)

    st.markdown("---")

    # =====================================================
    # FINAL STRATEGY
    # =====================================================
    st.subheader("🚀 Final Business Strategy")

    st.success(f"""
    🔥 Focus Product:
    {top_product['product_name']}

    🎯 Recommended Actions:
    • Increase focus on top products
    • Retain valuable customers
    • Reduce risk inventory
    • Improve marketing efficiency
    """)