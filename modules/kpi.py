import streamlit as st
import pandas as pd
from db import get_conn


def show_kpi(df):

    # =====================================================
    # DATABASE
    # =====================================================
    cust_df = pd.read_sql(
        "SELECT * FROM customers",
        get_conn()
    )

    # =====================================================
    # BASIC KPI
    # =====================================================
    total_revenue = df['total_price'].sum()

    total_orders = df['order_id'].nunique()

    total_customers = cust_df[
        'customer_id'
    ].nunique()

    active_customers = df[
        'customer_id'
    ].nunique()

    inactive_customers = (
        total_customers -
        active_customers
    )

    avg_order = (
        total_revenue / total_orders
        if total_orders != 0 else 0
    )

    # =====================================================
    # GROWTH KPI
    # =====================================================
    revenue_by_date = df.groupby(
        'order_date'
    )['total_price'].sum().reset_index()

    revenue_by_date = revenue_by_date.sort_values(
        by='order_date'
    )

    if not revenue_by_date.empty:

        first_revenue = revenue_by_date['total_price'].iloc[0]

        last_revenue = revenue_by_date['total_price'].iloc[-1]

    else:

        first_revenue = 0
        last_revenue = 0

    if first_revenue == 0:

        revenue_growth = 0

    else:

        revenue_growth = (
            (
                last_revenue -
                first_revenue
            )
            /
            first_revenue
        ) * 100

    # =====================================================
    # KPI CARDS
    # =====================================================
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "💰 Revenue",
        f"{total_revenue:,.0f}",
        f"{revenue_growth:.1f}%"
    )

    col2.metric(
        "📦 Orders",
        total_orders
    )

    col3.metric(
        "👥 Customers",
        total_customers
    )

    col4.metric(
        "🛍 Avg Order",
        f"{avg_order:,.0f}"
    )

    st.markdown("---")

    # =====================================================
    # EXTRA KPI
    # =====================================================
    e1, e2, e3 = st.columns(3)

    e1.metric(
        "🟢 Active Customers",
        active_customers
    )

    e2.metric(
        "😴 Inactive Customers",
        inactive_customers
    )

    repeat_df = df.groupby(
        'customer_id'
    )['order_id'].nunique()

    repeat_customers = len(
        repeat_df[repeat_df > 1]
    )

    repeat_rate = (
        repeat_customers /
        active_customers
    ) * 100 if active_customers != 0 else 0

    e3.metric(
        "🔁 Repeat Rate",
        f"{repeat_rate:.1f}%"
    )

    st.markdown("---")

    # =====================================================
    # LATEST CUSTOMER
    # =====================================================
    with st.expander(
        "👥 View Latest Customers"
    ):

        latest_customers = cust_df.sort_values(
            by='customer_id',
            ascending=False
        ).head(5)

        st.dataframe(
            latest_customers,
            use_container_width=True
        )