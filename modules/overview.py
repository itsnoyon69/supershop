import streamlit as st
import pandas as pd
import plotly.express as px

from modules.kpi import show_kpi


def show_overview(df):

    st.markdown("## 📊 Business Overview")

    # =====================================================
    # KPI
    # =====================================================
    show_kpi(df)

    st.markdown("---")

    # =====================================================
    # SALES TREND
    # =====================================================
    st.subheader("📈 Sales Trend")

    sales = df.groupby(
        'order_date'
    )['total_price'].sum().reset_index()

    fig = px.line(
        sales,
        x='order_date',
        y='total_price',
        markers=True,
        title="Revenue Trend"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    # =====================================================
    # PRODUCT INSIGHT
    # =====================================================
    st.subheader("🧠 Product Insight")

    product_perf = df.groupby(
        'product_name'
    )['quantity'].sum()


    if not product_perf.empty:

        top_product = product_perf.idxmax()

    else:

        top_product = "No Data"

    low_product = product_perf.idxmin()

    col1, col2 = st.columns(2)

    with col1:

        st.success(
            f"🔥 Top Product: {top_product}"
        )

    with col2:

        st.warning(
            f"⚠️ Low Product: {low_product}"
        )

    st.info(f"""
    🎯 Suggested Strategy

    Bundle '{low_product}'
    with '{top_product}'
    to improve weak sales.
    """)

    st.markdown("---")

    # =====================================================
    # TOP PRODUCTS
    # =====================================================
    st.subheader("📦 Top Selling Products")

    top5 = product_perf.sort_values(
        ascending=False
    ).head(5)

    fig2 = px.bar(
        x=top5.index,
        y=top5.values,
        labels={
            'x': 'Product',
            'y': 'Sales'
        },
        title="Top 5 Products"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.markdown("---")

    # =====================================================
    # BUSINESS STATUS
    # =====================================================
    st.subheader("🚀 Business Status")

    revenue_growth = (
        sales['total_price'].iloc[-1]
        -
        sales['total_price'].iloc[0]
    )

    if revenue_growth > 0:

        st.success("""
        📈 Business performance improving

        • Revenue trend positive
        • Strong product demand
        • Customer activity healthy
        """)

    else:

        st.warning("""
        📉 Revenue growth slowing

        • Need marketing strategy
        • Focus on customer retention
        • Improve weak product sales
        """)