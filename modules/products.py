import streamlit as st
import pandas as pd
import plotly.express as px


def show_products(df):

    st.markdown("## 📦 Product Analytics (Demand)")

    # ================= DATA =================
    prod = df.groupby('product_name').agg({
        'quantity': 'sum',
        'total_price': 'sum'
    }).reset_index().sort_values(
        by='quantity',
        ascending=False
    )

    size_df = df.groupby([
        'product_name',
        'size'
    ])['quantity'].sum().reset_index()

    size_total = df.groupby(
        'size'
    )['quantity'].sum().reset_index().sort_values(
        by='quantity',
        ascending=False
    )

    # ================= KPI =================
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "📦 Total Products",
        prod['product_name'].nunique()
    )

    col2.metric(
        "🔥 Best Seller",
        prod.iloc[0]['product_name']
    )

    col3.metric(
        "⚠️ Low Seller",
        prod.iloc[-1]['product_name']
    )

    st.markdown("---")

    # ================= PRODUCT SEARCH =================
    st.subheader("🔎 Search Product")

    product_names = sorted(
        df['product_name']
        .dropna()
        .unique()
    )

    selected_product = st.selectbox(
        "Select Product",
        ["Search Product"] + product_names
    )

    search_btn = st.button(
        "🔍 Confirm Product Search"
    )

    st.markdown("---")

    # ============================================
    # SHOW PRODUCT DETAILS
    # ============================================
    if search_btn:

        if selected_product == "Search Product":

            st.warning("Please select a product")

        else:

            # ====================================
            # PRODUCT DATA
            # ====================================
            product_data = df[
                df['product_name']
                == selected_product
            ]

            total_qty = product_data[
                'quantity'
            ].sum()

            total_revenue = product_data[
                'total_price'
            ].sum()

            total_orders = product_data[
                'order_id'
            ].nunique()

            total_customers = product_data[
                'customer_id'
            ].nunique()

            st.subheader(
                f"📦 {selected_product}"
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "📦 Quantity Sold",
                int(total_qty)
            )

            c2.metric(
                "💰 Revenue",
                f"{total_revenue:.0f}"
            )

            c3.metric(
                "🛒 Orders",
                total_orders
            )

            c4.metric(
                "👥 Customers",
                total_customers
            )

            st.markdown("---")

            # ====================================
            # SIZE ANALYSIS
            # ====================================
            st.subheader(
                "📏 Product Size Analysis"
            )

            product_sizes = product_data.groupby(
                'size',
                as_index=False
            )['quantity'].sum()

            fig_size = px.bar(
                product_sizes,
                x='size',
                y='quantity',
                color='quantity',
                title="Product Size Demand"
            )

            st.plotly_chart(
                fig_size,
                use_container_width=True,
                key=f"size_analysis_{selected_product}"
            )

            st.markdown("---")

            # ====================================
            # CUSTOMER ANALYSIS
            # ====================================
            st.subheader(
                "👑 Top Customers"
            )

            customer_summary = product_data.groupby(
                'customer_id',
                as_index=False
            ).agg({
                'quantity': 'sum',
                'total_price': 'sum',
                'order_id': 'nunique'
            })

            customer_summary.columns = [
                'Customer ID',
                'Quantity Bought',
                'Total Spending',
                'Times Purchased'
            ]

            customer_summary = customer_summary.sort_values(
                by='Quantity Bought',
                ascending=False
            )

            st.dataframe(
                customer_summary.head(10),
                use_container_width=True
            )

            st.markdown("---")

            # ====================================
            # CUSTOMER BAR CHART
            # ====================================
            st.subheader(
                "📈 Customer Purchase Chart"
            )

            fig_customer = px.bar(
                customer_summary.head(10),
                x='Customer ID',
                y='Quantity Bought',
                color='Quantity Bought',
                title="Top Customers"
            )

            st.plotly_chart(
                fig_customer,
                use_container_width=True,
                key=f"customer_chart_{selected_product}"
            )

            st.markdown("---")

            # ====================================
            # SALES TREND
            # ====================================
            st.subheader(
                "📊 Product Sales Trend"
            )

            trend_df = product_data.groupby(
                'order_date',
                as_index=False
            )['quantity'].sum()

            fig_trend = px.line(
                trend_df,
                x='order_date',
                y='quantity',
                markers=True,
                title="Sales Trend"
            )

            st.plotly_chart(
                fig_trend,
                use_container_width=True,
                key=f"trend_chart_{selected_product}"
            )

            st.markdown("---")

            # ====================================
            # AI INSIGHT
            # ====================================
            st.subheader(
                "🧠 AI Product Insight"
            )

            top_customer = customer_summary.iloc[0][
                'Customer ID'
            ]

            best_size = product_sizes.sort_values(
                by='quantity',
                ascending=False
            ).iloc[0]['size']

            avg_sales = prod['quantity'].mean()

            if total_qty > avg_sales * 1.5:

                st.success(f"""
                🔥 High Demand Product

                • Most popular size:
                  {best_size}

                • Top customer:
                  {top_customer}

                Recommended Action:
                • Increase stock
                • Push marketing
                • Create combo offers
                """)

            elif total_qty < avg_sales:

                st.warning(f"""
                ⚠️ Low Demand Product

                • Most active customer:
                  {top_customer}

                Recommended Action:
                • Run discounts
                • Bundle with best sellers
                • Reduce inventory
                """)

            else:

                st.info(f"""
                📈 Stable Product Performance

                • Most popular size:
                  {best_size}

                Recommended Action:
                • Maintain inventory
                • Monitor demand trends
                """)

    st.markdown("---")

    # ================= SALES CHART =================
    st.subheader("📈 Top Selling Products")

    fig1 = px.bar(
        prod.head(10),
        x='product_name',
        y='quantity',
        title="Top Products by Sales"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True,
        key="prod_bar"
    )

    st.markdown("---")

    # ================= LOW SELLING =================
    st.subheader("⚠️ Low Selling Products")

    low_products = prod.sort_values(
        by='quantity',
        ascending=True
    ).head(10)

    fig_low = px.bar(
        low_products,
        x='product_name',
        y='quantity',
        title="Low Selling Products",
        color='quantity'
    )

    st.plotly_chart(
        fig_low,
        use_container_width=True,
        key="low_selling_chart"
    )

    st.markdown("---")

    # ================= SIZE DEMAND =================
    st.subheader("📏 Overall Size Demand")

    fig2 = px.bar(
        size_total,
        x='size',
        y='quantity',
        title="Overall Size Demand"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True,
        key="size_total"
    )

    st.markdown("---")

    # ================= INSIGHT =================
    st.subheader("🧠 Product Insight")

    top_product = prod.iloc[0]['product_name']

    low_product = prod.iloc[-1]['product_name']

    st.success(
        f"🔥 Best Selling Product: {top_product}"
    )

    st.warning(
        f"⚠️ Low Performing Product: {low_product}"
    )

    st.info(
        f"🎯 Strategy: Promote '{low_product}' or bundle with '{top_product}'"
    )