import streamlit as st
import pandas as pd
import plotly.express as px


def show_inventory(df):

    st.markdown("## 📦 Inventory Management (Stock)")

    # ================= DATA =================
    stock = df.groupby([
        'product_name',
        'size'
    ])['quantity'].sum().reset_index()

    low_stock = stock[
        stock['quantity'] < 10
    ]

    high_stock = stock[
        stock['quantity'] > 80
    ]

    size_stock = df.groupby(
        'size'
    )['quantity'].sum().reset_index()

    # ================= KPI =================
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "📦 Total Stock",
        int(stock['quantity'].sum())
    )

    col2.metric(
        "🛍 Products",
        stock['product_name'].nunique()
    )

    col3.metric(
        "🔴 Low Stock",
        len(low_stock)
    )

    col4.metric(
        "🟡 Overstock",
        len(high_stock)
    )

    st.markdown("---")

    # ================= ALERT =================
    st.subheader("⚠️ Stock Alerts")

    col1, col2 = st.columns(2)

    # =================================================
    # LOW STOCK
    # =================================================
    with col1:

        st.write("🔴 Low Stock")

        if len(low_stock) == 0:

            st.success(
                "No low stock issues"
            )

        else:

            for r in low_stock.itertuples():

                st.error(
                    f"{r.product_name} ({r.size}) → {r.quantity}"
                )

    # =================================================
    # OVER STOCK
    # =================================================
    with col2:

        st.write("🟡 Overstock")

        if len(high_stock) == 0:

            st.info(
                "No overstock issues"
            )

        else:

            for r in high_stock.itertuples():

                st.warning(
                    f"{r.product_name} ({r.size}) → {r.quantity}"
                )

    st.markdown("---")

    # ================= SIZE SEARCH =================
    st.subheader("📏 Search Size")

    available_sizes = sorted(
        df['size']
        .dropna()
        .astype(str)
        .unique()
    )

    size_options = [
        "Select Size"
    ] + available_sizes

    col1, col2 = st.columns([3, 1])

    with col1:

        selected_size = st.selectbox(
            "Select Size",
            size_options
        )

    with col2:

        st.write("")
        st.write("")

        size_btn = st.button(
            "🔍 Search Size",
            use_container_width=True
        )

    # ============================================
    # SIZE RESULT
    # ============================================
    if size_btn:

        if selected_size == "Select Size":

            st.warning("Please select a size")

        else:

            size_data = df[
                df['size'].astype(str)
                == selected_size
            ]

            st.markdown("---")

            st.subheader(
                f"📏 Size {selected_size} Analysis"
            )

            # ====================================
            # SIZE KPI
            # ====================================
            total_qty = size_data[
                'quantity'
            ].sum()

            total_orders = size_data[
                'order_id'
            ].nunique()

            total_customers = size_data[
                'customer_id'
            ].nunique()

            total_products = size_data[
                'product_name'
            ].nunique()

            k1, k2, k3, k4 = st.columns(4)

            k1.metric(
                "📦 Quantity",
                int(total_qty)
            )

            k2.metric(
                "🛒 Orders",
                total_orders
            )

            k3.metric(
                "👥 Customers",
                total_customers
            )

            k4.metric(
                "📦 Products",
                total_products
            )

            st.markdown("---")

            # ====================================
            # PRODUCT TABLE
            # ====================================
            size_summary = size_data.groupby(
                'product_name',
                as_index=False
            )['quantity'].sum()

            size_summary = size_summary.sort_values(
                by='quantity',
                ascending=False
            )

            st.subheader(
                "📦 Products in This Size"
            )

            st.dataframe(
                size_summary,
                use_container_width=True
            )

            st.markdown("---")

            # ====================================
            # PRODUCT BAR
            # ====================================
            st.subheader(
                "📈 Product Demand"
            )

            fig_size_products = px.bar(
                size_summary,
                x='product_name',
                y='quantity',
                color='quantity',
                title=f"Products Sold in Size {selected_size}"
            )

            st.plotly_chart(
                fig_size_products,
                use_container_width=True,
                key=f"size_product_chart_{selected_size}"
            )

            st.markdown("---")

            # ====================================
            # CUSTOMER TABLE
            # ====================================
            customer_size = size_data.groupby(
                'customer_id',
                as_index=False
            ).agg({
                'quantity': 'sum',
                'order_id': 'nunique'
            })

            customer_size.columns = [
                'Customer ID',
                'Quantity Bought',
                'Orders'
            ]

            customer_size = customer_size.sort_values(
                by='Quantity Bought',
                ascending=False
            )

            st.subheader(
                "👑 Top Customers"
            )

            st.dataframe(
                customer_size.head(10),
                use_container_width=True
            )

            st.markdown("---")

            # ====================================
            # CUSTOMER BAR
            # ====================================
            st.subheader(
                "📊 Customer Purchase Analysis"
            )

            fig_size_customer = px.bar(
                customer_size.head(10),
                x='Customer ID',
                y='Quantity Bought',
                color='Quantity Bought',
                title=f"Top Customers Buying Size {selected_size}"
            )

            st.plotly_chart(
                fig_size_customer,
                use_container_width=True,
                key=f"size_customer_chart_{selected_size}"
            )

            st.markdown("---")

            # ====================================
            # AI INSIGHT
            # ====================================
            st.subheader(
                "🧠 AI Size Insight"
            )

            top_product = size_summary.iloc[0][
                'product_name'
            ]

            top_customer = customer_size.iloc[0][
                'Customer ID'
            ]

            avg_qty = size_stock[
                'quantity'
            ].mean()

            if total_qty > avg_qty * 1.5:

                st.success(f"""
                🔥 High Demand Size

                • Most popular product:
                  {top_product}

                • Top customer:
                  {top_customer}

                Recommended Action:
                • Increase stock
                • Prioritize production
                • Improve availability
                """)

            elif total_qty < avg_qty:

                st.warning(f"""
                ⚠️ Low Demand Size

                • Weak demand detected
                • Most active customer:
                  {top_customer}

                Recommended Action:
                • Reduce stock
                • Offer discounts
                • Bundle products
                """)

            else:

                st.info(f"""
                📈 Stable Size Performance

                • Popular product:
                  {top_product}

                Recommended Action:
                • Maintain stock
                • Monitor demand
                """)

    st.markdown("---")

    # ================= SIZE STOCK =================
    st.subheader("📏 Size Stock Distribution")

    fig1 = px.bar(
        size_stock,
        x='size',
        y='quantity',
        title="Size Stock"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True,
        key="inv_size"
    )

    st.markdown("---")

    # ================= PRODUCT STOCK =================
    st.subheader("📦 Product Stock View")

    col1, col2 = st.columns([3, 1])

    with col1:

        product_options = [
            "Select Product"
        ] + list(
            df['product_name']
            .dropna()
            .unique()
        )

        selected_product = st.selectbox(
            "Select Product",
            product_options,
            index=0,
            key="inv_selectbox"
        )

    with col2:

        st.write("")
        st.write("")

        show_btn = st.button(
            "🔍 Show Product Stock",
            use_container_width=True
        )

    # =================================================
    # SHOW GRAPH
    # =================================================
    if show_btn:

        if selected_product == "Select Product":

            st.warning("""
            Please select a product
            """)

        else:

            product_stock = stock[
                stock['product_name']
                == selected_product
            ]

            fig2 = px.bar(
                product_stock,
                x='size',
                y='quantity',
                title=f"{selected_product} Stock by Size",
                color='quantity'
            )

            st.plotly_chart(
                fig2,
                use_container_width=True,
                key=f"inv_product_{selected_product}"
            )

            # =================================================
            # STOCK INSIGHT
            # =================================================
            total_stock = product_stock[
                'quantity'
            ].sum()

            avg_stock = stock[
                'quantity'
            ].mean()

            st.markdown("---")

            st.subheader("🧠 Product Stock Insight")

            if total_stock < avg_stock:

                st.error(f"""
                ⚠️ {selected_product} stock is low.

                Suggested Action:
                • Restock soon
                • Monitor fast-selling sizes
                """)

            elif total_stock > avg_stock * 2:

                st.warning(f"""
                🟡 {selected_product} has high stock.

                Suggested Action:
                • Run discounts
                • Create combo offers
                • Clear excess inventory
                """)

            else:

                st.success(f"""
                ✅ {selected_product} stock level is balanced.
                """)

    st.markdown("---")

    # ================= ACTION =================
    st.subheader("🎯 Suggested Actions")

    if len(low_stock) > 0:

        st.write("📦 Restock Needed:")

        for r in low_stock.itertuples():

            st.write(
                f"→ {r.product_name} ({r.size})"
            )

    if len(high_stock) > 0:

        st.write("💸 Discount / Clear Stock:")

        for r in high_stock.itertuples():

            st.write(
                f"→ {r.product_name} ({r.size})"
            )

    if (
        len(low_stock) == 0
        and
        len(high_stock) == 0
    ):

        st.success("""
        ✅ Stock levels are balanced
        """)