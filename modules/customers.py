import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import plotly.express as px
import pandas as pd
from db import get_conn


def show_customers(df):

    st.markdown("## 👥 Customer Dashboard")

    # ================= RFM =================
    today = datetime.today()

    rfm = df.groupby('customer_id').agg({
        'order_date': lambda x: (today - x.max()).days,
        'order_id': 'count',
        'total_price': 'sum'
    }).reset_index()

    rfm.columns = [
        'customer_id',
        'recency',
        'frequency',
        'monetary'
    ]

    # ================= INCLUDE ALL =================
    cust_df = pd.read_sql(
        "SELECT * FROM customers",
        get_conn()
    )

    rfm = cust_df[['customer_id']].merge(
        rfm,
        on='customer_id',
        how='left'
    )

    rfm['recency'] = rfm['recency'].fillna(999)
    rfm['frequency'] = rfm['frequency'].fillna(0)
    rfm['monetary'] = rfm['monetary'].fillna(0)

    # ================= CLUSTER =================
    scaler = StandardScaler()

    scaled = scaler.fit_transform(
        rfm[['recency', 'frequency', 'monetary']]
    )

    kmeans = KMeans(
        n_clusters=3,
        random_state=42
    )

    rfm['cluster'] = kmeans.fit_predict(
        scaled
    )

    # ================= SEGMENT =================
    rfm['segment'] = 'Normal'

    rfm.loc[
        rfm['frequency'] == 0,
        'segment'
    ] = 'Inactive'

    rfm.loc[
        rfm['monetary'] >
        rfm['monetary'].quantile(0.75),
        'segment'
    ] = 'VIP'

    rfm.loc[
        (
            rfm['frequency'] >
            rfm['frequency'].mean()
        )
        &
        (
            rfm['segment'] != 'VIP'
        ),
        'segment'
    ] = 'Loyal'

    # ================= KPI =================
    st.subheader("📊 Customer Summary")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "👥 Total",
        len(rfm)
    )

    col2.metric(
        "👑 VIP",
        len(
            rfm[
                rfm['segment'] == 'VIP'
            ]
        )
    )

    col3.metric(
        "🔁 Loyal",
        len(
            rfm[
                rfm['segment'] == 'Loyal'
            ]
        )
    )

    col4.metric(
        "⚠️ Inactive",
        len(
            rfm[
                rfm['segment'] == 'Inactive'
            ]
        )
    )

    st.markdown("---")

    # ================= VISUAL =================
    st.subheader("📈 Customer Segmentation")

    fig = px.scatter(
        rfm,
        x='frequency',
        y='monetary',
        color='segment',
        size='monetary',
        hover_data=['customer_id']
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="customer_segment_chart"
    )

    st.markdown("---")

    # ================= FILTER =================
    st.subheader("🔍 Filter Customers")

    segment_option = st.selectbox(
        "Select Segment",
        [
            "All",
            "VIP",
            "Loyal",
            "Normal",
            "Inactive"
        ]
    )

    if segment_option != "All":

        filtered = rfm[
            rfm['segment']
            == segment_option
        ]

    else:

        filtered = rfm

    # ================= TABLE =================
    st.subheader("📋 Customer List")

    st.dataframe(
        filtered.sort_values(
            by='monetary',
            ascending=False
        )[
            [
                'customer_id',
                'recency',
                'frequency',
                'monetary',
                'segment',
                'cluster'
            ]
        ].head(20),
        use_container_width=True
    )

    st.write(
        "Total:",
        len(filtered)
    )

    st.markdown("---")

    # ================= CUSTOMER SEARCH =================
    st.subheader("🔎 Search Customer")

    customer_input = st.text_input(
        "Search a Customer",
        placeholder="Enter Customer ID"
    )

    search_btn = st.button(
        "🔍 Confirm Search"
    )

    if customer_input != "":

        try:

            selected_customer = int(
                customer_input
            )

        except:

            selected_customer = None

    else:

        selected_customer = None

    # ============================================
    # SHOW CUSTOMER DETAILS
    # ============================================
    if search_btn:

        if (
            selected_customer is None
            or
            selected_customer not in rfm['customer_id'].values
        ):

            st.warning("Customer not found")

        else:

            # ====================================
            # CUSTOMER INFO
            # ====================================
            customer_info = rfm[
                rfm['customer_id']
                == selected_customer
            ].iloc[0]

            st.markdown("---")

            st.subheader(
                f"👤 Customer {selected_customer}"
            )

            c1, c2, c3, c4, c5 = st.columns(5)

            c1.metric(
                "🎯 Segment",
                customer_info['segment']
            )

            c2.metric(
                "🧠 Cluster",
                int(customer_info['cluster'])
            )

            c3.metric(
                "💰 Spending",
                f"{customer_info['monetary']:.0f}"
            )

            c4.metric(
                "📦 Orders",
                int(customer_info['frequency'])
            )

            c5.metric(
                "⏰ Recency",
                int(customer_info['recency'])
            )

            st.markdown("---")

            # ====================================
            # PRODUCT HISTORY
            # ====================================
            customer_products = df[
                df['customer_id']
                == selected_customer
            ]

            product_summary = customer_products.groupby(
                'product_name',
                as_index=False
            ).agg({
                'quantity': 'sum',
                'total_price': 'sum',
                'order_id': 'nunique'
            })

            product_summary.columns = [
                'Product',
                'Quantity Bought',
                'Total Spending',
                'Times Purchased'
            ]

            product_summary = product_summary.drop_duplicates()

            st.subheader(
                "🛒 Purchased Products"
            )

            st.dataframe(
                product_summary.sort_values(
                    by='Quantity Bought',
                    ascending=False
                ),
                use_container_width=True
            )

            st.markdown("---")

            # ====================================
            # PRODUCT CHART
            # ====================================
            st.subheader(
                "📈 Product Purchase Chart"
            )

            fig2 = px.bar(
                product_summary,
                x='Product',
                y='Quantity Bought',
                color='Quantity Bought',
                title="Customer Product Purchases"
            )

            st.plotly_chart(
                fig2,
                use_container_width=True,
                key=f"customer_chart_{selected_customer}"
            )

            st.markdown("---")

            # ====================================
            # SIMILAR CUSTOMERS
            # ====================================
            st.subheader(
                "👥 Similar Customers"
            )

            user_product = df.pivot_table(
                index='customer_id',
                columns='product_name',
                values='quantity',
                aggfunc='sum',
                fill_value=0
            )

            similarity = cosine_similarity(
                user_product
            )

            similarity_df = pd.DataFrame(
                similarity,
                index=user_product.index,
                columns=user_product.index
            )

            similar_users = similarity_df[
                selected_customer
            ].sort_values(
                ascending=False
            )

            similar_users = similar_users.iloc[1:6]

            similar_df = pd.DataFrame({
                'Customer ID': similar_users.index,
                'Similarity Score': similar_users.values
            })

            st.dataframe(
                similar_df,
                use_container_width=True
            )

            st.markdown("---")

            # ====================================
            # TOP PRODUCT
            # ====================================
            top_product = product_summary.sort_values(
                by='Quantity Bought',
                ascending=False
            ).iloc[0]['Product']

            # ====================================
            # AI INSIGHT
            # ====================================
            st.subheader("🧠 AI Customer Insight")

            if customer_info['segment'] == "VIP":

                st.success(f"""
                👑 VIP Customer

                • High-value customer
                • Strong purchasing power
                • Favorite Product:
                  {top_product}

                Recommended Action:
                • Premium rewards
                • Early-access campaigns
                • Personalized offers
                """)

            elif customer_info['segment'] == "Loyal":

                st.info(f"""
                🔁 Loyal Customer

                • Frequently purchases products
                • Strong engagement
                • Favorite Product:
                  {top_product}

                Recommended Action:
                • Loyalty rewards
                • Combo offers
                • Retention campaigns
                """)

            elif customer_info['segment'] == "Inactive":

                st.warning(f"""
                ⚠️ Inactive Customer

                • No recent purchase activity
                • Previously interested in:
                  {top_product}

                Recommended Action:
                • Discount campaigns
                • Reactivation emails
                • Personalized promotions
                """)

            else:

                st.info(f"""
                🙂 Normal Customer

                • Moderate purchase activity
                • Favorite Product:
                  {top_product}

                Recommended Action:
                • Product recommendations
                • Bundle offers
                • Upselling campaigns
                """)

    st.markdown("---")

    # ================= CAMPAIGN =================
    st.subheader("🎯 Campaign Manager")

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        if st.button("👑 VIP Campaign"):

            st.success(
                "VIP Offer Running"
            )

            st.write(
                "• 20% Discount"
            )

            st.write(
                "• Premium Access"
            )

    with col2:

        if st.button("🔁 Loyalty Program"):

            st.success(
                "Loyalty Activated"
            )

            st.write(
                "• Reward Points"
            )

            st.write(
                "• Cashback"
            )

    with col3:

        if st.button("🙂 Conversion Offer"):

            st.info(
                "Conversion Campaign"
            )

            st.write(
                "• 10% Discount"
            )

            st.write(
                "• Combo Deals"
            )

    with col4:

        if st.button("⚠️ Reactivation"):

            st.warning(
                "Reactivation Started"
            )

            st.write(
                "• 30% Discount"
            )

            st.write(
                "• Email Campaign"
            )

    st.markdown("---")

    # ================= SMART INSIGHT =================
    st.subheader("🧠 Smart Insight")

    if segment_option == "VIP":

        st.success(
            "👑 Focus on retention & premium services"
        )

    elif segment_option == "Loyal":

        st.success(
            "🔁 Boost repeat purchases with rewards"
        )

    elif segment_option == "Inactive":

        st.warning(
            "⚠️ Need re-engagement campaign"
        )

    elif segment_option == "Normal":

        st.info(
            "🙂 Convert into loyal customers"
        )

    else:

        vip_ratio = len(
            rfm[
                rfm['segment'] == 'VIP'
            ]
        ) / len(rfm)

        if vip_ratio > 0.2:

            st.success(
                "🔥 Strong high-value customer base"
            )

        else:

            st.warning(
                "⚠️ Need to increase VIP customers"
            )