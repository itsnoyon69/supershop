import streamlit as st
from load_data import load_data
from modules.admin import admin_panel
from modules.kpi import show_kpi
from modules.overview import show_overview
from modules.products import show_products
from modules.customers import show_customers
from modules.combo_and_discount import show_combo_engine
from modules.forecast import show_forecast
from modules.strategy import show_strategy
from modules.inventory import show_inventory
from modules.delivery import show_delivery_ai
from modules.prediction import show_prediction
st.set_page_config(page_title="Supershop Overview", layout="wide")

st.title("🛒 Supershop Dashboard")

df = load_data()

admin_panel(df)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "Overview", "Products", "Customers", "Inventory", "Combo & Discount", "Forecast", "Strategy", "Delivery", "Prediction"
])

with tab1:
    show_overview(df)

with tab2:
    show_products(df)

with tab3:
    show_customers(df)

with tab4:
    show_inventory(df)

with tab5:
    show_combo_engine(df)

with tab6:
    show_forecast(df)

with tab7:
    show_strategy(df)

with tab8:
    show_delivery_ai(df)
with tab9:
    show_prediction(df)
