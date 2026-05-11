import pandas as pd
from db import get_conn

def load_data():
    conn = get_conn()

    customers = pd.read_sql("SELECT * FROM customers", conn)
    products = pd.read_sql("SELECT * FROM products", conn)
    orders = pd.read_sql("SELECT * FROM orders", conn)
    sales = pd.read_sql("SELECT * FROM sales", conn)

    df = sales.merge(orders, on='order_id', how='left') \
              .merge(customers, on='customer_id', how='left') \
              .merge(products, on='product_id', how='left', suffixes=('', '_drop'))

    df = df[[c for c in df.columns if not c.endswith('_drop')]]

    df.to_csv(
        "merged_dataset.csv",
        index=False
    )

    print(df.head())

    if 'quantity_y' in df.columns:
        df['quantity'] = df['quantity_y']
    elif 'quantity_x' in df.columns:
        df['quantity'] = df['quantity_x']

    df['order_date'] = pd.to_datetime(df['order_date'])

    return df