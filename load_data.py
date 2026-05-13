# ==============================
# load_data.py
# ==============================

import pandas as pd
from db import get_conn

def load_data():

    conn = get_conn()

    # Use merged analytical table
    query = "SELECT * FROM merge"

    df = pd.read_sql(query, conn)

    # Convert date column
    if 'order_date' in df.columns:
        df['order_date'] = pd.to_datetime(df['order_date'])

    return df