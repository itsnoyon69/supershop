import pandas as pd
from db import get_conn

def load_data():
    conn = get_conn()

    query = "SELECT * FROM merge"

    df = pd.read_sql(query, conn)

    df['order_date'] = pd.to_datetime(df['order_date'])

    return df
```
