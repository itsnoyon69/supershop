import psycopg2

def get_conn():

    conn = psycopg2.connect(
        host="aws-1-ap-northeast-1.pooler.supabase.com",
        database="postgres",
        user="postgres.lrfshbpmjqevlookcrmx",
        password="#itsnoyon69#",
        port="5432"
    )

    return conn