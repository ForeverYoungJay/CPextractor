# db/pg.py
import psycopg
from psycopg.rows import dict_row

def connect_pg(host: str, port: int, dbname: str, user: str, password: str) -> psycopg.Connection:
    conn = psycopg.connect(
        host=host, port=port, dbname=dbname, user=user, password=password,
        row_factory=dict_row
    )
    conn.execute("SET statement_timeout TO '5min';")
    return conn
