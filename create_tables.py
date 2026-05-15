"""Drop and recreate all Sparkify Redshift warehouse tables."""

import configparser
from contextlib import closing

import psycopg2

from sql_queries import create_table_queries, drop_table_queries


def connect_to_redshift():
    """Open a psycopg2 connection using the values in dwh.cfg."""

    config = configparser.ConfigParser()
    config.read("dwh.cfg")
    return psycopg2.connect(
        host=config.get("CLUSTER", "HOST"),
        dbname=config.get("CLUSTER", "DB_NAME"),
        user=config.get("CLUSTER", "DB_USER"),
        password=config.get("CLUSTER", "DB_PASSWORD"),
        port=config.get("CLUSTER", "DB_PORT"),
    )


def drop_tables(cur, conn):
    """Drop tables in dependency-safe order."""

    for query in drop_table_queries:
        cur.execute(query)
        conn.commit()


def create_tables(cur, conn):
    """Create staging, dimension, and fact tables."""

    for query in create_table_queries:
        cur.execute(query)
        conn.commit()


def main():
    with closing(connect_to_redshift()) as conn:
        with conn.cursor() as cur:
            drop_tables(cur, conn)
            create_tables(cur, conn)
    print("Dropped and recreated Sparkify data warehouse tables.")


if __name__ == "__main__":
    main()

