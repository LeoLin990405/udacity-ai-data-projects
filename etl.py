"""Run the Sparkify S3-to-Redshift ETL pipeline."""

import configparser
from contextlib import closing

import psycopg2

from sql_queries import copy_table_queries, insert_table_queries


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


def load_staging_tables(cur, conn):
    """COPY raw log and song JSON files from S3 into Redshift staging tables."""

    for query in copy_table_queries:
        cur.execute(query)
        conn.commit()


def insert_tables(cur, conn):
    """Transform staging data into star-schema analytics tables."""

    for query in insert_table_queries:
        cur.execute(query)
        conn.commit()


def main():
    with closing(connect_to_redshift()) as conn:
        with conn.cursor() as cur:
            load_staging_tables(cur, conn)
            insert_tables(cur, conn)
    print("Loaded staging tables and inserted analytics tables.")


if __name__ == "__main__":
    main()

