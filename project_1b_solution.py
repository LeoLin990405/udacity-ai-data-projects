"""Sparkify Cassandra data modeling project.

Run this script from the project root after starting a local Cassandra instance.
It mirrors the completed notebook workflow:

1. Consolidate event CSV files into event_datafile_new.csv.
2. Create a Cassandra keyspace.
3. Create one table for each required analytics query.
4. Insert data into the query-specific tables.
5. Run verification SELECT statements.
6. Drop tables and close the session.
"""

from __future__ import annotations

import csv
import glob
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVENT_DATA_DIR = ROOT / "event_data"
OUTPUT_FILE = ROOT / "event_datafile_new.csv"
KEYSPACE = "sparkify"


def preprocess_event_data() -> int:
    """Create event_datafile_new.csv from the raw event_data CSV files."""

    file_path_list: list[str] = []
    for root, _dirs, _files in os.walk(EVENT_DATA_DIR):
        file_path_list.extend(glob.glob(os.path.join(root, "*.csv")))

    full_data_rows: list[list[str]] = []
    for file_path in sorted(file_path_list):
        with open(file_path, "r", encoding="utf8", newline="") as csvfile:
            csvreader = csv.reader(csvfile)
            next(csvreader)
            for line in csvreader:
                full_data_rows.append(line)

    csv.register_dialect("myDialect", quoting=csv.QUOTE_ALL, skipinitialspace=True)
    with open(OUTPUT_FILE, "w", encoding="utf8", newline="") as output:
        writer = csv.writer(output, dialect="myDialect")
        writer.writerow(
            [
                "artist",
                "firstName",
                "gender",
                "itemInSession",
                "lastName",
                "length",
                "level",
                "location",
                "sessionId",
                "song",
                "userId",
            ]
        )
        for row in full_data_rows:
            if row[0] == "":
                continue
            writer.writerow((row[0], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[12], row[13], row[16]))

    with open(OUTPUT_FILE, "r", encoding="utf8") as output:
        return sum(1 for _line in output)


def connect():
    """Connect to local Cassandra and select the Sparkify keyspace."""

    from cassandra.cluster import Cluster

    cluster = Cluster(["127.0.0.1"])
    session = cluster.connect()
    session.execute(
        f"""
        CREATE KEYSPACE IF NOT EXISTS {KEYSPACE}
        WITH REPLICATION = {{
            'class': 'SimpleStrategy',
            'replication_factor': 1
        }}
        """
    )
    session.set_keyspace(KEYSPACE)
    return cluster, session


def create_tables(session) -> None:
    """Create query-specific Cassandra tables."""

    session.execute(
        """
        CREATE TABLE IF NOT EXISTS songplays_by_session_item (
            session_id int,
            item_in_session int,
            artist text,
            song text,
            length double,
            PRIMARY KEY (session_id, item_in_session)
        )
        """
    )
    session.execute(
        """
        CREATE TABLE IF NOT EXISTS songplays_by_user_session (
            user_id int,
            session_id int,
            item_in_session int,
            artist text,
            song text,
            first_name text,
            last_name text,
            PRIMARY KEY ((user_id, session_id), item_in_session)
        ) WITH CLUSTERING ORDER BY (item_in_session ASC)
        """
    )
    session.execute(
        """
        CREATE TABLE IF NOT EXISTS users_by_song (
            song text,
            user_id int,
            first_name text,
            last_name text,
            PRIMARY KEY (song, user_id)
        )
        """
    )


def insert_rows(session) -> None:
    """Insert rows into all three denormalized tables."""

    query_1 = """
        INSERT INTO songplays_by_session_item
        (session_id, item_in_session, artist, song, length)
        VALUES (%s, %s, %s, %s, %s)
    """
    query_2 = """
        INSERT INTO songplays_by_user_session
        (user_id, session_id, item_in_session, artist, song, first_name, last_name)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """
    query_3 = """
        INSERT INTO users_by_song
        (song, user_id, first_name, last_name)
        VALUES (%s, %s, %s, %s)
    """

    with open(OUTPUT_FILE, encoding="utf8") as f:
        csvreader = csv.reader(f)
        next(csvreader)
        for line in csvreader:
            artist, first_name, _gender, item, last_name, length, _level, _location, session_id, song, user_id = line
            session.execute(query_1, (int(session_id), int(item), artist, song, float(length)))
            session.execute(query_2, (int(user_id), int(session_id), int(item), artist, song, first_name, last_name))
            session.execute(query_3, (song, int(user_id), first_name, last_name))


def run_verification_queries(session) -> None:
    """Print the three required project query results."""

    print("\nQuery 1")
    rows = session.execute(
        """
        SELECT artist, song, length
        FROM songplays_by_session_item
        WHERE session_id = 338 AND item_in_session = 4
        """
    )
    for row in rows:
        print(row.artist, row.song, row.length)

    print("\nQuery 2")
    rows = session.execute(
        """
        SELECT artist, song, first_name, last_name
        FROM songplays_by_user_session
        WHERE user_id = 10 AND session_id = 182
        """
    )
    for row in rows:
        print(row.artist, row.song, row.first_name, row.last_name)

    print("\nQuery 3")
    rows = session.execute(
        """
        SELECT first_name, last_name
        FROM users_by_song
        WHERE song = 'All Hands Against His Own'
        """
    )
    for row in rows:
        print(row.first_name, row.last_name)


def drop_tables(session) -> None:
    """Drop tables before shutting down the connection."""

    session.execute("DROP TABLE IF EXISTS songplays_by_session_item")
    session.execute("DROP TABLE IF EXISTS songplays_by_user_session")
    session.execute("DROP TABLE IF EXISTS users_by_song")


def main() -> None:
    row_count = preprocess_event_data()
    print(f"Created {OUTPUT_FILE.name} with {row_count} rows including header.")

    cluster, session = connect()
    try:
        create_tables(session)
        insert_rows(session)
        run_verification_queries(session)
        drop_tables(session)
    finally:
        session.shutdown()
        cluster.shutdown()


if __name__ == "__main__":
    main()
