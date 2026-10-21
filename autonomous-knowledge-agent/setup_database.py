from __future__ import annotations

from database import DB_PATH, initialize_database, get_connection


def main() -> None:
    db_path = initialize_database(DB_PATH, reset=True)
    with get_connection(db_path) as conn:
        counts = {}
        for table in ["Account", "User", "Ticket", "TicketMetadata", "TicketMessage", "Knowledge", "CustomerMemory", "ToolLog"]:
            counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    print(f"Database initialized at {db_path}")
    for table, count in counts.items():
        print(f"{table}: {count}")


if __name__ == "__main__":
    main()
