from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path


DB_PATH = Path(__file__).resolve().parents[1] / "data" / "energy_data.db"


@dataclass
class EnergyUsage:
    timestamp: str
    device_type: str
    consumption_kwh: float
    cost_usd: float
    user_id: str = "demo-user"


@dataclass
class SolarGeneration:
    timestamp: str
    generation_kwh: float
    weather_condition: str
    user_id: str = "demo-user"


def get_connection(db_path: str | Path = DB_PATH) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database(db_path: str | Path = DB_PATH) -> None:
    conn = get_connection(db_path)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS EnergyUsage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            device_type TEXT NOT NULL,
            consumption_kwh REAL NOT NULL,
            cost_usd REAL NOT NULL
        );

        CREATE TABLE IF NOT EXISTS SolarGeneration (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            generation_kwh REAL NOT NULL,
            weather_condition TEXT NOT NULL
        );
        """
    )
    conn.commit()
    conn.close()


def populate_sample_data(db_path: str | Path = DB_PATH) -> None:
    initialize_database(db_path)
    conn = get_connection(db_path)
    existing = conn.execute("SELECT COUNT(*) AS count FROM EnergyUsage").fetchone()["count"]
    if existing:
        conn.close()
        return

    start = datetime(2026, 5, 11, 0, 0)
    devices = {
        "ev_charger": (7.2, [0, 1, 2, 11, 12, 13, 22, 23]),
        "hvac": (2.4, list(range(13, 20))),
        "dishwasher": (1.3, [21]),
        "pool_pump": (1.1, [10, 11, 12, 13, 14]),
        "laundry": (1.8, [9, 18]),
    }
    for day in range(7):
        for hour in range(24):
            ts = (start + timedelta(days=day, hours=hour)).isoformat()
            price = 0.14 if hour < 7 else 0.22 if hour < 16 else 0.36 if hour < 21 else 0.18
            for device, (base_kwh, active_hours) in devices.items():
                if hour in active_hours:
                    kwh = base_kwh * (1 + 0.05 * ((day + hour) % 3))
                    conn.execute(
                        "INSERT INTO EnergyUsage (user_id, timestamp, device_type, consumption_kwh, cost_usd) VALUES (?, ?, ?, ?, ?)",
                        ("demo-user", ts, device, round(kwh, 3), round(kwh * price, 3)),
                    )
            daylight = max(0, 1 - abs(hour - 12) / 6)
            weather = ["sunny", "partly_cloudy", "cloudy", "sunny", "sunny", "rainy", "partly_cloudy"][day]
            weather_factor = {"sunny": 1.0, "partly_cloudy": 0.72, "cloudy": 0.45, "rainy": 0.28}[weather]
            solar = round(5.8 * daylight * weather_factor, 3)
            conn.execute(
                "INSERT INTO SolarGeneration (user_id, timestamp, generation_kwh, weather_condition) VALUES (?, ?, ?, ?)",
                ("demo-user", ts, solar, weather),
            )
    conn.commit()
    conn.close()


def query_rows(sql: str, params: tuple = (), db_path: str | Path = DB_PATH) -> list[dict]:
    conn = get_connection(db_path)
    rows = [dict(row) for row in conn.execute(sql, params).fetchall()]
    conn.close()
    return rows
