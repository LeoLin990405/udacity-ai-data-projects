"""Local validation helper for the STEDI project.

This script mirrors the Glue SQL logic with pandas against the included starter
JSON files. It is not required by Glue, but it documents and verifies that the
joins produce the Udacity rubric row counts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
STARTER = ROOT / "data" / "starter"


def load_json_lines(dataset: str) -> pd.DataFrame:
    rows = []
    for path in sorted((STARTER / dataset / "landing").glob("*.json")):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    rows.append(json.loads(line))
    return pd.DataFrame(rows)


def build_tables() -> dict[str, pd.DataFrame]:
    customer_landing = load_json_lines("customer")
    accelerometer_landing = load_json_lines("accelerometer")
    step_trainer_landing = load_json_lines("step_trainer")

    customer_trusted = customer_landing[customer_landing["shareWithResearchAsOfDate"].notna()].copy()

    accelerometer_trusted = accelerometer_landing.merge(
        customer_trusted[["email"]],
        left_on="user",
        right_on="email",
        how="inner",
    )[accelerometer_landing.columns].drop_duplicates()

    customers_curated = customer_trusted.merge(
        accelerometer_trusted[["user"]].drop_duplicates(),
        left_on="email",
        right_on="user",
        how="inner",
    )[customer_trusted.columns].drop_duplicates()

    step_trainer_trusted = step_trainer_landing.merge(
        customers_curated[["serialNumber"]].drop_duplicates(),
        on="serialNumber",
        how="inner",
    )[step_trainer_landing.columns].drop_duplicates()

    machine_learning_curated = step_trainer_trusted.merge(
        accelerometer_trusted,
        left_on="sensorReadingTime",
        right_on="timestamp",
        how="inner",
    )

    return {
        "customer_landing": customer_landing,
        "accelerometer_landing": accelerometer_landing,
        "step_trainer_landing": step_trainer_landing,
        "customer_trusted": customer_trusted,
        "accelerometer_trusted": accelerometer_trusted,
        "customers_curated": customers_curated,
        "step_trainer_trusted": step_trainer_trusted,
        "machine_learning_curated": machine_learning_curated,
    }


def main() -> None:
    tables = build_tables()
    expected = {
        "customer_landing": 956,
        "accelerometer_landing": 81273,
        "step_trainer_landing": 28680,
        "customer_trusted": 482,
        "accelerometer_trusted": 40981,
        "customers_curated": 482,
        "step_trainer_trusted": 14460,
        "machine_learning_curated": 43681,
    }

    for table_name, expected_count in expected.items():
        actual_count = len(tables[table_name])
        print(f"{table_name}: {actual_count}")
        assert actual_count == expected_count, (table_name, actual_count, expected_count)

    blank_consent = tables["customer_trusted"]["shareWithResearchAsOfDate"].isna().sum()
    print(f"blank consent rows in customer_trusted: {blank_consent}")
    assert blank_consent == 0


if __name__ == "__main__":
    main()

