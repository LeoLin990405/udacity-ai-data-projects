# Data Pipelines with Airflow

Completed Udacity **Data Pipelines** project for the Sparkify music analytics warehouse.

The project implements an Apache Airflow DAG that stages JSON data from S3 into Amazon Redshift, loads fact and dimension tables, and runs parameterized data quality checks.

## What Is Implemented

- `dags/final_project.py`
  - Defines `default_args` with owner, no dependency on past runs, start date, three retries, five minute retry delay, no retry email, and no catchup.
  - Schedules the DAG hourly with `schedule_interval='0 * * * *'`.
  - Uses Airflow Variables for S3 location configuration.
  - Wires all task dependencies according to the required graph.

- `plugins/operators/stage_redshift.py`
  - Implements `StageToRedshiftOperator`.
  - Uses `AwsBaseHook` and `PostgresHook`/`Redshift` connection via Airflow connections.
  - Builds a dynamic Redshift `COPY` statement from operator parameters.
  - Supports templated `s3_key` for backfills and timestamped S3 paths.

- `plugins/operators/load_fact.py`
  - Implements append-only fact table loading.

- `plugins/operators/load_dimension.py`
  - Implements dimension loading with configurable `append_only` behavior.
  - Uses truncate-insert by default.

- `plugins/operators/data_quality.py`
  - Implements parameterized SQL data quality checks.
  - Raises errors when checks fail so Airflow retries/fails the task.
  - Supports equality and comparison checks.

- `plugins/helpers/sql_queries.py`
  - Contains Sparkify star-schema transformation SQL for songplays, users, songs, artists, and time.

## Required DAG Flow

```text
Begin_execution
  -> [Stage_events, Stage_songs]
  -> Load_songplays_fact_table
  -> [Load_artist_dim_table, Load_song_dim_table, Load_time_dim_table, Load_user_dim_table]
  -> Run_data_quality_checks
  -> Stop_execution
```

## Airflow Connections

Configure these Airflow connections before running the DAG:

- `aws_credentials`: AWS access credentials that can read the S3 bucket.
- `redshift`: Redshift/Postgres connection for the warehouse.

The downloaded Udacity helper script `set_connections_and_variables.sh` explains how to add these in Airflow. Do not commit real credentials.

## Airflow Variables

The DAG reads the following variables. Defaults are provided for Udacity source data, but a copied private bucket is recommended.

| Variable | Default | Purpose |
|---|---|---|
| `s3_bucket` | `udacity-dend` | Bucket containing log/song data. |
| `s3_prefix` | empty | Optional prefix such as `data-pipelines`. |
| `s3_log_key` | `log_data` | Log data key under the bucket/prefix. |
| `s3_song_key` | `song-data` | Song data key under the bucket/prefix. |
| `s3_log_jsonpath` | `log_json_path.json` | JSONPaths file for event logs. |

For an own bucket copied as `s3://my-bucket/data-pipelines/...`, useful settings are:

```bash
airflow variables set s3_bucket my-bucket
airflow variables set s3_prefix data-pipelines
airflow variables set s3_log_key log-data
airflow variables set s3_song_key song-data
airflow variables set s3_log_jsonpath log_json_path.json
```

## Redshift Tables

Run `create_tables.sql` in Redshift before running the DAG. It creates:

- staging tables: `staging_events`, `staging_songs`
- fact table: `songplays`
- dimension tables: `users`, `songs`, `artists`, `time`

## Static Validation

The project files were syntax-checked with Python AST parsing:

```bash
python3 - <<'PY'
import ast
from pathlib import Path
for path in [*Path('dags').glob('*.py'), *Path('plugins').rglob('*.py')]:
    ast.parse(path.read_text())
    print('parsed', path)
PY
```

## Notes

- Real AWS credentials are intentionally not included.
- The four custom operators use parameters rather than hard-coded table/S3/check values.
- The data quality task checks that `songplays` has rows and that primary identifying fields in dimension tables are not null.
