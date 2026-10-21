# Sparkify Data Warehouse

This repository completes Udacity's **Project: Data Warehouse** for the Sparkify music streaming scenario. The pipeline stages JSON data from Amazon S3 into Amazon Redshift and transforms it into a star schema for song-play analytics.

## Project Goal

Sparkify's analytics team needs to answer questions about what songs users are listening to. The source data lives in S3:

- Song metadata: `s3://udacity-dend/song_data`
- Event logs: `s3://udacity-dend/log_data`
- Log JSON path: `s3://udacity-dend/log_json_path.json`

The ETL flow is:

```text
S3 JSON files -> Redshift staging tables -> Sparkify star schema
```

## Schema Design

### Staging Tables

- `staging_events`: raw app event log records.
- `staging_songs`: raw song metadata records.

### Fact Table

- `songplays`: records in event data associated with song plays.
  - Columns: `songplay_id`, `start_time`, `user_id`, `level`, `song_id`, `artist_id`, `session_id`, `location`, `user_agent`.

### Dimension Tables

- `users`: app users.
  - Columns: `user_id`, `first_name`, `last_name`, `gender`, `level`.
- `songs`: songs in the music database.
  - Columns: `song_id`, `title`, `artist_id`, `year`, `duration`.
- `artists`: artists in the music database.
  - Columns: `artist_id`, `name`, `location`, `latitude`, `longitude`.
- `time`: timestamps of songplays broken into analytical units.
  - Columns: `start_time`, `hour`, `day`, `week`, `month`, `year`, `weekday`.

The fact table is sorted by `start_time` and distributed by `user_id`, which supports common time-series and user-behavior queries. Songs and artists include keys that support joins from the fact table.

## Files

- `dwh.cfg`: Redshift, IAM role, and S3 configuration template.
- `sql_queries.py`: all drop, create, copy, and insert SQL statements.
- `create_tables.py`: connects to Redshift, drops existing tables, and recreates staging/final tables.
- `etl.py`: loads S3 data into staging tables, then inserts records into the analytics schema.
- `requirements.txt`: Python dependencies.

## Setup

1. Create a Redshift cluster and attach an IAM role with read access to the Udacity S3 bucket.
2. Fill in `dwh.cfg`:

```ini
[CLUSTER]
HOST=your-redshift-endpoint
DB_NAME=dev
DB_USER=awsuser
DB_PASSWORD=your-password
DB_PORT=5439

[IAM_ROLE]
ARN=arn:aws:iam::...:role/your-redshift-s3-role
```

3. Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

4. Recreate tables:

```bash
python create_tables.py
```

5. Run the ETL pipeline:

```bash
python etl.py
```

## Data Quality Checks

After running the pipeline, useful validation queries include:

```sql
SELECT COUNT(*) FROM staging_events;
SELECT COUNT(*) FROM staging_songs;
SELECT COUNT(*) FROM songplays;
SELECT COUNT(*) FROM users;
SELECT COUNT(*) FROM songs;
SELECT COUNT(*) FROM artists;
SELECT COUNT(*) FROM time;
```

Example analytical query:

```sql
SELECT
    u.level,
    t.hour,
    COUNT(*) AS plays
FROM songplays sp
JOIN users u ON sp.user_id = u.user_id
JOIN time t ON sp.start_time = t.start_time
GROUP BY u.level, t.hour
ORDER BY plays DESC
LIMIT 10;
```

## Notes

- `dwh.cfg` intentionally contains placeholders only; no credentials are committed.
- `staging_events.userId` is loaded as text because log records may contain blank user IDs. It is safely cast during final inserts.
- User rows are deduplicated with the latest observed level by timestamp.
- Songplay rows use a left join to keep valid `NextSong` events even when a song metadata match is unavailable.

