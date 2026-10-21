"""SQL statements for the Sparkify Redshift data warehouse project.

The pipeline loads raw JSON files from S3 into staging tables, then transforms
those rows into a star schema optimized for song-play analytics.
"""

import configparser


config = configparser.ConfigParser()
config.read("dwh.cfg")

LOG_DATA = config.get("S3", "LOG_DATA")
LOG_JSONPATH = config.get("S3", "LOG_JSONPATH")
SONG_DATA = config.get("S3", "SONG_DATA")
REGION = config.get("S3", "REGION", fallback="us-west-2")
IAM_ROLE = config.get("IAM_ROLE", "ARN")


# DROP TABLES

staging_events_table_drop = "DROP TABLE IF EXISTS staging_events;"
staging_songs_table_drop = "DROP TABLE IF EXISTS staging_songs;"
songplay_table_drop = "DROP TABLE IF EXISTS songplays;"
user_table_drop = "DROP TABLE IF EXISTS users;"
song_table_drop = "DROP TABLE IF EXISTS songs;"
artist_table_drop = "DROP TABLE IF EXISTS artists;"
time_table_drop = "DROP TABLE IF EXISTS time;"


# CREATE STAGING TABLES

staging_events_table_create = """
CREATE TABLE IF NOT EXISTS staging_events (
    artist          VARCHAR(500),
    auth            VARCHAR(100),
    firstName       VARCHAR(200),
    gender          VARCHAR(10),
    itemInSession   INTEGER,
    lastName        VARCHAR(200),
    length          DOUBLE PRECISION,
    level           VARCHAR(20),
    location        VARCHAR(500),
    method          VARCHAR(20),
    page            VARCHAR(100),
    registration    BIGINT,
    sessionId       INTEGER,
    song            VARCHAR(500),
    status          INTEGER,
    ts              BIGINT,
    userAgent       VARCHAR(1000),
    userId          VARCHAR(50)
);
"""

staging_songs_table_create = """
CREATE TABLE IF NOT EXISTS staging_songs (
    num_songs        INTEGER,
    artist_id        VARCHAR(50),
    artist_latitude  DOUBLE PRECISION,
    artist_longitude DOUBLE PRECISION,
    artist_location  VARCHAR(500),
    artist_name      VARCHAR(500),
    song_id          VARCHAR(50),
    title            VARCHAR(500),
    duration         DOUBLE PRECISION,
    year             INTEGER
);
"""


# CREATE ANALYTICS TABLES

songplay_table_create = """
CREATE TABLE IF NOT EXISTS songplays (
    songplay_id BIGINT IDENTITY(0, 1) NOT NULL,
    start_time  TIMESTAMP NOT NULL,
    user_id     INTEGER NOT NULL,
    level       VARCHAR(20),
    song_id     VARCHAR(50),
    artist_id   VARCHAR(50),
    session_id  INTEGER,
    location    VARCHAR(500),
    user_agent  VARCHAR(1000),
    PRIMARY KEY (songplay_id)
)
DISTSTYLE KEY
DISTKEY (user_id)
SORTKEY (start_time);
"""

user_table_create = """
CREATE TABLE IF NOT EXISTS users (
    user_id    INTEGER NOT NULL,
    first_name VARCHAR(200),
    last_name  VARCHAR(200),
    gender     VARCHAR(10),
    level      VARCHAR(20),
    PRIMARY KEY (user_id)
)
DISTSTYLE KEY
DISTKEY (user_id)
SORTKEY (user_id);
"""

song_table_create = """
CREATE TABLE IF NOT EXISTS songs (
    song_id   VARCHAR(50) NOT NULL,
    title     VARCHAR(500) NOT NULL,
    artist_id VARCHAR(50),
    year      INTEGER,
    duration  DOUBLE PRECISION,
    PRIMARY KEY (song_id)
)
DISTSTYLE ALL
SORTKEY (song_id);
"""

artist_table_create = """
CREATE TABLE IF NOT EXISTS artists (
    artist_id VARCHAR(50) NOT NULL,
    name      VARCHAR(500) NOT NULL,
    location  VARCHAR(500),
    latitude  DOUBLE PRECISION,
    longitude DOUBLE PRECISION,
    PRIMARY KEY (artist_id)
)
DISTSTYLE ALL
SORTKEY (artist_id);
"""

time_table_create = """
CREATE TABLE IF NOT EXISTS time (
    start_time TIMESTAMP NOT NULL,
    hour       INTEGER,
    day        INTEGER,
    week       INTEGER,
    month      INTEGER,
    year       INTEGER,
    weekday    INTEGER,
    PRIMARY KEY (start_time)
)
DISTSTYLE ALL
SORTKEY (start_time);
"""


# STAGING TABLE COPY COMMANDS

staging_events_copy = f"""
COPY staging_events
FROM '{LOG_DATA}'
IAM_ROLE '{IAM_ROLE}'
REGION '{REGION}'
FORMAT AS JSON '{LOG_JSONPATH}'
TIMEFORMAT AS 'epochmillisecs'
TRUNCATECOLUMNS
BLANKSASNULL
EMPTYASNULL;
"""

staging_songs_copy = f"""
COPY staging_songs
FROM '{SONG_DATA}'
IAM_ROLE '{IAM_ROLE}'
REGION '{REGION}'
FORMAT AS JSON 'auto'
TRUNCATECOLUMNS
BLANKSASNULL
EMPTYASNULL;
"""


# FINAL TABLE INSERTS

user_table_insert = """
INSERT INTO users (user_id, first_name, last_name, gender, level)
SELECT user_id, first_name, last_name, gender, level
FROM (
    SELECT
        NULLIF(userId, '')::INTEGER AS user_id,
        firstName AS first_name,
        lastName AS last_name,
        gender,
        level,
        ROW_NUMBER() OVER (
            PARTITION BY NULLIF(userId, '')::INTEGER
            ORDER BY ts DESC
        ) AS row_num
    FROM staging_events
    WHERE page = 'NextSong'
      AND userId IS NOT NULL
      AND userId <> ''
) latest_users
WHERE row_num = 1;
"""

song_table_insert = """
INSERT INTO songs (song_id, title, artist_id, year, duration)
SELECT DISTINCT
    song_id,
    title,
    artist_id,
    year,
    duration
FROM staging_songs
WHERE song_id IS NOT NULL;
"""

artist_table_insert = """
INSERT INTO artists (artist_id, name, location, latitude, longitude)
SELECT artist_id, name, location, latitude, longitude
FROM (
    SELECT
        artist_id,
        artist_name AS name,
        artist_location AS location,
        artist_latitude AS latitude,
        artist_longitude AS longitude,
        ROW_NUMBER() OVER (
            PARTITION BY artist_id
            ORDER BY artist_name
        ) AS row_num
    FROM staging_songs
    WHERE artist_id IS NOT NULL
) artists_deduped
WHERE row_num = 1;
"""

time_table_insert = """
INSERT INTO time (start_time, hour, day, week, month, year, weekday)
SELECT DISTINCT
    start_time,
    EXTRACT(hour FROM start_time) AS hour,
    EXTRACT(day FROM start_time) AS day,
    EXTRACT(week FROM start_time) AS week,
    EXTRACT(month FROM start_time) AS month,
    EXTRACT(year FROM start_time) AS year,
    EXTRACT(dayofweek FROM start_time) AS weekday
FROM (
    SELECT TIMESTAMP 'epoch' + (ts / 1000) * INTERVAL '1 second' AS start_time
    FROM staging_events
    WHERE page = 'NextSong'
      AND ts IS NOT NULL
) event_times;
"""

songplay_table_insert = """
INSERT INTO songplays (
    start_time,
    user_id,
    level,
    song_id,
    artist_id,
    session_id,
    location,
    user_agent
)
SELECT
    TIMESTAMP 'epoch' + (e.ts / 1000) * INTERVAL '1 second' AS start_time,
    NULLIF(e.userId, '')::INTEGER AS user_id,
    e.level,
    s.song_id,
    s.artist_id,
    e.sessionId AS session_id,
    e.location,
    e.userAgent AS user_agent
FROM staging_events e
LEFT JOIN staging_songs s
  ON e.song = s.title
 AND e.artist = s.artist_name
 AND ABS(e.length - s.duration) < 0.01
WHERE e.page = 'NextSong'
  AND e.userId IS NOT NULL
  AND e.userId <> ''
  AND e.ts IS NOT NULL;
"""


# QUERY LISTS

create_table_queries = [
    staging_events_table_create,
    staging_songs_table_create,
    user_table_create,
    song_table_create,
    artist_table_create,
    time_table_create,
    songplay_table_create,
]

drop_table_queries = [
    songplay_table_drop,
    user_table_drop,
    song_table_drop,
    artist_table_drop,
    time_table_drop,
    staging_events_table_drop,
    staging_songs_table_drop,
]

copy_table_queries = [staging_events_copy, staging_songs_copy]

insert_table_queries = [
    user_table_insert,
    song_table_insert,
    artist_table_insert,
    time_table_insert,
    songplay_table_insert,
]
