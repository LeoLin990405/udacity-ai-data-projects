from datetime import datetime, timedelta
import pendulum
import os
from airflow.decorators import dag
from airflow.models import Variable
from airflow.operators.dummy import DummyOperator
from operators import (StageToRedshiftOperator, LoadFactOperator,
                       LoadDimensionOperator, DataQualityOperator)
from helpers import SqlQueries

default_args = {
    'owner': 'udacity',
    'depends_on_past': False,
    'start_date': pendulum.datetime(2018, 11, 1, tz="UTC"),
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'email_on_retry': False,
    'catchup': False,
}


S3_BUCKET = Variable.get("s3_bucket", default_var="udacity-dend")
S3_PREFIX = Variable.get("s3_prefix", default_var="")
S3_LOG_KEY = Variable.get("s3_log_key", default_var="log_data")
S3_SONG_KEY = Variable.get("s3_song_key", default_var="song-data")
S3_LOG_JSONPATH = Variable.get("s3_log_jsonpath", default_var="log_json_path.json")


def prefixed_key(key):
    prefix = S3_PREFIX.strip("/")
    return f"{prefix}/{key.strip('/')}" if prefix else key.strip("/")


@dag(
    default_args=default_args,
    description='Load and transform data in Redshift with Airflow',
    schedule_interval='0 * * * *',
    catchup=False,
)
def final_project():

    start_operator = DummyOperator(task_id='Begin_execution')

    stage_events_to_redshift = StageToRedshiftOperator(
        task_id='Stage_events',
        redshift_conn_id='redshift',
        aws_credentials_id='aws_credentials',
        table='staging_events',
        s3_bucket=S3_BUCKET,
        s3_key=prefixed_key(S3_LOG_KEY),
        json_path=f"'s3://{S3_BUCKET}/{prefixed_key(S3_LOG_JSONPATH)}'",
        region='us-west-2',
    )

    stage_songs_to_redshift = StageToRedshiftOperator(
        task_id='Stage_songs',
        redshift_conn_id='redshift',
        aws_credentials_id='aws_credentials',
        table='staging_songs',
        s3_bucket=S3_BUCKET,
        s3_key=prefixed_key(S3_SONG_KEY),
        json_path="'auto'",
        region='us-west-2',
    )

    load_songplays_table = LoadFactOperator(
        task_id='Load_songplays_fact_table',
        redshift_conn_id='redshift',
        table='songplays',
        sql=SqlQueries.songplay_table_insert,
    )

    load_user_dimension_table = LoadDimensionOperator(
        task_id='Load_user_dim_table',
        redshift_conn_id='redshift',
        table='users',
        sql=SqlQueries.user_table_insert,
        append_only=False,
    )

    load_song_dimension_table = LoadDimensionOperator(
        task_id='Load_song_dim_table',
        redshift_conn_id='redshift',
        table='songs',
        sql=SqlQueries.song_table_insert,
        append_only=False,
    )

    load_artist_dimension_table = LoadDimensionOperator(
        task_id='Load_artist_dim_table',
        redshift_conn_id='redshift',
        table='artists',
        sql=SqlQueries.artist_table_insert,
        append_only=False,
    )

    load_time_dimension_table = LoadDimensionOperator(
        task_id='Load_time_dim_table',
        redshift_conn_id='redshift',
        table='"time"',
        sql=SqlQueries.time_table_insert,
        append_only=False,
    )

    run_quality_checks = DataQualityOperator(
        task_id='Run_data_quality_checks',
        redshift_conn_id='redshift',
        dq_checks=[
            {'check_sql': "SELECT COUNT(*) FROM songplays", 'expected_result': 0, 'comparison': 'greater_than'},
            {'check_sql': "SELECT COUNT(*) FROM users WHERE userid IS NULL", 'expected_result': 0},
            {'check_sql': "SELECT COUNT(*) FROM songs WHERE songid IS NULL", 'expected_result': 0},
            {'check_sql': "SELECT COUNT(*) FROM artists WHERE artistid IS NULL", 'expected_result': 0},
            {'check_sql': 'SELECT COUNT(*) FROM "time" WHERE start_time IS NULL', 'expected_result': 0},
        ],
    )

    end_operator = DummyOperator(task_id='Stop_execution')

    start_operator >> [stage_events_to_redshift, stage_songs_to_redshift]
    [stage_events_to_redshift, stage_songs_to_redshift] >> load_songplays_table
    load_songplays_table >> [
        load_user_dimension_table,
        load_song_dimension_table,
        load_artist_dimension_table,
        load_time_dimension_table,
    ]
    [
        load_user_dimension_table,
        load_song_dimension_table,
        load_artist_dimension_table,
        load_time_dimension_table,
    ] >> run_quality_checks
    run_quality_checks >> end_operator

final_project_dag = final_project()
