from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults

try:
    from airflow.providers.postgres.hooks.postgres import PostgresHook
except ImportError:  # Airflow 1 compatibility
    from airflow.hooks.postgres_hook import PostgresHook


class LoadFactOperator(BaseOperator):

    ui_color = '#F98866'

    @apply_defaults
    def __init__(self,
                 redshift_conn_id="redshift",
                 table="",
                 sql="",
                 *args, **kwargs):

        super(LoadFactOperator, self).__init__(*args, **kwargs)
        self.redshift_conn_id = redshift_conn_id
        self.table = table
        self.sql = sql

    def execute(self, context):
        self.log.info("Loading fact table %s with append-only insert", self.table)
        redshift_hook = PostgresHook(postgres_conn_id=self.redshift_conn_id)
        insert_sql = f"INSERT INTO {self.table} {self.sql};"
        redshift_hook.run(insert_sql)
        self.log.info("Fact table %s loaded", self.table)
