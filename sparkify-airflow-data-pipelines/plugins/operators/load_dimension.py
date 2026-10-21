from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults

try:
    from airflow.providers.postgres.hooks.postgres import PostgresHook
except ImportError:  # Airflow 1 compatibility
    from airflow.hooks.postgres_hook import PostgresHook


class LoadDimensionOperator(BaseOperator):

    ui_color = '#80BD9E'

    @apply_defaults
    def __init__(self,
                 redshift_conn_id="redshift",
                 table="",
                 sql="",
                 append_only=False,
                 *args, **kwargs):

        super(LoadDimensionOperator, self).__init__(*args, **kwargs)
        self.redshift_conn_id = redshift_conn_id
        self.table = table
        self.sql = sql
        self.append_only = append_only

    def execute(self, context):
        redshift_hook = PostgresHook(postgres_conn_id=self.redshift_conn_id)
        if not self.append_only:
            self.log.info("Truncating dimension table %s before insert", self.table)
            redshift_hook.run(f"TRUNCATE TABLE {self.table};")
        else:
            self.log.info("Appending to dimension table %s", self.table)

        self.log.info("Loading dimension table %s", self.table)
        redshift_hook.run(f"INSERT INTO {self.table} {self.sql};")
        self.log.info("Dimension table %s loaded", self.table)
