from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults

try:
    from airflow.providers.amazon.aws.hooks.base_aws import AwsBaseHook
    from airflow.providers.postgres.hooks.postgres import PostgresHook
except ImportError:  # Airflow 1 compatibility used by older Udacity workspaces
    from airflow.contrib.hooks.aws_hook import AwsHook
    from airflow.hooks.postgres_hook import PostgresHook

    class AwsBaseHook(AwsHook):
        def __init__(self, aws_conn_id=None, client_type=None, *args, **kwargs):
            super().__init__(aws_conn_id=aws_conn_id, *args, **kwargs)


class StageToRedshiftOperator(BaseOperator):
    ui_color = '#358140'
    template_fields = ("s3_key",)

    copy_sql = """
        COPY {table}
        FROM '{s3_path}'
        ACCESS_KEY_ID '{access_key}'
        SECRET_ACCESS_KEY '{secret_key}'
        REGION '{region}'
        {session_token_clause}
        FORMAT AS JSON {json_path}
        TRUNCATECOLUMNS
        BLANKSASNULL
        EMPTYASNULL;
    """

    @apply_defaults
    def __init__(self,
                 redshift_conn_id="redshift",
                 aws_credentials_id="aws_credentials",
                 table="",
                 s3_bucket="",
                 s3_key="",
                 region="us-west-2",
                 json_path="'auto'",
                 *args, **kwargs):

        super(StageToRedshiftOperator, self).__init__(*args, **kwargs)
        self.redshift_conn_id = redshift_conn_id
        self.aws_credentials_id = aws_credentials_id
        self.table = table
        self.s3_bucket = s3_bucket
        self.s3_key = s3_key
        self.region = region
        self.json_path = json_path

    def execute(self, context):
        self.log.info("Staging S3 data from s3://%s/%s into %s", self.s3_bucket, self.s3_key, self.table)
        aws_hook = AwsBaseHook(aws_conn_id=self.aws_credentials_id, client_type="s3")
        credentials = aws_hook.get_credentials()
        redshift_hook = PostgresHook(postgres_conn_id=self.redshift_conn_id)

        s3_path = f"s3://{self.s3_bucket}/{self.s3_key}"
        session_token = getattr(credentials, "token", None)
        session_token_clause = f"SESSION_TOKEN '{session_token}'" if session_token else ""
        formatted_sql = self.copy_sql.format(
            table=self.table,
            s3_path=s3_path,
            access_key=credentials.access_key,
            secret_key=credentials.secret_key,
            region=self.region,
            json_path=self.json_path,
            session_token_clause=session_token_clause,
        )

        self.log.info("Clearing destination table %s", self.table)
        redshift_hook.run(f"TRUNCATE TABLE {self.table};")
        self.log.info("Running Redshift COPY command for %s", self.table)
        redshift_hook.run(formatted_sql)
        self.log.info("Successfully staged %s", self.table)
