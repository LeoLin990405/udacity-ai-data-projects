from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults

try:
    from airflow.providers.postgres.hooks.postgres import PostgresHook
except ImportError:  # Airflow 1 compatibility
    from airflow.hooks.postgres_hook import PostgresHook


class DataQualityOperator(BaseOperator):

    ui_color = '#89DA59'

    @apply_defaults
    def __init__(self,
                 redshift_conn_id="redshift",
                 dq_checks=None,
                 *args, **kwargs):

        super(DataQualityOperator, self).__init__(*args, **kwargs)
        self.redshift_conn_id = redshift_conn_id
        self.dq_checks = dq_checks or []

    def execute(self, context):
        if not self.dq_checks:
            raise ValueError("No data quality checks were provided.")

        redshift_hook = PostgresHook(postgres_conn_id=self.redshift_conn_id)
        self.log.info("Running %s data quality checks", len(self.dq_checks))

        for check in self.dq_checks:
            sql = check.get("check_sql")
            expected_result = check.get("expected_result")
            comparison = check.get("comparison", "equals")
            if sql is None:
                raise ValueError(f"Invalid data quality check without check_sql: {check}")

            records = redshift_hook.get_records(sql)
            if not records or len(records[0]) < 1:
                raise ValueError(f"Data quality check returned no results: {sql}")

            actual_result = records[0][0]
            self.log.info(
                "Data quality check SQL: %s, actual result: %s, expected result: %s",
                sql,
                actual_result,
                expected_result,
            )
            if comparison == "equals":
                passed = actual_result == expected_result
            elif comparison == "greater_than":
                passed = actual_result > expected_result
            elif comparison == "greater_than_or_equal":
                passed = actual_result >= expected_result
            elif comparison == "less_than":
                passed = actual_result < expected_result
            elif comparison == "less_than_or_equal":
                passed = actual_result <= expected_result
            else:
                raise ValueError(f"Unsupported data quality comparison: {comparison}")

            if not passed:
                raise ValueError(
                    f"Data quality check failed. SQL: {sql}. "
                    f"Comparison {comparison}, expected {expected_result}, got {actual_result}."
                )

        self.log.info("All data quality checks passed")
