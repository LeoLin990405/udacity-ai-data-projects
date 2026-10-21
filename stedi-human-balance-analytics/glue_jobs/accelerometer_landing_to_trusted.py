import sys

from awsglue.context import GlueContext
from awsglue.dynamicframe import DynamicFrame
from awsglue.job import Job
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext


args = getResolvedOptions(
    sys.argv,
    ["JOB_NAME", "S3_TRUSTED_PATH", "DATABASE_NAME"],
)

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

accelerometer_landing = glueContext.create_dynamic_frame.from_catalog(
    database=args["DATABASE_NAME"],
    table_name="accelerometer_landing",
    transformation_ctx="accelerometer_landing",
)

customer_trusted = glueContext.create_dynamic_frame.from_catalog(
    database=args["DATABASE_NAME"],
    table_name="customer_trusted",
    transformation_ctx="customer_trusted",
)

accelerometer_landing.createOrReplaceTempView("accelerometer_landing")
customer_trusted.createOrReplaceTempView("customer_trusted")

accelerometer_trusted_df = spark.sql(
    """
    SELECT DISTINCT
        a.user,
        a.timestamp,
        a.x,
        a.y,
        a.z
    FROM accelerometer_landing a
    INNER JOIN customer_trusted c
        ON a.user = c.email
    """
)

accelerometer_trusted = DynamicFrame.fromDF(accelerometer_trusted_df, glueContext, "accelerometer_trusted")

sink = glueContext.getSink(
    path=args["S3_TRUSTED_PATH"],
    connection_type="s3",
    updateBehavior="UPDATE_IN_DATABASE",
    partitionKeys=[],
    enableUpdateCatalog=True,
    transformation_ctx="accelerometer_trusted_sink",
)
sink.setCatalogInfo(catalogDatabase=args["DATABASE_NAME"], catalogTableName="accelerometer_trusted")
sink.setFormat("json")
sink.writeFrame(accelerometer_trusted)

job.commit()

