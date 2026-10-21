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

step_trainer_landing = glueContext.create_dynamic_frame.from_catalog(
    database=args["DATABASE_NAME"],
    table_name="step_trainer_landing",
    transformation_ctx="step_trainer_landing",
)

customers_curated = glueContext.create_dynamic_frame.from_catalog(
    database=args["DATABASE_NAME"],
    table_name="customers_curated",
    transformation_ctx="customers_curated",
)

step_trainer_landing.createOrReplaceTempView("step_trainer_landing")
customers_curated.createOrReplaceTempView("customers_curated")

step_trainer_trusted_df = spark.sql(
    """
    SELECT DISTINCT
        s.sensorReadingTime,
        s.serialNumber,
        s.distanceFromObject
    FROM step_trainer_landing s
    INNER JOIN customers_curated c
        ON s.serialNumber = c.serialNumber
    """
)

step_trainer_trusted = DynamicFrame.fromDF(step_trainer_trusted_df, glueContext, "step_trainer_trusted")

sink = glueContext.getSink(
    path=args["S3_TRUSTED_PATH"],
    connection_type="s3",
    updateBehavior="UPDATE_IN_DATABASE",
    partitionKeys=[],
    enableUpdateCatalog=True,
    transformation_ctx="step_trainer_trusted_sink",
)
sink.setCatalogInfo(catalogDatabase=args["DATABASE_NAME"], catalogTableName="step_trainer_trusted")
sink.setFormat("json")
sink.writeFrame(step_trainer_trusted)

job.commit()

