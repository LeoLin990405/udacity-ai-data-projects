import sys

from awsglue.context import GlueContext
from awsglue.dynamicframe import DynamicFrame
from awsglue.job import Job
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext


args = getResolvedOptions(
    sys.argv,
    ["JOB_NAME", "S3_CURATED_PATH", "DATABASE_NAME"],
)

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args["JOB_NAME"], args)

step_trainer_trusted = glueContext.create_dynamic_frame.from_catalog(
    database=args["DATABASE_NAME"],
    table_name="step_trainer_trusted",
    transformation_ctx="step_trainer_trusted",
)

accelerometer_trusted = glueContext.create_dynamic_frame.from_catalog(
    database=args["DATABASE_NAME"],
    table_name="accelerometer_trusted",
    transformation_ctx="accelerometer_trusted",
)

step_trainer_trusted.createOrReplaceTempView("step_trainer_trusted")
accelerometer_trusted.createOrReplaceTempView("accelerometer_trusted")

machine_learning_curated_df = spark.sql(
    """
    SELECT
        s.sensorReadingTime,
        s.serialNumber,
        s.distanceFromObject,
        a.user,
        a.x,
        a.y,
        a.z
    FROM step_trainer_trusted s
    INNER JOIN accelerometer_trusted a
        ON s.sensorReadingTime = a.timestamp
    """
)

machine_learning_curated = DynamicFrame.fromDF(
    machine_learning_curated_df,
    glueContext,
    "machine_learning_curated",
)

sink = glueContext.getSink(
    path=args["S3_CURATED_PATH"],
    connection_type="s3",
    updateBehavior="UPDATE_IN_DATABASE",
    partitionKeys=[],
    enableUpdateCatalog=True,
    transformation_ctx="machine_learning_curated_sink",
)
sink.setCatalogInfo(catalogDatabase=args["DATABASE_NAME"], catalogTableName="machine_learning_curated")
sink.setFormat("json")
sink.writeFrame(machine_learning_curated)

job.commit()

