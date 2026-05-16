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

customer_trusted = glueContext.create_dynamic_frame.from_catalog(
    database=args["DATABASE_NAME"],
    table_name="customer_trusted",
    transformation_ctx="customer_trusted",
)

accelerometer_trusted = glueContext.create_dynamic_frame.from_catalog(
    database=args["DATABASE_NAME"],
    table_name="accelerometer_trusted",
    transformation_ctx="accelerometer_trusted",
)

customer_trusted.createOrReplaceTempView("customer_trusted")
accelerometer_trusted.createOrReplaceTempView("accelerometer_trusted")

customers_curated_df = spark.sql(
    """
    SELECT DISTINCT
        c.customerName,
        c.email,
        c.phone,
        c.birthDay,
        c.serialNumber,
        c.registrationDate,
        c.lastUpdateDate,
        c.shareWithResearchAsOfDate,
        c.shareWithPublicAsOfDate,
        c.shareWithFriendsAsOfDate
    FROM customer_trusted c
    INNER JOIN accelerometer_trusted a
        ON c.email = a.user
    """
)

customers_curated = DynamicFrame.fromDF(customers_curated_df, glueContext, "customers_curated")

sink = glueContext.getSink(
    path=args["S3_CURATED_PATH"],
    connection_type="s3",
    updateBehavior="UPDATE_IN_DATABASE",
    partitionKeys=[],
    enableUpdateCatalog=True,
    transformation_ctx="customers_curated_sink",
)
sink.setCatalogInfo(catalogDatabase=args["DATABASE_NAME"], catalogTableName="customers_curated")
sink.setFormat("json")
sink.writeFrame(customers_curated)

job.commit()

