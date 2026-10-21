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

customer_landing = glueContext.create_dynamic_frame.from_catalog(
    database=args["DATABASE_NAME"],
    table_name="customer_landing",
    transformation_ctx="customer_landing",
)

customer_landing.createOrReplaceTempView("customer_landing")
customer_trusted_df = spark.sql(
    """
    SELECT
        customerName,
        email,
        phone,
        birthDay,
        serialNumber,
        registrationDate,
        lastUpdateDate,
        shareWithResearchAsOfDate,
        shareWithPublicAsOfDate,
        shareWithFriendsAsOfDate
    FROM customer_landing
    WHERE shareWithResearchAsOfDate IS NOT NULL
    """
)

customer_trusted = DynamicFrame.fromDF(customer_trusted_df, glueContext, "customer_trusted")

sink = glueContext.getSink(
    path=args["S3_TRUSTED_PATH"],
    connection_type="s3",
    updateBehavior="UPDATE_IN_DATABASE",
    partitionKeys=[],
    enableUpdateCatalog=True,
    transformation_ctx="customer_trusted_sink",
)
sink.setCatalogInfo(catalogDatabase=args["DATABASE_NAME"], catalogTableName="customer_trusted")
sink.setFormat("json")
sink.writeFrame(customer_trusted)

job.commit()

