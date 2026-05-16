-- Athena/Glue DDL for the STEDI Step Trainer landing zone.
-- Replace ${bucket} with the project bucket name.

CREATE EXTERNAL TABLE IF NOT EXISTS stedi.step_trainer_landing (
    sensorReadingTime bigint,
    serialNumber string,
    distanceFromObject int
)
ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
WITH SERDEPROPERTIES (
    'case.insensitive' = 'false',
    'mapping.sensorReadingTime' = 'sensorReadingTime',
    'mapping.serialNumber' = 'serialNumber',
    'mapping.distanceFromObject' = 'distanceFromObject'
)
LOCATION 's3://${bucket}/step_trainer_landing/'
TBLPROPERTIES ('has_encrypted_data'='false');

