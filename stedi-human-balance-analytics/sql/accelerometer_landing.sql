-- Athena/Glue DDL for the STEDI accelerometer landing zone.
-- Replace ${bucket} with the project bucket name.

CREATE EXTERNAL TABLE IF NOT EXISTS stedi.accelerometer_landing (
    user string,
    timestamp bigint,
    x double,
    y double,
    z double
)
ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
LOCATION 's3://${bucket}/accelerometer_landing/'
TBLPROPERTIES ('has_encrypted_data'='false');

