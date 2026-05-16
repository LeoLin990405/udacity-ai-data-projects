-- Athena/Glue DDL for the STEDI customer landing zone.
-- Replace ${bucket} with the project bucket name.

CREATE EXTERNAL TABLE IF NOT EXISTS stedi.customer_landing (
    customerName string,
    email string,
    phone string,
    birthDay string,
    serialNumber string,
    registrationDate bigint,
    lastUpdateDate bigint,
    shareWithResearchAsOfDate bigint,
    shareWithPublicAsOfDate bigint,
    shareWithFriendsAsOfDate bigint
)
ROW FORMAT SERDE 'org.openx.data.jsonserde.JsonSerDe'
WITH SERDEPROPERTIES (
    'case.insensitive' = 'false',
    'mapping.customerName' = 'customerName',
    'mapping.birthDay' = 'birthDay',
    'mapping.serialNumber' = 'serialNumber',
    'mapping.registrationDate' = 'registrationDate',
    'mapping.lastUpdateDate' = 'lastUpdateDate',
    'mapping.shareWithResearchAsOfDate' = 'shareWithResearchAsOfDate',
    'mapping.shareWithPublicAsOfDate' = 'shareWithPublicAsOfDate',
    'mapping.shareWithFriendsAsOfDate' = 'shareWithFriendsAsOfDate'
)
LOCATION 's3://${bucket}/customer_landing/'
TBLPROPERTIES ('has_encrypted_data'='false');

