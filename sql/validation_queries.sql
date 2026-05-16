-- Landing zone checks
SELECT COUNT(*) AS customer_landing_count FROM stedi.customer_landing;
SELECT * FROM stedi.customer_landing WHERE shareWithResearchAsOfDate IS NULL LIMIT 10;
SELECT COUNT(*) AS accelerometer_landing_count FROM stedi.accelerometer_landing;
SELECT COUNT(*) AS step_trainer_landing_count FROM stedi.step_trainer_landing;

-- Trusted zone checks
SELECT COUNT(*) AS customer_trusted_count FROM stedi.customer_trusted;
SELECT * FROM stedi.customer_trusted WHERE shareWithResearchAsOfDate IS NULL LIMIT 10;
SELECT COUNT(*) AS accelerometer_trusted_count FROM stedi.accelerometer_trusted;
SELECT COUNT(*) AS step_trainer_trusted_count FROM stedi.step_trainer_trusted;

-- Curated zone checks
SELECT COUNT(*) AS customers_curated_count FROM stedi.customers_curated;
SELECT COUNT(*) AS machine_learning_curated_count FROM stedi.machine_learning_curated;

