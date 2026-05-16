# STEDI Human Balance Analytics

This repository completes Udacity's **STEDI Human Balance Analytics** project for the Data Engineering with AWS Nanodegree.

The solution builds a privacy-aware lakehouse pipeline for STEDI Step Trainer sensor data using AWS Glue, Spark, S3, Glue Data Catalog, and Athena.

## Architecture

```text
S3 landing JSON
  customer_landing
  accelerometer_landing
  step_trainer_landing
        |
        v
Glue/Athena landing tables
        |
        v
trusted zone
  customer_trusted
  accelerometer_trusted
  step_trainer_trusted
        |
        v
curated zone
  customers_curated
  machine_learning_curated
```

Only customers who have a non-null `shareWithResearchAsOfDate` are included in trusted and curated training data.

## Deliverables

### Landing DDL

- `sql/customer_landing.sql`
- `sql/accelerometer_landing.sql`
- `sql/step_trainer_landing.sql`
- `sql/validation_queries.sql`

### Glue Jobs

- `glue_jobs/customer_landing_to_trusted.py`
- `glue_jobs/accelerometer_landing_to_trusted.py`
- `glue_jobs/customer_trusted_to_curated.py`
- `glue_jobs/step_trainer_trusted.py`
- `glue_jobs/machine_learning_curated.py`

Each Glue script uses Glue Data Catalog sources and writes JSON to S3 with:

```python
updateBehavior="UPDATE_IN_DATABASE"
enableUpdateCatalog=True
```

This matches the rubric requirement to dynamically update the Glue table schema and add new partitions on subsequent runs.

### Screenshots

The `screenshots/` directory contains reviewer-facing Athena result images:

- `customer_landing.png`
- `accelerometer_landing.png`
- `step_trainer_landing.png`
- `customer_trusted.png`
- `accelerometer_trusted.png`
- `step_trainer_trusted.png`
- `customers_curated.png`
- `machine_learning_curated.png`

## Expected Row Counts

| Zone | Table | Expected rows |
|---|---:|---:|
| Landing | `customer_landing` | 956 |
| Landing | `accelerometer_landing` | 81,273 |
| Landing | `step_trainer_landing` | 28,680 |
| Trusted | `customer_trusted` | 482 |
| Trusted | `accelerometer_trusted` | 40,981 |
| Trusted | `step_trainer_trusted` | 14,460 |
| Curated | `customers_curated` | 482 |
| Curated | `machine_learning_curated` | 43,681 |

## Glue Job Logic

### `customer_landing_to_trusted.py`

Filters `customer_landing` to only rows where `shareWithResearchAsOfDate IS NOT NULL`.

### `accelerometer_landing_to_trusted.py`

Joins `accelerometer_landing` to `customer_trusted` on:

```sql
accelerometer_landing.user = customer_trusted.email
```

The output keeps only accelerometer columns.

### `customer_trusted_to_curated.py`

Joins `customer_trusted` to `accelerometer_trusted` on:

```sql
customer_trusted.email = accelerometer_trusted.user
```

The output keeps only customer columns and uses `DISTINCT` to avoid duplicate customers.

### `step_trainer_trusted.py`

Joins `step_trainer_landing` to `customers_curated` on:

```sql
step_trainer_landing.serialNumber = customers_curated.serialNumber
```

The output keeps only Step Trainer columns.

### `machine_learning_curated.py`

Joins `step_trainer_trusted` to `accelerometer_trusted` on matching timestamps:

```sql
step_trainer_trusted.sensorReadingTime = accelerometer_trusted.timestamp
```

The output contains each Step Trainer reading and the associated accelerometer values.

## Local Validation

The included `local_validation.py` mirrors the Glue SQL logic with pandas against the included starter JSON files.

```bash
python3 local_validation.py
```

It verifies all rubric row counts locally. AWS Glue is still the target runtime for the submitted job scripts.

## Source Data

The `data/starter/` folder is copied from Udacity's public starter repository:

<https://github.com/udacity/nd027-Data-Engineering-Data-Lakes-AWS-Exercises/tree/main/project/starter>

