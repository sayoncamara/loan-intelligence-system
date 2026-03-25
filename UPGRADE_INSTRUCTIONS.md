# Loan Intelligence System — Production Upgrade Files

## What's in this package

All the files you need to add to your existing repo to upgrade it
from a demo to a production-grade system.

## How to add these to your repo

1. Unzip this package
2. Copy ALL files and folders into your existing loan-intelligence-system repo
3. Your repo should now look like this:

```
loan-intelligence-system/
├── .github/
│   └── workflows/
│       └── ci.yml              ← NEW: CI/CD pipeline
├── dbt/
│   ├── dbt_project.yml         ← NEW: dbt config
│   ├── profiles.yml            ← NEW: DuckDB connection
│   └── models/
│       ├── schema.yml          ← NEW: data quality tests
│       ├── staging/
│       │   └── stg_loans.sql   ← NEW: raw data cleaning
│       ├── intermediate/
│       │   └── int_loan_features.sql  ← NEW: feature engineering
│       └── marts/
│           └── mart_training.sql      ← NEW: final training table
├── pipeline/
│   ├── __init__.py             ← NEW
│   └── training_flow.py        ← NEW: Prefect orchestration
├── tests/
│   ├── __init__.py             ← NEW
│   └── test_system.py          ← NEW: unit tests
├── policies/                   ← EXISTING (keep as-is)
├── docs/                       ← EXISTING (keep as-is)
├── Dockerfile                  ← NEW: containerization
├── docker-compose.yml          ← NEW: run full stack
├── app.py                      ← EXISTING (keep as-is)
├── requirements.txt            ← EXISTING (update — see below)
├── xgboost_loan_model.json     ← EXISTING (keep as-is)
└── README.md                   ← EXISTING (update with new architecture)
```

## Update your requirements.txt

Add these lines to your existing requirements.txt:

```
prefect>=2.14.0
duckdb>=0.9.0
dbt-core>=1.7.0
dbt-duckdb>=1.7.0
pytest>=7.0.0
boto3>=1.28.0
```

## Test locally

```bash
# Run tests
pip install pytest
pytest tests/ -v

# Run the training pipeline
pip install prefect
python pipeline/training_flow.py

# Build and run with Docker
docker-compose up --build
```

## To add dbt tests

```bash
cd dbt
pip install dbt-core dbt-duckdb
dbt run       # builds the models
dbt test      # runs data quality checks
```

## Git commands to push everything

```bash
cd loan-intelligence-system
git add -A
git commit -m "Add production infrastructure: Docker, CI/CD, dbt pipeline, Prefect orchestration, tests"
git push origin main
```

That single commit message tells hiring managers everything they need to know.
