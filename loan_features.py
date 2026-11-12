"""Shared feature contract for training, serving and monitoring."""

NUMERIC_FEATURES = [
    "amount_usd",
    "term_months",
    "annual_income_usd",
    "credit_score",
    "existing_debt_usd",
    "employment_years",
    "num_late_payments_12m",
    "age",
]
CATEGORICAL_FEATURES = ["purpose"]
FEATURE_COLUMNS = [
    "amount_usd",
    "term_months",
    "purpose",
    "annual_income_usd",
    "credit_score",
    "existing_debt_usd",
    "employment_years",
    "num_late_payments_12m",
    "age",
]
TARGET_COLUMN = "default"
TERM_MONTHS = (3, 6, 12, 18, 24)
