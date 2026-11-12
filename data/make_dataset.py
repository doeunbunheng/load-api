"""Generate reproducible synthetic loan-default data for the MLflow lab.

The reference file is used for training and drift monitoring. The production
window is deliberately shifted so the monitor has something to detect.
"""

from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_STATE = 42
DATA_DIR = Path(__file__).resolve().parent
N_REFERENCE = 8000
N_PRODUCTION = 1200

PURPOSES = ["equipment", "school_fees", "medical", "home_repair", "working_capital", "vehicle"]
PURPOSE_PROBS = [0.22, 0.18, 0.15, 0.15, 0.18, 0.12]
PURPOSE_RISK = {
    "equipment": 0.00,
    "school_fees": 0.10,
    "medical": 0.20,
    "home_repair": 0.05,
    "working_capital": 0.30,
    "vehicle": 0.15,
}


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _applicant_frame(rng: np.random.Generator, n: int, drifted: bool) -> pd.DataFrame:
    amount = rng.integers(100, 5001, size=n)
    term = rng.choice([3, 6, 12, 18, 24], size=n, p=[0.15, 0.20, 0.35, 0.15, 0.15])
    purpose = rng.choice(PURPOSES, size=n, p=PURPOSE_PROBS)
    income = np.clip(rng.lognormal(mean=np.log(9000), sigma=0.55, size=n), 1500, 60000)
    credit = np.clip(rng.normal(640, 85, size=n), 300, 850)
    debt = np.clip(rng.lognormal(mean=np.log(3000), sigma=0.85, size=n), 0, 40000)
    employment = np.clip(rng.exponential(scale=4.0, size=n), 0, 35)
    late = np.clip(rng.poisson(1.5, size=n), 0, 12)
    age = np.clip(rng.normal(38, 10, size=n), 18, 75)

    if drifted:
        amount = np.clip(amount * 1.20, 100, 5000)
        income = np.clip(income * 0.70, 1500, 60000)
        credit = np.clip(credit - 80, 300, 850)
        debt = np.clip(debt * 1.80, 0, 40000)
        late = np.clip(late + 2, 0, 12)
        purpose = rng.choice(PURPOSES, size=n, p=[0.12, 0.12, 0.24, 0.10, 0.30, 0.12])

    risk = (
        -3.40
        + 1.30 * (amount / 5000.0)
        + 0.70 * (term / 24.0)
        + 0.014 * (650.0 - credit)
        + 1.00 * (debt / 10000.0)
        - 0.90 * (income / 20000.0)
        - 0.070 * employment
        + 0.50 * late
        + np.vectorize(PURPOSE_RISK.get)(purpose)
        + 0.70 * ((credit < 600) & (debt > 5000)).astype(float)
        + rng.normal(0, 0.20, size=n)
    )
    probability = np.clip(_sigmoid(risk), 0.01, 0.95)
    default = rng.binomial(1, probability)

    return pd.DataFrame(
        {
            "amount_usd": amount.astype(int),
            "term_months": term.astype(int),
            "purpose": purpose,
            "annual_income_usd": income.round(2),
            "credit_score": credit.astype(int),
            "existing_debt_usd": debt.round(2),
            "employment_years": employment.round(2),
            "num_late_payments_12m": late.astype(int),
            "age": age.astype(int),
            "default": default.astype(int),
        }
    )


def main() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    reference = _applicant_frame(rng, N_REFERENCE, drifted=False)
    production = _applicant_frame(rng, N_PRODUCTION, drifted=True)

    reference_path = DATA_DIR / "loan_default.csv"
    production_path = DATA_DIR / "production_window.csv"
    reference.to_csv(reference_path, index=False)
    production.to_csv(production_path, index=False)

    print(f"reference rows={len(reference)} default_rate={reference['default'].mean():.3f} -> {reference_path}")
    print(f"production rows={len(production)} default_rate={production['default'].mean():.3f} -> {production_path}")
    print(reference.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
