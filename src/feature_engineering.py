"""
Feature engineering: composite scores and financial ratios.
"""

import pandas as pd

from .config import SURVEY_ITEMS


def compute_composite_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean composite scores for each survey construct
    (financial literacy, impulse buying, stress spending,
    saving discipline, budgeting habit).
    """
    df = df.copy()
    for construct, items in SURVEY_ITEMS.items():
        col_name = f"{construct}_score"
        df[col_name] = df[items].mean(axis=1)
    print("  Computed 5 composite scores")
    return df


def compute_financial_ratios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived financial metrics:
      - total_expenses = essential + discretionary + debt_payment
      - savings_rate   = monthly_savings / monthly_income
      - expense_to_income_ratio = total_expenses / monthly_income
    """
    df = df.copy()
    df["total_expenses"] = (
        df["essential_expenses"]
        + df["discretionary_expenses"]
        + df["debt_payment"]
    )
    df["savings_rate"] = df["monthly_savings"] / df["monthly_income"]
    df["expense_to_income_ratio"] = df["total_expenses"] / df["monthly_income"]
    print("  Computed financial ratios (savings_rate, expense_to_income_ratio)")
    return df
