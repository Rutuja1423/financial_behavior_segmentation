"""
Internal consistency reliability testing (Cronbach's Alpha).
"""

from typing import List

import numpy as np
import pandas as pd


def cronbachs_alpha(df: pd.DataFrame, items: List[str]) -> float:
    """
    Compute Cronbach's Alpha for a set of Likert-scale items.

    Parameters
    ----------
    df : DataFrame containing the item columns.
    items : List of column names for the construct.

    Returns
    -------
    alpha : float in [0, 1]. Values above 0.7 are generally acceptable;
            above 0.8 is good; above 0.9 is excellent.
    """
    item_data = df[items].dropna().values
    n_items = len(items)
    item_variances = item_data.var(axis=0, ddof=1)
    total_variance = item_data.sum(axis=1).var(ddof=1)

    alpha = (n_items / (n_items - 1)) * (1 - item_variances.sum() / total_variance)
    return float(alpha)


def reliability_report(df: pd.DataFrame, survey_items: dict) -> pd.DataFrame:
    """
    Compute Cronbach's Alpha for every survey construct and return
    a summary DataFrame.
    """
    rows = []
    for construct, items in survey_items.items():
        alpha = cronbachs_alpha(df, items)
        rows.append({
            "Construct": construct.replace("_", " ").title(),
            "Items": len(items),
            "Cronbach Alpha": round(alpha, 4),
            "Reliability": (
                "Excellent" if alpha >= 0.9
                else "Good" if alpha >= 0.8
                else "Acceptable" if alpha >= 0.7
                else "Questionable"
            ),
        })
    report = pd.DataFrame(rows)
    print("  Reliability report:")
    print(report.to_string(index=False))
    return report
