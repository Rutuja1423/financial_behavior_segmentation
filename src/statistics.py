"""
Statistical analysis: correlation, ANOVA, and regression.
"""

from typing import Dict, List, Tuple

import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def correlation_matrix(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Compute Pearson correlation matrix for the given columns."""
    corr = df[cols].corr()
    print(f"  Computed {len(cols)}x{len(cols)} correlation matrix")
    return corr


def run_anova(
    df: pd.DataFrame, group_col: str, value_col: str
) -> Dict[str, float]:
    """
    Run one-way ANOVA and return F-statistic and p-value.
    """
    groups = [group[value_col].values for _, group in df.groupby(group_col)]
    f_stat, p_value = stats.f_oneway(*groups)
    result = {"F_statistic": round(f_stat, 4), "p_value": round(p_value, 4)}
    sig = "significant" if p_value < 0.05 else "not significant"
    print(f"  ANOVA ({group_col} -> {value_col}): F={f_stat:.4f}, p={p_value:.4f} ({sig})")
    return result


def run_regression(df: pd.DataFrame, formula: str) -> Tuple:
    """
    Fit an OLS regression model and return (model, summary_dict).

    summary_dict contains r_squared, adj_r_squared, f_pvalue, and n_obs.
    """
    model = smf.ols(formula, data=df).fit()
    summary = {
        "r_squared": round(model.rsquared, 4),
        "adj_r_squared": round(model.rsquared_adj, 4),
        "f_pvalue": round(model.f_pvalue, 6),
        "n_obs": int(model.nobs),
    }
    print(f"  Regression: R²={summary['r_squared']}, Adj R²={summary['adj_r_squared']}, n={summary['n_obs']}")
    return model, summary
