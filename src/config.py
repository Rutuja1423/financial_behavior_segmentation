"""
Central configuration for the financial behavior segmentation pipeline.
All constants, file paths, column definitions, and hyperparameters are
defined here so that every module draws from a single source of truth.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "synthetic_financial_behavior_data.csv"
ASSETS_DIR = ROOT_DIR / "assets"

# ── Random seed ────────────────────────────────────────────────────────
RANDOM_STATE = 42

# ── Likert-scale survey item groups ───────────────────────────────────
SURVEY_ITEMS = {
    "financial_literacy": ["FL1", "FL2", "FL3", "FL4", "FL5"],
    "impulse_buying": ["IB1", "IB2", "IB3", "IB4", "IB5"],
    "stress_spending": ["SS1", "SS2", "SS3", "SS4"],
    "saving_discipline": ["SD1", "SD2", "SD3", "SD4", "SD5"],
    "budgeting_habit": ["BH1", "BH2", "BH3", "BH4"],
}

# ── Composite score column names ──────────────────────────────────────
COMPOSITE_SCORE_COLS = [
    "financial_literacy_score",
    "impulse_buying_score",
    "stress_spending_score",
    "saving_discipline_score",
    "budgeting_habit_score",
]

# ── Columns used for clustering ───────────────────────────────────────
CLUSTER_FEATURE_COLS = COMPOSITE_SCORE_COLS + [
    "savings_rate",
    "expense_to_income_ratio",
]

# ── PCA settings ──────────────────────────────────────────────────────
PCA_N_COMPONENTS = 2

# ── K-Means search range ─────────────────────────────────────────────
KMEANS_K_RANGE = range(2, 8)
KMEANS_N_INIT = 10

# ── Regression formula ────────────────────────────────────────────────
REGRESSION_FORMULA = (
    "savings_rate ~ financial_literacy_score + impulse_buying_score + "
    "stress_spending_score + saving_discipline_score + budgeting_habit_score + "
    "age + C(gender) + C(employment_status) + C(city_tier)"
)

REGRESSION_COEF_DISPLAY = [
    "financial_literacy_score",
    "impulse_buying_score",
    "stress_spending_score",
    "saving_discipline_score",
    "budgeting_habit_score",
]
