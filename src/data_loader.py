"""
Data loading and schema validation utilities.
"""

from pathlib import Path

import pandas as pd

from .config import SURVEY_ITEMS


# Minimum expected columns (demographic + financial + survey items)
_REQUIRED_COLUMNS = {
    "participant_id",
    "age",
    "gender",
    "employment_status",
    "city_tier",
    "monthly_income",
    "essential_expenses",
    "discretionary_expenses",
    "monthly_savings",
    "debt_payment",
}
# Add all Likert items
for _items in SURVEY_ITEMS.values():
    _REQUIRED_COLUMNS.update(_items)


def load_data(path: Path) -> pd.DataFrame:
    """Read the CSV dataset and return a DataFrame."""
    df = pd.read_csv(path)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns from {path.name}")
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Assert that all required columns are present in the DataFrame."""
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    print(f"  Schema validation passed — all {len(_REQUIRED_COLUMNS)} required columns present")
