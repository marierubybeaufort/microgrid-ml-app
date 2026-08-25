from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "timestamp",
    "household_id",
    "generation_kw",
    "fault",
}


def load_household_data(path: str | Path) -> pd.DataFrame:
    """Load and validate household-level microgrid observations."""

    df = pd.read_csv(path, parse_dates=["timestamp"])

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    if df[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Dataset contains missing values")

    df = df.sort_values(
        ["household_id", "timestamp"]
    ).reset_index(drop=True)

    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical time-of-day features for forecasting."""

    df = df.copy()

    minutes = (
        df["timestamp"].dt.hour * 60
        + df["timestamp"].dt.minute
    )

    angle = 2.0 * np.pi * minutes / (24 * 60)

    df["hour_sin"] = np.sin(angle)
    df["hour_cos"] = np.cos(angle)

    return df


def prepare_forecasting_data(path: str | Path) -> pd.DataFrame:
    """Load household observations and create forecasting features."""

    df = load_household_data(path)
    df = add_time_features(df)

    return df
