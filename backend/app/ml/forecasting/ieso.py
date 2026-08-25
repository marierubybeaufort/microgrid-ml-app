from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_COLUMNS = {
    "FSA",
    "DATE",
    "HOUR",
    "CUSTOMER_TYPE",
    "PRICE_PLAN",
    "TOTAL_CONSUMPTION",
    "PREMISE_COUNT",
}


def load_ieso_residential(path: str | Path) -> pd.DataFrame:
    """Load and aggregate one IESO hourly-consumption file."""

    df = pd.read_csv(path, skiprows=3)

    missing = EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"IESO dataset missing columns: {sorted(missing)}"
        )

    df = df[df["CUSTOMER_TYPE"] == "Residential"].copy()

    df["DATE"] = pd.to_datetime(df["DATE"])

    # IESO reports hours in hour-ending format:
    # HOUR=1 ends at 01:00, HOUR=24 ends at midnight next day.
    df["timestamp"] = (
        df["DATE"]
        + pd.to_timedelta(df["HOUR"], unit="h")
    )

    aggregated = (
        df.groupby(
            ["FSA", "timestamp"],
            as_index=False,
        )
        .agg(
            total_consumption_kwh=(
                "TOTAL_CONSUMPTION",
                "sum",
            ),
            premise_count=(
                "PREMISE_COUNT",
                "sum",
            ),
        )
    )

    aggregated = aggregated[
        aggregated["premise_count"] > 0
    ].copy()

    aggregated["consumption_per_premise_kwh"] = (
        aggregated["total_consumption_kwh"]
        / aggregated["premise_count"]
    )

    return aggregated.sort_values(
        ["FSA", "timestamp"]
    ).reset_index(drop=True)


def keep_complete_series(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Keep only geographic series with every expected timestamp."""

    expected_timestamps = df["timestamp"].nunique()

    counts = df.groupby("FSA")["timestamp"].nunique()

    complete_fsas = counts[
        counts == expected_timestamps
    ].index

    return (
        df[df["FSA"].isin(complete_fsas)]
        .copy()
        .reset_index(drop=True)
    )


def add_forecasting_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Add cyclical calendar features."""

    df = df.copy()

    hour = df["timestamp"].dt.hour
    day_of_week = df["timestamp"].dt.dayofweek

    df["hour_sin"] = np.sin(
        2.0 * np.pi * hour / 24.0
    )
    df["hour_cos"] = np.cos(
        2.0 * np.pi * hour / 24.0
    )

    df["dow_sin"] = np.sin(
        2.0 * np.pi * day_of_week / 7.0
    )
    df["dow_cos"] = np.cos(
        2.0 * np.pi * day_of_week / 7.0
    )

    return df
