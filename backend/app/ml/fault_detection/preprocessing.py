from pathlib import Path

import pandas as pd


FEATURE_COLUMNS = (
    "delta_1",
    "drop_from_4",
    "drop_from_12",
    "relative_delta_1",
    "relative_drop_4",
    "relative_drop_12",
    "z_drop_4",
    "z_drop_12",
)


def load_fault_data(path: str | Path) -> pd.DataFrame:
    """Load household fault observations."""

    df = pd.read_csv(
        path,
        parse_dates=["timestamp"],
    )

    required = {
        "timestamp",
        "household_id",
        "generation_kw",
        "fault",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"Fault dataset missing columns: {sorted(missing)}"
        )

    return (
        df.sort_values(
            ["household_id", "timestamp"]
        )
        .reset_index(drop=True)
    )


def add_fault_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Create scale-robust temporal degradation features."""

    df = df.copy()

    grouped = df.groupby(
        "household_id",
        sort=False,
    )["generation_kw"]

    df["lag_1"] = grouped.shift(1)

    df["rolling_mean_4"] = grouped.transform(
        lambda s: s.shift(1).rolling(4).mean()
    )

    df["rolling_mean_12"] = grouped.transform(
        lambda s: s.shift(1).rolling(12).mean()
    )

    df["rolling_std_4"] = grouped.transform(
        lambda s: s.shift(1).rolling(4).std()
    )

    df["rolling_std_12"] = grouped.transform(
        lambda s: s.shift(1).rolling(12).std()
    )

    df["delta_1"] = (
        df["generation_kw"]
        - df["lag_1"]
    )

    df["drop_from_4"] = (
        df["rolling_mean_4"]
        - df["generation_kw"]
    )

    df["drop_from_12"] = (
        df["rolling_mean_12"]
        - df["generation_kw"]
    )

    lag_scale = df["lag_1"].abs().clip(lower=0.1)

    mean_4_scale = (
        df["rolling_mean_4"]
        .abs()
        .clip(lower=0.1)
    )

    mean_12_scale = (
        df["rolling_mean_12"]
        .abs()
        .clip(lower=0.1)
    )

    std_4_scale = (
        df["rolling_std_4"]
        .clip(lower=0.05)
    )

    std_12_scale = (
        df["rolling_std_12"]
        .clip(lower=0.05)
    )

    df["relative_delta_1"] = (
        df["delta_1"] / lag_scale
    ).clip(-10.0, 10.0)

    df["relative_drop_4"] = (
        df["drop_from_4"] / mean_4_scale
    ).clip(-10.0, 10.0)

    df["relative_drop_12"] = (
        df["drop_from_12"] / mean_12_scale
    ).clip(-10.0, 10.0)

    df["z_drop_4"] = (
        df["drop_from_4"] / std_4_scale
    ).clip(-10.0, 10.0)

    df["z_drop_12"] = (
        df["drop_from_12"] / std_12_scale
    ).clip(-10.0, 10.0)

    return df.dropna(
        subset=list(FEATURE_COLUMNS)
    ).reset_index(drop=True)


def prepare_fault_data(
    path: str | Path,
) -> pd.DataFrame:
    df = load_fault_data(path)
    return add_fault_features(df)
