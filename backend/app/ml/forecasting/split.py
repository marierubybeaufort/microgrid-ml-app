from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    train_end: pd.Timestamp
    val_end: pd.Timestamp


def make_time_split(
    df: pd.DataFrame,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
) -> TimeSplit:
    """Create chronological train/validation/test boundaries."""

    if train_fraction <= 0 or val_fraction <= 0:
        raise ValueError("Split fractions must be positive")

    if train_fraction + val_fraction >= 1:
        raise ValueError(
            "train_fraction + val_fraction must be less than 1"
        )

    timestamps = pd.Series(
        sorted(df["timestamp"].unique())
    )

    n = len(timestamps)

    train_idx = int(n * train_fraction)
    val_idx = int(n * (train_fraction + val_fraction))

    if train_idx == 0 or val_idx <= train_idx or val_idx >= n:
        raise ValueError("Dataset is too small for requested split")

    return TimeSplit(
        train_end=pd.Timestamp(timestamps.iloc[train_idx - 1]),
        val_end=pd.Timestamp(timestamps.iloc[val_idx - 1]),
    )


def label_split(
    timestamp: pd.Timestamp,
    split: TimeSplit,
) -> str:
    """Return the chronological partition for a target timestamp."""

    if timestamp <= split.train_end:
        return "train"

    if timestamp <= split.val_end:
        return "validation"

    return "test"
