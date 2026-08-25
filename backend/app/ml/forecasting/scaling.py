from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TargetScaler:
    mean: float
    std: float
    target_column: str

    def transform(self, values):
        return (values - self.mean) / self.std

    def inverse_transform(self, values):
        return values * self.std + self.mean


def fit_target_scaler(
    df: pd.DataFrame,
    train_end: pd.Timestamp,
    target_column: str,
) -> TargetScaler:
    """Fit target normalization using training-period rows only."""

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found"
        )

    train_rows = df[df["timestamp"] <= train_end]

    if train_rows.empty:
        raise ValueError("No training rows available for scaling")

    mean = float(train_rows[target_column].mean())
    std = float(train_rows[target_column].std())

    if std <= 0:
        raise ValueError(
            f"{target_column} standard deviation must be positive"
        )

    return TargetScaler(
        mean=mean,
        std=std,
        target_column=target_column,
    )


def apply_target_scaler(
    df: pd.DataFrame,
    scaler: TargetScaler,
) -> pd.DataFrame:
    """Return a copy with the target column normalized."""

    df = df.copy()

    df[scaler.target_column] = scaler.transform(
        df[scaler.target_column]
    )

    return df


def fit_generation_scaler(
    df: pd.DataFrame,
    train_end: pd.Timestamp,
) -> TargetScaler:
    """Backward-compatible scaler for household generation."""

    return fit_target_scaler(
        df,
        train_end,
        "generation_kw",
    )


def apply_generation_scaler(
    df: pd.DataFrame,
    scaler: TargetScaler,
) -> pd.DataFrame:
    """Backward-compatible household scaling helper."""

    return apply_target_scaler(df, scaler)
