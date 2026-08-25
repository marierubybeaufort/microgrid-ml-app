from dataclasses import dataclass

from torch.utils.data import DataLoader

from backend.app.ml.forecasting.dataset import MultiSeriesForecastDataset
from backend.app.ml.forecasting.split import TimeSplit


@dataclass
class ForecastLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader


def make_forecast_loaders(
    df,
    split: TimeSplit,
    series_column: str,
    feature_columns: tuple[str, ...],
    target_column: str,
    sequence_length: int = 24,
    batch_size: int = 64,
) -> ForecastLoaders:
    """Create chronological train, validation, and test DataLoaders."""

    train_dataset = MultiSeriesForecastDataset(
        df=df,
        split=split,
        partition="train",
        series_column=series_column,
        feature_columns=feature_columns,
        target_column=target_column,
        sequence_length=sequence_length,
    )

    validation_dataset = MultiSeriesForecastDataset(
        df=df,
        split=split,
        partition="validation",
        series_column=series_column,
        feature_columns=feature_columns,
        target_column=target_column,
        sequence_length=sequence_length,
    )

    test_dataset = MultiSeriesForecastDataset(
        df=df,
        split=split,
        partition="test",
        series_column=series_column,
        feature_columns=feature_columns,
        target_column=target_column,
        sequence_length=sequence_length,
    )

    return ForecastLoaders(
        train=DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
        ),
        validation=DataLoader(
            validation_dataset,
            batch_size=batch_size,
            shuffle=False,
        ),
        test=DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
        ),
    )
