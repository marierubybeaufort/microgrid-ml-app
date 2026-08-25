from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from backend.app.ml.forecasting.split import TimeSplit, label_split


Partition = Literal["train", "validation", "test"]


class MultiSeriesForecastDataset(Dataset):
    """Sliding-window forecasting dataset for multiple independent time series."""

    def __init__(
        self,
        df: pd.DataFrame,
        split: TimeSplit,
        partition: Partition,
        series_column: str,
        feature_columns: tuple[str, ...],
        target_column: str,
        sequence_length: int = 24,
    ) -> None:
        if partition not in {"train", "validation", "test"}:
            raise ValueError(
                "partition must be 'train', 'validation', or 'test'"
            )

        if sequence_length < 1:
            raise ValueError("sequence_length must be at least 1")

        required = {
            "timestamp",
            series_column,
            target_column,
            *feature_columns,
        }

        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Dataset is missing required columns: {sorted(missing)}"
            )

        self.partition = partition
        self.sequence_length = sequence_length
        self.series_column = series_column
        self.feature_columns = feature_columns
        self.target_column = target_column

        self.series: dict[str, dict[str, object]] = {}
        self.samples: list[tuple[str, int]] = []

        for series_id, series_df in df.groupby(
            series_column,
            sort=True,
        ):
            series_df = (
                series_df
                .sort_values("timestamp")
                .reset_index(drop=True)
            )

            features = series_df[
                list(feature_columns)
            ].to_numpy(dtype=np.float32)

            targets = series_df[
                target_column
            ].to_numpy(dtype=np.float32)

            timestamps = pd.to_datetime(
                series_df["timestamp"]
            ).tolist()

            series_key = str(series_id)

            self.series[series_key] = {
                "features": features,
                "targets": targets,
                "timestamps": timestamps,
            }

            for target_index in range(
                sequence_length,
                len(series_df),
            ):
                target_timestamp = pd.Timestamp(
                    timestamps[target_index]
                )

                if label_split(target_timestamp, split) == partition:
                    self.samples.append(
                        (series_key, target_index)
                    )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        series_id, target_index = self.samples[index]

        series = self.series[series_id]

        features = series["features"]
        targets = series["targets"]

        start_index = target_index - self.sequence_length

        x = features[start_index:target_index]
        y = targets[target_index]

        return (
            torch.as_tensor(x, dtype=torch.float32),
            torch.tensor([y], dtype=torch.float32),
        )

    def metadata(self, index: int) -> dict[str, object]:
        series_id, target_index = self.samples[index]
        series = self.series[series_id]

        return {
            "series_id": series_id,
            "target_timestamp": series["timestamps"][target_index],
        }


class HouseholdForecastDataset(MultiSeriesForecastDataset):
    """Backward-compatible dataset for the original household data."""

    def __init__(
        self,
        df: pd.DataFrame,
        split: TimeSplit,
        partition: Partition,
        sequence_length: int = 24,
        feature_columns: tuple[str, ...] = (
            "generation_kw",
            "hour_sin",
            "hour_cos",
        ),
        target_column: str = "generation_kw",
    ) -> None:
        super().__init__(
            df=df,
            split=split,
            partition=partition,
            series_column="household_id",
            feature_columns=feature_columns,
            target_column=target_column,
            sequence_length=sequence_length,
        )
