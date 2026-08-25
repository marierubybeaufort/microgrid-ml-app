import math

import torch
from torch.utils.data import DataLoader

from backend.app.ml.forecasting.scaling import TargetScaler


@torch.no_grad()
def evaluate_real_units(
    model: torch.nn.Module,
    loader: DataLoader,
    scaler: TargetScaler,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate predictions in normalized and original target units."""

    model.eval()

    normalized_squared_error = 0.0
    absolute_error = 0.0
    squared_error = 0.0
    total_samples = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        predictions = model(x)

        normalized_squared_error += (
            (predictions - y) ** 2
        ).sum().item()

        predictions_real = scaler.inverse_transform(
            predictions
        )
        y_real = scaler.inverse_transform(y)

        errors = predictions_real - y_real

        absolute_error += errors.abs().sum().item()
        squared_error += (errors ** 2).sum().item()

        total_samples += y.numel()

    normalized_mse = (
        normalized_squared_error / total_samples
    )

    mae = absolute_error / total_samples
    rmse = math.sqrt(
        squared_error / total_samples
    )

    return {
        "normalized_mse": normalized_mse,
        "mae_kwh_per_premise": mae,
        "rmse_kwh_per_premise": rmse,
    }
