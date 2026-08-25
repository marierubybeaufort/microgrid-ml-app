import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def persistence_mse(loader: DataLoader) -> float:
    """Use the final generation value in each window as the next prediction."""

    total_squared_error = 0.0
    total_samples = 0

    for x, y in loader:
        # generation_kw is feature 0
        predictions = x[:, -1, 0].unsqueeze(1)

        squared_error = (predictions - y) ** 2

        total_squared_error += squared_error.sum().item()
        total_samples += y.size(0)

    return total_squared_error / total_samples
