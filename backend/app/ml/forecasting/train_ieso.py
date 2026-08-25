import json
from pathlib import Path

import torch
from torch import nn

from backend.app.ml.forecasting.baseline import persistence_mse
from backend.app.ml.forecasting.evaluation import evaluate_real_units
from backend.app.ml.forecasting.ieso import (
    add_forecasting_features,
    keep_complete_series,
    load_ieso_residential,
)
from backend.app.ml.forecasting.loaders import make_forecast_loaders
from backend.app.ml.forecasting.model import LSTMForecaster
from backend.app.ml.forecasting.scaling import (
    apply_target_scaler,
    fit_target_scaler,
)
from backend.app.ml.forecasting.split import make_time_split
from backend.app.ml.forecasting.train import fit


DATA_PATH = Path(
    "data/PUB_HourlyConsumptionByFSA_202505_v1.csv"
)

ARTIFACT_DIR = Path(
    "backend/artifacts/forecasting"
)

FEATURES = (
    "consumption_per_premise_kwh",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
)

TARGET = "consumption_per_premise_kwh"

SEQUENCE_LENGTH = 24
BATCH_SIZE = 128
HIDDEN_SIZE = 64
LEARNING_RATE = 0.001
MAX_EPOCHS = 8
PATIENCE = 2
SEED = 42


def main() -> None:
    torch.manual_seed(SEED)

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_ieso_residential(DATA_PATH)
    df = keep_complete_series(df)
    df = add_forecasting_features(df)

    split = make_time_split(df)

    scaler = fit_target_scaler(
        df,
        split.train_end,
        TARGET,
    )

    scaled_df = apply_target_scaler(
        df,
        scaler,
    )

    loaders = make_forecast_loaders(
        scaled_df,
        split,
        series_column="FSA",
        feature_columns=FEATURES,
        target_column=TARGET,
        sequence_length=SEQUENCE_LENGTH,
        batch_size=BATCH_SIZE,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = LSTMForecaster(
        input_size=len(FEATURES),
        hidden_size=HIDDEN_SIZE,
    ).to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print("Device:", device)
    print(
        "Parameters:",
        sum(p.numel() for p in model.parameters()),
    )

    persistence_validation = persistence_mse(
        loaders.validation
    )

    persistence_test = persistence_mse(
        loaders.test
    )

    history = fit(
        model=model,
        train_loader=loaders.train,
        validation_loader=loaders.validation,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
    )

    metrics = evaluate_real_units(
        model=model,
        loader=loaders.test,
        scaler=scaler,
        device=device,
    )

    metrics["persistence_validation_mse"] = (
        persistence_validation
    )

    metrics["persistence_test_mse"] = (
        persistence_test
    )

    metrics["mse_improvement_vs_persistence_pct"] = (
        100.0
        * (
            persistence_test
            - metrics["normalized_mse"]
        )
        / persistence_test
    )

    torch.save(
        model.state_dict(),
        ARTIFACT_DIR / "model.pt",
    )

    with open(
        ARTIFACT_DIR / "scaler.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "mean": scaler.mean,
                "std": scaler.std,
                "target_column": scaler.target_column,
            },
            file,
            indent=2,
        )

    with open(
        ARTIFACT_DIR / "config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "features": list(FEATURES),
                "target": TARGET,
                "sequence_length": SEQUENCE_LENGTH,
                "hidden_size": HIDDEN_SIZE,
                "batch_size": BATCH_SIZE,
                "learning_rate": LEARNING_RATE,
                "seed": SEED,
                "train_end": str(split.train_end),
                "validation_end": str(split.val_end),
                "training_history": history,
            },
            file,
            indent=2,
        )

    with open(
        ARTIFACT_DIR / "metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    print()
    print("=== Final Test Metrics ===")
    print(
        "Normalized MSE:",
        round(metrics["normalized_mse"], 6),
    )
    print(
        "MAE:",
        round(
            metrics["mae_kwh_per_premise"],
            6,
        ),
        "kWh/premise",
    )
    print(
        "RMSE:",
        round(
            metrics["rmse_kwh_per_premise"],
            6,
        ),
        "kWh/premise",
    )
    print(
        "Persistence test MSE:",
        round(persistence_test, 6),
    )
    print(
        "MSE improvement:",
        round(
            metrics[
                "mse_improvement_vs_persistence_pct"
            ],
            2,
        ),
        "%",
    )

    print()
    print("Saved:", ARTIFACT_DIR / "model.pt")
    print("Saved:", ARTIFACT_DIR / "scaler.json")
    print("Saved:", ARTIFACT_DIR / "config.json")
    print("Saved:", ARTIFACT_DIR / "metrics.json")


if __name__ == "__main__":
    main()
