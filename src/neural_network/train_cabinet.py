"""Training pipeline for the cabinet model (Etapa 5.2)."""

from __future__ import annotations

from pathlib import Path
import json
import sys
from typing import Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import tensorflow as tf

from src.neural_network.model import build_model


DATA_DIR = Path("data") / "cabinets"
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
MODEL_PATH = MODELS_DIR / "cabinet_model.h5"
HISTORY_PATH = RESULTS_DIR / "cabinet_training_history.csv"
METRICS_PATH = RESULTS_DIR / "cabinet_training_metrics.json"


def load_datasets() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load train and validation datasets from CSV files."""
    x_train = pd.read_csv(DATA_DIR / "train" / "X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "train" / "y_train.csv").squeeze()
    x_val = pd.read_csv(DATA_DIR / "validation" / "X_val.csv")
    y_val = pd.read_csv(DATA_DIR / "validation" / "y_val.csv").squeeze()
    return x_train, y_train, x_val, y_val


def train(epochs: int = 10, batch_size: int = 32) -> None:
    """Train the cabinet model and save outputs."""
    x_train, y_train, x_val, y_val = load_datasets()

    input_dim = x_train.shape[1]
    num_classes = int(pd.Series(y_train).nunique())

    model = build_model(input_dim=input_dim, num_classes=num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model.save(MODEL_PATH)

    history_df = pd.DataFrame(history.history)
    history_df.to_csv(HISTORY_PATH, index=False)

    metrics = {
        "train_accuracy": history.history.get("accuracy", [None])[-1],
        "train_loss": history.history.get("loss", [None])[-1],
        "val_accuracy": history.history.get("val_accuracy", [None])[-1],
        "val_loss": history.history.get("val_loss", [None])[-1],
    }
    with METRICS_PATH.open("w", encoding="utf-8") as file_handle:
        json.dump(metrics, file_handle, indent=2)

    print(f"Final train accuracy: {metrics['train_accuracy']:.4f}")
    print(f"Final validation accuracy: {metrics['val_accuracy']:.4f}")


if __name__ == "__main__":
    train()
