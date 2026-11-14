from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.training.train_cnn import train_cnn
from src.training.train_random_forest import train_random_forest
from src.training.train_svm import train_svm

DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_MODELS_DIR = Path("models")
DEFAULT_EXPERIMENTS_DIR = Path("experiments")


def train_all_models(
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
    output_summary_path: str | Path = DEFAULT_EXPERIMENTS_DIR / "model_comparison_ml.csv",
) -> pd.DataFrame:
    _, rf_metrics = train_random_forest(processed_dir=processed_dir, models_dir=models_dir)
    _, _, svm_metrics = train_svm(processed_dir=processed_dir, models_dir=models_dir)
    _, cnn_metrics = train_cnn(processed_dir=processed_dir, models_dir=models_dir)

    summary = pd.DataFrame([
        {
            "Model": rf_metrics["model"],
            "Validation Accuracy": rf_metrics["accuracy"],
            "Validation Macro F1": rf_metrics["macro_f1"],
        },
        {
            "Model": svm_metrics["model"],
            "Validation Accuracy": svm_metrics["accuracy"],
            "Validation Macro F1": svm_metrics["macro_f1"],
        },
        {
            "Model": cnn_metrics["model"],
            "Validation Accuracy": cnn_metrics["accuracy"],
            "Validation Macro F1": cnn_metrics["macro_f1"],
        },
    ])

    output_summary_path = Path(output_summary_path)
    output_summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_summary_path, index=False)

    full_metrics_path = output_summary_path.parent / "training_metrics.json"
    with open(full_metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "random_forest": rf_metrics,
                "svm": svm_metrics,
                "cnn": cnn_metrics,
            },
            f,
            indent=2,
        )

    return summary

