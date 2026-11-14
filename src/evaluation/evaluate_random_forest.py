from __future__ import annotations

from pathlib import Path

import joblib

from src.data.load_data import load_ml_arrays
from src.evaluation.metrics import full_report
from src.utils.config import load_label_info


def evaluate_random_forest(
    processed_dir: str | Path = Path("data/processed"),
    models_dir: str | Path = Path("models"),
) -> dict:
    processed_dir = Path(processed_dir)
    models_dir = Path(models_dir)
    model_path = next(
        (
            path
            for path in (
                models_dir / "random_forest.pkl",
                models_dir / "randomforest.pkl",
            )
            if path.exists()
        ),
        None,
    )
    if model_path is None:
        raise FileNotFoundError(f"Random Forest model not found in {models_dir}.")

    label_info = load_label_info(processed_dir)
    arrays = load_ml_arrays(processed_dir)
    model = joblib.load(model_path)
    predictions = model.predict(arrays["Xtest"])
    probabilities = model.predict_proba(arrays["Xtest"])
    return full_report(
        arrays["ytest"], predictions, probabilities,
        label_info["class_names"], "Random Forest",
    )
