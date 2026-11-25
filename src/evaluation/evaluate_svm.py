from __future__ import annotations

from pathlib import Path

import joblib

from src.data.load_data import load_ml_test_arrays
from src.evaluation.metrics import full_report
from src.utils.config import load_label_info


def evaluate_svm(
    processed_dir: str | Path = Path("data/processed"),
    models_dir: str | Path = Path("models"),
) -> dict:
    processed_dir = Path(processed_dir)
    models_dir = Path(models_dir)
    model_path = models_dir / "svm.pkl"
    scaler_path = models_dir / "scaler.pkl"
    for path in (model_path, scaler_path):
        if not path.exists():
            raise FileNotFoundError(f"Required SVM artifact not found: {path}")

    label_info = load_label_info(processed_dir)
    arrays = load_ml_test_arrays(processed_dir)
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    features = scaler.transform(arrays["Xtest"])
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)
    return full_report(
        arrays["ytest"], predictions, probabilities,
        label_info["class_names"], "SVM",
    )
