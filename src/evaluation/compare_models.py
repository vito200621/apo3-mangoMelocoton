from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.evaluation.evaluate_cnn import evaluate_cnn
from src.evaluation.evaluate_random_forest import evaluate_random_forest
from src.evaluation.evaluate_svm import evaluate_svm


def evaluate_all_models(
    processed_dir: str | Path = Path("data/processed"),
    models_dir: str | Path = Path("models"),
    results_path: str | Path = Path("experiments/test_results.csv"),
    detailed_path: str | Path = Path("experiments/detailed_test_results.json"),
) -> pd.DataFrame:
    reports = {
        "random_forest": evaluate_random_forest(processed_dir, models_dir),
        "svm": evaluate_svm(processed_dir, models_dir),
        "cnn": evaluate_cnn(processed_dir, models_dir),
    }
    results = pd.DataFrame([
        {
            key: value
            for key, value in report.items()
            if not isinstance(value, (dict, list))
        }
        for report in reports.values()
    ])

    results_path = Path(results_path)
    detailed_path = Path(detailed_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    detailed_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False)
    with detailed_path.open("w", encoding="utf-8") as file:
        json.dump(reports, file, indent=2, allow_nan=True)
    return results
