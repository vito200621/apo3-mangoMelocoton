from __future__ import annotations

from pathlib import Path

from sklearn.model_selection import GridSearchCV

from src.data.load_data import load_ml_arrays
from src.evaluation.metrics import evaluate_predictions
from src.models.random_forest import build_random_forest, get_random_forest_param_grid
from src.utils.config import load_label_info
from src.utils.helpers import save_sklearn_model

DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_MODELS_DIR = Path("models")

def train_random_forest(
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    models_dir: str | Path = DEFAULT_MODELS_DIR,
) -> tuple[object, dict]:
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    label_info = load_label_info(processed_dir)
    class_names = label_info["class_names"]

    arrays = load_ml_arrays(processed_dir)
    Xtrain, ytrain = arrays["Xtrain"], arrays["ytrain"]
    Xval, yval = arrays["Xval"], arrays["yval"]

    rf = build_random_forest()
    rf_grid = GridSearchCV(
        estimator=rf,
        param_grid=get_random_forest_param_grid(),
        cv=3,
        scoring="f1_macro",
        verbose=1,
        n_jobs=-1,
    )
    rf_grid.fit(Xtrain, ytrain)

    best_rf = rf_grid.best_estimator_
    y_pred_val = best_rf.predict(Xval)

    metrics = evaluate_predictions(
        y_true=yval,
        y_pred=y_pred_val,
        class_names=class_names,
        model_name="Random Forest",
        split="Validation",
    )
    metrics["best_params"] = rf_grid.best_params_
    metrics["best_cv_f1_macro"] = float(rf_grid.best_score_)

    save_sklearn_model(best_rf, models_dir / "randomforest.pkl")

    return best_rf, metrics



