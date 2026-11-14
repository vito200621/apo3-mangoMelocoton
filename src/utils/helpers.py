from __future__ import annotations

from pathlib import Path

import joblib


def save_sklearn_model(model: object, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path
