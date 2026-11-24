
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_image(path: str | Path) -> np.ndarray | None:
    image = cv2.imread(str(path))
    return image if image is not None and image.size else None


def load_ml_arrays(
    processed_dir: str | Path = Path("data/processed"),
) -> dict[str, np.ndarray]:
    return _load_ml_arrays(
        processed_dir,
        ("Xtrain", "ytrain", "Xval", "yval", "Xtest", "ytest"),
    )


def load_ml_train_val_arrays(
    processed_dir: str | Path = Path("data/processed"),
) -> dict[str, np.ndarray]:
    return _load_ml_arrays(processed_dir, ("Xtrain", "ytrain", "Xval", "yval"))


def load_ml_test_arrays(
    processed_dir: str | Path = Path("data/processed"),
) -> dict[str, np.ndarray]:
    return _load_ml_arrays(processed_dir, ("Xtest", "ytest"))


def _load_ml_arrays(
    processed_dir: str | Path, keys: tuple[str, ...]
) -> dict[str, np.ndarray]:
    processed_dir = Path(processed_dir)
    names = {
        "Xtrain": ("Xtrain_ml.npy", "Xtrainml.npy"),
        "ytrain": ("ytrain_ml.npy", "ytrainml.npy"),
        "Xval": ("Xval_ml.npy", "Xvalml.npy"),
        "yval": ("yval_ml.npy", "yvalml.npy"),
        "Xtest": ("Xtest_ml.npy", "Xtestml.npy"),
        "ytest": ("ytest_ml.npy", "ytestml.npy"),
    }
    arrays = {}

    for key in keys:
        canonical, legacy = names[key]
        path = processed_dir / canonical
        if not path.exists():
            path = processed_dir / legacy
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {canonical} and legacy fallback {legacy}. "
                "Run the EDA/feature extraction pipeline first."
            )
        arrays[key] = np.load(path)

    return arrays
