from __future__ import annotations

import json
from pathlib import Path


def load_label_info(processed_dir: str | Path = Path("data/processed")) -> dict:
    path = Path(processed_dir) / "labelinfo.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the EDA/preprocessing pipeline first."
        )

    with path.open(encoding="utf-8") as file:
        label_info = json.load(file)

    required = {"label_map", "inverse_map", "n_classes", "class_names", "img_size"}
    missing = required - label_info.keys()
    if missing:
        raise ValueError(f"{path.name} is missing required keys: {sorted(missing)}")
    return label_info


def load_class_weights(processed_dir: str | Path = Path("data/processed")) -> dict[int, float]:
    path = Path(processed_dir) / "classweights.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the EDA/preprocessing pipeline first."
        )

    with path.open(encoding="utf-8") as file:
        return {int(label): float(weight) for label, weight in json.load(file).items()}
