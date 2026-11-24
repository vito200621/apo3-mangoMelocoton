from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def build_group_cv_splits(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 3,
    random_state: int = 42,
) -> list[tuple[np.ndarray, np.ndarray]]:
    y = np.asarray(y)
    groups = np.asarray(groups)
    if len(y) != len(groups):
        raise ValueError("Labels and group identifiers must have equal lengths.")

    classes = np.unique(y)
    group_counts = {
        int(label): int(np.unique(groups[y == label]).size) for label in classes
    }
    insufficient_groups = {
        label: count for label, count in group_counts.items() if count < n_splits
    }
    if insufficient_groups:
        raise ValueError(
            f"Group-aware {n_splits}-fold CV needs at least {n_splits} distinct "
            f"groups per class; found {insufficient_groups}."
        )

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    splits = list(splitter.split(np.zeros(len(y)), y, groups))
    expected_classes = set(classes.tolist())
    for fold_number, (train_indices, validation_indices) in enumerate(splits, start=1):
        if set(y[train_indices].tolist()) != expected_classes:
            raise ValueError(f"Training fold {fold_number} is missing a class.")
        if set(y[validation_indices].tolist()) != expected_classes:
            raise ValueError(f"Validation fold {fold_number} is missing a class.")
        overlap = set(groups[train_indices]) & set(groups[validation_indices])
        if overlap:
            raise ValueError(f"Group leakage detected in fold {fold_number}.")
    return splits
