from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    model_name: str,
    split: str = "Validation",
) -> dict:
    labels = list(range(len(class_names)))
    return {
        "model": model_name,
        "split": split,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        ),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def full_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_names: list[str],
    model_name: str,
) -> dict:
    labels = list(range(len(class_names)))
    observed_labels = np.unique(y_true)
    if len(observed_labels) != len(labels):
        auc = float("nan")
        auc_status = (
            "undefined: the test split does not contain every configured class"
        )
    elif y_proba.ndim != 2 or y_proba.shape != (len(y_true), len(labels)):
        auc = float("nan")
        auc_status = (
            "undefined: probability columns do not match the configured classes"
        )
    elif not np.isfinite(y_proba).all() or not np.allclose(
        y_proba.sum(axis=1), 1.0, atol=1e-5
    ):
        auc = float("nan")
        auc_status = "undefined: probabilities are non-finite or not normalized"
    elif len(labels) == 2:
        auc = float(roc_auc_score(y_true, y_proba[:, 1]))
        auc_status = "defined"
    else:
        auc = float(
            roc_auc_score(
                y_true,
                y_proba,
                labels=labels,
                multi_class="ovr",
                average="macro",
            )
        )
        auc_status = "defined"

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = matrix.sum(axis=1, keepdims=True)
    matrix_pct = np.divide(
        matrix.astype(float),
        row_sums,
        out=np.zeros_like(matrix, dtype=float),
        where=row_sums != 0,
    ) * 100
    return {
        "Model": model_name,
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Balanced Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "Macro F1": float(
            f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        ),
        "Macro Precision": float(
            precision_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        ),
        "Macro Recall": float(
            recall_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
        ),
        "AUC-ROC": auc,
        "AUC-ROC status": auc_status,
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_pct": matrix_pct.tolist(),
    }
