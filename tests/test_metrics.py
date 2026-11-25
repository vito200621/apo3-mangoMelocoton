import numpy as np

from src.evaluation.metrics import full_report


def test_auc_is_undefined_when_test_data_omits_a_configured_class():
    report = full_report(
        y_true=np.array([0, 2, 0, 2]),
        y_pred=np.array([0, 2, 2, 2]),
        y_proba=np.array(
            [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.6, 0.4],
                [0.1, 0.9],
            ]
        ),
        class_names=["Good", "Regular", "Bad"],
        model_name="test",
    )

    assert np.isnan(report["AUC-ROC"])
    assert "does not contain every configured class" in report["AUC-ROC status"]


def test_auc_is_computed_for_complete_multiclass_probabilities():
    report = full_report(
        y_true=np.array([0, 1, 2]),
        y_pred=np.array([0, 1, 2]),
        y_proba=np.eye(3),
        class_names=["Good", "Regular", "Bad"],
        model_name="test",
    )

    assert report["AUC-ROC"] == 1.0
    assert report["AUC-ROC status"] == "defined"
