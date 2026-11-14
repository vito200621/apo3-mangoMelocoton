from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.models.svm import build_svm, build_svm_pipeline, get_svm_param_grid


def test_svm_pipeline_keeps_scaling_inside_cross_validation():
    pipeline = build_svm_pipeline()

    assert isinstance(pipeline, Pipeline)
    assert isinstance(pipeline.named_steps["scaler"], StandardScaler)
    assert isinstance(pipeline.named_steps["svm"], SVC)
    assert set(get_svm_param_grid()) == {
        "svm__C",
        "svm__kernel",
        "svm__gamma",
    }


def test_svm_defaults_are_reproducible_and_balanced():
    model = build_svm()

    assert model.class_weight == "balanced"
    assert model.random_state == 42
    assert model.probability
