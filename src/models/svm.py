from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


def build_svm(random_state: int = 42, probability: bool = True) -> SVC:
    return SVC(
        class_weight="balanced",
        random_state=random_state,
        probability=probability,
    )


def get_svm_param_grid() -> dict:
    return {
        "svm__C": [0.1, 1, 10],
        "svm__kernel": ["rbf", "linear"],
        "svm__gamma": ["scale", "auto"],
    }


def build_scaler() -> StandardScaler:
    return StandardScaler()


def build_svm_pipeline(random_state: int = 42) -> Pipeline:
    return Pipeline([
        ("scaler", build_scaler()),
        ("svm", build_svm(random_state=random_state)),
    ])
