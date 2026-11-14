from sklearn.ensemble import RandomForestClassifier

from src.models.random_forest import build_random_forest, get_random_forest_param_grid


def test_random_forest_defaults_are_reproducible_and_balanced():
    model = build_random_forest()

    assert isinstance(model, RandomForestClassifier)
    assert model.class_weight == "balanced"
    assert model.random_state == 42
    assert model.n_jobs == -1
    assert set(get_random_forest_param_grid()) == {
        "n_estimators",
        "max_depth",
        "min_samples_split",
    }
