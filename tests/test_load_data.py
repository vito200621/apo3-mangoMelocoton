import numpy as np
import pytest

from src.data.load_data import (
    load_image,
    load_ml_arrays,
    load_ml_test_arrays,
    load_ml_train_val_arrays,
)


def test_load_image_returns_none_for_a_missing_path(tmp_path):
    assert load_image(tmp_path / "missing.png") is None


@pytest.mark.parametrize("legacy", [False, True])
def test_load_ml_arrays_reads_all_splits(tmp_path, legacy):
    suffix = "ml" if legacy else "_ml"
    expected = {}
    for name in ("Xtrain", "ytrain", "Xval", "yval", "Xtest", "ytest"):
        expected[name] = np.array([1, 2, 3])
        np.save(tmp_path / f"{name}{suffix}.npy", expected[name])

    actual = load_ml_arrays(tmp_path)

    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])


def test_load_ml_arrays_reports_missing_artifacts(tmp_path):
    with pytest.raises(FileNotFoundError, match="Xtrain_ml.npy"):
        load_ml_arrays(tmp_path)


def test_training_array_loader_does_not_require_test_artifacts(tmp_path):
    expected = {}
    for name in ("Xtrain", "ytrain", "Xval", "yval"):
        expected[name] = np.array([1, 2, 3])
        np.save(tmp_path / f"{name}ml.npy", expected[name])

    actual = load_ml_train_val_arrays(tmp_path)

    assert set(actual) == set(expected)
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])


def test_test_array_loader_does_not_require_training_artifacts(tmp_path):
    expected = {"Xtest": np.array([1, 2]), "ytest": np.array([0, 1])}
    for name, value in expected.items():
        np.save(tmp_path / f"{name}ml.npy", value)

    actual = load_ml_test_arrays(tmp_path)

    assert set(actual) == set(expected)
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])
