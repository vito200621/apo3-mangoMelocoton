import numpy as np
import pytest

from src.data.load_data import load_image, load_ml_arrays


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
