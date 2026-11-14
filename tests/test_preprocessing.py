import numpy as np
import pytest

from src.data.preprocess import preprocess_for_cnn, preprocess_image


def test_preprocessing_preserves_contract_shapes_and_ranges():
    image = np.zeros((32, 48, 3), dtype=np.uint8)

    processed = preprocess_image(image, size=(64, 64))
    batch = preprocess_for_cnn(image, size=(64, 64))

    assert processed.shape == (64, 64, 3)
    assert processed.dtype == np.uint8
    assert batch.shape == (1, 64, 64, 3)
    assert batch.dtype == np.float32
    assert 0 <= batch.min() <= batch.max() <= 1


def test_preprocessing_rejects_non_bgr_input():
    with pytest.raises(ValueError, match="three channels"):
        preprocess_image(np.zeros((20, 20), dtype=np.uint8))
