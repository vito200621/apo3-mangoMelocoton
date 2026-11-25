import cv2
import numpy as np
import pandas as pd
import pytest

from src.data.size_estimation import add_size_estimations, estimate_size_from_array


def test_size_estimation_uses_normalized_foreground_diameter():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(image, (50, 50), 25, (0, 180, 0), -1)

    estimate = estimate_size_from_array(image)

    assert estimate["size_label"] == "medium"
    assert 0.45 <= estimate["diameter_norm"] <= 0.55


def test_size_estimation_rejects_non_bgr_arrays():
    with pytest.raises(ValueError, match="three channels"):
        estimate_size_from_array(np.zeros((10, 10), dtype=np.uint8))


def test_size_estimation_preserves_a_manual_size_label(tmp_path):
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(image, (50, 50), 25, (0, 180, 0), -1)
    image_path = tmp_path / "fruit.png"
    assert cv2.imwrite(str(image_path), image)
    frame = pd.DataFrame(
        [
            {
                "path": str(image_path),
                "size_label": "large",
                "diameter_norm": np.nan,
            }
        ]
    )

    result = add_size_estimations(frame)

    assert result.iloc[0]["size_label"] == "large"
    assert 0.45 <= result.iloc[0]["diameter_norm"] <= 0.55
