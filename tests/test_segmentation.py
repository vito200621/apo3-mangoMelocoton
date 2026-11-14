import cv2
import numpy as np

from src.data.segmentation import segment_fruits_from_image


def test_segmentation_writes_a_crop_for_a_synthetic_foreground(tmp_path):
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(image, (50, 50), 25, (0, 180, 0), -1)
    source = tmp_path / "fruit.png"
    assert cv2.imwrite(str(source), image)

    records = segment_fruits_from_image(source, tmp_path / "crops", "sample")

    assert len(records) == 1
    crop = cv2.imread(str(tmp_path / "crops" / "sample_crop_01.png"))
    assert crop is not None
    assert crop.shape[0] < image.shape[0]
    assert records[0]["source_image_id"] == "sample"
