from src.models.cnn import build_cnn


def test_cnn_matches_the_configured_image_and_class_dimensions():
    model = build_cnn(input_shape=(64, 64, 3), n_classes=4)

    assert model.input_shape == (None, 64, 64, 3)
    assert model.output_shape == (None, 4)
    assert model.loss == "sparse_categorical_crossentropy"
