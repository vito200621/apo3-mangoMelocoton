from src.prediction.predictor import AgroQualityPredictor, ModelBackend


def test_prediction_api_is_in_the_prediction_layer():
    assert AgroQualityPredictor.__module__ == "src.prediction.predictor"
    assert ModelBackend.RANDOM_FOREST.value == "random_forest"
