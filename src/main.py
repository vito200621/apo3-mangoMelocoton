from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.data.load_data import load_image
from src.data.prepare_data import run_eda_pipeline
from src.evaluation.compare_models import evaluate_all_models
from src.prediction.predictor import AgroQualityPredictor, ModelBackend
from src.training.train_models import train_all_models

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
EXPERIMENTS_DIR = ROOT / "experiments"

def run_training_pipeline() -> None:
    print(train_all_models(processed_dir=PROCESSED_DIR, models_dir=MODELS_DIR))


def run_evaluation_pipeline() -> pd.DataFrame:
    print("=" * 80)
    print("STEP 4/4 - FINAL EVALUATION")
    print("=" * 80)
    results = evaluate_all_models(
        processed_dir=PROCESSED_DIR,
        models_dir=MODELS_DIR,
        results_path=EXPERIMENTS_DIR / "test_results.csv",
        detailed_path=EXPERIMENTS_DIR / "detailed_test_results.json",
    )
    print(results)
    return results


def run_full_pipeline(reset_processed: bool = False) -> None:
    run_eda_pipeline(reset_processed=reset_processed)
    run_training_pipeline()
    run_evaluation_pipeline()
    print("=" * 80)
    print("FULL PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 80)


def run_predict_demo(image_path: str, backend: str = "random_forest") -> None:
    backend_map = {
        "random_forest": ModelBackend.RANDOM_FOREST,
        "svm": ModelBackend.SVM,
        "cnn": ModelBackend.CNN,
    }

    predictor = AgroQualityPredictor(backend=backend_map[backend])
    img = load_image(image_path)

    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    result = predictor.predict_frame(img)

    if result is None:
        print("Prediction failed.")
        return

    print("Prediction result")
    print(f"Quality: {result.class_name}")
    print(f"Confidence: {result.confidence:.4f}")
    print(f"Size: {result.size_label}")
    print(f"Diameter norm: {result.diameter_norm}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="AgroQuality pipeline CLI. Without arguments, runs the full pipeline."
    )

    subparsers = parser.add_subparsers(dest="command")

    predict_parser = subparsers.add_parser(
        "predict", help="Run prediction on one image"
    )
    predict_parser.add_argument("--image", required=True, help="Path to input image")
    predict_parser.add_argument(
        "--backend",
        choices=["random_forest", "svm", "cnn"],
        default="random_forest",
    )

    train_parser = subparsers.add_parser("train", help="Run training only")
    train_parser.add_argument("--skip-eda", action="store_true")
    train_parser.add_argument("--reset-processed", action="store_true")

    subparsers.add_parser("evaluate", help="Run evaluation only")
    eda_parser = subparsers.add_parser("eda", help="Run EDA/preparation only")
    eda_parser.add_argument("--reset-processed", action="store_true")
    all_parser = subparsers.add_parser("all", help="Run the full pipeline")
    all_parser.add_argument("--reset-processed", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        run_full_pipeline()
        return

    if args.command == "all":
        run_full_pipeline(reset_processed=args.reset_processed)
    elif args.command == "eda":
        run_eda_pipeline(reset_processed=args.reset_processed)

    elif args.command == "train":
        if not args.skip_eda:
            run_eda_pipeline(reset_processed=args.reset_processed)
        run_training_pipeline()

    elif args.command == "evaluate":
        run_evaluation_pipeline()

    elif args.command == "predict":
        run_predict_demo(args.image, args.backend)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()




