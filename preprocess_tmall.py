import argparse
import yaml
from data.preprocessor import TmallPreprocessor


def main():
    parser = argparse.ArgumentParser(description="Preprocess Tmall dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to raw Tmall data file (CSV)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/tmall/",
        help="Output directory for preprocessed data",
    )

    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Create preprocessor
    preprocessor = TmallPreprocessor(config)

    # Process data
    print(f"Processing Tmall data from {args.input}")
    print(f"Output directory: {args.output}")

    train_data, val_data, test_data = preprocessor.process(args.input, args.output)

    print("\nPreprocessing completed successfully!")


if __name__ == "__main__":
    main()
