import argparse
import yaml
from data.preprocessor import YoochoosePreprocessor # Import the new preprocessor

def main():
    parser = argparse.ArgumentParser(description="Preprocess Yoochoose dataset")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_yoochoose.yaml", # Point to a new config file
        help="Path to configuration file for Yoochoose",
    )
    parser.add_argument(
        "--input", 
        type=str, 
        required=True, 
        help="Path to raw Yoochoose data file (yoochoose-clicks.dat)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="datasets/yoochoose/",
        help="Output directory for preprocessed Yoochoose data",
    )

    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Create preprocessor instance for Yoochoose
    preprocessor = YoochoosePreprocessor(config)

    # Process the data
    print(f"Processing Yoochoose data from {args.input}")
    print(f"Output directory: {args.output}")

    preprocessor.process(args.input, args.output)

    print("\nPreprocessing for Yoochoose completed successfully!")


if __name__ == "__main__":
    main()
