"""Entry point for the Assistive Vision System.

Usage:
    python -m src.main
    python -m src.main --config path/to/config.yaml
"""

import argparse
import sys

from src.config import load_config
from src.pipeline import Pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Assistive Vision System - V1"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file (default: config.yaml)",
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Create and run pipeline (uses placeholder components for now)
    pipeline = Pipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()