#!/usr/bin/env python3

import argparse
import numpy as np

from cats import Cats
from genes import Genes


def parse_args() -> argparse.Namespace:
    """
    Function to parse arguments.

    Returns:
    parser: Argument parser containing arguments.
    """

    parser = argparse.ArgumentParser(description="Run the Pattern Recognition Pipeline")
    parser.add_argument("pipeline", help="options are cats and genes")
    parser.add_argument(
        "command",
        help="options are tune, test, cross-val, ensemble (for classification) and cluster (for clustering)",
    )

    return parser.parse_args()


def genes_pipeline(args: argparse.Namespace) -> None:
    """
    Pipline for performing clustering/classification on
    gene dataset based on specified model operation.

    Arguments:
    args: Parser object containing necessary arguments.
    """

    genes = Genes()
    genes.load_data()
    genes.feature_extraction()
    genes.visualize_data()
    if args.command == "cross-val":
        genes.classification(command="cross-val")
    elif args.command == "tune":
        genes.tune_classification_params()
    elif args.command == "tune-big":
        genes.tune_classification_params("big")
    elif args.command == "test":
        genes.classification(command="test")
    elif args.command == "cluster":
        genes.clustering()
    elif args.command == "full-run":
        genes.classification(command="test")
        genes.clustering()


def cats_pipeline(args: argparse.Namespace) -> None:
    """
    Pipline for performing clustering/classification on
    cats dataset based on specified model operation.

    Arguments:
    args: Parser object containing necessary arguments.
    """

    cats = Cats()
    cats.load_data()
    if args.command == "augment":
        cats.augmented_run()
    else:
        cats.feature_extraction()
        cats.visualize_data()
        if args.command == "cross-val":
            cats.classification(command="cross-val")
        elif args.command == "tune":
            cats.tune_classification_params()
        elif args.command == "tune-big":
            cats.tune_classification_params("big")
        elif args.command == "test":
            cats.classification(command="test")
        elif args.command == "ensemble":
            cats.ensemble()
        elif args.command == "cluster":
            cats.clustering()
        elif args.command == "full-run":
            cats.classification(command="test")
            cats.ensemble()
            cats.clustering()


def main():
    """
    Main function for choosing pipeline
    """

    args = parse_args()
    if args.pipeline == "genes":
        genes_pipeline(args)
    elif args.pipeline == "cats":
        cats_pipeline(args)


if __name__ == "__main__":
    main()
