#!/usr/bin/env python3

import argparse
import numpy as np

from cats import Cats
from genes import Genes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Pattern Recognition Pipeline")
    parser.add_argument("pipeline", help="options are cats and genes")
    parser.add_argument(
        "command",
        help="options are tune, test, cross-val (for classification) and cluster (for clustering)",
    )

    return parser.parse_args()


def genes_pipeline(args: argparse.Namespace) -> None:
    genes = Genes()
    genes.load_data()
    genes.feature_extraction()
    genes.visualize_data()
    if args.command == "cross-val":
        genes.cross_val()
    elif args.command == "cluster":
        genes.clustering()
    elif args.command == "ensemble":
        genes.ensemble()
    else:
        genes.classification(command=args.command)


def cats_pipeline(args: argparse.Namespace) -> None:
    cats = Cats()
    cats.load_data()
    cats.feature_extraction()
    cats.visualize_data()
    if args.command == "cross-val":
        cats.cross_val()
    elif args.command == "cluster":
        cats.clustering()
    else:
        cats.classification(command=args.command)


def main():

    args = parse_args()
    if args.pipeline == "genes":
        genes_pipeline(args)
    elif args.pipeline == "cats":
        cats_pipeline(args)


if __name__ == "__main__":
    main()
