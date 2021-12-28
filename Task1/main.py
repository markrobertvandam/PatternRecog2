#!/usr/bin/env python3

import argparse
import numpy as np

from cats import Cats
from genes import Genes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Pattern Recognition Pipeline")
    parser.add_argument("command")

    return parser.parse_args()


def genes_pipeline(args: argparse.Namespace) -> None:
    genes = Genes(0.98, 0.2)
    genes.load_data()
    genes.feature_extraction()
    genes.classification()


def cats_pipeline(args: argparse.Namespace) -> None:
    cats = Cats()
    cats.load_data()
    cats.feature_extraction()

    # Classify original dataset
    print("Original performance: \n")
    old_shape = cats.images.shape
    flattened_original = cats.images.reshape(
        old_shape[0], old_shape[1] * old_shape[2] * old_shape[3]
    )
    cats.classification(flattened_original)
    print("--------------")

    # Classify sift dataset
    print("Sift performance: \n")
    cats.classification(cats.sift_data)
    print("--------------\n")

    # Classify fourier dataset
    print("Fourier performance: \n")
    cats.classification(cats.fourier_data)
    print("--------------")


def main():

    args = parse_args()
    if args.command == "genes":
        genes_pipeline(args)
    elif args.command == "cats":
        cats_pipeline(args)


if __name__ == "__main__":
    main()
