#!/usr/bin/env python3

import argparse
from cats import Cats
from genes import Genes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Pattern Recognition Pipeline")
    parser.add_argument("command")

    return parser.parse_args()


def genes_pipeline(args: argparse.Namespace) -> None:
    print(args)
    print("genes")
    genes = Genes()
    genes.load_data()


def cats_pipeline(args: argparse.Namespace) -> None:
    print(args)
    print("cats")
    cats = Cats()
    cats.load_data()
    cats.feature_extraction()
    cats.classification()
    cats.clustering()


def main():

    args = parse_args()
    print(args)
    if args.command == "genes":
        genes_pipeline(args)
    elif args.command == "cats":
        cats_pipeline(args)


if __name__ == "__main__":
    main()
