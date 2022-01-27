import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ss import SemiSupervised


def parse_args() -> argparse.Namespace:
    """
    Argument parser

    Returns:
    parser: Parser with input arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("data", help="Folder of creditcard.csv")
    parser.add_argument("res", help="Folder where the plots will be saved")

    return parser.parse_args()


def plot(args, scores: list, type: str) -> None:
    """
    Plots the F-score.

    Arguments:
    args: Parser containing result directory.
    scores: List of scores over various runs.
    type: Type of plot. Options: "line", "box"
    """
    titles = [
        "Baseline model (KNN): Epochs vs F1 score",
        "Semi-supervised model (LinearPropogation): Epochs vs F1 score",
        "Baseline model (KNN) training with Semi-supervised model (LinearPropogation) labels: Epochs vs F1 score",
    ]

    if type == "line":
        for i in range(len(scores)):
            fig, ax = plt.subplots(figsize=(15, 8))
            ax.plot(scores[i], c="r")
            ax.set_title(titles[i])
            ax.set_xlabel("Epochs")
            ax.set_ylabel("F1 score")

            fig.savefig(f"{args.res}/q{i}.png", bbox_inches="tight")

    elif type == "box":
        labels = ["Knn", "SS", "Knn+SS"]
        fig, ax = plt.subplots(figsize=(16, 15))

        plt.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

        bp = ax.boxplot(
            x=scores,
            vert=True,
            whis=0.5,
            labels=labels,
            manage_ticks=True,
            showmeans=True,
        )
        plt.setp(bp["boxes"], color="red")
        plt.setp(bp["whiskers"], color="black")
        plt.setp(bp["means"], marker="o")
        plt.setp(bp["fliers"], marker=".")

        ax.set_ylabel("F score")
        ax.set_xlabel("Models")
        ax.grid(False)

        fig.savefig(f"{args.res}/boxplot.png", bbox_inches="tight")


def main():
    """
    Main function to run the pipeline.
    """
    args = parse_args()
    df = pd.read_csv(f"{args.data}/creditcard.csv")
    df = df.drop(["Time", "Amount"], axis=1)

    ss = SemiSupervised(df, neighbors=3, resdir=args.res)

    s1, s2, s3 = ss.run()

    print("================\n")

    print(f"Average q2 score: {np.mean(s1)}")
    print(f"Average q3 score: {np.mean(s2)}")
    print(f"Average q4 score: {np.mean(s3)}")

    print("Plotting and Saving the results...")

    plots = ["line", "box"]

    for i in range(len(plots)):
        plot(args, [s1, s2, s3], plots[i])


if __name__ == "__main__":
    main()
