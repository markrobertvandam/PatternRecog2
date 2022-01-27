from platform import processor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.semi_supervised import LabelPropagation
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


class SemiSupervised:
    def __init__(self, df: pd.DataFrame, neighbors: int, resdir: str) -> None:
        """
        Initialize baseline and semisupervised learning models for training.

        Arguments:
        df: Dataset containing input features and class labels.
        neighbors: Number of neighbors for Baseline model.
        resdir: Directory for saving plots
        """
        self.df = df
        self.resdir = resdir

        over = SMOTE(sampling_strategy=0.1)
        under = RandomUnderSampler(sampling_strategy=0.5)
        steps = [("o", over), ("u", under)]
        self.pipeline = Pipeline(steps=steps)

        self.scalar = StandardScaler()

        self.baseline_model = KNeighborsClassifier(n_neighbors=neighbors)
        self.semi_supervised = LabelPropagation(kernel="knn", n_neighbors=neighbors)
        self.baseline_2 = KNeighborsClassifier(n_neighbors=neighbors)

    def processData(self):
        """
        Function to preprocess the data.

        Returns:
        X1: Input features.
        y1: Class labels.
        """
        X1 = self.df.iloc[:, 0:-1].values
        y1 = self.df.iloc[:, -1].values

        return X1, y1

    def balanceData(self, X, y, plot = False, f1 = 10, f2 = 10):
        """
        Function to balance the data with data sampling.

        Arguments:
        X: Input features.
        y: Class labels.
        plot: Visualize balanced data.
        f1: First feature for plot.
        f2: Second feature for plot.

        Returns:
        X_scale: Balanced and scaled input features.
        y_bal: Balanced class labels.
        """
        X_bal, y_bal = self.pipeline.fit_resample(X, y)
        X_scale = self.scalar.fit_transform(X_bal)

        if plot:
            counter1 = Counter(y)
            counter2 = Counter(y_bal)

            f1 = 10
            f2 = 20
            fig, ax = plt.subplots(1, 2, figsize = (19, 9))
            for label, _ in counter1.items():
                row_ix = np.where(y == label)[0]
                ax[0].scatter(X[row_ix, f1], X[row_ix, f2], label=f"Class {str(label)}")

            ax[0].set_title("Original data")
            ax[0].set_xlabel(f"Feature {f1}")
            ax[0].set_ylabel(f"Feature {f2}")
            ax[0].legend()

            for label, _ in counter2.items():
                row_ix = np.where(y_bal == label)[0]
                ax[1].scatter(X_bal[row_ix, f1], X_bal[row_ix, f2], label=f"Class {str(label)}")

            ax[1].legend()
            ax[1].set_title("After Sampling")
            ax[0].set_xlabel(f"Feature {f1}")
            ax[0].set_ylabel(f"Feature {f2}")
            plt.legend()
            fig.savefig(f"{self.resdir}/sampling.png", bbox_inches = "tight")

        return X_scale, y_bal

    def splitData(self, X, y):
        """
        Function to split the data in stratified manner.

        Arguments:
        X: Input features.
        y: Class labels.

        Returns:
        baseline_data: Train and test data from initial split.
        semi_supervised_data: Labelled and unlabelled data
                              for semi-supervised learning.
        """
        X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(
            X, y, test_size=0.2, stratify=y
        )
        X_TRAIN_UNLAB, X_TRAIN_LAB, Y_TRAIN_UNLAB, Y_TRAIN_LAB = train_test_split(
            X_TRAIN, Y_TRAIN, test_size=0.3, stratify=Y_TRAIN
        )

        baseline_data = (X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
        semi_supervised_data = (X_TRAIN_UNLAB, X_TRAIN_LAB, Y_TRAIN_UNLAB, Y_TRAIN_LAB)

        return baseline_data, semi_supervised_data

    def task2(self, X, y, X_test, y_test):
        """
        Function to run task 2 (Training baseline model).

        Arguments:
        X: Input train features.
        y: Train class labels.
        X_test: Input test features.
        y_test: Test class labels.

        Returns:
        score: F-score of baseline model
        """
        self.baseline_model.fit(X, y)
        pred = self.baseline_model.predict(X_test)
        score = f1_score(y_test, pred)

        return score

    def task3(self, train_data, X_test, y_test):
        """
        Function to run task 3 (Training semi-supervised model).

        Arguments:
        train_data: list containing unlabelled and labelled train data.
        X_test: Input test features.
        y_test: Test class labels.

        Returns:
        score: F-score of semi-supervised model.
        X: Labeled and unlabeled input features.
        y_transduction: Class labels obtained using transduction.
        """
        X_TRAIN_UNLAB, X_TRAIN_LAB, Y_TRAIN_UNLAB, Y_TRAIN_LAB = train_data

        Y_TRAIN_UNLAB[:] = -1
        X = np.concatenate((X_TRAIN_LAB, X_TRAIN_UNLAB))
        y = np.concatenate((Y_TRAIN_LAB, Y_TRAIN_UNLAB))

        self.semi_supervised.fit(X, y)
        pred = self.semi_supervised.predict(X_test)
        score = f1_score(y_test, pred)

        y_transduction = self.semi_supervised.transduction_

        return score, X, y_transduction

    def task4(self, X, y, X_test, y_test):
        """
        Function to run task 4 (Training baseline with transduction).

        Arguments:
        X: Input train data.
        y: Train class labels.
        X_test: Input test data.
        y_test: Test class labels.

        Returns:
        score: F-score of baseline model with semi-supervised labels.
        """
        self.baseline_2.fit(X, y)
        pred = self.baseline_2.predict(X_test)
        score = f1_score(y_test, pred)

        return score

    def getClassRatio(self, y):
        """
        Returns class ratio.

        Arguments:
        y: Class labels
        """
        res = []

        for x in y:
            c = Counter(x)
            res.append(c[0] / c[1])

        print(f"Train ratio: {res[0]} \nTest ratio: {res[1]}")

    def run(self, epochs=100):
        """
        Function to run the main experiment.

        Arguments:
        epochs: Number of runs for performing experiment.

        Returns:
        s1, s2, s3: F-scores of Task 2, 3 & 4 over number of experiments.
        """
        s1 = []
        s2 = []
        s3 = []

        for i in range(epochs):
            print("------------")
            print(f"Run: {i + 1}")

            X, y = self.processData()

            if i == 0:
                X_bal, y_bal = self.balanceData(X, y, plot = True)
            
            else:
                X_bal, y_bal = self.balanceData(X, y)

            baseline_data, semi_supervised_data = self.splitData(X_bal, y_bal)
            (X_TRAIN, X_TEST, Y_TRAIN, Y_TEST) = baseline_data
            (
                X_TRAIN_UNLAB,
                X_TRAIN_LAB,
                Y_TRAIN_UNLAB,
                Y_TRAIN_LAB,
            ) = semi_supervised_data

            self.getClassRatio([Y_TRAIN, Y_TEST])

            score = self.task2(X_TRAIN_LAB, Y_TRAIN_LAB, X_TEST, Y_TEST)
            print(f"Baseline score: {score}")
            s1.append(score)

            train_data = [X_TRAIN_UNLAB, X_TRAIN_LAB, Y_TRAIN_UNLAB, Y_TRAIN_LAB]
            score, X_transduction, y_transduction = self.task3(
                train_data, X_TEST, Y_TEST
            )
            print(f"Semi-supervised score: {score}")
            s2.append(score)

            score = self.task4(X_transduction, y_transduction, X_TEST, Y_TEST)
            print(f"Baseline with labels score: {score}")
            s3.append(score)

        return s1, s2, s3
