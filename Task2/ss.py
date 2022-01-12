from platform import processor
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score
from sklearn.semi_supervised import LabelPropagation
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42


class SemiSupervised:
    def __init__(self, df: pd.DataFrame, neighbors: int) -> None:
        self.df = df

        over = SMOTE(sampling_strategy=0.1)
        under = RandomUnderSampler(sampling_strategy=0.5)
        steps = [("o", over), ("u", under)]
        self.pipeline = Pipeline(steps=steps)

        self.scalar = StandardScaler()

        self.baseline_model = KNeighborsClassifier(n_neighbors=neighbors)
        self.semi_supervised = LabelPropagation(kernel="knn", n_neighbors=neighbors)
        self.baseline_2 = KNeighborsClassifier(n_neighbors=neighbors)

    def processData(self):
        X1 = self.df.iloc[:, 0:-1].values
        y1 = self.df.iloc[:, -1].values

        return X1, y1

    def balanceData(self, X, y):
        X_bal, y_bal = self.pipeline.fit_resample(X, y)
        X_scale = self.scalar.fit_transform(X_bal)

        return X_scale, y_bal

    def splitData(self, X, y):
        X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(
            X, y, test_size=0.2, stratify=y
        )
        X_TRAIN_UNLAB, X_TRAIN_LAB, Y_TRAIN_UNLAB, Y_TRAIN_LAB = train_test_split(
            X_TRAIN, Y_TRAIN, test_size=0.3, stratify=Y_TRAIN
        )

        baseline_data = (X_TRAIN, X_TEST, Y_TRAIN, Y_TEST)
        semi_supervised_data = (X_TRAIN_UNLAB, X_TRAIN_LAB, Y_TRAIN_UNLAB, Y_TRAIN_LAB)

        return baseline_data, semi_supervised_data

    def task1(self, X, y, X_test, y_test):
        self.baseline_model.fit(X, y)
        pred = self.baseline_model.predict(X_test)
        score = f1_score(y_test, pred)

        return score

    def task2(self, train_data, X_test, y_test):
        X_TRAIN_UNLAB, X_TRAIN_LAB, Y_TRAIN_UNLAB, Y_TRAIN_LAB = train_data

        Y_TRAIN_UNLAB[:] = -1
        X = np.concatenate((X_TRAIN_LAB, X_TRAIN_UNLAB))
        y = np.concatenate((Y_TRAIN_LAB, Y_TRAIN_UNLAB))

        self.semi_supervised.fit(X, y)
        pred = self.semi_supervised.predict(X_test)
        score = f1_score(y_test, pred)

        y_transduction = self.semi_supervised.transduction_

        return score, X, y_transduction

    def task3(self, X, y, X_test, y_test):
        self.baseline_2.fit(X, y)
        pred = self.baseline_2.predict(X_test)
        score = f1_score(y_test, pred)

        return score

    def getClassRatio(self, y):
        res = []

        for x in y:
            c = Counter(x)
            res.append(c[0] / c[1])

        print(f"Train ratio: {res[0]} \nTest ratio: {res[1]}")

    def run(self, epochs=100):
        s1 = []
        s2 = []
        s3 = []

        for i in range(epochs):
            print("------------")
            print(f"Run: {i + 1}")

            X, y = self.processData()
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

            score = self.task1(X_TRAIN_LAB, Y_TRAIN_LAB, X_TEST, Y_TEST)
            print(f"Baseline score: {score}")
            s1.append(score)

            train_data = [X_TRAIN_UNLAB, X_TRAIN_LAB, Y_TRAIN_UNLAB, Y_TRAIN_LAB]
            score, X_transduction, y_transduction = self.task2(
                train_data, X_TEST, Y_TEST
            )
            print(f"Semi-supervised score: {score}")
            s2.append(score)

            score = self.task3(X_transduction, y_transduction, X_TEST, Y_TEST)
            print(f"Baseline with labels score: {score}")
            s3.append(score)

        return s1, s2, s3
