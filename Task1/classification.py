#!/usr/bin/env python3
import numpy as np
from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC


class Classification:
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=0.2, random_state=42
        )
        self.x_train_full = self.x_train
        self.y_train_full = self.y_train
        self.x_train, self.x_val, self.y_train, self.y_val = train_test_split(
            self.x_train, self.y_train, test_size=0.2, random_state=42
        )
        self.k = None
        self.iter_svc = None
        self.iter_log = None
        self.n_trees = None

    def general_classify(self, clf) -> None:
        cross_val_scores = cross_val_score(
            clf, self.x_train_full, self.y_train_full, cv=5
        )
        print(
            "cross-val scores: ",
            cross_val_scores,
            " Average: ",
            np.average(cross_val_scores),
        )

        # one validation run
        clf.fit(self.x_train, self.y_train)
        y_pred = clf.predict(self.x_val)
        conf_matrix = metrics.confusion_matrix(self.y_val, y_pred)
        print(conf_matrix)

    def knn_classify(self, k=5) -> None:
        # cross-val using KNN means
        print("KNN classifier:\n -----------------")
        clf = KNeighborsClassifier(k)
        self.k = k
        self.general_classify(clf)

    def logistic_regression(self, max_iter=10000) -> None:
        print("Logistic Regression classifier:\n -----------------")
        clf = LogisticRegression(max_iter=max_iter, random_state=42)
        self.iter_log = max_iter
        self.general_classify(clf)

    def nb_classify(self) -> None:
        print("Naive-Bayes classifier:\n -----------------")
        clf = GaussianNB()
        self.general_classify(clf)

    def random_forest(self, n_trees=200) -> None:
        print("Random Forest classifier:\n -----------------")
        clf = RandomForestClassifier(n_trees, random_state=42)
        self.n_trees = n_trees
        self.general_classify(clf)

    def svm_classify(self, max_iter=10000) -> None:
        print("Linear SVC classifier:\n -----------------")
        clf = LinearSVC(max_iter=max_iter, random_state=42)
        self.iter_svc = max_iter
        self.general_classify(clf)
