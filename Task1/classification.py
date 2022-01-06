#!/usr/bin/env python3
import numpy as np
from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC


class Classification:
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x, self.y = x, y
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

        self.models = []
        self.max_cv_score = 0
        self.best_model = 0

    def evaluate(self, y_true, y_pred):
        conf_matrix = metrics.confusion_matrix(y_true, y_pred)
        print(conf_matrix)
        print(
            f"F1-score: {f1_score(y_true, y_pred, average='macro')}, Accuracy: {accuracy_score(y_true, y_pred)}"
        )

    def grid_search(self, clf) -> None:
        # one validation run
        clf.fit(self.x_train, self.y_train)
        y_pred = clf.predict(self.x_val)
        self.evaluate(self.y_val, y_pred)

    def test_run(self, clf) -> None:
        clf.fit(self.x_train_full, self.y_train_full)
        y_pred = clf.predict(self.x_test)
        self.evaluate(self.y_test, y_pred)

    def cross_val_run(self, clf) -> None:
        cross_val_scores = cross_validate(
            clf, self.x, self.y, scoring=["f1_macro", "accuracy"]
        )
        print(
            "cross-val F1-scores: ",
            cross_val_scores["test_f1_macro"],
            f" (Avg: {np.average(cross_val_scores['test_f1_macro'])})",
        )
        print(
            "cross-val Accuracy scores: ",
            cross_val_scores["test_accuracy"],
            f" (Avg: {np.average(cross_val_scores['test_accuracy'])})",
        )

    def knn_classify(self, k=5, command="tune") -> None:
        # cross-val using KNN means
        print("KNN classifier:\n -----------------")
        clf = KNeighborsClassifier(k)
        self.k = k
        if command == "tune":
            self.grid_search(clf)
        elif command == "test":
            self.test_run(clf)
        elif command == "cross-val":
            self.cross_val_run(clf)

    def logistic_regression(self, max_iter=10000, command="tune") -> None:
        print("Logistic Regression classifier:\n -----------------")
        clf = LogisticRegression(max_iter=max_iter, random_state=42)
        self.iter_log = max_iter
        if command == "tune":
            self.grid_search(clf)
        elif command == "test":
            self.test_run(clf)
        elif command == "cross-val":
            self.cross_val_run(clf)

    def nb_classify(self, command="tune") -> None:
        print("Naive-Bayes classifier:\n -----------------")
        clf = GaussianNB()
        if command == "tune":
            self.grid_search(clf)
        elif command == "test":
            self.test_run(clf)
        elif command == "cross-val":
            self.cross_val_run(clf)

    def random_forest(self, n_trees=200, command="tune") -> None:
        print("Random Forest classifier:\n -----------------")
        clf = RandomForestClassifier(n_trees, random_state=42)
        self.n_trees = n_trees
        if command == "tune":
            self.grid_search(clf)
        elif command == "test":
            self.test_run(clf)
        elif command == "cross-val":
            self.cross_val_run(clf)

    def svm_classify(self, max_iter=100000, command="tune") -> None:
        print("Linear SVC classifier:\n -----------------")
        clf = LinearSVC(max_iter=max_iter, random_state=42)
        self.iter_svc = max_iter
        if command == "tune":
            self.grid_search(clf)
        elif command == "test":
            self.test_run(clf)
        elif command == "cross-val":
            self.cross_val_run(clf)

    def save_model(self, score, model):
        self.models.append(model)
        if score > self.max_cv_score:
            self.max_cv_score = score
            self.best_model = len(self.models) - 1
            print(len(self.models))

    # Only works for 3 ensembles
    def ensemble(self) -> None:
        print("Ensemble:\n -----------------")
        model_predictions = []
        ensemble_predictions = []
        for model in self.models:
            model_predictions.append(model.predict(self.x_val))
        print(len(model_predictions), len(model_predictions[0]))
        for i in range(len(model_predictions[0])):
            if (
                model_predictions[0][i] is model_predictions[1][i]
                or model_predictions[2][i]
            ):
                ensemble_predictions.append(model_predictions[0][i])
            elif model_predictions[1][i] is model_predictions[2][i]:
                ensemble_predictions.append(model_predictions[1][i])
            else:
                ensemble_predictions.append(model_predictions[self.best_model][i])
        conf_matrix = metrics.confusion_matrix(self.y_val, ensemble_predictions)
        print(conf_matrix)
