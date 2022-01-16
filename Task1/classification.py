#!/usr/bin/env python3
import numpy as np

from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


class Classification:
    def __init__(self, x: np.ndarray, y: np.ndarray, file_names) -> None:
        self.x, self.y = x, y
        self.x_train, self.x_test, self.y_train, self.y_test, self.files_train, self.files_test = train_test_split(
            x, y, file_names, test_size=0.2, random_state=42, stratify=self.y
        )

        # x: 100%, x_train_full: 80%, x_test: 20%, x_train: 70%, x_val: 10%
        self.x_train_full = self.x_train
        self.y_train_full = self.y_train
        self.files_train_full = self.files_train
        self.x_train, self.x_val, self.y_train, self.y_val, self.files_train, self.files_val = train_test_split(
            self.x_train,
            self.y_train,
            self.files_train,
            test_size=0.125,
            random_state=42,
            stratify=self.y_train,
        )

        self.k = None
        self.iter_svc = None
        self.iter_log = None
        self.n_trees = None

        self.models = []

        self.models_dict = {}

        for i in self.files_val:
            print(i)

    def evaluate(self, y_true, y_pred, probs = None):
        if probs is not None:
            print("Probability of classes")
            for i in probs:
                print(i)
        else:
            conf_matrix = metrics.confusion_matrix(y_true, y_pred)
            print(conf_matrix)
            print(
                f"F1-score: {f1_score(y_true, y_pred, average='macro')}, Accuracy: {accuracy_score(y_true, y_pred)}"
            )

    def grid_search(self, clf):
        # one validation run
        clf.fit(self.x_train, self.y_train)
        y_pred = clf.predict(self.x_val)
        # self.evaluate(self.y_val, y_pred)
        probs = clf.predict_proba(self.x_val)
        self.evaluate(self.y_val, y_pred, probs)
        return f1_score(self.y_val, y_pred, average='macro'), accuracy_score(self.y_val, y_pred)

    def test_run(self, clf):
        clf.fit(self.x_train_full, self.y_train_full)
        y_pred = clf.predict(self.x_test)
        self.evaluate(self.y_test, y_pred)
        return clf

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

    def save_classifier(self, clf):
        clf.fit(self.x_train, self.y_train)
        self.models.append(clf)

    def select_command_action(self, clf, command: str):
        if command == "tune":
            return self.grid_search(clf)
        elif command == "test":
            self.test_run(clf)
        elif command == "cross-val":
            self.cross_val_run(clf)
        elif command == "save classifier":
            self.save_classifier(clf)

    def knn_classify(self, k: int, command: str):
        # cross-val using KNN means
        print("KNN classifier:\n -----------------")
        clf = KNeighborsClassifier(k)
        self.k = k
        return self.select_command_action(clf, command)

    def logistic_regression(self, max_iter: int, command: str):
        print("\nLogistic Regression classifier:\n -----------------")
        clf = LogisticRegression(max_iter=max_iter, random_state=42)
        self.iter_log = max_iter
        return self.select_command_action(clf, command)

    def nb_classify(self, command: str):
        print("\nNaive-Bayes classifier:\n -----------------")
        clf = GaussianNB()
        return self.select_command_action(clf, command)

    def random_forest(self, n_trees: int, command: str):
        print("\nRandom Forest classifier:\n -----------------")
        clf = RandomForestClassifier(n_trees, random_state=42)
        self.n_trees = n_trees
        return self.select_command_action(clf, command)

    def svm_classify(self, max_iter: int, command: str):
        print("\nLinear SVC classifier:\n -----------------")
        clf = LinearSVC(max_iter=max_iter, random_state=42)
        self.iter_svc = max_iter
        return self.select_command_action(clf, command)

    def svc(self, kernel: str, command: str):
        print("\n SV classifier:\n -----------------")
        clf = SVC(kernel=kernel, random_state=42)
        return self.select_command_action(clf, command)

    def gp_classify(self, command: str):
        print("\n GP classifier:\n -----------------")
        kernel = 1.0 * RBF(1.0)
        clf = GaussianProcessClassifier(kernel=kernel, random_state=42)
        return self.select_command_action(clf, command)

    def adb(self, n_trees: int, command: str):
        clf = AdaBoostClassifier(n_estimators=n_trees, random_state=42)
        return self.select_command_action(clf, command)

    def qda(self, command: str):
        clf = QuadraticDiscriminantAnalysis()
        return self.select_command_action(clf, command)

    def train_ensemble_classifiers(
        self, clf1="KNN", param1=5, clf2="NB", param2=None, clf3="RF", param3=200
    ) -> None:
        params = [param1, param2, param3]
        clfs = [clf1, clf2, clf3]
        for i in range(3):
            if clfs[i] == "KNN":
                self.knn_classify(params[i], "save classifier")
            elif clfs[i] == "LG":
                self.logistic_regression(params[i], "save classifier")
            elif clfs[i] == "NB":
                self.nb_classify("save classifier")
            elif clfs[i] == "RF":
                self.random_forest(params[i], "save classifier")
            elif clfs[i] == "SVM":
                self.svm_classify(params[i], "save classifier")

    # Only works for 3 ensembles, assumes classifier 1 is the best performing classifier
    def old_ensemble(
        self, clf1="KNN", param1=5, clf2="NB", param2=None, clf3="RF", param3=200
    ) -> None:
        print("Ensemble:\n -----------------")
        self.train_ensemble_classifiers(clf1, param1, clf2, param2, clf3, param3)
        model_predictions = []
        ensemble_predictions = []
        for model in self.models:
            model_predictions.append(model.predict(self.x_val))
        print(len(model_predictions), len(model_predictions[0]))
        for i in range(len(model_predictions[0])):
            if model_predictions[1][i] == model_predictions[2][i]:
                ensemble_predictions.append(model_predictions[1][i])
            else:
                ensemble_predictions.append((model_predictions[0][i]))
        self.evaluate(self.y_val, ensemble_predictions)

    def ensemble(self, model1, model2, model3=None) -> None:
        estimators = [("model1", model1), ("model2", model2)]
        if model3 is not None:
            estimators.append(("model3", model3))
        ensemble_model = VotingClassifier(estimators=estimators, voting="soft")
        self.test_run(ensemble_model)
