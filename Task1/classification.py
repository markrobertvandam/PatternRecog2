#!/usr/bin/env python3
import numpy as np

from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn_lvq import GlvqModel


class Classification:
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Initialize dataset, splitting and model parameters.

        Arguments:
        X: Input features.
        y: Class labels.

        Returns:
        None
        """
        
        self.x, self.y = x, y
        (
            self.x_train,
            self.x_test,
            self.y_train,
            self.y_test,
        ) = train_test_split(x, y, test_size=0.2, random_state=42, stratify=self.y)

        # x: 100%, x_train_full: 80%, x_test: 20%, x_train: 70%, x_val: 10%
        self.x_train_full = self.x_train
        self.y_train_full = self.y_train
        (self.x_train, self.x_val, self.y_train, self.y_val,) = train_test_split(
            self.x_train,
            self.y_train,
            test_size=0.125,
            random_state=42,
            stratify=self.y_train,
        )

        self.k = None
        self.kernel = None
        self.c = None
        self.gamma = None
        self.iter_log = None
        self.n_trees = None

        self.models = []
        self.models_dict = {}

    @staticmethod
    def evaluate(y_true, y_pred, probs=None):
        """
        Static function to evaluate F-score and Accuracy.

        Arguments:
        y_true: True class labels.
        y_pred: Predicted class labels.

        Returns:
        None
        """
        
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
        """
        Function to train model with splitted data.

        Arguments:
        clf: Classifier model.

        Returns:
        None
        """
        
        # one validation run
        clf.fit(self.x_train, self.y_train)
        y_pred = clf.predict(self.x_val)
        return f1_score(self.y_val, y_pred, average="macro"), accuracy_score(
            self.y_val, y_pred
        )

    def test_run(self, clf):
        """
        Function to train model with entire train data 
        and test on test data.

        Arguments:
        clf: Classifier model.
        """
        
        clf.fit(self.x_train_full, self.y_train_full)
        y_pred = clf.predict(self.x_test)
        Classification.evaluate(self.y_test, y_pred)
        return clf

    def cross_val_run(self, clf) -> None:
        """
        Function to train model on cross validated data
        and print cross validated test scores.

        Arguments:
        clf: Classifier model

        Return:
        None
        """
        
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

    def select_command_action(self, clf, command: str):
        """
        Function to run specific model operation.

        Arguments:
        clf: Classifier model.
        command: Command specifying model operation.

        Returns:
        None
        """
        
        if command == "tune":
            return self.grid_search(clf)
        elif command == "test":
            self.test_run(clf)
        elif command == "cross-val":
            self.cross_val_run(clf)

    def knn_classify(self, k: int, command: str):
        """
        Function to perform classification with KNN.

        Arguments:
        K: Neighbor size.
        command: Command specifying model operation.

        Returns:
        Function: Performing the selected operation.
        """
        # cross-val using KNN means
        print("KNN classifier:\n -----------------")
        clf = KNeighborsClassifier(k)
        self.k = k
        return self.select_command_action(clf, command)

    def logistic_regression(self, max_iter: int, command: str):
        """
        Function to perform classification with Logistic Regression.

        Arguments:
        max_iter: Maximum iterations for LR model.
        command: Command specifying model operation.

        Returns:
        Function: Performing the selected operation.
        """
        
        print("\nLogistic Regression classifier:\n -----------------")
        clf = LogisticRegression(max_iter=max_iter, random_state=42)
        self.iter_log = max_iter
        return self.select_command_action(clf, command)

    def nb_classify(self, command: str):
        """
        Function to perform classification using Naive-Bayes.

        Arguments:
        command: Command specifying model operation.

        Returns:
        Function: Performing the selected operation.
        """
        
        print("\nNaive-Bayes classifier:\n -----------------")
        clf = GaussianNB()
        return self.select_command_action(clf, command)

    def random_forest(self, n_trees: int, command: str):
        """
        Function to perform classification using Random Forests.

        Arguments:
        n_trees: Number of decision trees.
        command: Command specifying model operation.

        Returns:
        Function: Performing the selected operation.
        """
        
        print("\nRandom Forest classifier:\n -----------------")
        clf = RandomForestClassifier(n_trees, random_state=42)
        self.n_trees = n_trees
        return self.select_command_action(clf, command)

    def svm_classify(self, kernel: str, c: float, gamma, degree: int, command: str):
        """
        Function to perform classification using SVM.

        Arguments:
        kernel: Type of kernel for SVM.
        c: Regularization parameter.
        gamma: Gamma for influencing distance in training.
        command: Command specifying model operation.

        Returns:
        Function: Performing the selected operation.
        """
        
        print("\nSVC classifier:\n -----------------")
        clf = SVC(C=c, kernel=kernel, gamma=gamma, degree=degree, random_state=42)
        self.kernel = kernel
        self.c = c
        self.gamma = gamma
        return self.select_command_action(clf, command)

    def glvq_classify(self, prototypes_per_class: int, command: str):
        """
        Function to perform classification using Generalized LVQ.

        Arguments:
        prototypes_per_class: Number of prototypes per class.
        command: Command specifying model operation.

        Returns:
        Function: Performing the selected operation.
        """
        
        print("\nGLVQ classifier:\n -----------------")
        # The creation of the model object used to fit the data to.
        clf = GlvqModel(prototypes_per_class=prototypes_per_class, random_state=42)
        return self.select_command_action(clf, command)

    def ensemble(self, model1, model2, model3=None) -> None:
        """
        Function to perform classification using Ensemble classifier.

        Arguments:
        model1: First classification model.
        model2: Second classification model.
        model3: Third classification model.
        """
        
        estimators = [("model1", model1), ("model2", model2)]
        if model3 is not None:
            estimators.append(("model3", model3))
        ensemble_model = VotingClassifier(estimators=estimators, voting="soft")
        self.test_run(ensemble_model)
