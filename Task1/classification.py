#!/usr/bin/env python3
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


class Classification:
    def __init__(self, x: np.ndarray, y: list, n_trees=100) -> None:
        self.n_trees = n_trees
        self.random_forest_classifier = None
        self.naive_bayes_classifier = None
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            x, y, test_size=0.2, random_state=42
        )

    def random_forest(self):
        self.random_forest_classifier = RandomForestClassifier(self.n_trees)
        self.random_forest_classifier.fit(self.x_train,self.y_train)
        y_pred = self.random_forest_classifier.predict(self.x_test)
        accuracy = metrics.accuracy_score(self.y_test, y_pred)
        print(accuracy)

    def naive_bayes(self):
        self.naive_bayes_classifier = GaussianNB()
        self.naive_bayes_classifier.fit(self.x_train,self.y_train)
        y_pred = self.naive_bayes_classifier.predict(self.x_test)
        accuracy = metrics.accuracy_score(self.y_test, y_pred)
        print(accuracy)