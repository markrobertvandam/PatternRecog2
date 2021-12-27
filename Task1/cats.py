#!/usr/bin/env python3
import glob
import cv2
import sklearn
import numpy as np
import matplotlib.pyplot as plt

from feature_extraction import FeatureExtraction
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC



class Cats:
    def __init__(self):
        self.images = []
        self.gray_images = []
        self.labels = []
        self.feature_extractor = None
        self.fourier_data = None
        self.sift_data = None

    def load_data(self) -> None:
        animals = ["Cheetah", "Jaguar", "Leopard", "Lion", "Tiger"]
        for animal in animals:
            for img in glob.glob(f"data/BigCats/{animal}/*.jp*g"):
                image = cv2.imread(img)
                resized_image = cv2.resize(image, (250, 250))
                grayscale_img = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
                self.images.append(resized_image)
                self.gray_images.append(grayscale_img)
                self.labels.append(animal)
        # images shape is (170, 300, 500, 3)
        self.images = np.asarray(self.images)
        self.gray_images = np.asarray(self.gray_images)
        self.labels = np.asarray(self.labels)

    def feature_extraction(self) -> None:
        self.feature_extractor = FeatureExtraction(
            self.gray_images, self.labels, "cats"
        )
        self.fourier_data = self.feature_extractor.fourier_transform()

        # shape (170, 240, 128), 170 images, 200 keypoints, 128 length of descriptors
        self.sift_data = self.feature_extractor.sift()

    def general_classify(self, clf, x_train, x_val, x_test, y_train, y_val, y_test):
        cross_val_scores = cross_val_score(clf, x_train, y_train, cv=5)
        print("cross-val scores: ", cross_val_scores)

        # one validation run
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_val)
        conf_matrix = sklearn.metrics.confusion_matrix(y_val, y_pred)
        print(conf_matrix)

    def k_means_classify(self, x_train, x_val, x_test, y_train, y_val, y_test, k=5) -> None:
        # cross-val using KNN means
        print("K-means classifier:\n -----------------")
        clf = KNeighborsClassifier(k)
        self.general_classify(clf, x_train, x_val, x_test, y_train, y_val, y_test)

    def svm_classify(self, x_train, x_val, x_test, y_train, y_val, y_test, max_iter=800) -> None:
        print("Linear SVC classifier:\n -----------------")
        clf = LinearSVC(max_iter=max_iter)
        self.general_classify(clf, x_train, x_val, x_test, y_train, y_val, y_test)

    def nb_classify(self, x_train, x_val, x_test, y_train, y_val, y_test) -> None:
        print("Naive-Bayes classifier:\n -----------------")
        clf = GaussianNB()
        self.general_classify(clf, x_train, x_val, x_test, y_train, y_val, y_test)

    def classification(self, x) -> None:
        # resize to 2d if data is 3d
        if len(x.shape) == 3:
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
        print("Data shape: ", x.shape)

        # Split the data for classification:
        x_train, x_test, y_train, y_test = train_test_split(
            x, self.labels, test_size=0.2, random_state=42
        )
        x_train, x_val, y_train, y_val = train_test_split(
            x_train, y_train, test_size=0.2, random_state=42
        )
        self.k_means_classify(x_train, x_val, x_test, y_train, y_val, y_test)

        # classify using method1
        if x_train.shape[1] < 1000:
            self.svm_classify(x_train, x_val, x_test, y_train, y_val, y_test, 20000)

        # classify using method2
        self.nb_classify(x_train, x_val, x_test, y_train, y_val, y_test)

    def clustering(self) -> None:
        # clustering using .. Fuzzy c-means?
        pass
