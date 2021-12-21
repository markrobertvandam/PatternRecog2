#!/usr/bin/env python3
import glob
import cv2
import sklearn

from feature_extraction import FeatureExtraction
from sklearn.model_selection import train_test_split


class Cats:
    def __init__(self):
        self.images = []
        self.labels = []
        self.feature_extractor = None
        self.fourier_data = []
        self.sift_data = []

    def load_data(self) -> None:
        animals = ["Cheetah", "Jaguar", "Leopard", "Lion", "Tiger"]
        for animal in animals:
            for img in glob.glob(f"data/BigCats/{animal}/*.jp*g"):
                self.images.append(cv2.imread(img))
                self.labels.append(animal)

    def feature_extraction(self) -> None:
        self.feature_extractor = FeatureExtraction(self.images, self.labels, "cats")
        self.fourier_data = self.feature_extractor.fourier_transform()
        self.sift_data = self.feature_extractor.sift()

    def classification(self) -> None:
        # classify using KNN means
        kmeans_classifier = sklearn.neighbors.KNeighborsClassifier
        # classify using method1
        # classify using method2
        pass

    def clustering(self) -> None:
        # clustering using .. Fuzzy c-means?
        pass
