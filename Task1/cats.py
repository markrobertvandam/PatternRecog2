#!/usr/bin/env python3

import glob
import cv2
import numpy as np

from classification import Classification
from feature_extraction import FeatureExtraction


class Cats:
    def __init__(self):
        self.images = []
        self.gray_images = []
        self.labels = []
        self.feature_extractor = None
        self.fourier_data = None
        self.sift_data = None
        self.normal_classifier = None
        self.sift_classifier = None
        self.fourier_classifier = None

    def load_data(self) -> None:
        animals = ["Cheetah", "Jaguar", "Leopard", "Lion", "Tiger"]
        for animal in animals:
            for img in glob.glob(f"data/BigCats_filtered/{animal}/*.jp*g"):
                image = cv2.imread(img)
                resized_image = cv2.resize(image, (250, 250))
                grayscale_img = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
                self.images.append(resized_image)
                self.gray_images.append(grayscale_img)
                self.labels.append(animal)
        # images shape is (170, 250, 250, 3)
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

    def train_validation(self) -> None:
        flattened_original = self.images.reshape(
            self.images.shape[0],
            self.images.shape[1] * self.images.shape[2] * self.images.shape[3],
        )

        print(f"Original performance (shape: {flattened_original.shape}): \n")

        self.normal_classifier = Classification(flattened_original, self.labels)
        self.normal_classifier.knn_classify()
        self.normal_classifier.nb_classify()
        self.normal_classifier.random_forest()
        print("--------------")

        # Classify sift dataset
        print(f"Sift performance (shape: {self.sift_data.shape}): \n")
        self.sift_classifier = Classification(self.sift_data, self.labels)
        self.sift_classifier.knn_classify()
        self.sift_classifier.nb_classify()
        self.sift_classifier.random_forest()
        print("--------------\n")

        # Classify fourier dataset
        flattened_fourier = self.fourier_data.reshape(
            self.fourier_data.shape[0],
            self.fourier_data.shape[1] * self.fourier_data.shape[2],
        )
        print(f"Fourier performance (shape: {flattened_fourier.shape}): \n")

        self.fourier_classifier = Classification(flattened_fourier, self.labels)
        self.fourier_classifier.knn_classify()
        self.fourier_classifier.nb_classify()
        self.fourier_classifier.random_forest()
        print("--------------")

    def test_run(self):
        print("\nTesting run\n")
