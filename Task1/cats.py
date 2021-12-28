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

    def classification(self, x) -> None:
        # resize to 2d if data is 3d
        if len(x.shape) == 3:
            x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])
        print("Data shape: ", x.shape)

        classifiers = Classification(x, self.labels)

        # classify using k_means
        classifiers.k_means_classify()

        # classify using method1
        if x.shape[1] < 1000:
            classifiers.svm_classify(80000)

        # classify using method2
        classifiers.nb_classify()
