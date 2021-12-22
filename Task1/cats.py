#!/usr/bin/env python3
import glob
import cv2
import sklearn
import numpy as np
import matplotlib.pyplot as plt

from feature_extraction import FeatureExtraction
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


class Cats:
    def __init__(self):
        self.images = []
        self.labels = []
        self.feature_extractor = None
        self.fourier_data = None
        self.sift_data = None

    def load_data(self) -> None:
        animals = ["Cheetah", "Jaguar", "Leopard", "Lion", "Tiger"]
        for animal in animals:
            for img in glob.glob(f"data/BigCats/{animal}/*.jp*g"):
                self.images.append(cv2.imread(img))
                self.labels.append(animal)

    def feature_extraction(self) -> None:
        self.feature_extractor = FeatureExtraction(self.images, self.labels, "cats")
        self.fourier_data = np.array(self.feature_extractor.fourier_transform(), dtype="object")

        # shape (170, 240, 128), 170 images, 240 keypoints, 128 length of descriptor
        self.sift_data = np.array(self.feature_extractor.sift(), dtype="object")
        # cv2.imshow("img1", self.images[0])
        # #cv2.imshow("sift", self.sift_data[0][1])
        #
        # plt.imshow(self.fourier_data[0], cmap='gray')
        # plt.title('Fourier'), plt.xticks([]), plt.yticks([])
        # plt.show()
        #
        # cv2.waitKey(0)

        #print([len(x) for x in self.sift_data])
        print(self.sift_data.shape)

    def classification(self) -> None:
        # classify using KNN means
        k_means_classifier = KNeighborsClassifier()
        # classify using method1
        # classify using method2
        pass

    def clustering(self) -> None:
        # clustering using .. Fuzzy c-means?
        pass
