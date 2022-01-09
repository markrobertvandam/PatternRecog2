#!/usr/bin/env python3

import glob
import cv2
import matplotlib.pyplot as plt
import numpy as np

from classification import Classification
from clustering import Clustering
from feature_extraction import FeatureExtraction

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


class Cats:
    def __init__(self):
        self.file_names = []
        self.images = []
        self.flattened_original = None
        self.gray_images = []
        self.labels = []
        self.feature_extractor = None
        self.fourier_data = None
        self.sift_data = None
        self.normal_classifier = None
        self.sift_classifier = None
        self.fourier_classifier = None

    def load_data(self) -> None:
        print("Loading data...")
        animals = ["Cheetah", "Jaguar", "Leopard", "Lion", "Tiger"]
        for animal in animals:
            for img in glob.glob(f"data/cats_projekat_filtered/{animal}/*.jp*g"):
                self.file_names.append(img)
                image = cv2.imread(img)
                resized_image = cv2.resize(image, (250, 250))
                grayscale_img = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
                self.images.append(resized_image)
                self.gray_images.append(grayscale_img)
                self.labels.append(animal)
        # images shape is (170, 250, 250, 3)
        self.images = np.asarray(self.images)
        self.flattened_original = self.images.reshape(
            self.images.shape[0],
            self.images.shape[1] * self.images.shape[2] * self.images.shape[3],
        )
        self.gray_images = np.asarray(self.gray_images)
        self.labels = np.asarray(self.labels)

    def visualize_data(self):
        print("Visualizing data...")
        unique, counts = np.unique(self.labels, return_counts=True)
        plt.bar(unique, counts, 0.4)
        plt.title("Class Frequency")
        plt.xlabel("Class")
        plt.ylabel("Frequency")
        plt.savefig("plots/cats_histo.png")
        plt.close()

    def feature_extraction(self) -> None:
        print("Doing feature extraction...")
        self.feature_extractor = FeatureExtraction(
            self.gray_images, self.labels, "cats"
        )
        self.fourier_data = self.feature_extractor.fourier_transform()

        # flatten fourier
        self.fourier_data = self.fourier_data.reshape(
            self.fourier_data.shape[0],
            self.fourier_data.shape[1] * self.fourier_data.shape[2],
        )

        self.sift_data = self.feature_extractor.sift()

    def classification(self, command="tuning") -> None:
        """
        function to run grid-search/test-run depending on command
        """
        print(f"Original performance (shape: {self.flattened_original.shape}): \n")

        self.normal_classifier = Classification(self.flattened_original, self.labels)
        self.normal_classifier.knn_classify(command=command)
        self.normal_classifier.nb_classify(command=command)
        self.normal_classifier.random_forest(command=command)
        print("--------------")

        # Classify sift dataset
        print(f"Sift performance (shape: {self.sift_data.shape}): \n")
        self.sift_classifier = Classification(self.sift_data, self.labels)
        self.sift_classifier.knn_classify(command=command)
        self.sift_classifier.nb_classify(command=command)
        self.sift_classifier.random_forest(command=command)
        print("--------------\n")

        # Classify fourier dataset

        print(f"Fourier performance (shape: {self.fourier_data.shape}): \n")

        self.fourier_classifier = Classification(self.fourier_data, self.labels)
        self.fourier_classifier.knn_classify(command=command)
        self.fourier_classifier.nb_classify(command=command)
        self.fourier_classifier.random_forest(command=command)
        print("--------------")

    def cross_val(self) -> None:
        """
        function to run cross-val with full data with best pipeliness
        """
        print(f"Original performance (shape: {self.flattened_original.shape}): \n")

        self.normal_classifier = Classification(self.flattened_original, self.labels)
        self.normal_classifier.nb_classify(command="cross-val")
        self.normal_classifier.random_forest(command="cross-val")
        print("--------------")

        # Classify sift dataset
        print(f"Sift performance (shape: {self.sift_data.shape}): \n")
        self.sift_classifier = Classification(self.sift_data, self.labels)
        self.sift_classifier.nb_classify(command="cross-val")
        self.sift_classifier.random_forest(command="cross-val")

        print("--------------")

    def ensemble(self) -> None:
        """
        function to run all possible ensembles with full data
        """
        # Classify sift dataset
        print(f"Sift performance (shape: {self.sift_data.shape}): \n")
        self.sift_classifier = Classification(self.sift_data, self.labels)
        self.sift_classifier.nb_classify(command="test")
        self.sift_classifier.random_forest(command="test")
        self.sift_classifier.knn_classify(command="test")

        print("\nEnsemble using Naive Bayes and Random Forest:")
        print("--------------")
        self.sift_classifier.ensemble(GaussianNB(), RandomForestClassifier())

        print("\nEnsemble using Naive Bayes and KNN:")
        print("--------------")
        self.sift_classifier.ensemble(GaussianNB(), KNeighborsClassifier())

        print("\nEnsemble using KNN and Random Forest:")
        print("--------------")
        self.sift_classifier.ensemble(KNeighborsClassifier(), RandomForestClassifier())

        print("\nEnsemble using KNN, Naive Bayes and Random Forest:")
        print("--------------")
        self.sift_classifier.ensemble(
            KNeighborsClassifier(), GaussianNB(), RandomForestClassifier()
        )

        print("--------------")

    def clustering(self) -> None:
        print("Clustering: \n")
        print("Original performance: \n")
        normal_clustering = Clustering(self.flattened_original)
        normal_clustering.k_means(n_clusters=5)
        print("--------------\n")

        print("SIFT performance: \n")
        sift_clustering = Clustering(self.sift_data)
        sift_clustering.cluster_with_plots()
        exit()
        sift_cluster_labels = sift_clustering.k_means(n_clusters=5)
        # holds the cluster id and the images { id: [images] }
        # groups = {}
        # for file, cluster in zip(self.file_names, sift_cluster_labels):
        #     if cluster not in groups.keys():
        #         groups[cluster] = []
        #         groups[cluster].append(file)
        #     else:
        #         groups[cluster].append(file)
        # print(groups)
        print("--------------\n")

        print("Fourier performance: \n")
        fourier_clustering = Clustering(self.fourier_data)
        fourier_clustering.k_means(n_clusters=5)
        print("--------------\n")
