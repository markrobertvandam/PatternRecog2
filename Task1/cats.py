#!/usr/bin/env python3

import cv2
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import shutil

from classification import Classification
from clustering import Clustering
from feature_extraction import FeatureExtraction

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


class Cats:
    def __init__(self):
        self.file_names = []
        self.images = None
        self.flattened_original = None
        self.gray_images = None
        self.labels = None
        self.feature_extractor = None
        self.fourier_data = None
        self.sift_data = None
        self.normal_classifier = None
        self.sift_classifier = None
        self.fourier_classifier = None

    def load_data(self) -> None:
        print("Loading data...")
        animals = ["Cheetah", "Jaguar", "Leopard", "Lion", "Tiger"]
        images = []
        labels = []
        gray_images = []
        for animal in animals:
            files_path = os.path.join(
                "data", "cats_projekat_filtered", animal, "*.jp*g"
            )
            for img in glob.glob(files_path):
                self.file_names.append(img)
                image = cv2.imread(img)
                resized_image = cv2.resize(image, (250, 250))
                grayscale_img = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
                images.append(resized_image)
                gray_images.append(grayscale_img)
                labels.append(animal)
        # images shape is (170, 250, 250, 3)
        self.images = np.asarray(images)
        self.flattened_original = self.images.reshape(
            self.images.shape[0],
            self.images.shape[1] * self.images.shape[2] * self.images.shape[3],
        )
        self.gray_images = np.asarray(gray_images)
        self.labels = np.asarray(labels)

    def visualize_data(self):
        print("Visualizing data...")
        unique, counts = np.unique(self.labels, return_counts=True)
        plt.bar(unique, counts, 0.4)
        plt.title("Class Frequency")
        plt.xlabel("Class")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join("plots", "cats_histo.png"))
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
        self.normal_classifier.knn_classify(k=5, command=command)
        self.normal_classifier.nb_classify(command=command)
        self.normal_classifier.random_forest(n_trees=200, command=command)
        print("--------------")

        # Classify sift dataset
        print(f"Sift performance (shape: {self.sift_data.shape}): \n")
        self.sift_classifier = Classification(self.sift_data, self.labels)
        self.sift_classifier.knn_classify(k=5, command=command)
        self.sift_classifier.nb_classify(command=command)
        self.sift_classifier.random_forest(n_trees=200, command=command)
        print("--------------\n")

        # Classify fourier dataset

        print(f"Fourier performance (shape: {self.fourier_data.shape}): \n")

        self.fourier_classifier = Classification(self.fourier_data, self.labels)
        self.fourier_classifier.knn_classify(k=5, command=command)
        self.fourier_classifier.nb_classify(command=command)
        self.fourier_classifier.random_forest(n_trees=200, command=command)
        print("--------------")

    def cross_val(self) -> None:
        """
        function to run cross-val with full data with best pipeliness
        """
        print(f"Original performance (shape: {self.flattened_original.shape}): \n")

        self.normal_classifier = Classification(self.flattened_original, self.labels)
        self.normal_classifier.nb_classify(command="cross-val")
        self.normal_classifier.random_forest(n_trees=200, command="cross-val")
        print("--------------")

        # Classify sift dataset
        print(f"Sift performance (shape: {self.sift_data.shape}): \n")
        self.sift_classifier = Classification(self.sift_data, self.labels)
        self.sift_classifier.nb_classify(command="cross-val")
        self.sift_classifier.random_forest(n_trees=200, command="cross-val")

        print("--------------")

    def ensemble(self) -> None:
        """
        function to run all possible ensembles with full data
        """
        random_state = 42  # seed

        # Classify sift dataset
        print(f"Sift performance (shape: {self.sift_data.shape}): \n")
        self.sift_classifier = Classification(self.sift_data, self.labels)
        self.sift_classifier.knn_classify(k=5, command="test")
        self.sift_classifier.nb_classify(command="test")
        self.sift_classifier.random_forest(n_trees=200, command="test")

        print("\nEnsemble using Naive Bayes and Random Forest:")
        print("--------------")
        self.sift_classifier.ensemble(
            GaussianNB(), RandomForestClassifier(random_state=random_state)
        )

        print("\nEnsemble using Naive Bayes and KNN:")
        print("--------------")
        self.sift_classifier.ensemble(GaussianNB(), KNeighborsClassifier(n_neighbors=5))

        print("\nEnsemble using KNN and Random Forest:")
        print("--------------")
        self.sift_classifier.ensemble(
            KNeighborsClassifier(n_neighbors=5),
            RandomForestClassifier(random_state=random_state),
        )

        print("\nEnsemble using KNN, Naive Bayes and Random Forest:")
        print("--------------")
        self.sift_classifier.ensemble(
            KNeighborsClassifier(n_neighbors=5),
            GaussianNB(),
            RandomForestClassifier(random_state=random_state),
        )

        print("--------------")

    def clustering(self) -> None:
        print("Clustering: \n")
        print("Original performance: \n")
        normal_clustering = Clustering(self.flattened_original, self.labels, 4, 5)
        normal_clustering.k_means()
        print("--------------\n")

        print("SIFT performance: \n")
        sift_clustering = Clustering(self.sift_data, self.labels, 4, 5)
        sift_cluster_labels = sift_clustering.k_means()
        print("--------------\n")

        print("Fourier performance: \n")
        fourier_clustering = Clustering(self.fourier_data, self.labels, 4, 5)
        fourier_clustering.k_means()
        print("--------------\n")

    def save_clustering(self, cluster_labels, n_clusters) -> None:
        # holds the cluster id and the images { id: [images] }
        clustering_folder = os.path.join("data", "clustering")
        if os.path.exists(clustering_folder):
            print("Removing old clustering folder...")
            shutil.rmtree(clustering_folder)
        os.mkdir(clustering_folder)
        print(f"Making new clustering folder at {clustering_folder}...")
        groups = {}
        for file, cluster in zip(self.file_names, cluster_labels):
            if cluster not in groups.keys():
                groups[cluster] = []
                groups[cluster].append(file)
            else:
                groups[cluster].append(file)
        for cluster in range(1, n_clusters + 1):
            cluster_path = os.path.join(clustering_folder, "cluster" + str(cluster))
            os.mkdir(cluster_path)
            for img in groups[cluster - 1]:
                img_name = os.path.basename(img)
                shutil.copy(img, os.path.join(cluster_path, img_name))
