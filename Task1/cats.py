#!/usr/bin/env python3

import skimage
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
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from tuning import Tuning


class Cats:
    def __init__(self):
        """
        Intialize classifiers and extractors for cats dataset.
        """

        self.file_names = []
        self.images = None
        self.flattened_original = None
        self.gray_images = None
        self.labels = None
        self.feature_extractor = None
        self.fourier_unmasked = None
        self.fourier_data = None
        self.sift_data = None
        self.normal_classifier = None
        self.sift_classifier = None
        self.fourier_classifier = None

    def load_data(self) -> None:
        """
        Function for loading image data.
        """

        print("Loading data...")
        animals = ["Cheetah", "Jaguar", "Leopard", "Lion", "Tiger"]
        images = []
        labels = []
        gray_images = []
        for animal in animals:
            files_path = os.path.join(
                "data", "cats_projekat_filtered", animal, "*.jp*g"
            )
            files = glob.glob(files_path)
            sorted_files = sorted(files)
            for img in sorted_files:
                self.file_names.append(img)
                image = cv2.imread(img)
                grayscale_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                images.append(image)
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

    def augmented_run(self):
        """
        Test run with augmentation
        """

        augmented_gray = []
        (x_train, x_test, y_train, y_test,) = train_test_split(
            self.gray_images,
            self.labels,
            test_size=0.2,
            random_state=42,
            stratify=self.labels,
        )
        print("Augmenting images...")
        for i in range(len(x_train)):
            image = x_train[i]
            augmented_gray += Cats.augment_image(image)
        test_len = len(x_test)
        augmented_labels = np.repeat(y_train, 3, axis=0)
        full_x = np.concatenate((augmented_gray, x_test))
        full_y = np.concatenate((augmented_labels, y_test))

        print("Running SIFT Feature Extraction...")
        feature_extractor = FeatureExtraction(full_x, full_y, "cats")
        sift_data, bad_imgs = feature_extractor.sift(225)
        augmented_labels = np.delete(augmented_labels, bad_imgs)

        print(f"Sift performance (shape: {sift_data.shape}): \n")
        sift_classifier_rf = RandomForestClassifier(280, random_state=42)
        sift_classifier_rf.fit(sift_data[:-test_len], augmented_labels)
        y_pred = sift_classifier_rf.predict(sift_data[-test_len:])
        Classification.evaluate(y_test, y_pred)

    @staticmethod
    def augment_image(image) -> list:
        """
        Function for augmenting images.

        Arguments:
        image: Numpy array containing image data.

        Returns:
        list: [the original image, flipped image, contrast adjusted image]
        """

        flipped_x = image[:, ::-1]
        gamma = skimage.exposure.adjust_gamma(image, gamma=0.4, gain=0.9)
        return [image, flipped_x, gamma]

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
        """
        Function to perform feature extraction
        using fourier transform
        """

        print("Doing feature extraction...")
        self.feature_extractor = FeatureExtraction(
            self.gray_images, self.labels, "cats"
        )
        (
            self.fourier_data,
            self.fourier_unmasked,
        ) = self.feature_extractor.fourier_transform()

        # flatten fourier
        self.fourier_data = self.fourier_data.reshape(
            self.fourier_data.shape[0],
            self.fourier_data.shape[1] * self.fourier_data.shape[2],
        )

        self.sift_data, _ = self.feature_extractor.sift()

    def tune_classification_params(self, option=None) -> None:
        """
        Function to run grid-search for all data
        """
        if option:
            cats_tuner = Tuning(
                self.flattened_original,
                self.labels,
                self.sift_data,
                self.fourier_unmasked,
                "cats",
                25.,
            )
            cats_tuner.tune_gene_params()
        else:
            cats_tuner = Tuning(
                self.flattened_original,
                self.labels,
                self.sift_data,
                self.fourier_unmasked,
                "cats",
                6,
            )
            cats_tuner.tune_cats_params()

    def classification(self, command) -> None:
        """
        function to run cross-val or test-run depending on command with best pipelines

        Arguments:
        command: String for specifying the type of model operation.

        Returns:
        None
        """

        print(f"Original performance (shape: {self.flattened_original.shape}): \n")

        self.normal_classifier = Classification(self.flattened_original, self.labels)
        self.normal_classifier.random_forest(n_trees=160, command=command)
        print("--------------")

        # Classify sift dataset
        print(f"Sift performance (shape: {self.sift_data.shape}): \n")
        sift_classifier_rf = Classification(self.sift_data[:, 0:225], self.labels)
        sift_classifier_rf.random_forest(n_trees=280, command=command)
        print("--------------\n")

    def ensemble(self) -> None:
        """
        function to run all possible ensembles with full data
        """

        # Classify sift dataset
        print(f"Sift performance (shape: {self.sift_data.shape}): \n")
        self.sift_classifier = Classification(self.sift_data, self.labels)
        self.sift_classifier.knn_classify(k=5, command="test")
        self.sift_classifier.svm_classify(
            kernel="linear", c=1, gamma="scale", command="test"
        )
        self.sift_classifier.random_forest(n_trees=200, command="test")

        print("\nEnsemble using SVM and Random Forest:")
        print("--------------")
        self.sift_classifier.ensemble(
            SVC(kernel="linear", C=1, gamma="scale"), RandomForestClassifier()
        )

        print("\nEnsemble using SVC and KNN:")
        print("--------------")
        self.sift_classifier.ensemble(
            SVC(kernel="linear", C=1, gamma="scale"),
            KNeighborsClassifier(n_neighbors=5),
        )

        print("\nEnsemble using KNN and Random Forest:")
        print("--------------")
        self.sift_classifier.ensemble(
            KNeighborsClassifier(n_neighbors=5),
            RandomForestClassifier(),
        )

        print("\nEnsemble using KNN, SVC and Random Forest:")
        print("--------------")
        self.sift_classifier.ensemble(
            KNeighborsClassifier(n_neighbors=5),
            SVC(kernel="linear", C=1, gamma="scale"),
            RandomForestClassifier(),
        )

        print("--------------")

    def clustering(self) -> None:
        """
        Function to perform clustering of data.
        """

        print("Clustering: \n")
        print("Original performance: \n")
        normal_clustering = Clustering(self.flattened_original, self.labels, 4, 5, 5)
        normal_clustering.agglomerative_clustering()
        normal_clustering.k_means()
        normal_clustering.spectral()
        normal_clustering.optics()
        print("--------------\n")

        print("SIFT performance: \n")
        sift_clustering = Clustering(self.sift_data, self.labels, 4, 5, 5)
        sift_clustering.agglomerative_clustering()
        sift_clustering.k_means()
        sift_clustering.spectral()
        sift_clustering.optics()
        # normal_clustering.h_dbscan()
        print("--------------\n")

        # print("Fourier performance: \n")
        # fourier_clustering = Clustering(self.fourier_data, self.labels, 4, 5)
        # fourier_clustering.agglomerative_clustering()
        # print("--------------\n")

    def save_clustering(self, cluster_labels, n_clusters) -> None:
        """
        Function to save clustering results.

        Arguments:
        cluster_labels: List containing strings of clustering names.
        n_clusters: Interger specifying number of clusters.

        Returns:
        None
        """

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
