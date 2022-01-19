#!/usr/bin/env python3

import cv2
import glob
import sys
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

    def tune_classification_params(self) -> None:
        """
        Function to run grid-search for all data
        """
        self.normal_classifier = Classification(self.flattened_original, self.labels)
        self.fourier_classifier = Classification(self.fourier_data, self.labels)
        print("Original:")
        self.original_fourier_classification_params(
            self.normal_classifier, "original", 12, 140
        )
        print("Sift:")
        self.sift_classification_params()
        print("Fourier:")
        self.original_fourier_classification_params(
            self.fourier_classifier, "fourier", 5, 140
        )

    # Disable
    def block_print(self):
        sys.stdout = open(os.devnull, "w")
        sys.stdout = open(os.devnull, "w")

    # Restore
    def enable_print(self):
        sys.stdout = sys.__stdout__

    def save_tune_results(
        self, f1_results: list, acc_results: list, models: list, name: str
    ) -> None:
        result_path = os.path.join("data", "results", "cats")
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        self.enable_print()
        for i in range(len(models)):
            print(
                f"Max {models[i]} f1-score = "
                + str(np.amax(f1_results[i]))
                + f", max {models[i]} acc = "
                + str(np.amax(acc_results[i]))
            )
            np.savetxt(
                os.path.join(result_path, f"{name}_f1_{models[i]}.csv"),
                f1_results[i],
                delimiter=",",
            )
            np.savetxt(
                os.path.join(result_path, f"{name}_acc_{models[i]}.csv"),
                acc_results[i],
                delimiter=",",
            )

    def original_fourier_classification_params(
        self, clf, name: str, knn_offset, rf_offset
    ) -> None:
        """
        Helper function to run grid-search for original data or fourier data
        """
        self.block_print()
        results_f1_knn, results_acc_knn, results_f1_rf, results_acc_rf = [
            np.zeros(6) for _ in range(4)
        ]
        results_f1_nb, results_acc_nb = [np.zeros(1), np.zeros(1)]

        # k-value loop
        for k in range(knn_offset, knn_offset + 6):
            (
                results_f1_knn[(k - knn_offset)],
                results_acc_knn[(k - knn_offset)],
            ) = clf.knn_classify(k, command="tune")

        results_f1_nb[0], results_acc_nb[0] = clf.nb_classify(command="tune")

        # n-trees loop
        for n in range(0, 6):
            n_trees = rf_offset + n * 10
            (
                results_f1_rf[n],
                results_acc_rf[n],
            ) = clf.random_forest(n_trees=n_trees, command="tune")

        self.save_tune_results(
            [results_f1_knn, results_f1_nb, results_f1_rf],
            [results_acc_knn, results_acc_nb, results_acc_rf],
            ["knn", "nb", "rf"],
            name,
        )

    def sift_classification_params(self) -> None:
        """
        Helper function to run grid-search for sift data
        """
        self.block_print()
        results_f1_knn, results_acc_knn, results_f1_rf, results_acc_rf = [
            np.zeros((6, 6)) for _ in range(4)
        ]
        results_f1_nb, results_acc_nb = [np.zeros(6), np.zeros(6)]

        # Max keypoints loop
        for i in range(0, 6):
            key_points = i * 5
            knn_reduced_sift = self.sift_data[:, 0 : key_points + 250]
            self.sift_classifier = Classification(knn_reduced_sift, self.labels)

            # k-value loop
            for k in range(14, 20):
                (
                    results_f1_knn[i][(k - 14)],
                    results_acc_knn[i][(k - 14)],
                ) = self.sift_classifier.knn_classify(k, command="tune")

            # Naive Bayes
            nb_reduced_sift = self.sift_data[:, 0 : key_points + 75]
            self.sift_classifier = Classification(nb_reduced_sift, self.labels)
            results_f1_nb[i], results_acc_nb[i] = self.sift_classifier.nb_classify(
                command="tune"
            )

            # n-trees loop
            rf_reduced_sift = self.sift_data[:, 0 : key_points + 205]
            self.sift_classifier = Classification(rf_reduced_sift, self.labels)
            for n in range(0, 6):
                n_trees = 200 + n * 20
                (
                    results_f1_rf[i][n],
                    results_acc_rf[i][n],
                ) = self.sift_classifier.random_forest(n_trees=n_trees, command="tune")

        self.save_tune_results(
            [results_f1_knn, results_f1_nb, results_f1_rf],
            [results_acc_knn, results_acc_nb, results_acc_rf],
            ["knn", "nb", "rf"],
            "sift",
        )

    def classification(self, command) -> None:
        """
        function to run cross-val or test-run depending on command with best pipelines
        """
        print(f"Original performance (shape: {self.flattened_original.shape}): \n")

        self.normal_classifier = Classification(self.flattened_original, self.labels)
        self.normal_classifier.knn_classify(k=12, command=command)
        self.normal_classifier.random_forest(n_trees=160, command=command)
        print("--------------")

        # Classify sift dataset
        print(f"Sift performance (shape: {self.sift_data.shape}): \n")
        sift_classifier_knn = Classification(self.sift_data[:, 0:265], self.labels)
        sift_classifier_rf = Classification(self.sift_data[:, 0:225], self.labels)
        sift_classifier_knn.knn_classify(k=19, command=command)
        sift_classifier_rf.random_forest(n_trees=280, command=command)
        print("--------------\n")

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
        sift_clustering.k_means()
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
