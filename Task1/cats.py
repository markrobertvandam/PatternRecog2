#!/usr/bin/env python3

import skimage
import cv2
import glob
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import shutil

from collections import Counter
from classification import Classification
from clustering import Clustering
from feature_extraction import FeatureExtraction

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


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
        self.fourier_data = self.feature_extractor.fourier_transform()

        # flatten fourier
        self.fourier_data = self.fourier_data.reshape(
            self.fourier_data.shape[0],
            self.fourier_data.shape[1] * self.fourier_data.shape[2],
        )

        self.sift_data, _ = self.feature_extractor.sift()

    def tune_classification_params(self) -> None:
        """
        Function to run grid-search for all data
        """

        # print("Original:")
        # self.original_classification_params()
        print("Sift:")
        self.sift_classification_params("sift")
        # print("Fourier:")
        # self.fourier_data = None
        # self.sift_fourier_classification_params("fourier")

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
        """
        Function to save tuning results.

        Arguments:
        f1_results: List containing F1-scores.
        acc_results: List containing accuracies.
        model: List containing models names.
        name: String for choice for saving filename.

        Returns:
        None
        """

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

    def original_classification_params(self) -> None:
        """
        Helper function to run grid-search for original data or fourier data
        """
        clf = Classification(self.flattened_original, self.labels)
        self.block_print()
        results_f1_knn, results_acc_knn, results_f1_rf, results_acc_rf = [
            np.zeros(25) for _ in range(4)
        ]
        results_f1_svm, results_acc_svm = [np.zeros(96), np.zeros(96)]

        # k-value loop
        for k in range(1, 26):
            (
                results_f1_knn[(k - 1)],
                results_acc_knn[(k - 1)],
            ) = clf.knn_classify(k, command="tune")

        kernels = ['linear', 'poly', 'rbf', 'sigmoid']
        c = [0.1, 1, 10, 100]
        gamma = ['scale', 'auto', 0.0001, 0.001, 0.1, 1]

        for i in range(4):
            for j in range(4):
                for k in range(6):
                    (
                        results_f1_svm[(i * 24 + j * 6 + k)],
                        results_acc_svm[(i * 24 + j * 6 + k)],
                    ) = clf.svm_classify(
                        kernel=kernels[i], c=c[j], gamma=gamma[k], command="tune"
                    )

        # n-trees loop
        for n in range(25):
            n_trees = 20 + n * 20
            (
                results_f1_rf[n],
                results_acc_rf[n],
            ) = clf.random_forest(n_trees=n_trees, command="tune")

        self.save_tune_results(
            [results_f1_knn, results_f1_svm, results_f1_rf],
            [results_acc_knn, results_acc_svm, results_acc_rf],
            ["knn", "svm", "rf"],
            "original",
        )

    def fourier_classification_params(self, name: str) -> None:
        """
        Helper function to run grid-search for sift data

        Arguments:
        name: Name of preprocessing. (Options: "sift", "fourier")

        Returns:
        None
        """
        self.block_print()
        results_f1_knn, results_acc_knn, results_f1_rf, results_acc_rf = [
            np.zeros((13, 25)) for _ in range(4)
        ]
        results_f1_svm, results_acc_svm = [np.zeros((13, 12)), np.zeros((13, 12))]

        # Max keypoints loop
        for i in range(13):
            self.enable_print()
            print("filter radius: ", (i*2))
            self.block_print()
            if self.fourier_data is None:
                self.fourier_data = self.feature_extractor.fourier_transform(i*5)
                fourier_data = self.fourier_data
            else:
                fourier_data = self.feature_extractor.fourier_transform(i*5, self.fourier_data)
                # knn_fourier_data = self.feature_extractor.fourier_transform(i + knn_offset, self.fourier_data)
                # svm_fourier_data = self.feature_extractor.fourier_transform(i + svm_offset, self.fourier_data)
                # rf_fourier_data = self.feature_extractor.fourier_transform(i+ rf_offset, self.fourier_data)
            fourier_data = fourier_data.reshape(
                fourier_data.shape[0],
                fourier_data.shape[1] * fourier_data.shape[2],
            )
            knn_classifier = Classification(fourier_data, self.labels)
            svm_classifier = Classification(fourier_data, self.labels)
            rf_classifier = Classification(fourier_data, self.labels)

            # k-value loop knn
            for k in range(1, 26):
                (
                    results_f1_knn[i][(k - 1)],
                    results_acc_knn[i][(k - 1)],
                ) = knn_classifier.knn_classify(k, command="tune")

            # SVM
            kernels = ['linear', 'poly', 'rbf', 'sigmoid']
            c = [0.1, 1, 10, 100]
            gamma = ['scale', 'auto', 0.0001, 0.001, 0.1, 1]
            degree = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

            for j in range(2):
                for k in range(1, 2):
                    for m in range(1):
                        if j == 1:
                            degree_range = 11
                        else:
                            degree_range = 1
                        for n in range(degree_range):
                            results_f1_svm[i][(j + n)], results_acc_svm[i][(j + n)] = \
                                svm_classifier.svm_classify(
                                kernel=kernels[j], c=c[k], gamma=gamma[m], degree=degree[n], command="tune"
                            )

            # n-trees loop
            for n in range(25):
                n_trees = 20 + n * 20
                (
                    results_f1_rf[i][n],
                    results_acc_rf[i][n],
                ) = rf_classifier.random_forest(n_trees=n_trees, command="tune")

        self.save_tune_results(
            [results_f1_knn, results_f1_svm, results_f1_rf],
            [results_acc_knn, results_acc_svm, results_acc_rf],
            ["knn", "svm", "rf"],
            name,
        )

    def sift_classification_params(self, name: str) -> None:
        """
        Helper function to run grid-search for sift data

        Arguments:
        name: Name of preprocessing. (Options: "sift", "fourier")

        Returns:
        None
        """
        self.block_print()
        results_f1_knn, results_acc_knn, results_f1_rf, results_acc_rf = [
            np.zeros((6, 6)) for _ in range(4)
        ]
        results_f1_svm, results_acc_svm = [np.zeros((6, 11)), np.zeros((6, 11))]

        # Max keypoints loop
        for i in range(6):
            key_points = i * 5
            self.enable_print()
            print("number of keypoints: ", (key_points + 5))
            self.block_print()

            knn_reduced_sift = self.sift_data[:, 255: key_points + 5]
            knn_classifier = Classification(knn_reduced_sift, self.labels)

            svm_reduced_sift = self.sift_data[:, 0: key_points + 5]
            svm_classifier = Classification(svm_reduced_sift, self.labels)

            rf_reduced_sift = self.sift_data[:, 205: key_points + 5]
            rf_classifier = Classification(rf_reduced_sift, self.labels)

            # k-value loop knn
            for k in range(19, 26):
                (
                    results_f1_knn[i][(k - 19)],
                    results_acc_knn[i][(k - 19)],
                ) = knn_classifier.knn_classify(k, command="tune")

            # SVM
            kernels = ['linear', 'poly', 'rbf', 'sigmoid']
            c = [0.1, 1, 10, 100]
            gamma = ['scale', 'auto', 0.0001, 0.001, 0.1, 1]
            degree = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

            for n in range(11):
                results_f1_svm[i][n], results_acc_svm[i][n] = \
                    svm_classifier.svm_classify(
                    kernel=kernels[3], c=c[2], gamma=gamma[0], degree=degree[n], command="tune"
                )

            # n-trees loop
            for n in range(9, 16):
                n_trees = 20 + n * 20
                (
                    results_f1_rf[i][n-9],
                    results_acc_rf[i][n-9],
                ) = rf_classifier.random_forest(n_trees=n_trees, command="tune")

        self.save_tune_results(
            [results_f1_knn, results_f1_svm, results_f1_rf],
            [results_acc_knn, results_acc_svm, results_acc_rf],
            ["knn", "svm", "rf"],
            name,
        )

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

        random_state = 42  # seed

        # Classify sift dataset
        print(f"Sift performance (shape: {self.sift_data.shape}): \n")
        self.sift_classifier = Classification(self.sift_data, self.labels)
        self.sift_classifier.knn_classify(k=5, command="test")
        self.sift_classifier.svm_classify(
            kernel="linear", c=1, gamma="scale", command="test"
        )
        self.sift_classifier.random_forest(n_trees=200, command="test")

        print("\nEnsemble using Naive Bayes and Random Forest:")
        print("--------------")
        self.sift_classifier.ensemble(
            SVC(), RandomForestClassifier(random_state=random_state)
        )

        print("\nEnsemble using Naive Bayes and KNN:")
        print("--------------")
        self.sift_classifier.ensemble(SVC(), KNeighborsClassifier(n_neighbors=5))

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
            SVC(),
            RandomForestClassifier(random_state=random_state),
        )

        print("--------------")

    def clustering(self) -> None:
        """
        Function to perform clustering of data.
        """

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
