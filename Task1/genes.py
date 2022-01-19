#!/usr/bin/env python3
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from classification import Classification
from clustering import Clustering
from feature_extraction import FeatureExtraction

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


class Genes:
    def __init__(self, pca_min_variance=0.6, mi_min_information=0.55):
        self.samples = None
        self.labels = None
        self.label_names = {"PRAD": 0, "LUAD": 1, "BRCA": 2, "KIRC": 3, "COAD": 4}
        self.feature_extractor = None
        self.pca = None
        self.pca_data = None
        self.mi = None
        self.mi_data = None
        self.pca_min_variance = pca_min_variance
        self.mi_min_information = mi_min_information
        self.normal_classifier = None
        self.pca_classifier = None
        self.mi_classifier = None
        self.normal_clustering = None
        self.pca_clustering = None
        self.mi_clustering = None

    def load_data(self) -> None:
        print("Loading data...")
        filename_samples = f"data/Genes/data.csv"
        self.samples = np.loadtxt(
            filename_samples, delimiter=",", skiprows=1, usecols=range(1, 1024)
        )

        filename_labels = f"data/Genes/labels.csv"

        labels = []
        with open(filename_labels) as labels_csv:
            labels_csv_reader = csv.reader(labels_csv)
            next(labels_csv_reader, None)
            for row in labels_csv_reader:
                labels.append(self.label_names[row[1]])
        self.labels = np.asarray(labels, dtype=int)

    def biplot_helper(self, name: str, vars: str, data: np.ndarray, offset=0) -> None:
        plt.figure()
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        for i in range(0 + offset, 3 + offset):
            j = i + 1
            scatter = axs[i - offset].scatter(
                data[:, i], data[:, j], c=self.labels, alpha=0.5
            )
            axs[i - offset].set_xlabel(f"{vars}_{i}")
            axs[i - offset].set_ylabel(f"{vars}_{j}")
        elems = list(scatter.legend_elements())
        # by default, the legend labels are the values
        # of the target, 0, 1, 2, etc.
        # we replace that with the target names:
        elems[1] = self.label_names
        fig.legend(*elems)
        plt.savefig(f"plots/biplots_{name}")
        plt.close(fig)

    def visualize_data(self) -> None:
        print("Visualizing the data...")
        # Biplots to show scatter using 2 random genes
        self.biplot_helper("original", "gene", self.samples, 1)

        # Class distribution
        unique, counts = np.unique(self.labels, return_counts=True)
        plt.figure(2)
        plt.bar(["PRAD", "LUAD", "BRCA", "KIRC", "COAD"], counts, 0.4)
        plt.title("Class Frequency")
        plt.xlabel("Class")
        plt.ylabel("Frequency")

        plt.savefig("plots/genes_histo.png")
        plt.close()

        # Biplots using most informaive PCA-components
        self.biplot_helper("pca", "pca", self.pca_data)

        # Biplots using most informaive MI-components
        self.biplot_helper("mi", "mi", self.mi_data)

    def feature_extraction(self) -> None:
        print("Doing feature extraction...")
        self.feature_extractor = FeatureExtraction(self.samples, self.labels, "genes")
        self.pca_data, self.pca = self.feature_extractor.pca(self.pca_min_variance)
        self.mi_data, self.mi = self.feature_extractor.mutual_information(
            self.mi_min_information
        )

    def tune_classification_params(self) -> None:
        """
        Function to run grid-search for all data
        """
        self.normal_classifier = Classification(self.samples, self.labels)

        print("Original:")
        self.original_classification_params(self.normal_classifier, "original")
        print("PCA:")
        self.pca_mi_classification_params("pca")
        print("MI:")
        self.pca_mi_classification_params("mi")

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
        result_path = os.path.join("data", "results", "genes")
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

    def original_classification_params(self, clf, name: str) -> None:
        """
        Helper function to run grid-search for original data
        """
        self.block_print()
        results_f1_knn, results_acc_knn = [np.zeros(20), np.zeros(20)]
        results_f1_lr, results_acc_lr, results_f1_svm, results_acc_svm = [
            np.zeros(1) for _ in range(4)
        ]

        # k-value loop
        for k in range(1, 21):
            (
                results_f1_knn[(k - 1)],
                results_acc_knn[(k - 1)],
            ) = clf.knn_classify(k, command="tune")

        results_f1_svm[0], results_acc_svm[0] = clf.svm_classify(
            max_iter=100000, command="tune"
        )
        results_f1_lr[0], results_acc_lr[0] = clf.logistic_regression(
            max_iter=10000, command="tune"
        )

        self.save_tune_results(
            [results_f1_knn, results_f1_svm, results_f1_lr],
            [results_acc_knn, results_acc_svm, results_acc_lr],
            ["knn", "svm", "lr"],
            name,
        )

    def pca_mi_classification_params(self, name: str) -> None:
        """
        Helper function to run grid-search for pca data
        """
        self.block_print()
        results_f1_knn, results_acc_knn = [np.zeros((20, 20)), np.zeros((20, 20))]
        results_f1_lr, results_acc_lr, results_f1_svm, results_acc_svm = [
            np.zeros(20) for _ in range(4)
        ]

        # min-variance loop
        for i in range(0, 20):

            if name == "pca":
                min_variance = 0.45 + 0.01 * i
                data, _ = self.feature_extractor.pca(min_variance, self.pca)
            elif name == "mi":
                min_info = 0.45 + 0.01 * i
                data, _ = self.feature_extractor.mutual_information(min_info, self.mi)
            clf = Classification(data, self.labels)
            # k-value loop
            for k in range(1, 21):
                (
                    results_f1_knn[i][(k - 1)],
                    results_acc_knn[i][(k - 1)],
                ) = clf.knn_classify(k, command="tune")

            results_f1_svm[i], results_acc_svm[i] = clf.svm_classify(
                max_iter=100000, command="tune"
            )
            results_f1_lr[i], results_acc_lr[i] = clf.logistic_regression(
                max_iter=10000, command="tune"
            )

        self.save_tune_results(
            [results_f1_knn, results_f1_svm, results_f1_lr],
            [results_acc_knn, results_acc_svm, results_acc_lr],
            ["knn", "svm", "lr"],
            name,
        )

    def classification(self, command="tuning") -> None:
        """
        function to run grid-search/test-run depending on command
        """
        print(f"Original performance (shape: {self.samples.shape}): \n")

        self.normal_classifier = Classification(self.samples, self.labels)
        self.normal_classifier.knn_classify(k=5, command=command)
        self.normal_classifier.svm_classify(max_iter=100000, command=command)
        self.normal_classifier.logistic_regression(max_iter=10000, command=command)
        print("--------------")

        # Classify PCA dataset
        print(f"PCA performance (shape: {self.pca_data.shape}): \n")
        self.pca_classifier = Classification(self.pca_data, self.labels)
        self.pca_classifier.knn_classify(k=5, command=command)
        self.pca_classifier.svm_classify(max_iter=100000, command=command)
        self.pca_classifier.logistic_regression(max_iter=10000, command=command)
        print("--------------\n")

        # Classify Mutual Information dataset

        print(f"MI performance (shape: {self.mi_data.shape}): \n")

        self.mi_classifier = Classification(self.mi_data, self.labels)
        self.mi_classifier.knn_classify(k=5, command=command)
        self.mi_classifier.svm_classify(max_iter=100000, command=command)
        self.mi_classifier.logistic_regression(max_iter=10000, command=command)
        print("--------------")

    def cross_val(self) -> None:
        """
        function to run cross-val with full data with best pipelines
        """
        print(f"Original performance (shape: {self.samples.shape}): \n")

        self.normal_classifier = Classification(self.samples, self.labels)
        self.normal_classifier.knn_classify(k=5, command="cross-val")
        self.normal_classifier.svm_classify(max_iter=100000, command="cross-val")
        self.normal_classifier.logistic_regression(max_iter=10000, command="cross-val")
        print("--------------")

        # Classify PCA dataset
        print(f"PCA performance (shape: {self.pca_data.shape}): \n")
        self.pca_classifier = Classification(self.pca_data, self.labels)
        self.pca_classifier.knn_classify(k=5, command="cross-val")
        self.pca_classifier.svm_classify(max_iter=100000, command="cross-val")
        self.pca_classifier.logistic_regression(max_iter=10000, command="cross-val")
        print("--------------")

    def clustering(self) -> None:
        print("Clustering: \n")
        print("Original performance: \n")
        normal_clustering = Clustering(
            self.samples, y=self.labels, k_means_clusters=4, spectral_clusters=5
        )
        normal_clustering.spectral()
        print("--------------\n")

        print("PCA performance: \n")
        pca_clustering = Clustering(
            self.pca_data, y=self.labels, k_means_clusters=4, spectral_clusters=5
        )
        pca_clustering.spectral()
        print("--------------\n")

        print("Mutual Information performance: \n")
        mi_clustering = Clustering(
            x=self.mi_data, y=self.labels, k_means_clusters=4, spectral_clusters=5
        )
        mi_clustering.spectral()
        print("--------------\n")

    def ensemble(self) -> None:
        """
        function to run all possible ensembles with full data
        """
        random_state = 42  # seed

        # Classify pca dataset
        print(f"Sift performance (shape: {self.pca_data.shape}): \n")
        self.pca_classifier = Classification(self.pca_data, self.labels)
        self.pca_classifier.knn_classify(k=5, command="test")
        self.pca_classifier.svm_classify(max_iter=100000, command="test")
        self.pca_classifier.logistic_regression(max_iter=10000, command="test")

        print("\nEnsemble using KNN and Logistic Regression:")
        print("--------------")
        self.pca_classifier.ensemble(
            KNeighborsClassifier(n_neighbors=5),
            LogisticRegression(max_iter=10000, random_state=random_state),
        )
