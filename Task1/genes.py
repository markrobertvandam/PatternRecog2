#!/usr/bin/env python3
import numpy as np
import csv
import matplotlib.pyplot as plt

from feature_extraction import FeatureExtraction
from classification import Classification
from clustering import Clustering


class Genes:
    def __init__(self, pca_min_variance=0.6, mi_min_information=0.55):
        self.samples = None
        self.labels = []
        self.label_names = {"PRAD": 0, "LUAD": 1, "BRCA": 2, "KIRC": 3, "COAD": 4}
        self.feature_extractor = None
        self.pca_data = None
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
        filename_samples = f"data/Genes/data.csv"
        self.samples = np.loadtxt(
            filename_samples, delimiter=",", skiprows=1, usecols=range(1, 1024)
        )

        filename_labels = f"data/Genes/labels.csv"

        with open(filename_labels) as labels_csv:
            labels_csv_reader = csv.reader(labels_csv)
            next(labels_csv_reader, None)
            for row in labels_csv_reader:
                self.labels.append(self.label_names[row[1]])
        self.labels = np.asarray(self.labels)

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

        # Biplots to show scatter using 2 random genes
        self.biplot_helper("original", "gene", self.samples, 1)

        # Class distribution
        unique, counts = np.unique(self.labels, return_counts=True)
        plt.figure(2)
        plt.bar(unique, counts, 0.4)
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
        self.feature_extractor = FeatureExtraction(self.samples, self.labels, "genes")
        self.pca_data = self.feature_extractor.pca(self.pca_min_variance)
        self.mi_data = self.feature_extractor.mutual_information(
            self.mi_min_information
        )

    def train_validation(self) -> None:
        print("Classification: \n")
        print(f"Original performance (shape: {self.samples.shape}): \n")
        self.normal_classifier = Classification(self.samples, self.labels)
        self.normal_classifier.knn_classify()
        self.normal_classifier.svm_classify()
        self.normal_classifier.logistic_regression()
        print("--------------\n")

        print(f"PCA performance (shape: {self.pca_data.shape}): \n")
        self.pca_classifier = Classification(self.pca_data, self.labels)
        self.normal_classifier.knn_classify()
        self.normal_classifier.svm_classify()
        self.normal_classifier.logistic_regression()
        print("--------------\n")

        print(f"Mutual Information performance (shape: {self.mi_data.shape}): \n")
        self.mi_classifier = Classification(self.mi_data, self.labels)
        self.normal_classifier.knn_classify()
        self.normal_classifier.svm_classify()
        self.normal_classifier.logistic_regression()
        print("--------------\n")

    def clustering(self) -> None:
        print("Clustering: \n")
        print("Original performance: \n")
        self.normal_clustering = Clustering(self.samples)
        self.normal_clustering.k_means()
        self.normal_clustering.dbscan(eps=50, min_samples=2)
        self.normal_clustering.fuzzy_c_means()
        print("--------------\n")

        print("PCA performance: \n")
        self.pca_clustering = Clustering(self.pca_data)
        self.pca_clustering.k_means()
        self.pca_clustering.dbscan(eps=50, min_samples=2)
        self.pca_clustering.fuzzy_c_means()
        print("--------------\n")

        print("Mutual Information performance: \n")
        self.mi_clustering = Clustering(self.mi_data)
        self.mi_clustering.k_means()
        self.mi_clustering.dbscan(eps=50, min_samples=2)
        self.mi_clustering.fuzzy_c_means()
        print("--------------\n")
