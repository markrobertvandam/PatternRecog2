#!/usr/bin/env python3
import numpy as np
import csv
import matplotlib.pyplot as plt

from classification import Classification
from clustering import Clustering
from feature_extraction import FeatureExtraction


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
        print("Loading data...")
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
        print("Visualizing the data...")
        # Biplots to show scatter using 2 random genes
        self.biplot_helper("original", "gene", self.samples, 1)

        # Class distribution
        unique, counts = np.unique(self.labels, return_counts=True)
        plt.figure(2)
        print(unique)
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
        self.pca_data = self.feature_extractor.pca(self.pca_min_variance)
        self.mi_data = self.feature_extractor.mutual_information(
            self.mi_min_information
        )

    def classification(self, command="tuning") -> None:
        """
        function to run grid-search/test-run depending on command
        """
        print(f"Original performance (shape: {self.samples.shape}): \n")

        self.normal_classifier = Classification(self.samples, self.labels)
        self.normal_classifier.knn_classify(command=command)
        self.normal_classifier.nb_classify(command=command)
        self.normal_classifier.random_forest(command=command)
        print("--------------")

        # Classify PCA dataset
        print(f"PCA performance (shape: {self.pca_data.shape}): \n")
        self.pca_classifier = Classification(self.pca_data, self.labels)
        self.pca_classifier.knn_classify(command=command)
        self.pca_classifier.nb_classify(command=command)
        self.pca_classifier.random_forest(command=command)
        print("--------------\n")

        # Classify Mutual Information dataset

        print(f"MI performance (shape: {self.mi_data.shape}): \n")

        self.mi_classifier = Classification(self.mi_data, self.labels)
        self.mi_classifier.knn_classify(command=command)
        self.mi_classifier.nb_classify(command=command)
        self.mi_classifier.random_forest(command=command)
        print("--------------")

    def cross_val(self) -> None:
        """
        function to run cross-val with full data with best pipelines
        """
        print(f"Original performance (shape: {self.samples.shape}): \n")

        self.normal_classifier = Classification(self.samples, self.labels)
        self.normal_classifier.knn_classify(command="cross-val")
        self.normal_classifier.nb_classify(command="cross-val")
        self.normal_classifier.random_forest(command="cross-val")
        print("--------------")

        # Classify PCA dataset
        print(f"PCA performance (shape: {self.pca_data.shape}): \n")
        self.pca_classifier = Classification(self.pca_data, self.labels)
        self.pca_classifier.knn_classify(command="cross-val")
        self.pca_classifier.nb_classify(command="cross-val")
        self.pca_classifier.random_forest(command="cross-val")
        print("--------------")

    def clustering(self) -> None:
        print("Clustering: \n")
        print("Original performance: \n")
        normal_clustering = Clustering(self.samples)
        normal_clustering.k_means()
        print("--------------\n")

        print("PCA performance: \n")
        pca_clustering = Clustering(self.pca_data)
        pca_clustering.k_means()
        print("--------------\n")

        print("Mutual Information performance: \n")
        mi_clustering = Clustering(self.mi_data)
        mi_clustering.k_means(n_clusters=4)
        print("--------------\n")

    def ensemble(self) -> None:
        self.pca_classifier = Classification(self.pca_data, self.labels)
        self.pca_classifier.ensemble()
