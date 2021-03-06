#!/usr/bin/env python3
import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

from classification import Classification
from clustering import Clustering
from feature_extraction import FeatureExtraction

from tuning import Tuning


class Genes:
    def __init__(self, pca_min_variance=0.65, mi_min_information=0.55):
        """
        Initialize dataset, feature extractorsa and clustering algorithms
        for gene dataset.
        """

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
        """
        Function to load the data.
        """

        print("Loading data...")
        genes_path = os.path.join("data", "Genes")
        filename_samples = os.path.join(genes_path, "data.csv")
        self.samples = np.loadtxt(
            filename_samples, delimiter=",", skiprows=1, usecols=range(1, 1024)
        )

        filename_labels = os.path.join(genes_path, "labels.csv")

        labels = []
        with open(filename_labels) as labels_csv:
            labels_csv_reader = csv.reader(labels_csv)
            next(labels_csv_reader, None)
            for row in labels_csv_reader:
                labels.append(self.label_names[row[1]])
        self.labels = np.asarray(labels, dtype=int)

    def biplot_helper(self, name: str, vars: str, data: np.ndarray, offset=0) -> None:
        """
        Function to plot gene features.

        Arguments:
        name: Filename for saving plots.
        vars: Class labels of gene dataset.
        data: Input feature array.
        offset: Offset for shifting plot data.

        Returns:
        None
        """

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
        plt.savefig(os.path.join("plots", f"biplots_{name}"))
        plt.close(fig)

    def visualize_data(self) -> None:
        """
        Function to visualize gene data.
        """

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

        plt.savefig(os.path.join("plots", "genes_histo.png"))
        plt.close()

        # Biplots using most informaive PCA-components
        self.biplot_helper("pca", "pca", self.pca_data)

        # Biplots using most informaive MI-components
        self.biplot_helper("mi", "mi", self.mi_data)

    def feature_extraction(self) -> None:
        """
        Function to extract features using PCA and
        mutual information.
        """

        print("Doing feature extraction...")
        self.feature_extractor = FeatureExtraction(self.samples, self.labels, "genes")
        self.pca_data, self.pca = self.feature_extractor.pca(self.pca_min_variance)
        self.mi_data, self.mi = self.feature_extractor.mutual_information(
            self.mi_min_information
        )

    def tune_classification_params(self, option=None) -> None:
        """
        Function to run grid-search for all data
        """
        if option:
            genes_tuner = Tuning(
                self.samples, self.labels, self.pca, self.mi, "genes", 20
            )
            genes_tuner.tune_gene_params()
        else:
            genes_tuner = Tuning(
                self.samples, self.labels, self.pca, self.mi, "genes", 6
            )
            genes_tuner.tune_gene_params()

    def classification(self, command: str) -> None:
        """
        function to run cross-val/test-run depending on command

        Arguments:
        command: Commant specific for model operations.
        """

        print(f"Original performance (shape: {self.samples.shape}): \n")

        self.normal_classifier = Classification(self.samples, self.labels)
        self.normal_classifier.knn_classify(k=5, command=command)
        print("--------------")

        # Classify PCA dataset
        print(f"PCA performance (shape: {self.pca_data.shape}): \n")
        self.pca_classifier = Classification(self.pca_data, self.labels)
        self.pca_classifier.knn_classify(k=5, command=command)
        print("--------------\n")

    # Disable
    def block_print(self):
        sys.stdout = open(os.devnull, "w")
        sys.stdout = open(os.devnull, "w")

    # Restore
    def enable_print(self):
        sys.stdout = sys.__stdout__

    def clustering(self) -> None:
        """
        Function to perform clustering.
        """
        small = 1
        result_path = os.path.join("data", "results", "clustering")
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if small:
            normal_km_s, normal_km_mi = [np.zeros(6), np.zeros(6)]
            pca_km_s_max_s, pca_km_mi_max_s, pca_km_s_max_mi, pca_km_mi_max_mi = [
                np.zeros((6, 6)) for _ in range(4)
            ]

            pca_s = [pca_km_s_max_s, pca_km_s_max_mi]
            pca_mi = [pca_km_mi_max_s, pca_km_mi_max_mi]
            var_regions = [0.45, 0.59]
            start_neighbors = [3, 4]

            print("Clustering: \n")
            print("Original performance: \n")
            self.block_print()
            normal_clustering = Clustering(self.samples, y=self.labels)
            for k in range(4, 10):
                normal_km_s[(k - 4)], normal_km_mi[(k - 4)] = normal_clustering.k_means(
                    n_clusters=k
                )
            self.enable_print()
            print("--------------\n")
            print(
                "Max sil score normal km = "
                + str(np.max(normal_km_s))
                + ", Max mi score normal km = "
                + str(np.max(normal_km_mi))
            )

            np.savetxt(os.path.join(result_path, "normal_km_s.csv"), normal_km_s)
            np.savetxt(os.path.join(result_path, "normal_km_mi.csv"), normal_km_mi)

            print("PCA performance: \n")
            self.block_print()
            for region in range(2):
                for i in range(6):
                    current_pca_data, _ = self.feature_extractor.pca(
                        var_regions[region] + 0.01 * i, self.pca
                    )
                    pca_clustering = Clustering(current_pca_data, y=self.labels)
                    for k in range(
                        start_neighbors[region], start_neighbors[region] + 6
                    ):
                        (
                            pca_s[region][i][(k - start_neighbors[region])],
                            pca_mi[region][i][(k - start_neighbors[region])],
                        ) = pca_clustering.k_means(n_clusters=k)
            self.enable_print()
            print("--------------\n")

            print(
                "Max sil score pca km sil region = "
                + str(np.max(pca_s[0]))
                + ", Max mi score pca km sil region = "
                + str(np.max(pca_mi[0]))
            )
            print(
                "Max sil score pca km mi region = "
                + str(np.max(pca_s[1]))
                + ", Max mi score pca km mi region = "
                + str(np.max(pca_mi[1]))
            )

            np.savetxt(os.path.join(result_path, "pca_km_s_max_s.csv"), pca_s[0])
            np.savetxt(os.path.join(result_path, "pca_km_mi_max_s.csv"), pca_mi[0])

            np.savetxt(os.path.join(result_path, "pca_km_s_max_mi.csv"), pca_s[1])
            np.savetxt(os.path.join(result_path, "pca_km_mi_max_mi.csv"), pca_mi[1])
        else:
            normal_km_s, normal_km_mi = [np.zeros(24), np.zeros(24)]
            pca_km_s, pca_km_mi = [np.zeros((20, 24)), np.zeros((20, 24))]

            print("Clustering: \n")
            print("Original performance: \n")
            self.block_print()
            normal_clustering = Clustering(self.samples, y=self.labels)
            for k in range(2, 26):
                normal_km_s[(k - 2)], normal_km_mi[(k - 2)] = normal_clustering.k_means(
                    n_clusters=k
                )
            self.enable_print()
            print("--------------\n")
            print(
                "Max sil score normal km = "
                + str(np.max(normal_km_s))
                + ", Max mi score normal km = "
                + str(np.max(normal_km_mi))
            )

            np.savetxt(os.path.join(result_path, "normal_km_s.csv"), normal_km_s)
            np.savetxt(os.path.join(result_path, "normal_km_mi.csv"), normal_km_mi)

            print("PCA performance: \n")
            self.block_print()
            for i in range(20):
                current_pca_data, _ = self.feature_extractor.pca(
                    0.45 + 0.01 * i, self.pca
                )
                pca_clustering = Clustering(current_pca_data, y=self.labels)
                for k in range(2, 26):
                    (
                        pca_km_s[i][(k - 2)],
                        pca_km_mi[i][(k - 2)],
                    ) = pca_clustering.k_means(n_clusters=k)
            self.enable_print()
            print("--------------\n")

            print(
                "Max sil score pca km = "
                + str(np.max(pca_km_s))
                + ", Max mi score pca km = "
                + str(np.max(pca_km_mi))
            )

            np.savetxt(os.path.join(result_path, "pca_km_s.csv"), pca_km_s)
            np.savetxt(os.path.join(result_path, "pca_km_mi.csv"), pca_km_mi)
