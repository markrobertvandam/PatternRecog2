#!/usr/bin/env python3
import numpy as np
import csv

from feature_extraction import FeatureExtraction
from classification import Classification
from clustering import Clustering


class Genes:
    def __init__(self, pca_min_variance=0.95, mi_min_information=0.2):
        self.samples = None
        self.labels = []
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
        labels = {"PRAD": 0, "LUAD": 1, "BRCA": 2, "KIRC": 3, "COAD": 4}
        with open(filename_labels) as labels_csv:
            labels_csv_reader = csv.reader(labels_csv)
            next(labels_csv_reader, None)
            for row in labels_csv_reader:
                self.labels.append(labels[row[1]])
        self.labels = np.asarray(self.labels)

    def feature_extraction(self) -> None:
        self.feature_extractor = FeatureExtraction(self.samples, self.labels, "genes")
        self.pca_data = self.feature_extractor.pca(self.pca_min_variance)
        self.mi_data = self.feature_extractor.mutual_information(
            self.mi_min_information
        )

    def classification(self) -> None:
        print("Classification: \n")
        print("Original performance: \n")
        self.normal_classifier = Classification(self.samples, self.labels)
        self.normal_classifier.random_forest()
        self.normal_classifier.nb_classify()
        print("--------------\n")

        print("PCA performance: \n")
        self.pca_classifier = Classification(self.pca_data, self.labels)
        self.pca_classifier.random_forest()
        self.pca_classifier.nb_classify()
        print("--------------\n")

        print("Mutual Information performance: \n")
        self.mi_classifier = Classification(self.mi_data, self.labels)
        self.mi_classifier.random_forest()
        self.mi_classifier.nb_classify()
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

