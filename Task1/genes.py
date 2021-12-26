#!/usr/bin/env python3
import numpy as np
import csv
import sklearn

from sklearn.model_selection import train_test_split
from feature_extraction import FeatureExtraction


class Genes:
    def __init__(self, pca_min_variance=0.95):
        self.samples = []
        self.labels = []
        self.feature_extractor = None
        self.pca_data = None
        self.sift_data = None
        self.pca_min_variance = pca_min_variance

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

    def feature_extraction(self) -> None:
        self.feature_extractor = FeatureExtraction(self.samples, self.labels, "genes")
        self.pca_data = self.feature_extractor.pca(self.pca_min_variance)
