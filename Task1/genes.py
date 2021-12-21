#!/usr/bin/env python3
import numpy as np
import csv
import sklearn

from sklearn.model_selection import train_test_split


class Genes:
    def __init__(self):
        self.samples = []
        self.labels = []

    def load_data(self) -> None:
        filename_samples = f"data/Genes/data.csv"
        filename_labels = f"data/Genes/labels.csv"
        labels = {"PRAD":0, "LUAD":1, "BRCA":2, "KIRC":3, "COAD":4}
        with open(filename_samples, "r") as samples_csv, open(filename_labels) as labels_csv:
            samples_csv_reader = csv.reader(samples_csv)
            labels_csv_reader = csv.reader(labels_csv)
            next(samples_csv_reader, None)
            next(labels_csv_reader, None)
            for row in samples_csv_reader:
                self.samples.append(np.array(row[1:]))
            for row in labels_csv_reader:
                self.labels.append(labels[row[1]])