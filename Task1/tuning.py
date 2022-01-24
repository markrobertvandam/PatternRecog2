#!/usr/bin/env python3

import numpy as np
import os
import pandas as pd
import sys

from classification import Classification
from feature_extraction import FeatureExtraction


class Tuning:
    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        reduced_1: np.ndarray,
        reduced_2: np.ndarray,
        name: str,
        steps: int
    ) -> None:
        """
        Initialize dataset, splitting and model parameters.

        Arguments:
        X: Input features.
        y: Class labels.

        Returns:
        None
        """
        self.data = data
        self.labels = labels
        self.feature_extractor = FeatureExtraction(data, labels, name)
        if name == "genes":
            self.pca = reduced_1
            self.mi = reduced_2
        if name == "cats":
            self.sift = reduced_1
            self.fourier = reduced_2
        self.steps = steps

    def tune_gene_params(self) -> None:
        """
        Function to run grid-search for all data
        """

        self.normal_classifier = Classification(self.data, self.labels)

        print("Original:")
        self.original_gene_params(self.normal_classifier, "original")
        print("PCA:")
        self.pca_mi_params("pca", k_offset=1, glvq_offset=0, lr_offset=12)
        print("MI:")
        self.pca_mi_params("mi", k_offset=0, glvq_offset=14, lr_offset=0)

    # Disable
    def block_print(self):
        sys.stdout = open(os.devnull, "w")
        sys.stdout = open(os.devnull, "w")

    # Restore
    def enable_print(self):
        sys.stdout = sys.__stdout__

    def save_tune_results(
        self,
        f1_results: list,
        acc_results: list,
        models: list,
        name: str,
        cols=None,
        rows=None,
        col_names=None,
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
            index_lab = None
            if rows:
                if rows[i]:
                    index_lab = rows[i][0]
                df_acc = pd.DataFrame(
                    data=acc_results[i],
                    index=rows[i][1] if rows[i] else None,
                    columns=cols[i] if cols else None,
                )
                df_f1 = pd.DataFrame(
                    data=f1_results[i],
                    index=rows[i][1] if rows[i] else None,
                    columns=cols[i] if cols else None,
                )

            else:
                df_acc = pd.DataFrame(
                    data=acc_results[i], columns=cols[i] if cols else None
                )
                df_f1 = pd.DataFrame(
                    data=f1_results[i], columns=cols[i] if cols else None
                )

            if cols:
                if cols[i]:
                    df_acc = pd.concat([df_acc], keys=[col_names[i]], axis=1)
                    df_f1 = pd.concat([df_f1], keys=[col_names[i]], axis=1)
            save_f1 = os.path.join(result_path, f"{name}_f1_{models[i]}.csv")
            save_acc = os.path.join(result_path, f"{name}_acc_{models[i]}.csv")

            df_acc.to_csv(save_acc, index_label=index_lab)
            df_f1.to_csv(save_f1, index_label=index_lab)

    def original_gene_params(self, clf, name: str) -> None:
        """
        Helper function to run grid-search for original data

        Arguments:
        clf: Classifier model.
        name: Name for saving tuning results.

        Returns:
        None
        """

        self.block_print()
        results_f1_knn, results_acc_knn, results_f1_glvq, results_acc_glvq = [
            np.zeros(self.steps) for _ in range(4)
        ]
        results_f1_lr, results_acc_lr = [np.zeros(1), np.zeros(1)]

        # k-value loop
        for k in range(1, 1+self.steps):
            (
                results_f1_knn[(k - 1)],
                results_acc_knn[(k - 1)],
            ) = clf.knn_classify(k, command="tune")

        for n in range(1, 1+self.steps):
            (
                results_f1_glvq[n - 1],
                results_acc_glvq[n - 1],
            ) = clf.glvq_classify(prototypes_per_class=n, command="tune")

        results_f1_lr[0], results_acc_lr[0] = clf.logistic_regression(
            max_iter=10000, command="tune"
        )

        self.save_tune_results(
            [results_f1_knn, results_f1_glvq, results_f1_lr],
            [results_acc_knn, results_acc_glvq, results_acc_lr],
            ["knn", "glvq", "lr"],
            name,
            rows=[
                ("K-neighbors", list(range(1, 7))),
                ("Prototypes", list(range(1, 7))),
                None,
            ],
        )

    def pca_mi_params(
        self, name: str, k_offset: int, glvq_offset: int, lr_offset: int
    ) -> None:
        """
        Helper function to run grid-search for pca data

        Arguments:
        name: Name of feature extractor (Options: "pca", "mi")
        k_offset: Offset for neighbor value.
        glvq_offset: Offset for LVQ.
        lr_offset: Offset for PCA.

        Returns:
        None
        """

        self.block_print()
        results_f1_knn, results_acc_knn, results_f1_glvq, results_acc_glvq = [
            np.zeros((self.steps, self.steps)) for _ in range(4)
        ]
        results_f1_lr, results_acc_lr = [np.zeros(self.steps), np.zeros(self.steps)]

        # min-variance loop
        min_values = [0.45 + 0.01 * i for i in range(0, self.steps)]
        for i in range(0, self.steps):
            if name == "pca":
                min_variance = min_values[i]
                data, _ = self.feature_extractor.pca(min_variance, self.pca)
                lr_data, _ = self.feature_extractor.pca(
                    min_variance + 0.01 * lr_offset, self.pca
                )
            elif name == "mi":
                min_info = min_values[i]
                data, _ = self.feature_extractor.mutual_information(min_info, self.mi)
                lr_data, _ = self.feature_extractor.mutual_information(
                    min_info + 0.01 * lr_offset, self.mi
                )
            clf = Classification(data, self.labels)
            clf_lr = Classification(lr_data, self.labels)
            # k-value loop
            for k in range(1, 1+self.steps):
                (
                    results_f1_knn[i][(k - 1)],
                    results_acc_knn[i][(k - 1)],
                ) = clf.knn_classify(k + k_offset, command="tune")

            for n in range(1, 1+self.steps):
                (
                    results_f1_glvq[i][(n - 1)],
                    results_acc_glvq[i][(n - 1)],
                ) = clf.glvq_classify(
                    prototypes_per_class=n + glvq_offset, command="tune"
                )

            results_f1_lr[i], results_acc_lr[i] = clf_lr.logistic_regression(
                max_iter=10000, command="tune"
            )
        if name == "pca":
            row_name = "Min-variance"
        else:
            row_name = "Min-info"
        self.save_tune_results(
            [results_f1_knn, results_f1_glvq, results_f1_lr],
            [results_acc_knn, results_acc_glvq, results_acc_lr],
            ["knn", "glvq", "lr"],
            name,
            rows=[
                (row_name, min_values),
                (row_name, min_values),
                (row_name, [i + 0.01 * lr_offset for i in min_values]),
            ],
            cols=[
                [i + k_offset for i in range(1, 1+self.steps)],
                [i + glvq_offset for i in range(1, 1+self.steps)],
                None,
            ],
            col_names=["K-neighbors", "Prototypes", None],
        )
