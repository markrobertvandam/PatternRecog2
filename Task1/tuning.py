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
        steps: int,
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
            self.sift_data = reduced_1
            self.fourier = reduced_2
        self.steps = steps

    def tune_gene_params(self) -> None:
        """
        Function to run grid-search for all data
        """

        self.normal_classifier = Classification(self.data, self.labels)

        print("Original:")
        self.original_gene_params(self.normal_classifier, "original")

        # small sweep
        if self.steps == 6:
            print("PCA:")
            self.pca_mi_params("pca", k_offset=1, lr_offset=12)
            print("MI:")
            self.pca_mi_params("mi", glvq_offset=14)
        # big sweep
        else:
            self.pca_mi_params("pca")
            print("MI:")
            self.pca_mi_params("mi")

    def tune_cats_params(self) -> None:
        """
        Function to run grid-search for all data
        """
        if self.steps == 6:
            sift_kernels = ["sigmoid"]
            sift_c = [0.8, 0.9, 1, 1.1, 1.2, 1.3]
            sift_gamma = ["scale"]

            fourier_kernels = ["poly"]
            fourier_c = [1]
            fourier_gamma = ["scale"]
            fourier_degree = [2, 3, 4, 5, 6, 7]

            kernels = ["linear"]
            c = [0.8, 0.9, 1, 1.1, 1.2, 1.3]
            gamma = ["scale"]

            print("Original:")
            self.original_cats_params(k_offset=3, rf_offset=18,
                                      kernels=kernels, c=c, gamma=gamma)
            print("Sift:")
            self.sift_params(k_offset=20, rf_offset=10, key_pts=[255, 170, 205],
                                kernels=sift_kernels, c=sift_c, gamma=sift_gamma)
            print("Fourier:")
            self.fourier_params(k_offset=16, rf_offset=7, masks=[0, 4, 4],
                                kernels=fourier_kernels, c=fourier_c, gamma=fourier_gamma, degree=fourier_degree)
        else:
            kernels = ["linear", "poly", "rbf", "sigmoid"]
            c = [0.1, 1, 10, 100]
            gamma = ["scale", "auto", 0.0001, 0.001, 0.1, 1]

            print("Original:")
            self.original_cats_params(k_offset=1, rf_offset=1, kernels=kernels, c=c, gamma=gamma)
            print("Sift:")
            self.sift_params(key_pts=[5, 5, 5], kernels=kernels, c=c, gamma=gamma)
            print("Fourier:")
            self.fourier_params(masks=[0, 0, 0], kernels=kernels, c=c, gamma=gamma, degree=[3])

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
        dataset=None,
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

        result_path = os.path.join("data", "results", dataset)
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
        for k in range(1, 1 + self.steps):
            (
                results_f1_knn[(k - 1)],
                results_acc_knn[(k - 1)],
            ) = clf.knn_classify(k, command="tune")

        for n in range(1, 1 + self.steps):
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
                ("K-neighbors", list(range(1, 1+self.steps))),
                ("Prototypes", list(range(1, 1+self.steps))),
                None,
            ], dataset="genes",
        )

    def pca_mi_params(
        self, name: str, k_offset=0, glvq_offset=0, lr_offset=0
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
            for k in range(1, 1 + self.steps):
                (
                    results_f1_knn[i][(k - 1)],
                    results_acc_knn[i][(k - 1)],
                ) = clf.knn_classify(k + k_offset, command="tune")

            for n in range(1, 1 + self.steps):
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
                [i + k_offset for i in range(1, 1 + self.steps)],
                [i + glvq_offset for i in range(1, 1 + self.steps)],
                None,
            ],
            col_names=["K-neighbors", "Prototypes", None],
            dataset="genes",
        )

    def original_cats_params(self, kernels: list, c: list, gamma: list,
                             k_offset=0, rf_offset=0) -> None:
        """
        Helper function to run grid-search for original data or fourier data
        """
        clf = Classification(self.data, self.labels)
        kernel_size = len(kernels)
        c_size = len(c)
        gamma_size = len(gamma)
        total_size = kernel_size * c_size * gamma_size
        self.block_print()
        (
            results_f1_knn,
            results_acc_knn,
            results_f1_rf,
            results_acc_rf,
        ) = [np.zeros(self.steps) for _ in range(4)]
        results_f1_svm, results_acc_svm = [np.zeros(total_size) for _ in range(4)]

        # k-value loop
        for k in range(self.steps):
            (
                results_f1_knn[(k)],
                results_acc_knn[(k)],
            ) = clf.knn_classify(k+k_offset, command="tune")

            # SVM
            count = 0
            for j in range(len(kernels)):
                for k in range(len(c)):
                    for l in range(len(gamma)):
                        (
                            results_f1_svm[count],
                            results_acc_svm[count],
                        ) = clf.svm_classify(
                            kernel=kernels[j], c=c[k], gamma=gamma[l], degree=3, command="tune"
                        )
                        count += 1
        # n-trees loop
        for n in range(self.steps):
            n_trees = (n + rf_offset) * 20
            (
                results_f1_rf[n],
                results_acc_rf[n],
            ) = clf.random_forest(n_trees=n_trees, command="tune")

        self.save_tune_results(
            [results_f1_knn, results_f1_svm, results_f1_rf],
            [results_acc_knn, results_acc_svm, results_acc_rf],
            ["knn", "svm", "rf"],
            "original",
            rows=[("K-neighbors", list(range(k_offset, k_offset+self.steps))),
                  None,
                  ("n_trees", [(i + rf_offset) * 20 for i in range(self.steps)])],
            dataset="cats",
        )

    def fourier_params(self,  masks: list, kernels: list, c: list, gamma: list, degree: list,
                       k_offset=0, rf_offset=0) -> None:
        """
        Helper function to run grid-search for sift data

        Arguments:
        name: Name of preprocessing. (Options: "sift", "fourier")

        Returns:
        None
        """
        outer_loop = 6 if self.steps == 6 else 13
        kernel_size = len(kernels)
        c_size = len(c)
        gamma_size = len(gamma)
        deg_size = len(degree)
        total_size = kernel_size * c_size * gamma_size * deg_size
        self.block_print()
        (
            results_f1_knn,
            results_acc_knn,
            results_f1_rf,
            results_acc_rf,
        ) = [np.zeros((outer_loop, self.steps)) for _ in range(4)]
        results_f1_svm, results_acc_svm = [np.zeros((outer_loop, total_size)) for _ in range(2)]

        # Fourier masking loop
        for i in range(outer_loop):
            knn_fourier_data, _ = self.feature_extractor.fourier_transform(
                i * 2 + masks[0], self.fourier
            )
            knn_fourier_data = knn_fourier_data.reshape(
                knn_fourier_data.shape[0],
                knn_fourier_data.shape[1] * knn_fourier_data.shape[2],
            )

            svm_fourier_data, _ = self.feature_extractor.fourier_transform(
                i * 2 + masks[1], self.fourier
            )
            svm_fourier_data = svm_fourier_data.reshape(
                svm_fourier_data.shape[0],
                svm_fourier_data.shape[1] * svm_fourier_data.shape[2],
            )

            rf_fourier_data, _ = self.feature_extractor.fourier_transform(
                i * 2 + masks[2], self.fourier
            )
            rf_fourier_data = rf_fourier_data.reshape(
                rf_fourier_data.shape[0],
                rf_fourier_data.shape[1] * rf_fourier_data.shape[2],
            )

            knn_classifier = Classification(knn_fourier_data, self.labels)
            svm_classifier = Classification(svm_fourier_data, self.labels)
            rf_classifier = Classification(rf_fourier_data, self.labels)

            # k-value loop knn
            for k in range(self.steps):
                (
                    results_f1_knn[i][(k)],
                    results_acc_knn[i][(k)],
                ) = knn_classifier.knn_classify(k + k_offset, command="tune")

            # SVM
            count=0
            for j in range(len(kernels)):
                for k in range(len(c)):
                    for l in range(len(gamma)):
                        for m in range(len(degree)):
                            (
                                results_f1_svm[i][count],
                                results_acc_svm[i][count],
                            ) = svm_classifier.svm_classify(
                                kernel=kernels[j], c=c[k], gamma=gamma[l], degree=degree[m], command="tune"
                            )
                            count+=1

            # n-trees loop
            for n in range(self.steps):
                n_trees = (n+rf_offset) * 20
                (
                    results_f1_rf[i][n],
                    results_acc_rf[i][n],
                ) = rf_classifier.random_forest(n_trees=n_trees, command="tune")

        self.save_tune_results(
            [results_f1_knn, results_f1_svm, results_f1_rf],
            [results_acc_knn, results_acc_svm, results_acc_rf],
            ["knn", "svm", "rf"],
            "fourier",
            rows=[("Mask radius", [2 * i + masks[0] for i in range(self.steps)]),
                  ("Mask radius", [2 * i + masks[1] for i in range(self.steps)]),
                  ("Mask radius", [2 * i + masks[2] for i in range(self.steps)])],
            cols=[[i + k_offset for i in range(self.steps)],
                  None,
                  [(i + rf_offset) * 20 for i in range(self.steps)]],
            col_names=["K-Neighbors", "Degree", "n_trees"],
            dataset="cats",
        )

    def sift_params(self, key_pts: list, kernels: list, c: list, gamma: list,
                    k_offset=0, rf_offset=0) -> None:
        """
        Helper function to run grid-search for sift data

        Arguments:
        name: Name of preprocessing. (Options: "sift", "fourier")

        Returns:
        None
        """
        outer_loop = 6 if self.steps == 6 else 60
        kernel_size = len(kernels)
        c_size = len(c)
        gamma_size = len(gamma)
        total_size = kernel_size * c_size * gamma_size
        self.block_print()
        (
            results_f1_knn,
            results_acc_knn,
            results_f1_rf,
            results_acc_rf,
        ) = [np.zeros((outer_loop, self.steps)) for _ in range(4)]
        results_f1_svm, results_acc_svm = [np.zeros((outer_loop, total_size)) for _ in range(2)]

        # Max keypoints loop

        for i in range(outer_loop):
            key_points = i * 5

            knn_reduced_sift = self.sift_data[:, 0 : key_points + key_pts[0]]
            knn_classifier = Classification(knn_reduced_sift, self.labels)

            svm_reduced_sift = self.sift_data[:, 0 : key_points + key_pts[1]]
            svm_classifier = Classification(svm_reduced_sift, self.labels)

            rf_reduced_sift = self.sift_data[:, 0 : key_points + key_pts[2]]
            rf_classifier = Classification(rf_reduced_sift, self.labels)

            # k-value loop knn
            for k in range(self.steps):
                (
                    results_f1_knn[i][(k)],
                    results_acc_knn[i][(k)],
                ) = knn_classifier.knn_classify(k + k_offset, command="tune")

            # SVM
            count = 0
            for j in range(len(kernels)):
                for k in range(len(c)):
                    for l in range(len(gamma)):
                        (
                            results_f1_svm[i][count],
                            results_acc_svm[i][count],
                        ) = svm_classifier.svm_classify(
                            kernel=kernels[j], c=c[k], gamma=gamma[l], degree=3, command="tune"
                        )
                        count+=1

            # n-trees loop
            for n in range(self.steps):
                n_trees = (n+rf_offset) * 20
                (
                    results_f1_rf[i][n],
                    results_acc_rf[i][n],
                ) = rf_classifier.random_forest(n_trees=n_trees, command="tune")

        self.save_tune_results(
            [results_f1_knn, results_f1_svm, results_f1_rf],
            [results_acc_knn, results_acc_svm, results_acc_rf],
            ["knn", "svm", "rf"],
            "sift",
            rows=[("Max_Keypoints", [5*i + key_pts[0] for i in range(self.steps)]),
                  ("Max_Keypoints", [5*i + key_pts[1] for i in range(self.steps)]),
                  ("Max_Keypoints", [5*i + key_pts[2] for i in range(self.steps)])],
            cols=[[i+k_offset for i in range(self.steps)],
                  None,
                  [(i+rf_offset)*20 for i in range(self.steps)]],
            col_names=["K-Neighbors", "C", "n_trees"],
            dataset="cats",
        )
