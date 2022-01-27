#!/usr/bin/env python3

import skimage
import cv2
import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import shutil
import sys

from classification import Classification
from clustering import Clustering
from feature_extraction import FeatureExtraction

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from tuning import Tuning


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
        self.fourier_unmasked = None
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
        skf = StratifiedKFold(n_splits=5)
        f1_arr = []
        acc_arr = []
        f1_avg = 0
        acc_avg = 0
        f1_arr_aug = []
        acc_arr_aug = []
        f1_avg_aug = 0
        acc_avg_aug = 0
        k = 1
        for train_index, test_index in skf.split(self.gray_images, self.labels):
            print(f"K-fold {k}...")
            x_train, x_test = (
                self.gray_images[train_index],
                self.gray_images[test_index],
            )
            y_train, y_test = self.labels[train_index], self.labels[test_index]

            f1, acc = self.augmented_fold(x_train, x_test, y_train, y_test)

            f1_arr.append(f1)
            f1_avg += f1 / 5
            acc_arr.append(acc)
            acc_avg += acc / 5

            f1, acc = self.augmentation(x_train, x_test, y_train, y_test)

            f1_arr_aug.append(f1)
            f1_avg_aug += f1 / 5
            acc_arr_aug.append(acc)
            acc_avg_aug += acc / 5
            k += 1

        print(f"F1-scores: {f1_arr}, Average: {f1_avg}")
        print(f"Accuracy scores: {acc_arr}, Average: {acc_avg}")
        print(f"F1-scores augmented: {f1_arr_aug}, Average: {f1_avg_aug}")
        print(f"Accuracy scores augmented: {acc_arr_aug}, Average: {acc_avg_aug}")

        print("\n Single test runs: \n")
        x_train, x_test, y_train, y_test = train_test_split(
            self.gray_images,
            self.labels,
            test_size=0.2,
            random_state=42,
            stratify=self.labels,
        )
        f1, acc = self.augmented_fold(x_train, x_test, y_train, y_test)
        print(f"Normal run F1: {f1}, Acc: {acc}")
        f1, acc = self.augmentation(x_train, x_test, y_train, y_test)
        print(f"Augmented run F1: {f1}, Acc: {acc}")

    def augmentation(self, x_train, x_test, y_train, y_test):
        """
        Helper function for augment_run
        """
        augmented_gray = []
        print(f"Augmenting images...")
        for i in range(len(x_train)):
            image = x_train[i]
            augmented_gray += Cats.augment_image(image)
        augmented_labels = np.repeat(y_train, 3, axis=0)
        f1, acc = self.augmented_fold(
            augmented_gray, x_test, augmented_labels, y_test, "augmented"
        )
        return f1, acc

    def augmented_fold(self, x_train, x_test, y_train, y_test, fold="un-agumented"):
        """
        Helper function for augment_run
        """
        full_x = np.concatenate((x_train, x_test))
        full_y = np.concatenate((y_train, y_test))
        test_len = len(y_test)

        print(f"Running SIFT Feature Extraction for {fold} data...")
        feature_extractor = FeatureExtraction(full_x, full_y, "cats")
        sift_data, bad_imgs = feature_extractor.sift(225)
        if fold == "augmented":
            print("\n")
            y_train = np.delete(y_train, bad_imgs)

        classifier_rf = RandomForestClassifier(280, random_state=42)
        classifier_rf.fit(sift_data[:-test_len], y_train)

        y_pred = classifier_rf.predict(sift_data[-test_len:])
        f1 = f1_score(y_test, y_pred, average="macro")
        acc = accuracy_score(y_test, y_pred)

        return f1, acc

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
        print(counts)
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
        (
            self.fourier_data,
            self.fourier_unmasked,
        ) = self.feature_extractor.fourier_transform()

        # flatten fourier
        self.fourier_data = self.fourier_data.reshape(
            self.fourier_data.shape[0],
            self.fourier_data.shape[1] * self.fourier_data.shape[2],
        )

        self.sift_data, _ = self.feature_extractor.sift()

    def tune_classification_params(self, option=None) -> None:
        """
        Function to run grid-search for all data
        """
        if option:
            cats_tuner = Tuning(
                self.flattened_original,
                self.labels,
                self.sift_data,
                self.fourier_unmasked,
                "cats",
                25,
            )
            cats_tuner.tune_cats_params()
        else:
            cats_tuner = Tuning(
                self.flattened_original,
                self.labels,
                self.sift_data,
                self.fourier_unmasked,
                "cats",
                6,
            )
            cats_tuner.tune_cats_params()

    def classification(self, command) -> None:
        """
        function to run cross-val or test-run depending on command with best pipelines

        Arguments:
        command: String for specifying the type of model operation.

        Returns:
        None
        """

        print(f"Original performance (shape: {self.flattened_original.shape}): \n")

        normal_classifier = Classification(self.flattened_original, self.labels)
        print("Best number of trees for original data")
        normal_classifier.random_forest(n_trees=460, command=command)
        print("\nBest number of trees for SIFT data")
        normal_classifier.random_forest(n_trees=220, command=command)
        print("--------------")

        # Classify sift dataset
        print(f"Sift performance (shape: {self.sift_data.shape}): \n")
        sift_classifier_rf = Classification(self.sift_data[:, 0:210], self.labels)
        print("Best number of trees for original data")
        sift_classifier_rf.random_forest(n_trees=460, command=command)
        print("\nBest number of trees for SIFT data")
        sift_classifier_rf.random_forest(n_trees=220, command=command)
        print("--------------\n")

    def ensemble(self) -> None:
        """
        function to run all possible ensembles with full data
        """

        # Classify sift dataset
        print(f"Sift performance (shape: {self.sift_data[:, 0:210].shape}): \n")
        self.sift_classifier = Classification(self.sift_data[:, 0:210], self.labels)
        self.sift_classifier.knn_classify(k=25, command="test")
        self.sift_classifier.knn_classify(k=25, command="cross-val")

        self.sift_classifier.svm_classify(
            kernel="sigmoid", c=1.1, gamma="scale", degree=3, command="test"
        )
        self.sift_classifier.svm_classify(
            kernel="sigmoid", c=1.1, gamma="scale", degree=3, command="cross-val"
        )

        print("\nEnsemble using SVM and Random Forest:")
        print("--------------")
        self.sift_classifier.ensemble(
            SVC(
                kernel="sigmoid",
                C=1.1,
                gamma="scale",
                degree=3,
                probability=True,
                random_state=42,
            ),
            RandomForestClassifier(n_estimators=220, random_state=42),
        )

        print("\nEnsemble using SVC and KNN:")
        print("--------------")
        self.sift_classifier.ensemble(
            SVC(
                kernel="sigmoid",
                C=1.1,
                gamma="scale",
                degree=3,
                probability=True,
                random_state=42,
            ),
            KNeighborsClassifier(n_neighbors=25),
        )

        print("\nEnsemble using KNN and Random Forest:")
        print("--------------")
        self.sift_classifier.ensemble(
            KNeighborsClassifier(n_neighbors=25),
            RandomForestClassifier(n_estimators=220, random_state=42),
        )

        print("\nEnsemble using KNN, SVC and Random Forest:")
        print("--------------")
        self.sift_classifier.ensemble(
            KNeighborsClassifier(n_neighbors=25),
            SVC(
                kernel="sigmoid",
                C=1.1,
                degree=3,
                gamma="scale",
                probability=True,
                random_state=42,
            ),
            RandomForestClassifier(n_estimators=220, random_state=42),
        )

        print("--------------")

    # Disable
    def block_print(self):
        sys.stdout = open(os.devnull, "w")
        sys.stdout = open(os.devnull, "w")

    # Restore
    def enable_print(self):
        sys.stdout = sys.__stdout__

    def clustering(self) -> None:
        """
        Function to perform clustering of data.
        """
        small = 1

        if small:
            (
                normal_spec_s,
                normal_spec_mi,
                sift_spec_s_max_s,
                sift_spec_mi_max_s,
                sift_spec_s_max_mi,
                sift_spec_mi_max_mi,
            ) = [np.zeros((6, 6)) for _ in range(6)]
            # normal_spec_s, normal_spec_mi, sift_spec_s, sift_spec_mi = [np.zeros((6, 6)) for _ in range(4)]

            # sift_s = [sift_spec_s_max_s, sift_spec_s_max_mi]
            # sift_mi = [sift_spec_mi_max_s, sift_spec_mi_max_mi]
            # key_regions = [1, 9]
            # n_clusters = [2, 2]
            # n_neighbors = [5, 12]

            print("Clustering: \n")
            print("Original performance: \n")
            self.block_print()
            normal_clustering = Clustering(self.flattened_original, self.labels)

            for i in range(10, 16):
                for j in range(3, 9):
                    (
                        normal_spec_s[(i - 10)][(j - 3)],
                        normal_spec_mi[(i - 10)][(j - 3)],
                    ) = normal_clustering.spectral(n_clusters=i, n_neighbors=j)

            self.enable_print()
            print("--------------\n")
            print(
                "Max sil score normal spec = "
                + str(np.max(normal_spec_s))
                + ", Max mi score normal spec = "
                + str(np.max(normal_spec_mi))
            )

            np.savetxt("data/results/clustering/normal_spec_s.csv", normal_spec_s)
            np.savetxt("data/results/clustering/normal_spec_mi.csv", normal_spec_mi)

            for i in range(8, 14):
                key_points = i * 5

                reduced_sift = self.sift_data[:, 0 : key_points * 5 + 5]

                # print('Max keypoints = ' + str(((i*5) + 5)))
                self.block_print()
                sift_clustering = Clustering(reduced_sift, self.labels)
                k = 12
                for j in range(2, 8):
                    (
                        sift_spec_s_max_mi[i - 8][(j - 2)],
                        sift_spec_mi_max_mi[i - 8][(j - 2)],
                    ) = sift_clustering.spectral(n_clusters=j, n_neighbors=k)

                self.enable_print()
            print("--------------\n")

            print(
                "Max sil score sift spec = "
                + str(np.max(sift_spec_s_max_mi))
                + ", Max mi score sift spec = "
                + str(np.max(sift_spec_mi_max_mi))
            )

            np.savetxt(
                "data/results/clustering/sift_spec_s_max_mi.csv", sift_spec_s_max_mi
            )
            np.savetxt(
                "data/results/clustering/sift_spec_mi_max_mi.csv", sift_spec_mi_max_mi
            )

            for i in range(0, 6):
                key_points = i * 5

                reduced_sift = self.sift_data[:, 0 : key_points * 5 + 5]

                # print('Max keypoints = ' + str(((i*5) + 5)))
                self.block_print()
                sift_clustering = Clustering(reduced_sift, self.labels)
                k = 9
                for j in range(2, 8):
                    (
                        sift_spec_s_max_s[i][(j - 2)],
                        sift_spec_mi_max_s[i][(j - 2)],
                    ) = sift_clustering.spectral(n_clusters=j, n_neighbors=k)

                self.enable_print()
            print("--------------\n")

            print(
                "Max sil score sift spec s = "
                + str(np.max(sift_spec_s_max_s))
                + ", Max mi score sift spec s = "
                + str(np.max(sift_spec_mi_max_s))
            )

            np.savetxt(
                "data/results/clustering/sift_spec_s_max_s.csv", sift_spec_s_max_s
            )
            np.savetxt(
                "data/results/clustering/sift_spec_mi_max_s.csv", sift_spec_mi_max_s
            )

            # print("SIFT performance: \n")
            # self.block_print()
            # for region in range(2):
            #     for i in range(key_regions[region], key_regions[region] + 6):
            #         reduced_sift = self.sift_data[:, 0: i * 5]
            #         sift_clustering = Clustering(reduced_sift, self.labels)
            #         for j in range(n_clusters[region], n_clusters[region] + 6):
            #
            #             sift_s[region][i-key_regions[region]][j-n_clusters[region]], \
            #                 sift_mi[region][i-key_regions[region]][j-n_clusters[region]] = \
            #                 sift_clustering.spectral(n_clusters=j, n_neighbors=n_neighbors[region])
            #
            # self.enable_print()
            # print("--------------\n")
            #
            # print('Max sil score sift spec sil region = ' + str(np.max(sift_s[0])) +
            #       ', Max mi score sift spec sil region = ' + str(np.max(sift_mi[0])))
            # print('Max sil score sift spec mi region = ' + str(np.max(sift_s[1])) +
            #       ', Max mi score sift spec mi region = ' + str(np.max(sift_mi[1])))
            #
            # np.savetxt('data/results/clustering/sift_spec_s_max_s.csv', sift_s[0])
            # np.savetxt('data/results/clustering/sift_spec_mi_max_s.csv', sift_mi[0])
            #
            # np.savetxt('data/results/clustering/sift_spec_s_max_mi.csv', sift_s[1])
            # np.savetxt('data/results/clustering/sift_spec_mi_max_mi.csv', sift_mi[1])

        else:
            normal_spec_s, normal_spec_mi = [np.zeros((14, 25)), np.zeros((14, 25))]
            sift_spec_s, sift_spec_mi = [np.zeros((60, 350)), np.zeros((60, 350))]

            print("Clustering: \n")
            print("Original performance: \n")
            self.block_print()
            normal_clustering = Clustering(self.flattened_original, self.labels)

            for i in range(2, 16):
                for j in range(1, 26):
                    self.enable_print()
                    print("orignal loop iteration " + str(i))
                    self.block_print()
                    (
                        normal_spec_s[(i - 2)][(j - 1)],
                        normal_spec_mi[(i - 2)][(j - 1)],
                    ) = normal_clustering.spectral(n_clusters=i, n_neighbors=j)

            self.enable_print()
            print("--------------\n")
            print(
                "Max sil score normal spec = "
                + str(np.max(normal_spec_s))
                + ", Max mi score normal spec = "
                + str(np.max(normal_spec_mi))
            )

            np.savetxt("data/results/clustering/normal_spec_s.csv", normal_spec_s)
            np.savetxt("data/results/clustering/normal_spec_mi.csv", normal_spec_mi)

            print("SIFT performance: \n")
            for i in range(60):
                key_points = i * 5

                reduced_sift = self.sift_data[:, 0 : key_points * 5 + 5]

                print("Max keypoints = " + str(((i * 5) + 5)))
                self.block_print()
                sift_clustering = Clustering(reduced_sift, self.labels)
                for j in range(2, 16):
                    for k in range(1, 26):
                        (
                            sift_spec_s[i][((j - 2) * 25) + (k - 1)],
                            sift_spec_mi[i][((j - 2) * 25) + (k - 1)],
                        ) = sift_clustering.spectral(n_clusters=j, n_neighbors=k)

                self.enable_print()
            print("--------------\n")

            print(
                "Max sil score sift spec = "
                + str(np.max(sift_spec_s))
                + ", Max mi score sift spec = "
                + str(np.max(sift_spec_mi))
            )

            np.savetxt("data/results/clustering/sift_spec_s.csv", sift_spec_s)
            np.savetxt("data/results/clustering/sift_spec_mi.csv", sift_spec_mi)

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
