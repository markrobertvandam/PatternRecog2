#!/usr/bin/env python3

import cv2
import glob
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import shutil

from classification import Classification
from clustering import Clustering
from feature_extraction import FeatureExtraction

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier


class Cats:
    def __init__(self):
        self.file_names = []
        self.images = None
        self.flattened_original = None
        self.gray_images = None
        self.labels = None
        self.feature_extractor = None
        self.fourier_data = None
        self.sift_data = None
        self.normal_classifier = None
        self.sift_classifier = None
        self.fourier_classifier = None

    def load_data(self) -> None:
        print("Loading data...")
        animals = ["Cheetah", "Jaguar", "Leopard", "Lion", "Tiger"]
        images = []
        labels = []
        gray_images = []
        for animal in animals:
            files_path = os.path.join(
                "data", "cats_projekat_filtered", animal, "*.jp*g"
            )
            for img in glob.glob(files_path):
                self.file_names.append(img)
                image = cv2.imread(img)
                resized_image = cv2.resize(image, (250, 250))
                grayscale_img = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
                images.append(resized_image)
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

    def visualize_data(self):
        print("Visualizing data...")
        unique, counts = np.unique(self.labels, return_counts=True)
        plt.bar(unique, counts, 0.4)
        plt.title("Class Frequency")
        plt.xlabel("Class")
        plt.ylabel("Frequency")
        plt.savefig(os.path.join("plots", "cats_histo.png"))
        plt.close()

    def feature_extraction(self) -> None:
        print("Doing feature extraction...")
        self.feature_extractor = FeatureExtraction(
            self.gray_images, self.labels, "cats"
        )
        self.fourier_data = self.feature_extractor.fourier_transform()

        # flatten fourier
        self.fourier_data = self.fourier_data.reshape(
            self.fourier_data.shape[0],
            self.fourier_data.shape[1] * self.fourier_data.shape[2],
        )

        self.sift_data = self.feature_extractor.sift()

    # Disable
    def block_print(self):
        sys.stdout = open(os.devnull, 'w')

    # Restore
    def enable_print(self):
        sys.stdout = sys.__stdout__

    def tune_classification_parameters(self, command="tune") -> None:
        """
        New function to run grid-search on command
        """
        results_f1_knn = np.zeros((60, 25))
        results_acc_knn = np.zeros((60, 25))

        results_f1_lr = np.zeros(60)
        results_acc_lr = np.zeros(60)

        results_f1_nb = np.zeros(60)
        results_acc_nb = np.zeros(60)

        results_f1_rf = np.zeros((60, 20))
        results_acc_rf = np.zeros((60, 20))

        results_f1_svm = np.zeros(60)
        results_acc_svm = np.zeros(60)

        results_f1_svc = np.zeros((60, 3))
        results_acc_svc = np.zeros((60, 3))

        results_f1_gp = np.zeros(60)
        results_acc_gp = np.zeros(60)

        results_f1_adb = np.zeros((60, 20))
        results_acc_adb = np.zeros((60, 20))

        results_f1_qda = np.zeros(60)
        results_acc_qda = np.zeros(60)

        step = 5
        reduced_sift = self.sift_data
        for i in range(60):
            print('iteration ' + str(i))
            self.block_print()
            self.sift_classifier = Classification(reduced_sift, self.labels)

            for j in range(1, 26):
                results_f1_knn[i][(j-1)], results_acc_knn[i][(j-1)] = \
                    self.sift_classifier.knn_classify(k=j, command="tune")

            results_f1_lr[i], results_acc_lr[i] = self.sift_classifier.logistic_regression(max_iter=10000, command="tune")

            results_f1_nb[i], results_acc_nb[i] = self.sift_classifier.nb_classify(command="tune")

            k = 0
            for j in range(20, 420, 20):
                results_f1_rf[i][k], results_acc_rf[i][k] = self.sift_classifier.random_forest(n_trees=j, command="tune")
                k += 1

            results_f1_svm[i], results_acc_svm[i] = self.sift_classifier.svm_classify(max_iter=10000, command="tune")

            results_f1_svc[i][0], results_acc_svc[i][0] = self.sift_classifier.svc(kernel='poly', command="tune")
            results_f1_svc[i][1], results_acc_svc[i][1] = self.sift_classifier.svc(kernel='rbf', command="tune")
            results_f1_svc[i][2], results_acc_svc[i][2] = self.sift_classifier.svc(kernel='sigmoid', command="tune")

            results_f1_gp[i], results_acc_gp[i] = self.sift_classifier.gp_classify(command="tune")

            k = 0
            for j in range(20, 420, 20):
                results_f1_adb[i][k], results_acc_adb[i][k] = self.sift_classifier.adb(n_trees=j, command="tune")
                k += 1

            results_f1_qda[i], results_acc_qda[i] = self.sift_classifier.qda(command="tune")

            reduced_sift = reduced_sift[:, 0:(len(reduced_sift[0])-step)]

            self.enable_print()

        print('Max knn f1-score = ' + str(np.amax(results_f1_knn)) + ', max knn acc = ' + str(np.amax(results_acc_knn)))
        np.savetxt('data/results/results_f1_knn.csv', results_f1_knn, delimiter=',')
        np.savetxt('data/results/results_acc_knn.csv', results_acc_knn, delimiter=',')

        print('Max lr f1-score = ' + str(np.amax(results_f1_lr)) + ', max lr acc = ' + str(np.amax(results_acc_lr)))
        np.savetxt('data/results/results_f1_lr.csv', results_f1_lr, delimiter=',')
        np.savetxt('data/results/results_acc_lr.csv', results_acc_lr, delimiter=',')

        print('Max nb f1-score = ' + str(np.amax(results_f1_nb)) + ', max nb acc = ' + str(np.amax(results_acc_nb)))
        np.savetxt('data/results/results_f1_nb.csv', results_f1_nb, delimiter=',')
        np.savetxt('data/results/results_acc_nb.csv', results_acc_nb, delimiter=',')

        print('Max rf f1-score = ' + str(np.amax(results_f1_rf)) + ', max rf acc = ' + str(np.amax(results_acc_rf)))
        np.savetxt('data/results/results_f1_rf.csv', results_f1_rf, delimiter=',')
        np.savetxt('data/results/results_acc_rf.csv', results_acc_rf, delimiter=',')

        print('Max svm f1-score = ' + str(np.amax(results_f1_svm)) + ', max svm acc = ' + str(np.amax(results_acc_svm)))
        np.savetxt('data/results/results_f1_svm.csv', results_f1_svm, delimiter=',')
        np.savetxt('data/results/results_acc_svm.csv', results_acc_svm, delimiter=',')

        print('Max svc f1-score = ' + str(np.amax(results_f1_svc)) + ', max svc acc = ' + str(np.amax(results_acc_svc)))
        np.savetxt('data/results/results_f1_svc.csv', results_f1_svc, delimiter=',')
        np.savetxt('data/results/results_acc_svc.csv', results_acc_svc, delimiter=',')

        print('Max gp f1-score = ' + str(np.amax(results_f1_gp)) + ', max gp acc = ' + str(np.amax(results_acc_gp)))
        np.savetxt('data/results/results_f1_gp.csv', results_f1_gp, delimiter=',')
        np.savetxt('data/results/results_acc_gp.csv', results_acc_gp, delimiter=',')

        print('Max adb f1-score = ' + str(np.amax(results_f1_adb)) + ', max adb acc = ' + str(np.amax(results_acc_adb)))
        np.savetxt('data/results/results_f1_adb.csv', results_f1_adb, delimiter=',')
        np.savetxt('data/results/results_acc_adb.csv', results_acc_adb, delimiter=',')

        print('Max qda f1-score = ' + str(np.amax(results_f1_qda)) + ', max qda acc = ' + str(np.amax(results_acc_qda)))
        np.savetxt('data/results/results_f1_qda.csv', results_f1_qda, delimiter=',')
        np.savetxt('data/results/results_acc_qda.csv', results_acc_qda, delimiter=',')

    def classification(self, command="tuning") -> None:
        """
        function to run grid-search/test-run depending on command
        """
        print(f"Original performance (shape: {self.flattened_original.shape}): \n")

        self.normal_classifier = Classification(self.flattened_original, self.labels)
        self.normal_classifier.knn_classify(k=5, command=command)
        self.normal_classifier.nb_classify(command=command)
        self.normal_classifier.random_forest(n_trees=200, command=command)
        print("--------------")

        # Classify sift dataset
        print(f"Sift performance (shape: {self.sift_data.shape}): \n")
        self.sift_classifier = Classification(self.sift_data, self.labels)
        self.sift_classifier.knn_classify(k=5, command=command)
        self.sift_classifier.nb_classify(command=command)
        self.sift_classifier.random_forest(n_trees=200, command=command)
        print("--------------\n")

        # Classify fourier dataset

        print(f"Fourier performance (shape: {self.fourier_data.shape}): \n")

        self.fourier_classifier = Classification(self.fourier_data, self.labels)
        self.fourier_classifier.knn_classify(k=5, command=command)
        self.fourier_classifier.nb_classify(command=command)
        self.fourier_classifier.random_forest(n_trees=200, command=command)
        print("--------------")

    def cross_val(self) -> None:
        """
        function to run cross-val with full data with best pipeliness
        """
        print(f"Original performance (shape: {self.flattened_original.shape}): \n")

        self.normal_classifier = Classification(self.flattened_original, self.labels)
        self.normal_classifier.nb_classify(command="cross-val")
        self.normal_classifier.random_forest(n_trees=200, command="cross-val")
        print("--------------")

        # Classify sift dataset
        print(f"Sift performance (shape: {self.sift_data.shape}): \n")
        self.sift_classifier = Classification(self.sift_data, self.labels)
        self.sift_classifier.nb_classify(command="cross-val")
        self.sift_classifier.random_forest(n_trees=200, command="cross-val")

        print("--------------")

    def ensemble(self) -> None:
        """
        function to run all possible ensembles with full data
        """
        random_state = 42  # seed

        # Classify sift dataset
        print(f"Sift performance (shape: {self.sift_data.shape}): \n")
        self.sift_classifier = Classification(self.sift_data, self.labels)
        self.sift_classifier.knn_classify(k=5, command="test")
        self.sift_classifier.nb_classify(command="test")
        self.sift_classifier.random_forest(n_trees=200, command="test")

        print("\nEnsemble using Naive Bayes and Random Forest:")
        print("--------------")
        self.sift_classifier.ensemble(
            GaussianNB(), RandomForestClassifier(random_state=random_state)
        )

        print("\nEnsemble using Naive Bayes and KNN:")
        print("--------------")
        self.sift_classifier.ensemble(GaussianNB(), KNeighborsClassifier(n_neighbors=5))

        print("\nEnsemble using KNN and Random Forest:")
        print("--------------")
        self.sift_classifier.ensemble(
            KNeighborsClassifier(n_neighbors=5),
            RandomForestClassifier(random_state=random_state),
        )

        print("\nEnsemble using KNN, Naive Bayes and Random Forest:")
        print("--------------")
        self.sift_classifier.ensemble(
            KNeighborsClassifier(n_neighbors=5),
            GaussianNB(),
            RandomForestClassifier(random_state=random_state),
        )

        print("--------------")

    def clustering(self) -> None:
        print("Clustering: \n")
        print("Original performance: \n")
        normal_clustering = Clustering(self.flattened_original, self.labels, 4, 5)
        normal_clustering.k_means()
        print("--------------\n")

        print("SIFT performance: \n")
        sift_clustering = Clustering(self.sift_data, self.labels, 4, 5)
        sift_cluster_labels = sift_clustering.k_means()
        print("--------------\n")

        print("Fourier performance: \n")
        fourier_clustering = Clustering(self.fourier_data, self.labels, 4, 5)
        fourier_clustering.k_means()
        print("--------------\n")

    def save_clustering(self, cluster_labels, n_clusters) -> None:
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
