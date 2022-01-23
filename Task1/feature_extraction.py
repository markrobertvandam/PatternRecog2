#!/usr/bin/env python3
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

from scipy.cluster.vq import kmeans, vq
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif as MIC


class FeatureExtraction:
    def __init__(self, x: np.ndarray, y: np.ndarray, dataset: str) -> None:
        """
        Initialize dataset for feature extraction.

        Arguments:
        x: Input features.
        y: Class labels.
        dataset: Name of dataset.

        Returns:
        None
        """

        self.data = x
        self.labels = y

    def sift(self, max_keypoints=300) -> (np.ndarray, list):
        """
        Function to perform SIFT feature extraction.

        Arguments:
        max_keypoints: Maximum keypoints to consider.

        Returns:
        bow_visual: Bag of features containing strongest features.
        low_info_imgs: List of underperforming images.
        """

        # sift
        sift = cv2.SIFT_create(max_keypoints)
        sift_data = []
        low_info_imgs = []
        for i in range(len(self.data)):
            image = self.data[i]
            # only keeps the descriptors of best keypoints
            kp, des = sift.detectAndCompute(image, None)
            if len(des) < max_keypoints:
                low_info_imgs.append(i)
            else:
                sift_data.append(des[:max_keypoints])

        final_descriptors = np.asarray(sift_data, dtype="object")
        descriptors_float = final_descriptors.astype(float).reshape(
            (len(sift_data) * max_keypoints, 128)
        )
        voc, variance = kmeans(descriptors_float, max_keypoints, 1, seed=42)

        # Create histograms of visual bow
        bow_visual = np.zeros((len(sift_data), max_keypoints))
        for i in range(len(sift_data)):
            words, distance = vq(sift_data[i], voc)
            for w in words:
                bow_visual[i][w] += 1

        return bow_visual, low_info_imgs

    def fourier_transform(self, filter_radius=0, fourier_data=None) -> np.ndarray:
        """
        Function to perform feature extraction using fourier transform.

        Arguments:
        filter_radius: Radius of specified filter.
        fourier_data: Fourier data containing spectral information.

        Returns:
        fourier_data: Fourier transformed data.
        """
        
        print("filter radius: ", filter_radius)
        if fourier_data is None:
            # Apply DFT and shift the zero frequency component to center
            fourier_data = []

            for img in self.data:
                f = np.fft.fft2(img)
                f_shift = np.fft.fftshift(f)
                magnitude_spectrum = 20 * np.log(np.abs(f_shift))
                fourier_data.append(magnitude_spectrum)
            fourier_data = np.asarray(fourier_data, dtype=object)

        print(fourier_data.shape)
        n_rows, n_cols = fourier_data[0].shape
        central_row, central_col = int(n_rows / 2), int(n_cols / 2)

        mask = np.ones((n_rows, n_cols), np.uint8)
        r = filter_radius
        center = [central_row, central_col]
        x, y = np.ogrid[:n_rows, :n_cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
        mask[mask_area] = 0

        for i in range(len(fourier_data)):
            fourier_data[i] = fourier_data[i] * mask

        return fourier_data

    def pca(self, min_variance, pca=None) -> (np.ndarray, PCA):
        """
        Function to perform feature extraction using PCA.

        Arguments:
        min_variance: Minimum variance for selecting components.

        Returns:
        np.array: Reduced components based on the specified variance.
        """

        if pca is None:
            # Apply PCA to dataset
            pca = PCA()
            pca.fit(self.data)

        total_variance = 0
        n_components = 0
        for var in pca.explained_variance_ratio_:
            total_variance += var
            n_components += 1
            if total_variance >= min_variance:
                break

        pca_reduction = PCA(n_components=n_components)
        data_reduced = pca_reduction.fit_transform(self.data)
        # self.scree_plot(
        #     pca_reduction.n_components, pca_reduction.explained_variance_ratio_ * 100
        # )

        return np.array(data_reduced, dtype="object"), pca

    def mutual_information(self, min_information, mi_values=None) -> (np.ndarray, MIC):
        """
        Function to perform feature extraction using Mutual Information.

        Arguments:
        min_information: Minimum information for thresholding.
        mi_values: Mutual information of data.

        Returns:
        np.array: Reduced data with Mutual information.
        """

        if mi_values is None:
            mi_values = MIC(self.data, self.labels)

        low_information_features = []
        # sorted_mi = []
        for i in range(np.size(self.data, 1)):
            # sorted_mi.append((i, mi_values[i]))
            if mi_values[i] < min_information:
                low_information_features.append(i)

        data_reduced = np.delete(self.data, low_information_features, axis=1)

        # sorted_mi.sort(key=lambda x: x[1], reverse=True)

        return np.array(data_reduced, dtype="object"), mi_values

    def scree_plot(self, n_components, explain_var) -> None:
        """
        Function to plot variance of components.

        Arguments:
        n_components: Number of components.
        explain_var: Variance of components.

        Returns:
        None
        """

        pc_values = np.arange(n_components) + 1
        plt.plot(pc_values, explain_var, "o-")
        plt.title("Scree Plot")
        plt.xlabel("Principal Component (n)")
        plt.ylabel("Variance Explained (%)")
        plt.savefig(os.path.join("plots", f"scree{n_components}.png"))
        plt.close()

        pc_values = np.arange(n_components) + 1
        plt.plot(pc_values[-3:], explain_var[-3:], "o-")
        plt.title("Scree Plot")
        plt.xlabel("Principal Component (n)")
        plt.ylabel("Variance Explained (%)")
        plt.savefig(os.path.join("plots", f"scree{n_components}_short"))
        plt.close()
