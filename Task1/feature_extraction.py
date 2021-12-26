#!/usr/bin/env python3
import cv2
import numpy as np
from sklearn.decomposition import PCA


class FeatureExtraction:
    def __init__(self, x: np.ndarray, y: list, dataset: str) -> None:
        self.data = x
        self.labels = y

    def sift(self) -> np.ndarray:
        # sift
        sift = cv2.SIFT_create(90)
        sift_data = []

        for image in self.data:
            # only keeps the descriptors of best 200 keypoints
            kp, des = sift.detectAndCompute(image, None)
            sift_data.append(des[:90])
        return np.array(sift_data, dtype="object")

    def fourier_transform(self) -> np.ndarray:
        # Apply DFT and shift the zero frequency component to center
        fourier_data = []

        for img in self.data:
            f = np.fft.fft2(img)
            f_shift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(f_shift))
            fourier_data.append(magnitude_spectrum)

        return np.array(fourier_data, dtype="object")

    def pca(self, min_variance) -> np.ndarray:
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

        return np.array(data_reduced, dtype="object")

