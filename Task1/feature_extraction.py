#!/usr/bin/env python3
import cv2
import numpy as np


class FeatureExtraction:
    def __init__(self, x: np.ndarray, y: list, dataset: str) -> None:
        self.data = x
        self.labels = y

    def sift(self) -> np.ndarray:
        # sift
        sift = cv2.SIFT_create(200)
        sift_data = []

        for image in self.data:
            # only keeps the descriptors of best 200 keypoints
            kp, des = sift.detectAndCompute(image, None)
            sift_data.append(des[:200])

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
