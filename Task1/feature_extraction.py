#!/usr/bin/env python3
import cv2
import numpy as np


class FeatureExtraction:
    def __init__(self, x: list, y: list, dataset: str) -> None:
        self.data = x
        self.labels = y
        if dataset == "cats":

            # convert to grayscale
            self.gray_images = []
            for img in self.data:
                grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                self.gray_images.append(grayscale_img)

    def sift(self) -> list:
        # sift
        sift = cv2.SIFT_create(240)
        sift_data = []

        for image in self.gray_images:

            # only keeps the descriptors
            sift_data.append(sift.detectAndCompute(image, None)[1][:240])

        return sift_data

    def fourier_transform(self) -> list:
        img = cv2.imread("messi5.jpg", 0)

        # Apply DFT and shift the zero frequency component to center
        fourier_data = []

        for img in self.gray_images:
            f = np.fft.fft2(img)
            f_shift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(f_shift))
            fourier_data.append(magnitude_spectrum)

        return fourier_data
