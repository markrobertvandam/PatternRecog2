#!/usr/bin/env python3
import cv2
import glob
import os
import shutil
import sys


def main(argv):
    path = argv[1]
    dir = f"{path[:-1]}_filtered"
    if os.path.exists(dir):
        print("Removing old filtered data")
        shutil.rmtree(dir)
    os.mkdir(dir)
    print(f"New clean dir made at {dir}")

    animals = ["Cheetah", "Jaguar", "Leopard", "Lion", "Tiger"]
    sift = cv2.SIFT_create()

    for animal in animals:
        os.mkdir(f"{dir}/{animal}")
        for img in glob.glob(f"{path}{animal}/*.jp*g"):
            filename = img.split("\\")[-1]
            image = cv2.imread(img)
            size = image.shape

            # if image is too small, ignore it
            if size[0] < 250 or size[1] < 250:
                pass
            else:
                # delete images with few keypoints
                image = cv2.resize(image, (250, 250))
                kp, des = sift.detectAndCompute(image, None)
                if len(des) >= 300:
                    print(img)
                    shutil.copy(img, f"{dir}/{animal}/{filename}")


if __name__ == "__main__":
    main(sys.argv)
