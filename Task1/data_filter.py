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

    animals = ["Cheetah", "Jaguar", "Leopard", "Panther", "Tiger"]
    bad_jag_images = [
        "fg4_5398_c_f-g_grandin_mnhn",
        "hqdefault",
        "image14-e1426482140449",
        "Jaguar-vs-Leopard-900x640",
        "Loro-Parque-celebra-el-nacimiento-de-los-mellizos-de-jaguar-1",
        "SIERRA-8-IMG_3170-WB",
    ]
    bad_leop_images = [
        "3cfeddfd17c7d1b1a999ac3937b6e17e",
        "37398 (1)",
        "37398",
        "578547dbd3f3b86f3d55efb702861aa5",
        "29212606-0-image-a-74_1591279612366",
        "gettyimages-1091967090_custom-214d47fd400ef5c608549569ab05bbab351c9278",
        "owrj81gbn_Julie_Larsen_Maher_1474_Snow_leopard_in_snow_02_22_08_hr",
        "panter_i_leopard_aps_407813201",
        "sn-leopard",
        "Snow_leopard_(Panthera_uncia),_Natural_History_Museum,_London,_Mammals_Gallery_04",
        "Snow_leopard_-_Uncia_uncia-1200x800",
        "snow-leopard (1)",
        "Snow-Leopard",
        "Snow-Leopard-Cub-Signed-Poster-by-Ian-Coleman-1-820x820",
        "SnowLeopardDay",
        "snow-leopard-tambako-the-jaguar-flickr-2",
        "snow-leopard-test",
        "tim_flach_opiom_gallery_1708_vg",
        "Yala National Park 01113",
        "update_snowleopard_winter2019",
    ]
    bad_panth_images = [
        "Black-Panther-1140x541",
        "black-panther-attack-8-east2west-news",
        "carolina-panther-extinct",
        "mithun-og",
        "shutterstock_707199295-1024x683",
    ]
    bad_tiger_images = [
        "hqdefault",
        "hwn4d-GH8W9HM1PCC-Full-Image_GalleryCover-en-US-1508348134678._UY500_UX667_RI_VxijV5YqtdCYfF8fPoweD5sNk1Snr3J_TTW_",
        "d41586-019-03264-2_17323966",
    ]
    sift = cv2.SIFT_create()

    for animal in animals:
        os.mkdir(f"{dir}/{animal}")
        for img in glob.glob(f"{path}{animal}/*.jp*g"):
            filename = img.split("\\")[-1]
            image = cv2.imread(img)
            size = image.shape

            if filename.endswith(".jpeg"):
                no_extension = filename[:-5]
            else:
                no_extension = filename[:-4]
            if animal == "Jaguar" and no_extension in bad_jag_images:
                continue
            if animal == "Leopard" and no_extension in bad_leop_images:
                continue
            if animal == "Panther" and no_extension in bad_panth_images:
                continue
            if animal == "Tiger" and no_extension in bad_tiger_images:
                continue

            # if image is too small, ignore it
            if size[0] < 250 or size[1] < 250:
                continue
            # delete images with few keypoints
            image = cv2.resize(image, (250, 250))
            kp, des = sift.detectAndCompute(image, None)
            if len(des) >= 300:
                shutil.copy(img, f"{dir}/{animal}/{filename}")


if __name__ == "__main__":
    main(sys.argv)
