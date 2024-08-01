from matplotlib import patches
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path
from skimage.io import imread, imsave
import shutil

subsets = ["train", "valid", "test"]

for subset in subsets:

    # !!!
    # edit the path with 'with' or 'without' to choose the corresponding dataset:
    # !!!
    source_folder = "INBreast Dataset (without preprocessing)/" + str(subset) + "/"

    if "with " in source_folder:
        dest_folder = "/home/lazaros/PycharmProjects/yolo_new_clone/datasets/INBreast_PP"
    elif "without " in source_folder:
        dest_folder = "/home/lazaros/PycharmProjects/yolo_new_clone/datasets/INBreast_noPP"

    Path(dest_folder+"/images/" + str(subset) + "/").mkdir(parents=True, exist_ok=True)
    Path(dest_folder+"/labels/" + str(subset) + "/").mkdir(parents=True, exist_ok=True)


    image_files = []
    for file in os.listdir(source_folder):

        # check if the image ends with png
        if (file.endswith(".png")):
            print()
            image = imread(source_folder + file)
            imsave(os.path.join(dest_folder, "images/" + str(subset) + "/", file), image)
            print()

        if (file.endswith(".txt")):
            shutil.copyfile(source_folder + file, dest_folder + "/labels/" + str(subset) + "/" + file)


