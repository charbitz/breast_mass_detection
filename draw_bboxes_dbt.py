from matplotlib import patches
import matplotlib.pyplot as plt
import cv2
import os
from pathlib import Path

subsets = ["train", "valid", "test"]

for subset in subsets:


    source_folder = "DBT Dataset (with preprocessing)/" + str(subset) + "/"
    dest_folder = "DBT Dataset (with preprocessing)/images with GT/"
    Path(dest_folder).mkdir(parents=True, exist_ok=True)


    image_files = []
    for images in os.listdir(source_folder):

        # check if the image ends with png
        if (images.endswith(".png")):
            image_files.append(source_folder + images)


    for image_file in image_files:

        name = image_file.split(subset + "/")[1]
        name = name.split(".")[0]

        img_0 = plt.imread(image_file)
        print()

    # img_path = "/home/lazaros/PycharmProjects/Breast_Mass_Detection/breast_mass_detection/INBreast Dataset (with transforms)/train/inbreast_mass_1.png"
    # name = img_path[-19:-4]
    # img_0 = plt.imread(img_path)


        txt_file = image_file[:-4] + ".txt"

        with open (txt_file, 'rt') as myfile:
            cntr = 0
            for myline in myfile:              # For each line, read to a string,

                # extract the yolo coordinates:
                bbox_class = myline.split(" ")[0]
                bbox_x_center = float(myline.split(" ")[1])
                bbox_y_center = float(myline.split(" ")[2])
                bbox_w = float(myline.split(" ")[3])
                bbox_h = float(myline.split(" ")[4])

                # denormalize back to image dims:
                bbox_x_center = bbox_x_center * img_0.shape[1]
                bbox_w = bbox_w * img_0.shape[1]
                bbox_y_center = bbox_y_center * img_0.shape[0]
                bbox_h = bbox_h * img_0.shape[0]

                # take the bbox upper left x, y coordinates:
                bbox_x = bbox_x_center - bbox_w // 2
                bbox_y = bbox_y_center - bbox_h // 2

                figure, ax = plt.subplots(1)
                rect = patches.Rectangle((bbox_x,bbox_y),bbox_w,bbox_h, edgecolor='r', facecolor="none")
                ax.imshow(img_0)
                ax.add_patch(rect)

                plt.savefig(dest_folder + name + '_box_' + str(cntr) + '.png')

                # ax.imsave("/saves/data.png")
                cntr += 1
                print(myline)


# figure, ax = plt.subplots(1)
# rect = patches.Rectangle((125,100),50,25, edgecolor='r', facecolor="none")
# ax.imshow(img_0)
# ax.add_patch(rect)