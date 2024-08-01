import numpy as np
import cv2
import glob
import pydicom as dicom
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil
import os
import albumentations as A
import plistlib
from skimage.draw import polygon
import matplotlib.pyplot as plt

seed = 40  # to generate a different dataset

def csv_to_yolo():
    return 0

def bbox_to_txt(bboxes):
    """
    Convert a list of bbox into a string in YOLO format (to write a file).
    @bboxes : numpy array of bounding boxes
    return : a string for each object in new line: <object-class> <x> <y> <width> <height>
    """
    txt = ''
    for l in bboxes:
        l = [str(x) for x in l[:4]]
        l = ' '.join(l)
        txt += '0 ' + l + '\n'
    return txt


def crop(img):
    """
    Crop breast ROI from image.
    @img : numpy array image
    @mask : numpy array mask of the lesions
    return: numpy array of the ROI extracted for the image,
            numpy array of the ROI extracted for the breast mask,
            numpy array of the ROI extracted for the masses mask
    """
    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, breast_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return img, breast_mask


def truncation_normalization(img, mask):
    """
    Clip and normalize pixels in the breast ROI.
    @img : numpy array image
    @mask : numpy array mask of the breast
    return: numpy array of the normalized image
    """
    Pmin = np.percentile(img[mask != 0], 5)
    Pmax = np.percentile(img[mask != 0], 99)

    truncated = np.clip(img, Pmin, Pmax)
    normalized = (truncated - Pmin) / (Pmax - Pmin)
    normalized[mask == 0] = 0
    return normalized


def clahe(img, clip):
    """
    Image enhancement.
    @img : numpy array image
    @clip : float, clip limit for CLAHE algorithm
    return: numpy array of the enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=clip)
    cl = clahe.apply(np.array(img * 255, dtype=np.uint8))
    return cl


def synthesized_images(image_path):
    """
    Create a 3-channel image composed of the truncated and normalized image,
    the contrast enhanced image with clip limit 1,
    and the contrast enhanced image with clip limit 2
    @patient_id : patient id to recover image and mask in the dataset
    return: numpy array of the breast region, numpy array of the synthesized images, numpy array of the masses mask
    """
    image = cv2.imread(image_path)[:,:,0]

    breast, mask = crop(image)

    normalized = truncation_normalization(breast, mask)

    cl1 = clahe(normalized, 1.0)
    cl2 = clahe(normalized, 2.0)

    synthesized = cv2.merge((np.array(normalized * 255, dtype=np.uint8), cl1, cl2))
    return breast, synthesized


if __name__ == "__main__":

    dest_folder = "DBT Dataset (with preprocessing)/"
    yolo_dataset_dest_folder = "/home/lazaros/PycharmProjects/yolo_new_clone/datasets/DBT_PP/"

    shutil.rmtree(dest_folder, ignore_errors=True)
    Path(dest_folder + 'train').mkdir(parents=True, exist_ok=True)
    Path(dest_folder + 'valid').mkdir(parents=True, exist_ok=True)
    Path(dest_folder + 'test').mkdir(parents=True, exist_ok=True)
    Path(dest_folder + 'image preprocessing').mkdir(parents=True, exist_ok=True)

    subsets = ["train", "valid", "test"]

    # for subset in subsets:

    for subset in subsets:
        source_folder = "/home/lazaros/PycharmProjects/yolo_new_clone/datasets/dbt_dataset_ONLY-BIOPSIED_masses_NO-CLASSES/images/" + str(subset) + "/"

        os.makedirs(yolo_dataset_dest_folder + "images/" + str(subset) + "/", exist_ok=True)
        os.makedirs(yolo_dataset_dest_folder + "labels/" + str(subset) + "/", exist_ok=True)

        # find all images in source_folder (images) :
        image_files = []
        for images in os.listdir(source_folder):

            # check if the file ends with png - it could be avoided but it'''s a general approach:
            if (images.endswith(".png")):
                image_files.append(source_folder + images)

        # find all txt files in source_folder (labels) :
        txt_files = []
        source_folder_txt = source_folder.split("images")[0] + "labels" + source_folder.split("images")[1]
        for txts in os.listdir(source_folder_txt):

            # check if the file ends with txt - it could be avoided but it'''s a general approach:
            if (txts.endswith(".txt")):
                txt_files.append(txts)

        for image_file in image_files:

            name = image_file.split(subset+"/")[1]
            name = name.split(".")[0]

            original, synthesized = synthesized_images(image_file)

            # save the image in a folder to check them:
            cv2.imwrite(os.path.join(dest_folder + '/image preprocessing/', name + "_original.png"), original)

            # save the image in a folder to check them:
            cv2.imwrite(os.path.join(dest_folder + str(subset) + '/', name + ".png"), synthesized)

            # save the preprocessed phases of the image:
            cv2.imwrite(os.path.join(dest_folder + '/image preprocessing/', name + "_synthesized.png" ), synthesized)
            cv2.imwrite(os.path.join(dest_folder + '/image preprocessing/', name + "_trunc-norm.png" ),synthesized[:, :, 0])
            cv2.imwrite(os.path.join(dest_folder + '/image preprocessing/', name + "_cl1.png" ),synthesized[:, :, 1])
            cv2.imwrite(os.path.join(dest_folder + '/image preprocessing/', name + "_cl2.png" ),synthesized[:, :, 2])

            # save also the image in yolov5 dataset required format:
            cv2.imwrite(os.path.join(yolo_dataset_dest_folder + "images/" + str(subset) + '/', name + ".png"), synthesized)

            # cntr += 1

        for txt_file in txt_files:
            # save the txt files in this repo to be able to use them to draw the bboxes later:
            shutil.copyfile(source_folder_txt + txt_file, dest_folder + str(subset) + '/' + txt_file)

            # save the txt files in yolov5 dataset format:
            shutil.copyfile(source_folder_txt + txt_file, yolo_dataset_dest_folder + "labels/" + str(subset) + '/' + txt_file)
