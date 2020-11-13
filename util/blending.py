import glob as glob
import os
from PIL import Image
import matplotlib.pyplot as plt
from dataset.data_util import pil_load_img
from pathlib import Path
import cv2 as cv
import numpy as np
local_data_path = Path('.').absolute()
local_data_path.mkdir(exist_ok=True)
import random
import albumentations as A

# img_path= 'data/total-text-original/Images/Test/img1.jpg'
# img= pil_load_img(img_path)
# file_list = glob.glob("data/dtd/images/*/*.jpg")

def aug_image(image, texture_file_list):
    # for texture_path in file_list:
    texture_path=random.choice(texture_file_list)
    #print("file list",file_list)
    texture = cv.resize(cv.imread(texture_path),(image.shape[1], image.shape[0]))

    augmentation = A.Compose([
        A.RandomBrightnessContrast(p=1),
        A.RandomGamma(p=0.50),
        A.CLAHE(p=1),
        A.RGBShift(r_shift_limit=105, g_shift_limit=45, b_shift_limit=40, p=1),
        A.HueSaturationValue(hue_shift_limit=172, sat_shift_limit=20, val_shift_limit=27, p=1),
        A.ChannelShuffle(p=1),
        A.RandomContrast(limit=0.9, p=0.5),
        A.RandomBrightness(p=1),
        A.Blur(p=1),
        A.MedianBlur(p=1),
        A.JpegCompression(p=1),
    ], p=1)

    augmented= augmentation(image=image,mask=None, bboxes=[], category_id=[])
    choice = random.randint(0,2)

    if choice==0:
        alpha = 0.85
        beta = (1.0 - alpha)
        dst = cv.addWeighted(augmented['image'], alpha, texture, beta, 0.0)
    else:
        dst = augmented['image']
    #print(dst.shape)
    #print(type(augmented),augmented.keys())
    # plt.imshow(dst)
    # plt.show()
    # print("done")
    return dst
