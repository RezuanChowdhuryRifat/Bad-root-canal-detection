import os

import bm3d
import cv2
from skimage import io, img_as_float

PATH = "C:/Users/rezua/Desktop/Radiographic Dataset/Plat ptf a11/"
new_PATH ="C:/Users/rezua/Desktop/Radiographic Dataset/Denoised Plat ptf a11/"
list = os.listdir("C:/Users/rezua/Desktop/Radiographic Dataset/Plat ptf a11/")

for file in list:
    noisy_image = img_as_float(io.imread(PATH + file , as_gray= False))
    denoised_image = bm3d.bm3d(noisy_image, sigma_psd=0.08, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
    io.imsave(new_PATH + file,denoised_image)

