+"""
Bilateral
"""
import cv2

# noisy_image = cv2.imread("C:/Users/rezua/Documents/GitHub/yolov7/custom_dataset/images/train/ptf (98).jpg" , 0)
noisy_image = cv2.imread("C:/Users/rezua/Documents/GitHub/yolov7/custom_dataset/images/train/ptf (119).jpg" , 0)

denoised_image = cv2.bilateralFilter(noisy_image, 5, 40, 100, borderType = cv2.BORDER_CONSTANT)

cv2.imshow("Orginal", noisy_image)
cv2.imshow("Denoised", denoised_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


"""
BM3D
"""

import bm3d
import cv2
from skimage import io, img_as_float

PATH = "C:/Users/rezua/Documents/GitHub/yolov7/custom_dataset/images/train/"
new_PATH ="C:/Users/rezua/Documents/GitHub/yolov7/custom_dataset/images/train2/"
list = os.listdir("C:/Users/rezua/Documents/GitHub/yolov7/custom_dataset/images/train")

for file in list:
    noisy_image = img_as_float(io.imread(PATH + file , as_gray= False))
    denoised_image = bm3d.bm3d(noisy_image, sigma_psd=0.05, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
    io.imsave(new_PATH + file,denoised_image)


PATH = "C:/Users/rezua/Documents/GitHub/yolov7/custom_dataset/images/val/"
new_PATH ="C:/Users/rezua/Documents/GitHub/yolov7/custom_dataset/images/val2/"
list = os.listdir("C:/Users/rezua/Documents/GitHub/yolov7/custom_dataset/images/val")

for file in list:
    noisy_image = img_as_float(io.imread(PATH + file , as_gray= False))
    denoised_image = bm3d.bm3d(noisy_image, sigma_psd=0.05, stage_arg=bm3d.BM3DStages.HARD_THRESHOLDING)
    io.imsave(new_PATH + file,denoised_image)
