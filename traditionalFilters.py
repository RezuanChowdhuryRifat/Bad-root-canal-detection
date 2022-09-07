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

noisy_image = img_as_float(io.imread("C:/Users/rezua/Documents/GitHub/yolov7/custom_dataset/images/train/ptf (98).jpg" , as_gray= True))
#noisy_image =img_as_float(io.imread("C:/Users/rezua/Documents/GitHub/yolov7/custom_dataset/images/train/ptf (119).jpg" , as_gray= True))


denoised_image = bm3d.bm3d(noisy_image, sigma_psd=0.05, stage_arg=bm3d.BM3DStages.ALL_STAGES)

cv2.imshow("Orginal", noisy_image)
cv2.imshow("Denoised", denoised_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
