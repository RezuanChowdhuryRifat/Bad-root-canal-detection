
import os
import random
from shutil import copyfile
from sklearn.model_selection import train_test_split

# Define paths and parameters
source_folder = "C:/Users/rezua/Desktop/Radiographic Dataset/Denoised Plat ptf a11/"  # path to folder containing images to be split
train_folder = "C:/Users/rezua/Desktop/Radiographic Dataset/Yolo dataset/images/train/"  # path to folder where train images will be saved
test_folder = "C:/Users/rezua/Desktop/Radiographic Dataset/Yolo dataset/images/test/"   # path to folder where test images will be saved
val_folder = "C:/Users/rezua/Desktop/Radiographic Dataset/Yolo dataset/images/val/"   # path to folder where validation images will be saved
train_split = 0.75  # percentage of data to use for training
test_split = 0.15  # percentage of data to use for testing
val_split = 0.1  # percentage of data to use for validation

# Get a list of all image filenames in the source folder
image_filenames = [filename for filename in os.listdir(source_folder)
                   if filename.endswith(".jpg") or filename.endswith(".png")]

# Split the image filenames into training, testing, and validation sets
train_images, test_val_images = train_test_split(image_filenames, test_size=(test_split+val_split))
test_images, val_images = train_test_split(test_val_images, test_size=val_split/(test_split+val_split))

# Copy the image files to their respective folders
for filename in train_images:
    src_path = os.path.join(source_folder, filename)
    dst_path = os.path.join(train_folder, filename)
    copyfile(src_path, dst_path)

for filename in test_images:
    src_path = os.path.join(source_folder, filename)
    dst_path = os.path.join(test_folder, filename)
    copyfile(src_path, dst_path)

for filename in val_images:
    src_path = os.path.join(source_folder, filename)
    dst_path = os.path.join(val_folder, filename)
    copyfile(src_path, dst_path)





