
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




# Set the paths to the image and text file folders
image_folder = "C:/Users/rezua/Desktop/Radiographic Dataset/Yolo dataset/images/train/"
text_folder = "C:/Users/rezua/Desktop/Radiographic Dataset/Labelled malpractice images ptf a11"
output_folder = "C:/Users/rezua/Desktop/Radiographic Dataset/Yolo dataset/labels/train"

# Read in the images from the image folder and store them in a list
image_list = []
for file_name in os.listdir(image_folder):
    if file_name.endswith(".jpg") or file_name.endswith(".png"):
        image_list.append(os.path.splitext(file_name)[0])

# Look for text files in the text folder that have a matching filename in the image list
for file_name in os.listdir(text_folder):
    if file_name.endswith(".txt"):
        base_name = os.path.splitext(file_name)[0]
        if base_name in image_list:
            # Copy the text file to the output folder
            source_path = os.path.join(text_folder, file_name)
            dest_path = os.path.join(output_folder, file_name)
            with open(source_path, 'rb') as source_file, open(dest_path, 'wb') as dest_file:
                dest_file.write(source_file.read())


image_folder = "C:/Users/rezua/Desktop/Radiographic Dataset/Yolo dataset/images/test/"
text_folder = "C:/Users/rezua/Desktop/Radiographic Dataset/Labelled malpractice images ptf a11"
output_folder = "C:/Users/rezua/Desktop/Radiographic Dataset/Yolo dataset/labels/test"

# Read in the images from the image folder and store them in a list
image_list = []
for file_name in os.listdir(image_folder):
    if file_name.endswith(".jpg") or file_name.endswith(".png"):
        image_list.append(os.path.splitext(file_name)[0])

# Look for text files in the text folder that have a matching filename in the image list
for file_name in os.listdir(text_folder):
    if file_name.endswith(".txt"):
        base_name = os.path.splitext(file_name)[0]
        if base_name in image_list:
            # Copy the text file to the output folder
            source_path = os.path.join(text_folder, file_name)
            dest_path = os.path.join(output_folder, file_name)
            with open(source_path, 'rb') as source_file, open(dest_path, 'wb') as dest_file:
                dest_file.write(source_file.read())

image_folder = "C:/Users/rezua/Desktop/Radiographic Dataset/Yolo dataset/images/val/"
text_folder = "C:/Users/rezua/Desktop/Radiographic Dataset/Labelled malpractice images ptf a11"
output_folder = "C:/Users/rezua/Desktop/Radiographic Dataset/Yolo dataset/labels/val"

# Read in the images from the image folder and store them in a list
image_list = []
for file_name in os.listdir(image_folder):
    if file_name.endswith(".jpg") or file_name.endswith(".png"):
        image_list.append(os.path.splitext(file_name)[0])

# Look for text files in the text folder that have a matching filename in the image list
for file_name in os.listdir(text_folder):
    if file_name.endswith(".txt"):
        base_name = os.path.splitext(file_name)[0]
        if base_name in image_list:
            # Copy the text file to the output folder
            source_path = os.path.join(text_folder, file_name)
            dest_path = os.path.join(output_folder, file_name)
            with open(source_path, 'rb') as source_file, open(dest_path, 'wb') as dest_file:
                dest_file.write(source_file.read())
