import albumentations as A
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from bbaug import policies
import cv2
import os
# from pascal_voc_writer import Writer
from xml.dom import minidom
import imgaug as ia
import imgaug.augmenters as iaa
import math
import random
import copy
import glob
imagespath = ""
random.seed(7)


def readImage(filename):
    # OpenCV uses BGR channels
    img = cv2.imread(imagespath+filename)
    return img


def getCoordinates(filename):
    allbb = []
    xmldoc = minidom.parse(imagespath+filename)
    itemlist = xmldoc.getElementsByTagName('object')

    size = xmldoc.getElementsByTagName('size')[0]
    width = int((size.getElementsByTagName('width')[0]).firstChild.data)
    height = int((size.getElementsByTagName('height')[0]).firstChild.data)

    for item in itemlist:
        classid = (item.getElementsByTagName('name')[0]).firstChild.data
        xmin = ((item.getElementsByTagName('bndbox')[
            0]).getElementsByTagName('xmin')[0]).firstChild.data
        ymin = ((item.getElementsByTagName('bndbox')[
            0]).getElementsByTagName('ymin')[0]).firstChild.data
        xmax = ((item.getElementsByTagName('bndbox')[
            0]).getElementsByTagName('xmax')[0]).firstChild.data
        ymax = ((item.getElementsByTagName('bndbox')[
            0]).getElementsByTagName('ymax')[0]).firstChild.data

        xmin = int(xmin)
        ymin = int(ymin)
        xmax = int(xmax)
        ymax = int(ymax)

        b = [xmin, ymin, xmax, ymax, classid]
        allbb.append(b)
    return allbb


def start():
    count = 1000
    for filename in sorted(os.listdir(imagespath)):

        if filename.endswith(".jpg") or filename.endswith(".JPG"):
            title, ext = os.path.splitext(os.path.basename(filename))
            image = readImage(filename)
        if filename.endswith(".txt"):
            xmlTitle, txtExt = os.path.splitext(os.path.basename(filename))
            if xmlTitle == title:
                # bboxes = getCoordinates(filename)
                bboxes = readYolo(imagespath+xmlTitle+'.txt')
                for i in range(0, 8):
                    img = copy.deepcopy(image)
                    transform = getTransform()
                    try:
                        transformed = transform(image=img, bboxes=bboxes)
                        transformed_image = transformed['image']
                        transformed_bboxes = transformed['bboxes']
                        name = title+str(count)+'.jpg'
                        cv2.imwrite(name, transformed_image)
                        # print(transformed_bboxes)
                        # writeVoc(transformed_bboxes, count, transformed_image)
                        writeYolo(transformed_bboxes, count, title)
                        count = count+1
                    except:
                        print("bounding box issues")
                        pass

                # bboxes = [[int(float(j)) for j in i] for i in bb]



def readYolo(filename):
    coords = []
    with open(filename, 'r') as fname:
        for file1 in fname:
            x = file1.strip().split(' ')
            x.append(x[0])
            x.pop(0)
            x[0] = float(x[0])
            x[1] = float(x[1])
            x[2] = float(x[2])
            x[3] = float(x[3])
            coords.append(x)
    return coords


def writeYolo(coords, count, name):

    with open('label/'+name+str(count)+'.txt', "w") as f:
        for x in coords:
            f.write("%s %s %s %s %s \n" % (x[-1], x[0], x[1], x[2], x[3]))


def getTransform():
    transform = A.Compose([
            A.HorizontalFlip(p=1),
            A.RandomBrightnessContrast(p=1),
            A.geometric.rotate.Rotate (limit=90, interpolation=1, border_mode=4, rotate_method='largest_box',
                                       crop_border=False, p=1),
            A.VerticalFlip(p=1),
            A.Blur(blur_limit=(10, 10), p=1),
            A.RandomRotate90(p=1),
            A.BBoxSafeRandomCrop(erosion_rate=0.0, p=1),
            A.Transpose(p=1)
        ], bbox_params=A.BboxParams(format='yolo'))

    return transform

start()
