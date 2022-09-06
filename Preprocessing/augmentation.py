import os
from clodsa.techniques.techniqueFactory import createTechnique
import cv2
import numpy as np

path = "C:/Users/rezua/PycharmProjects/pythonProject1/panoramic sbj a12/"


def noisy(noise_typ, image):
    if noise_typ == "gauss":
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 255

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ == "speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return noisy


# CloDSA library augmentation functions
t2 = createTechnique("dropout", {"percentage": 0.05})
t3 = createTechnique("blurring", {"ksize": 5})
t4 = createTechnique("raise_blue", {"power": 0.9})
t1 = createTechnique("flip", {"flip": 0})
t5 = createTechnique("rotate", {"angle": 90})
t6 = createTechnique("rotate", {"angle": 270})
t7 = createTechnique("crop", {"percentage": 0.6, "startFrom": "TOPLEFT"})
t8 = createTechnique("crop", {"percentage": 0.6, "startFrom": "TOPRIGHT"})
t9 = createTechnique("crop", {"percentage": 0.6, "startFrom": "BOTTOMLEFT"})
t10 = createTechnique("crop", {"percentage": 0.6, "startFrom": "BOTTOMRIGHT"})
t11 = createTechnique("gamma", {"gamma": 1.5})

o_path = "C:/Users/rezua/PycharmProjects/pythonProject1/Augmented data/1/"
h_path = "C:/Users/rezua/PycharmProjects/pythonProject1/high resolution/1/"
l_path = "C:/Users/rezua/PycharmProjects/pythonProject1/lower resolution/1/"
list = os.listdir(path)
print(list)

for file in list:
  image = cv2.imread(path + file)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  #data augmentation for main project
  o_image = image

  #save images
  cv2.imwrite(os.path.join(o_path, file), o_image)

  # data augmentation for GAN
  #higher resolution image
  image = cv2.resize(image,(256,256))

    #save images
  cv2.imwrite(os.path.join(h_path , file) , image)

  #lower resolution images
  lr_image = cv2.resize(image,(56,56))

  #save images
  cv2.imwrite(os.path.join(l_path, file) , lr_image)


o_path = "C:/Users/rezua/PycharmProjects/pythonProject1/Augmented data/2/"
h_path = "C:/Users/rezua/PycharmProjects/pythonProject1/high resolution/2/"
l_path = "C:/Users/rezua/PycharmProjects/pythonProject1/lower resolution/2/"

for file in list:
    image = cv2.imread(path + file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # data augmentation for main project
    o_image1 = t1.apply(image)

    # save images
    cv2.imwrite(os.path.join(o_path, file) , o_image1)

    # data augmentation for GAN
    # higher resolution image
    image = cv2.resize(image, (256, 256))
    image1 = t1.apply(image)

    # save images
    cv2.imwrite(os.path.join(h_path, file) , image1)

    # noisy image
    noise_image1 = noisy("gauss", image1)

    # lower resolution images
    lr_image1 = cv2.resize(noise_image1, (56, 56))

    # save images
    cv2.imwrite(os.path.join(l_path, file) , lr_image1)


o_path = "C:/Users/rezua/PycharmProjects/pythonProject1/Augmented data/3/"
h_path = "C:/Users/rezua/PycharmProjects/pythonProject1/high resolution/3/"
l_path = "C:/Users/rezua/PycharmProjects/pythonProject1/lower resolution/3/"

for file in list:
  image = cv2.imread(path + file)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  #data augmentation for main project
  o_image2 = t5.apply(image)

  #save images
  cv2.imwrite(os.path.join(o_path, file) , o_image2)

  # data augmentation for GAN
  #higher resolution image
  image = cv2.resize(image,(256,256))
  image2 = t5.apply(image)

    #save images
  cv2.imwrite(os.path.join(h_path ,file) , image2)

  #noisy image
  noise_image2 = noisy("s&p",image2)

  #lower resolution images
  lr_image2 = cv2.resize(noise_image2,(56,56))

  #save images
  cv2.imwrite(os.path.join(l_path, file) , lr_image2)


o_path = "C:/Users/rezua/PycharmProjects/pythonProject1/Augmented data/4/"
h_path = "C:/Users/rezua/PycharmProjects/pythonProject1/high resolution/4/"
l_path = "C:/Users/rezua/PycharmProjects/pythonProject1/lower resolution/4/"

for file in list:
  image = cv2.imread(path + file)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  #data augmentation for main project
  o_image3 = t6.apply(image)

  #save images
  cv2.imwrite(os.path.join(o_path, file) , o_image3)

  # data augmentation for GAN
  #higher resolution image
  image = cv2.resize(image,(256,256))
  image3 = t6.apply(image)

    #save images
  cv2.imwrite(os.path.join(h_path , file) , image3)

  #noisy image
  noise_image3 = noisy("poisson",image3)

  #lower resolution images
  lr_image3 = cv2.resize(noise_image3,(56,56))

  #save images
  cv2.imwrite(os.path.join(l_path, file) , lr_image3)


o_path = "C:/Users/rezua/PycharmProjects/pythonProject1/Augmented data/5/"
h_path = "C:/Users/rezua/PycharmProjects/pythonProject1/high resolution/5/"
l_path = "C:/Users/rezua/PycharmProjects/pythonProject1/lower resolution/5/"

for file in list:
    image = cv2.imread(path + file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # data augmentation for main project
    o_image4 = t7.apply(image)

    # save images
    cv2.imwrite(os.path.join(o_path, file) , o_image4)

    # data augmentation for GAN
    # higher resolution image
    image = cv2.resize(image, (256, 256))
    image4 = t7.apply(image)

    # save images
    cv2.imwrite(os.path.join(h_path, file) , image4)

    # noisy image
    noise_image4 = t11.apply(image4)

    # lower resolution images
    lr_image4 = cv2.resize(noise_image4, (56, 56))

    # save images
    cv2.imwrite(os.path.join(l_path, file) , lr_image4)


o_path = "C:/Users/rezua/PycharmProjects/pythonProject1/Augmented data/6/"
h_path = "C:/Users/rezua/PycharmProjects/pythonProject1/high resolution/6/"
l_path = "C:/Users/rezua/PycharmProjects/pythonProject1/lower resolution/6/"

for file in list:
  image = cv2.imread(path + file)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  #data augmentation for main project
  o_image5 = t8.apply(image)

  #save images
  cv2.imwrite(os.path.join(o_path, file) , o_image5)

  # data augmentation for GAN
  #higher resolution image
  image = cv2.resize(image,(256,256))
  image5 = t8.apply(image)

    #save images
  cv2.imwrite(os.path.join(h_path , file) , image5)

  #noisy image
  noise_image5 = t2.apply(image5)

  #lower resolution images
  lr_image5 = cv2.resize(noise_image5,(56,56))

  #save images
  cv2.imwrite(os.path.join(l_path, file) , lr_image5)


o_path = "C:/Users/rezua/PycharmProjects/pythonProject1/Augmented data/7/"
h_path = "C:/Users/rezua/PycharmProjects/pythonProject1/high resolution/7/"
l_path = "C:/Users/rezua/PycharmProjects/pythonProject1/lower resolution/7/"

for file in list:
  image = cv2.imread(path + file)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  #data augmentation for main project
  o_image6 = t9.apply(image)

  #save images
  cv2.imwrite(os.path.join(o_path, file) , o_image6)

  # data augmentation for GAN
  #higher resolution image
  image = cv2.resize(image,(256,256))
  image6 = t9.apply(image)

    #save images
  cv2.imwrite(os.path.join(h_path , file) , image6)

  #noisy image
  noise_image6 = t3.apply(image6)

  #lower resolution images
  lr_image6 = cv2.resize(noise_image6,(56,56))

  #save images
  cv2.imwrite(os.path.join(l_path, file) , lr_image6)


o_path = "C:/Users/rezua/PycharmProjects/pythonProject1/Augmented data/8/"
h_path = "C:/Users/rezua/PycharmProjects/pythonProject1/high resolution/8/"
l_path = "C:/Users/rezua/PycharmProjects/pythonProject1/lower resolution/8/"


for file in list:
  image = cv2.imread(path + file)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  #data augmentation for main project
  o_image7 =t10.apply(image)

  #save images
  cv2.imwrite(os.path.join(o_path, file) , o_image7)

  # data augmentation for GAN
  #higher resolution image
  image = cv2.resize(image,(256,256))
  image7 =t10.apply(image)

    #save images
  cv2.imwrite(os.path.join(h_path , file) , image7)

  #noisy image
  noise_image7 = t4.apply(image7)

  #lower resolution images
  lr_image7 = cv2.resize(noise_image7,(56,56))

  #save images
  cv2.imwrite(os.path.join(l_path, file) , lr_image7)
