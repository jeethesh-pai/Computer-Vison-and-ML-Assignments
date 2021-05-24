#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage.util import random_noise

from utils import *


#
# Task 1
#


# TODO: Implement the the following filter functions such that each implements the respective image filter. They
#  shall not modify the input image, but return a filtered copy. Implement at least one of them without using an
#  existing filter function, e.g. do not use the corresponding OpenCV functions `cv2._____Blur(..)`.

def filter_box(img, ksize=5):
    # TODO: Implement the Box filter.
    return PLACEHOLDER(img)
    #
    # ???
    #


def filter_sinc(img, mask_circle_diameter=.4):
    # TODO: Implement the Sinc filter using DFT.
    return PLACEHOLDER(img)
    #
    # ???
    #


def filter_gauss(img, ksize=5):
    # TODO: Implement the Gaussian filter.
    return PLACEHOLDER(img)
    #
    # ???
    #


def filter_median(img, ksize=5):
    # TODO: Implement the Median filter.
    return PLACEHOLDER(img)
    #
    # ???
    #


#
# (Task 2)
#

# def filter_XYZ(img):
#     ...


def applyFilter(filter, img):
    return globals()["filter_" + filter](img)


img1 = cv2.imread("img/geometric_shapes.png")

# Simulate image noise
noise_types = ["gaussian", "poisson", "s&p"]
imgs_noise = [from0_1to0_255asUint8(random_noise(img1, mode=n)) for n in noise_types]

imgs = [("original", img1)] + [(noise + " noise", img) for noise, img in zip(noise_types, imgs_noise)]
plt.figure(figsize=(10, 3))
showImages(imgs)

# Filter noise images
filter_types = ["box", "sinc", "gauss", "median"]  # , "XYZ"] # (Task 2)
imgs_noise_filtered = [(f, [(noise, applyFilter(f, img)) for noise, img in imgs]) for f in filter_types]

imgs = imgs + [(f + " filter" if noise == "original" else "", img) for f, imgs_noise in imgs_noise_filtered for
               noise, img in imgs_noise]
plt.figure(figsize=(15, 8))
showImages(imgs, 4, transpose=True)


#
# Task 3
#


# TODO: Simulate a picture captured in low light without noise.
#  Reduce the brightness of `img` about the provided darkening `factor`.
#  The data type of the returned image shall be the same as that of the input image.
#  Example (factor = 3): three times darker, i.e. a third of the original intensity.
def reduceBrightness(img, factor):
    return PLACEHOLDER(img)
    #
    # ???
    #


# TODO: "Restore" the brightness of a picture captured in low light, ignoring potential noise.
#  Apply the inverse operation to `reduceBrightness(..)`.
def restoreBrightness(img, factor):
    return PLACEHOLDER(img)
    #
    # ???
    #


img2 = cv2.imread("img/couch.jpg")
imgs = [("Original", img2)]

# Reduce image brightness
darkening_factor = 3
img_dark = reduceBrightness(img2, darkening_factor)

# Restore image brightness
img_restored = restoreBrightness(img_dark, darkening_factor)

imgs = imgs + [("Low light", img_dark), ("Low light restored", img_restored)]

# Simulate multiple pictures captured in low light with noise.
num_dark_noise_imgs = 10
imgs_dark_noise = [from0_1to0_255asUint8(random_noise(img_dark, mode="poisson")) for _ in range(num_dark_noise_imgs)]

# TODO: Now try to "restore" a picture captured in low light with noise (`img_dark_noise`) using the same function as for the picture without noise.
img_dark_noise = imgs_dark_noise[0]
img_noise_restored_simple = PLACEHOLDER(img_dark_noise)
#
# ???
#

imgs = imgs + [None, ("Low light with noise", img_dark_noise),
               ("Low light with noise restored", img_noise_restored_simple)]

# TODO: Explain with your own words why the "restored" picture shows that much noise, i.e. why the intensity of the noise in low light images is typically so high compared to the image signal.
'''

________________________________________________________________________________
'''

# TODO: Restore a picture from all the low light pictures with noise (`imgs_dark_noise`) by computing the "average image" of them.
#  Adjust the resulting brightness to the original image (using the `darkening_factor` and `num_dark_noise_imgs`).
img_noise_stack_restored = PLACEHOLDER(imgs_dark_noise[0])
#
# ???
#


imgs = imgs + [("Low light with noise 1 ...", imgs_dark_noise[0]),
               ("... Low light with noise " + str(num_dark_noise_imgs), imgs_dark_noise[-1]),
               ("Low light stack with noise restored", img_noise_stack_restored)]
plt.figure(figsize=(15, 8))
showImages(imgs, 3)


#
# Task 4
#


def filter_sobel(img, ksize=3):
    # TODO: Implement a sobel filter (x/horizontal + y/vertical) for the provided `img` with kernel size `ksize`.
    #  The values of the final (combined) image shall be normalized to the range [0, 1].
    #  Return the final result along with the two intermediate images.
    sobel_x = PLACEHOLDER(img)
    sobel_y = PLACEHOLDER(img)
    sobel = PLACEHOLDER(img)
    #
    # ???
    #
    return sobel, sobel_x, sobel_y


def applyThreshold(img, threshold):
    # TODO: Return an image whose values are 1 where the `img` values are > `threshold` and 0 otherwise.
    return PLACEHOLDER(img)
    #
    # ???
    #


def applyMask(img, mask):
    # TODO: Apply white color to the masked pixels, i.e. return an image whose values are 1 where `mask` values are 1 and unchanged otherwise.
    #  (All mask values can be assumed to be either 0 or 1)
    return PLACEHOLDER(img)
    #
    # ???
    #


img3 = img2
imgs3 = [('Noise', img_noise_restored_simple),
         ('Gauss filter', filter_gauss(img_noise_restored_simple, 3)),
         ('Image stack + Gauss filter', filter_gauss(img_noise_stack_restored, 3))]

initial_threshold = .25
imgs3_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for _, img in imgs3]
imgs_sobel = [filter_sobel(img_gray) for img_gray in imgs3_gray]
imgs_thresh = [applyThreshold(img_sobel, initial_threshold) for img_sobel, _, _ in imgs_sobel]
imgs_masked = [applyMask(img3, img_thresh) for img_thresh in imgs_thresh]


def header(label, imgs, i, j=None):
    if i == 0:
        return label, (imgs[i] if j is None else imgs[i][j])
    return imgs[i] if j is None else imgs[i][j]


imgs = [[imgs3[i], header('Sobel X', imgs_sobel, i, 0),
         header('Sobel Y', imgs_sobel, i, 1),
         header('Sobel', imgs_sobel, i, 2),
         header('Edge mask', imgs_thresh, i),
         header('Stylized image', imgs_masked, i)] for i in range(len(imgs3))]
imgs = [label_and_image for img_list in imgs for label_and_image in img_list]

plt.figure(figsize=(17, 7))
plt_imgs = showImages(imgs, 6, False, padding=(.05, .15, .05, .05))


def updateImg(threshold):
    imgs_thresh = [applyThreshold(img_sobel, threshold) for img_sobel, _, _ in imgs_sobel]
    imgs_masked = [applyMask(img3, img_thresh) for img_thresh in imgs_thresh]
    imgs_masked = [convertColorImagesBGR2RGB(img_masked)[0] for img_masked in imgs_masked]
    for i in range(len(imgs3)):
        cols = len(imgs) // len(imgs3)
        plt_imgs[i * cols + 4].set_data(imgs_thresh[i])
        plt_imgs[i * cols + 5].set_data(imgs_masked[i])


ax_threshold = plt.axes([.67, .05, .27, .06])
slider_threshold = Slider(ax=ax_threshold, label='Threshold', valmin=0, valmax=1, valinit=initial_threshold,
                          valstep=.01)
slider_threshold.on_changed(updateImg)

plt.show()
