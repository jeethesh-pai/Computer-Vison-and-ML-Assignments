import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage.util import random_noise
import filter_zoo
from utils import from0_1to0_255asUint8, showImage, showImages


def applyFilter(filter, img):
    return getattr(filter_zoo, "filter_" + filter)(img)


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
