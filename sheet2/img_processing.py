#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#
# Task 5
#

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def showImage(img, show_window_now=True):
    # TODO: Convert the channel order of an image from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt_img = plt.imshow(img)
    if show_window_now:
        plt.show()
    return plt_img


# Create an empty image (remove this when image loading works)
img = np.zeros((10, 10, 3), dtype=np.uint8)

# TODO: Load the image "img/hummingbird_from_pixabay.png" with OpenCV (`cv2`) to the variable `img` and show it with
#  `showImage(img)`.
img = cv2.imread('img/hummingbird_from_pixabay.png')
rgb_img = showImage(img, show_window_now=True)


def imageStats(img):
    print("Image stats:")
    print('Height: ', img.shape[0])
    print('Width: ', img.shape[1])
    print('Number of channels: ', img.shape[2])


# TODO: Print image stats of the hummingbird image.
imageStats(img)

# TODO: Change the color of the hummingbird to blue by swapping red and blue image channels.
swapped_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# TODO: Store the modified image as "blue_hummingbird.png" to your hard drive.
cv2.imwrite('img/blue_hummingbird.png', swapped_image)

#
# Task 6
# Prepare to show the original image and keep a reference so that we can update the image plot later.
plt.figure(figsize=(4, 6))
plt_img = showImage(img, False)


# TODO: Convert the original image to HSV color space.
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def img_update(hue_offset):
    print("Set hue offset to " + str(hue_offset))
    # TODO: Change the hue channel of the HSV image by `hue_offset`.
    offset_mask = np.zeros_like(hsv_img)
    offset_mask[:, :, 0] = hue_offset
    hsv_offset = hsv_img + offset_mask
    hsv_offset[:, :, 0] = np.clip(hsv_offset[:, :, 0], a_min=0, a_max=179)
    # TODO: Convert the modified HSV image back to RGB
    # and update the image in the plot window using `plt_img.set_data(img_rgb)`.
    img_rgb = cv2.cvtColor(hsv_offset, cv2.COLOR_HSV2RGB)
    plt_img.set_data(img_rgb)


# Create an interactive slider for the hue value offset.
ax_hue = plt.axes([0.1, 0.04, 0.8, 0.06])  # x, y, width, height
slider_hue = Slider(ax=ax_hue, label='Hue', valmin=0, valmax=180, valinit=0, valstep=1)
slider_hue.on_changed(img_update)

# Now actually show the plot window
plt.show()
