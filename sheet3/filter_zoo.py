import numpy as np
import cv2
import matplotlib.pyplot as plt


def filter_box(img, ksize=5):
    box_filter = np.ones(shape=(ksize, ksize), dtype=np.uint8) / (ksize ** 2)
    img_copy = np.copy(img)
    img_copy[:, :, 0] = cv2.filter2D(img[:, :, 0], ddepth=-1, kernel=box_filter, borderType=cv2.BORDER_DEFAULT)
    img_copy[:, :, 1] = cv2.filter2D(img[:, :, 1], ddepth=-1, kernel=box_filter, borderType=cv2.BORDER_DEFAULT)
    img_copy[:, :, 2] = cv2.filter2D(img[:, :, 2], ddepth=-1, kernel=box_filter, borderType=cv2.BORDER_DEFAULT)
    return img_copy


def filter_sinc(img, mask_circle_diameter=40.0):
    img_new = np.copy(img)
    if len(img.shape) < 2:
        img_new = filter_sinc_channel(img_new, mask_circle_diameter)
    else:
        img_length = img.shape
        for i in range(img.shape[2]):
            img_new[:, :, i] = filter_sinc_channel(img[:, :, i], mask_circle_diameter)
    return img_new


def filter_sinc_channel(img, mask_circle_diameter=40.0):
    """
    takes in single channel image and returns its filtered image
    :param img: single channel image
    :param mask_circle_diameter: low pass filter size
    :return: filtered image
    """
    dft_image = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft_image)
    mask = np.zeros((img.shape[0], img.shape[1], 2), dtype=np.uint8)
    circle_center = (int(img.shape[0] / 2), int(img.shape[1] / 2))
    points_x, points_y = np.ogrid[:img.shape[0], :img.shape[1]]
    mask_area = (points_x - circle_center[0]) ** 2 + (points_y - circle_center[1]) ** 2 <= \
                (mask_circle_diameter / 2) ** 2
    mask[mask_area] = 1
    filtered_dft = dft_shift * mask
    idft_image = np.fft.ifftshift(filtered_dft)
    img_filtered = cv2.idft(idft_image)
    img_filtered = cv2.magnitude(img_filtered[:, :, 0], img_filtered[:, :, 1])
    return img_filtered


def filter_gauss(img, ksize=5):
    filtered_img = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=3)
    return filtered_img


def filter_median(img, ksize=5):
    filtered_img = cv2.medianBlur(img, ksize)
    return filtered_img


image = cv2.imread("img/geometric_shapes.png")
new_img = filter_sinc(image, mask_circle_diameter=40)
# plt.imshow(new_img)
# plt.show()