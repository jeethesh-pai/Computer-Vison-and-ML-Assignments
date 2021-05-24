import numpy as np
import cv2
from utils import asUint8, from0_1to0_255asUint8, PLACEHOLDER, showImage, showImages, convertColorImagesBGR2RGB
from filter_zoo import filter_gauss
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from skimage.util import random_noise


# TODO: Simulate a picture captured in low light without noise.
#  Reduce the brightness of `img` about the provided darkening `factor`.
#  The data type of the returned image shall be the same as that of the input image.
#  Example (factor = 3): three times darker, i.e. a third of the original intensity.
def reduceBrightness(img, factor):
    img_hsi = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_new = img_hsi
    img_new[:, :, 2] = img_hsi[:, :, 2] / factor
    img_new = cv2.cvtColor(img_new, cv2.COLOR_HSV2BGR)
    return img_new


# TODO: "Restore" the brightness of a picture captured in low light, ignoring potential noise.
#  Apply the inverse operation to `reduceBrightness(..)`.
def restoreBrightness(img, factor):
    img_new = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img_new[:, :, 2] = np.clip(img_new[:, :, 2] * factor, a_min=0, a_max=255)
    img_new = cv2.cvtColor(img_new, cv2.COLOR_HSV2BGR)
    return img_new


def filter_sobel(img, ksize=3):
    # TODO: Implement a sobel filter (x/horizontal + y/vertical) for the provided `img` with kernel size `ksize`.
    #  The values of the final (combined) image shall be normalized to the range [0, 1].
    #  Return the final result along with the two intermediate images.
    central_difference_kernel = np.expand_dims([-1, 0, 1], axis=-1)
    gaussian_kernel = cv2.getGaussianKernel(ksize, sigma=-1)
    sobel_kernel_y = np.matmul(central_difference_kernel, np.transpose(gaussian_kernel))
    sobel_kernel_x = np.matmul(gaussian_kernel, np.transpose(central_difference_kernel))
    sobel_y = cv2.filter2D(img, ddepth=-1, kernel=sobel_kernel_y, borderType=cv2.BORDER_DEFAULT)
    sobel_x = cv2.filter2D(img, ddepth=-1, kernel=sobel_kernel_x, borderType=cv2.BORDER_DEFAULT)
    sobel = asUint8(np.sqrt(sobel_y * sobel_y + sobel_x * sobel_x))
    # sobel = np.clip(sobel, a_min=0, a_max=255)
    return sobel, sobel_x, sobel_y


def applyThreshold(img, threshold):
    # TODO: Return an image whose values are 1 where the `img` values are > `threshold` and 0 otherwise.
    new_img = np.zeros_like(img)
    new_img[img > threshold] = 1
    return new_img


def applyMask(img, mask):
    # TODO: Apply white color to the masked pixels, i.e. return an image whose values are 1 where `mask` values are 1
    #  and unchanged otherwise. (All mask values can be assumed to be either 0 or 1)
    img[mask == 1] = 1
    return img


img2 = cv2.imread("img/couch.jpg")
imgs = [("Original", img2)]
img3 = img2
# Reduce image brightness
darkening_factor = 3
img_dark = reduceBrightness(img2, darkening_factor)
# Restore image brightness
img_restored = restoreBrightness(img_dark, darkening_factor)

imgs = imgs + [("Low light", img_dark), ("Low light restored", img_restored)]
# Simulate multiple pictures captured in low light with noise.

num_dark_noise_imgs = 10
imgs_dark_noise = [from0_1to0_255asUint8(random_noise(img_dark, mode="poisson")) for _ in range(num_dark_noise_imgs)]


# TODO: Now try to "restore" a picture captured in low light with noise (`img_dark_noise`) using the same function as
#  for the picture without noise.
img_dark_noise = imgs_dark_noise[0]
img_noise_restored_simple = restoreBrightness(img_dark_noise, darkening_factor)
img_noise_stack_restored = np.copy(imgs_dark_noise[0])
for i in range(1, num_dark_noise_imgs):
    img_noise_stack_restored = cv2.add(img_noise_stack_restored, imgs_dark_noise[i])
    img_noise_stack_restored = asUint8(img_noise_stack_restored / 2)

img_noise_stack_restored = restoreBrightness(img_noise_stack_restored, darkening_factor)
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
