import cv2
import numpy as np
from utils import from0_1to0_255asUint8, showImages, asUint8
from skimage.util import random_noise
import matplotlib.pyplot as plt


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


img2 = cv2.imread("img/couch.jpg")
imgs = [("Original", img2)]

# Reduce image brightness
darkening_factor = 3
img_dark = reduceBrightness(img2, darkening_factor)
# Restore image brightness
img_restored = restoreBrightness(img_dark, darkening_factor)

imgs = imgs + [("Low light", img_dark), ("Low light restored", img_restored)]
# showImages(imgs)
# Simulate multiple pictures captured in low light with noise.

num_dark_noise_imgs = 10
imgs_dark_noise = [from0_1to0_255asUint8(random_noise(img_dark, mode="poisson")) for _ in range(num_dark_noise_imgs)]


# TODO: Now try to "restore" a picture captured in low light with noise (`img_dark_noise`) using the same function as
#  for the picture without noise.
img_dark_noise = imgs_dark_noise[0]
img_noise_restored_simple = restoreBrightness(img_dark_noise, darkening_factor)

imgs = imgs + [None, ("Low light with noise", img_dark_noise),
               ("Low light with noise restored", img_noise_restored_simple)]
# showImages(imgs)
# # TODO: Explain with your own words why the "restored" picture shows that much noise, i.e. why the intensity of the
# #  noise in low light images is typically so high compared to the image signal.
'''
Noise is generally high or low intensity pixels(0 or 255) scattered at random locations of the image. So when we
increase the brightnessof this noise injected image we simultaneously increase the noise value also which was already 
255 or 0 and hence results in undesired artefacts. Therefore we need to treat the noise before increasing the overall
brghtness of the image.
________________________________________________________________________________
'''

# TODO: Restore a picture from all the low light pictures with noise (`imgs_dark_noise`) by computing the "average
#  image" of them. Adjust the resulting brightness to the original image (using the `darkening_factor` and
#  `num_dark_noise_imgs`).
img_noise_stack_restored = np.copy(imgs_dark_noise[0])
for i in range(1, num_dark_noise_imgs):
    img_noise_stack_restored = cv2.add(img_noise_stack_restored, imgs_dark_noise[i])
    img_noise_stack_restored = asUint8(img_noise_stack_restored / 2)

img_noise_stack_restored = restoreBrightness(img_noise_stack_restored, darkening_factor)

imgs = imgs + [("Low light with noise 1 ...", imgs_dark_noise[0]),
               ("... Low light with noise " + str(num_dark_noise_imgs), imgs_dark_noise[-1]),
               ("Low light stack with noise restored", img_noise_stack_restored)]
plt.figure(figsize=(15, 8))
showImages(imgs, 3)
