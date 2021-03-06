#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import mediapipe as mp
import os.path
from scipy.spatial import Delaunay
import cv2
import dlib
import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from utils import *

# Load images.
image_dir = "E:/TU Braunschweig Data/Computer Vision and Machine Learning/Github_repo/sheet7/"
src_img, dst_img = [convertColorImagesBGR2RGB(cv2.imread(p)) for p in ("img/face1.jpg", "img/face2.jpg")]

if src_img.shape[1::-1] != dst_img.shape[1::-1]:
    print("Resolution of images does not match!")
    exit()

# Rescale images.
max_img_size = 512
size = src_img.shape[1::-1]
if max(size) > max_img_size:
    size = np.dot(size, max_img_size / max(size)).astype(np.int).tolist()
    src_img, dst_img = [cv2.resize(img, tuple(size)) for img in (src_img, dst_img)]
src_img = src_img[:, :-50]
dst_img = dst_img[:, :-50]
w, h = src_img.shape[1::-1]

# Find 68 landmark dlib face model.
predictor_file = "shape_predictor_68_face_landmarks.dat"
predictor_path = "E:/TU Braunschweig Data/Computer Vision and Machine " \
                 "Learning/Github_repo/sheet7//" + predictor_file
if not os.path.isfile(predictor_path):
    print("File not found: %s/nDownload from http://dlib.net/files/%s.bz2" % (predictor_path, predictor_file))
    exit()


#
# Task 1
#
# Complete the code for face morphing.

def weighted_average(img1, img2, alpha=.5):
    # TODO: Compute and return the weighted average (linear interpolation) of the two supplied images. Use the
    #  interpolation factor `alpha` such that the function returns `img1` if `alpha` == 0, `img2` if `alpha` == 1,
    #  and the interpolation otherwise.
    return cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0.0)


def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    # return a tuple of (x, y, w, h)
    return x, y, w, h


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords


def get_face_landmarks(image, predictor_path=predictor_path):
    # TODO: Use the `dlib` library for "Face Landmark Detection".
    #  The function shall return a numpy array of shape (68, 2), holding 68 landmarks as 2D integer pixel position.
    # initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_path)
    rect = detector(image, 1)
    for (i, rectangle) in enumerate(rect):
        shape = predictor(image, rectangle)
        shape = shape_to_np(shape)
    return shape


def weighted_average_points(src_points, dst_points, alpha=.5):
    # TODO: Compute and return the weighted average (linear interpolation) of the two sets of supplied points. Use
    #  the interpolation factor `alpha` such that the function returns `start_points` if `alpha` == 0, `end_points`
    #  if `alpha` == 1, and the interpolation otherwise.
    length = np.asarray([np.linalg.norm(dst_points[j] - src_points[j]) for j in range(len(src_points))])
    slope = [np.arctan(dst_points[j][1] - src_points[j][1]) / (dst_points[j][0] - src_points[j][0])
             for j in range(len(src_points))]
    x = np.round(alpha * length * np.cos(slope), decimals=0) + src_points[:, 0]
    y = np.round(alpha * length * np.sin(slope), decimals=0) + src_points[:, 1]
    indices = np.where(np.isnan(slope))
    x[indices[0]] = src_points[indices[0], 0]
    y[indices[0]] = src_points[indices[0], 1]
    return np.asarray(list(zip(x, y)))


# Warp each triangle from the `src_img` to the destination image.
def process_warp(src_img, result_img, tri_affines, dst_points, delaunay):
    # Generate x,y pixel coordinates
    pixel_coords = np.asarray([(x, y) for y in range(src_img.shape[0] - 2) for x in range(src_img.shape[1] - 2)],
                              np.uint32)
    # Indices to vertices. -1 if pixel is not in any triangle.
    triangle_indices = delaunay.find_simplex(pixel_coords)

    # # DEBUG visualization of triangle surfaces. triangle_surfaces = np.reshape(triangle_indices, (pixel_coords[-1][
    # 1] - pixel_coords[0][1] + 1, pixel_coords[-1][0] - pixel_coords[0][0] + 1)) showImage(triangle_surfaces.astype(
    # np.uint8))

    for simplex_index in range(len(delaunay.simplices)):
        coords = pixel_coords[triangle_indices == simplex_index]
        num_coords = len(coords)
        if num_coords > 0:
            out_coords = np.dot(tri_affines[simplex_index], np.vstack((coords.T, np.ones(num_coords))))
            x, y = coords.T
            result_img[y, x] = bilinear_interpolate(src_img, out_coords)


# Calculate the affine transformation matrix for each triangle vertex (x,y) from `dest_points` to `src_points`.
def gen_triangular_affine_matrices(vertices, src_points, dest_points):
    ones = [1, 1, 1]
    for tri_indices in vertices:
        src_tri = np.vstack((src_points[tri_indices, :].T, ones))
        dst_tri = np.vstack((dest_points[tri_indices, :].T, ones))
        mat = np.dot(src_tri, np.linalg.inv(dst_tri))[:2, :]
        yield mat


def warp_image(src_img, src_points, dest_points):
    result_img = src_img.copy()
    delaunay = Delaunay(dest_points)
    tri_affines = np.asarray(list(gen_triangular_affine_matrices(delaunay.simplices, src_points, dest_points)))
    process_warp(src_img, result_img, tri_affines, dest_points, delaunay)
    return result_img, delaunay


# Detect facial landmarks as control points for warps.
src_points, dst_points = [get_face_landmarks(img) for img in (src_img, dst_img)]


# # Tensorflow version of the model
# # The below function is replica of the github definition
# # https://github.com/google/mediapipe/blob/374f5e2e7e818bde5289fb3cffa616705cec6f73/mediapipe/python/solutions/drawing_utils.py
# def _normalized_to_pixel_coordinates(normalized_x: float, normalized_y: float, image_width: int, image_height: int):
#     """Converts normalized value pair to pixel coordinates."""
#     # Checks if the float value is between 0 and 1.
#     def is_valid_normalized_value(value: float) -> bool:
#         return (value > 0 or math.isclose(0, value)) and (value < 1 or
#                                                           math.isclose(1, value))
#
#     if not (is_valid_normalized_value(normalized_x) and
#             is_valid_normalized_value(normalized_y)):
#         return None
#     x_px = min(math.floor(normalized_x * image_width), image_width - 1)
#     y_px = min(math.floor(normalized_y * image_height), image_height - 1)
#     return x_px, y_px
#
#
# mp_face_mesh = mp.solutions.face_mesh
# with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2, min_detection_confidence=0.5) as face_mesh:
#     results_src, results_dst = [face_mesh.process(img).multi_face_landmarks for img in (src_img, dst_img)]
#
# for i, landmark in enumerate(results_src):
#     src_px = _normalized_to_pixel_coordinates(landmark.landmark.x, landmark.landmark.y, h, w)
#
# Task 2
#
# Improve morphing output.

# TODO: Extend the `src_points` and `dst_points` arrays such that also the surrounding parts of the images are warped.
#  Hint: The corners of both images shall note move when warping.
# Adding corners of image dimensions to the respective points. These points shall act as anchors for warping
src_points = np.append(src_points, [[1, 1]], axis=0)
src_points = np.append(src_points, [[w - 1, 1]], axis=0)
src_points = np.append(src_points, [[(w - 1) // 2, 1]], axis=0)
src_points = np.append(src_points, [[1, h - 1]], axis=0)
src_points = np.append(src_points, [[1, (h - 1) // 2]], axis=0)
src_points = np.append(src_points, [[w - 1, h - 1]], axis=0)
src_points = np.append(src_points, [[(w - 1) // 2, h - 1]], axis=0)
src_points = np.append(src_points, [[(w - 1), (h - 1) // 2]], axis=0)

dst_points = np.append(dst_points, [[1, 1]], axis=0)
dst_points = np.append(dst_points, [[w - 1, 1]], axis=0)
dst_points = np.append(dst_points, [[(w - 1) // 2, 1]], axis=0)
dst_points = np.append(dst_points, [[1, h - 1]], axis=0)
dst_points = np.append(dst_points, [[1, (h - 1) // 2]], axis=0)
dst_points = np.append(dst_points, [[w - 1, h - 1]], axis=0)
dst_points = np.append(dst_points, [[(w - 1) // 2, h - 1]], axis=0)
dst_points = np.append(dst_points, [[(w - 1), (h - 1) // 2]], axis=0)


def bilinear_interpolate(img, coords):
    # TODO: Implement bilinear interpolation. The function shall return an array of RGB values that correspond to the
    #  interpolated pixel colors in `img` at the positions in `coords`. As the coords are floating point values,
    #  use bilinear interpolation, such that the RGB color for each position in `coords` is interpolated from 4
    #  instead of just one pixels in `img`.
    # given that coords are floating point values we round off them.
    # coords = coords.T
    # new_pixels = []
    # for coord in coords:
    #     x = coord[0]
    #     y = coord[1]
    #     if math.isclose(x, math.ceil(x)) and math.isclose(y, math.ceil(y)):
    #         new_pixel = img[math.ceil(y), math.ceil(x), :]
    #         new_pixels.append(new_pixel)
    #     else:
    #         x1 = math.floor(coord[0])
    #         y1 = math.floor(coord[1])
    #         x2 = math.ceil(coord[0])
    #         y2 = math.ceil(coord[1])
    #         # pixel values in the square
    #         Ia = img[y1, x1, :]
    #         Ib = img[y2, x1, :]
    #         Ic = img[y2, x2, :]
    #         Id = img[y1, x2, :]
    #         wa = (x2 - x) * (y2 - y)
    #         wc = (x - x1) * (y - y1)
    #         wb = (x2 - x) * (y - y1)
    #         wd = (x - x1) * (y2 - y)
    #         Iab = Ia + (Ib - Ia) * (y - y1)
    #         Icd = Id + (Ic - Id) * (y - y1)
    #         # new_pixel = np.asarray(Iab + (Icd - Iab) * (x - x1), dtype=np.uint8)
    #         new_pixel = np.asarray(Ia * wa + Ib * wb + Ic * wc + Id * wd, dtype=np.uint8)
    #         new_pixels.append(new_pixel)
    # return np.asarray(new_pixels)
    return REPLACE_THIS([img[y, x] for i in range(coords.shape[1]) for x, y in (coords[:, i].astype(np.int),)])

# nearest neighbour approach
# REPLACE_THIS([img[y, x] for i in range(coords.shape[1]) for x, y in (coords[:, i].astype(np.int),)])


found_face_points = len(src_points) > 0 and len(dst_points) > 0

fig = plt.figure(figsize=(16, 8))
_, imgAxs1 = showImages([("img", src_img), ("Facial landmarks", src_img), ("Delaunay triangulation", src_img),
                         dst_img, dst_img, dst_img], 3, show_window_now=False, convertRGB2BGR=False)
imgAxs1[0].text(-30, h / 2, "src", rotation="vertical", va="center")
imgAxs1[3].text(-30, h / 2, "dst", rotation="vertical", va="center")

if found_face_points:
    imgAxs1[1].plot(src_points[:, 0], src_points[:, 1], 'o', markersize=3)
    imgAxs1[4].plot(dst_points[:, 0], dst_points[:, 1], 'o', markersize=3)

    alpha = .5
    points = weighted_average_points(src_points, dst_points, alpha)
    _, src_delaunay = warp_image(src_img, src_points, points)
    _, dst_delaunay = warp_image(dst_img, dst_points, points)

    imgAxs1[2].triplot(src_points[:, 0], src_points[:, 1], src_delaunay.simplices.copy())
    imgAxs1[5].triplot(dst_points[:, 0], dst_points[:, 1], dst_delaunay.simplices.copy())

imgs = []
num_imgs = 4
if found_face_points:
    for alpha in np.linspace(0, 1, num_imgs):
        print("progress: %.2f" % alpha)
        points = weighted_average_points(src_points, dst_points, alpha)
        src_face, _ = warp_image(src_img, src_points, points)
        dst_face, _ = warp_image(dst_img, dst_points, points)
        imgs.append((src_face, weighted_average(src_face, dst_face, alpha), dst_face))

imgs_to_show = [("src", src_img), ("avg", src_img), ("dst", dst_img)]
if found_face_points:
    imgs_to_show += [imgs[0][0], imgs[0][1], imgs[0][2]]

fig = plt.figure(figsize=(16, 8))
imgRefs2, imgAxs2 = showImages(imgs_to_show, 3, show_window_now=False, convertRGB2BGR=False, padding=(0, .1, 0, .05))

imgAxs2[0].text(-30, h / 2, "blend", rotation="vertical", va="center")
if found_face_points:
    imgAxs2[3].text(-30, h / 2, "warp + blend", rotation="vertical", va="center")


def updateImgs(percent):
    alpha = percent / 100
    imgRefs2[1].set_data(weighted_average(src_img, dst_img, alpha))
    if found_face_points:
        selectedImgs = imgs[int(round((num_imgs - 1) * alpha))]
        imgRefs2[3].set_data(selectedImgs[0])
        imgRefs2[4].set_data(selectedImgs[1])
        imgRefs2[5].set_data(selectedImgs[2])


ax_slider = plt.axes([.33, .01, .33, .05])
slider = Slider(ax=ax_slider, label='Percent', valmin=0, valmax=100, valinit=50, valstep=100 / num_imgs)
slider.on_changed(updateImgs)
updateImgs(50)

plt.show()
