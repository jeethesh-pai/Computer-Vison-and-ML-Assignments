#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import cv2
import matplotlib.pyplot as plt

from utils import *



#
# Task 1
#
# Stereo rectification.

img1 = cv2.imread('img/apt1.jpg')
img2 = cv2.imread('img/apt2.jpg')

h, w, _ = img1.shape

sift = cv2.SIFT_create()
kp1, fd1 = sift.detectAndCompute(img1, None)
kp2, fd2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher_create(cv2.NORM_L2)
matches = bf.knnMatch(fd1, fd2, k=2)

best_to_secondBest_ratio = .6
good_matches = []
for m1, m2 in matches:
    if m1.distance < best_to_secondBest_ratio * m2.distance:
        good_matches.append(m1)

# Visualization of the "good" SIFT features.
img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2)

if len(good_matches) > 10:
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

    # TODO: Find the fundamental matrix from the good feature points.
    #  Use the RANSAC algorithm to further sort out outliers based on epipolar geometry.
    ransac_reproj_threshold = 5
    # F, inlier_mask = ...
    #
    # ???
    #

    # TODO: From the "good" matches, filter out the SIFT features that were recognized as outliers (inlierMask) in the previous step.
    approved_matches = REPLACE_THIS(good_matches)
    #
    # ???
    #

    # Visualization of the "approved" SIFT features.
    img_matches2 = cv2.drawMatches(img1, kp1, img2, kp2, approved_matches, None, flags=2)

    num_samples = 10
    colors = cv2.applyColorMap(np.uint8(np.linspace(0, 256, num=num_samples, endpoint=False)), cv2.COLORMAP_JET).reshape(-1, 3).tolist()

    # TODO: Compute the epipolar lines in `img2` for the following randomly selected points in `img1`.
    epi_points1 = np.int32([kp1[m.queryIdx].pt for m in np.random.choice(approved_matches, num_samples)]).reshape(-1,2)
    epi_lines1 = REPLACE_THIS([[[0, 1, -h/2]]])
    #
    # ???
    #

    # TODO: For each epipolar line (`epi_lines1`) in `img2`, compute the pixel position of any point on that line.
    #  Then, for these `epi_points2`, compute the corresponding epipolar lines in `img1` (`epi_lines2`).
    epi_points2 = REPLACE_THIS([[w//2, h//2]])
    epi_lines2 = REPLACE_THIS([[[0, 1, -h//2]]])
    #
    # ???
    #

    # Visualize selected points and corresponding epipolar lines in both directions (img1 -> img2 and vice versa)
    img1_epilines = img1.copy()
    img2_epilines = img2.copy()
    for img, points in ((img1_epilines, epi_points1), (img2_epilines, epi_points2)):
        for i in range(len(points)):
            cv2.circle(img, list(points[i]), 3, colors[i], 2)
    for img, lines in ((img2_epilines, epi_lines1), (img1_epilines, epi_lines2)):
        for i in range(len(lines)):
            a, b, c = lines[i][0] # ax + by + c = 0
            cv2.line(img, (0, int(-c//b)), (w, int((-a*w-c)//b)), colors[i], 2) # y = (-ax - c) / b

    # TODO: Rectify both images to align their epipolar lines.
    #  Compute the two homography matricies to perspectively warp both images accordingly.
    #  You can assume the images to be lens distortion free, i.e. you don't need to do a camera calibration.
    img1_rectified, img2_rectified = [REPLACE_THIS(img1), REPLACE_THIS(img2)]
    #
    # ???
    #

    img_rectified = np.concatenate((img1_rectified, img2_rectified), axis=1)
    for y in range(15, img_rectified.shape[0], 30):
        cv2.line(img_rectified, (0, y), (img_rectified.shape[1]-1, y), (100, 100, 255), 2)

    showImages([
        ("SIFT (good)", img_matches, (2, 1)), ("SIFT (approved)", img_matches2, (2, 1)),
        ("epipolar (img1)", img1_epilines), ("epipolar (img2)", img2_epilines),
        ("Rectified", img_rectified, (2, 1))], 4)
