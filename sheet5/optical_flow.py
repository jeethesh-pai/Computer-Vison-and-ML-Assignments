#!/usr/bin/env python3
# -*- coding: utf-8 -*-



import numpy as np
import cv2

from flow_utils import *
from utils import *



#
# Task 2
#
# Implement Lucas-Kanade or Horn-Schunck Optical Flow.



# TODO: Implement Lucas-Kanade Optical Flow.
#
# Parameters:
# frames: the two consecutive frames
# Ix: Image gradient in the x direction
# Iy: Image gradient in the y direction
# It: Image gradient with respect to time
# kernel_size: kernel size
# eigen_threshold: threshold for determining if the optical flow is valid when performing Lucas-Kanade
# returns the Optical flow based on the Lucas-Kanade algorithm
def LucasKanadeFlow(frames, Ix, Iy, It, kernel_size, eigen_threshold = 0.01):
    return PLACEHOLDER_FLOW(frames)
    #
    # ???
    #



# TODO: Implement Horn-Schunck Optical Flow.
#
# Parameters:
# frames: the two consecutive frames
# Ix: Image gradient in the x direction
# Iy: Image gradient in the y direction
# It: Image gradient with respect to time
# window_size: the number of points taken in the neighborhood of each pixel
# max_iterations: maximum number of iterations allowed until convergence of the Horn-Schuck algorithm
# epsilon: the stopping criterion for the difference when performing the Horn-Schuck algorithm
# returns the Optical flow based on the Horn-Schunck algorithm
def HornSchunckFlow(frames, Ix, Iy, It, max_iterations = 1000, epsilon = 0.002):
    return PLACEHOLDER_FLOW(frames)
    #
    # ???
    #



# Load image frames
frames = [  cv2.imread("resources/frame1.png"),
            cv2.imread("resources/frame2.png")]

# Load ground truth flow data for evaluation
flow_gt = load_FLO_file("resources/groundTruthOF.flo")

# Grayscales
gray = [(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255.0).astype(np.float64) for frame in frames]

# Get derivatives in X and Y
xdk = np.array([[-1.0, 1.0],[-1.0, 1.0]])
ydk = xdk.T
fx =  cv2.filter2D(gray[0], cv2.CV_64F, xdk) + cv2.filter2D(gray[1], cv2.CV_64F, xdk)
fy = cv2.filter2D(gray[0], cv2.CV_64F, ydk) + cv2.filter2D(gray[1], cv2.CV_64F, ydk)

# Get time derivative in time (frame1 -> frame2)
tdk1 =  np.ones((2,2))
tdk2 = tdk1 * -1
ft = cv2.filter2D(gray[0], cv2.CV_64F, tdk2) + cv2.filter2D(gray[1], cv2.CV_64F, tdk1)

# Ground truth flow
plt.figure(figsize=(5, 8))
showImages([("Groundtruth flow", flowMapToBGR(flow_gt)),
            ("Groundtruth field", drawArrows(frames[0], flow_gt)) ], 1, False)

# Lucas-Kanade flow
flow_lk = LucasKanadeFlow(gray, fx, fy, ft, [15, 15])
error_lk = calculateAngularError(flow_lk, flow_gt)
plt.figure(figsize=(5, 8))
showImages([("LK flow - angular error: %.3f" % error_lk, flowMapToBGR(flow_lk)),
            ("LK field", drawArrows(frames[0], flow_lk)) ], 1, False)

# Horn-Schunk flow
flow_hs = HornSchunckFlow(gray, fx, fy, ft)
error_hs = calculateAngularError(flow_hs, flow_gt)
plt.figure(figsize=(5, 8))
showImages([("HS flow - angular error %.3f" % error_hs, flowMapToBGR(flow_hs)),
            ("HS field", drawArrows(frames[0], flow_hs)) ], 1, False)

plt.show()
