import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def PLACEHOLDER_FLOW(frames):
    return np.array(
        [[[x, y] for x in np.linspace(-1, 1, frames[0].shape[1])] for y in np.linspace(-1, 1, frames[0].shape[0])])


PLACEHOLDER_FLOW_VISUALIZATION = cv2.imread('resources/example_flow_visualization.png')


#
# Task 1
#
# Implement utility functions for flow visualization.

def flowMapToBGR(flow_map):
    # Flow vector (X, Y) to angle
    h, w = flow_map.shape[:2]
    X, Y = flow_map[:, :, 0], flow_map[:, :, 1]
    angle = np.arctan2(Y, X) + np.pi
    magnitude = np.sqrt(X * X + Y * Y)
    # Angle and vector size to HSV color
    hsv = np.zeros((h, w, 3), np.uint8)
    # Sets image hue according to the optical flow direction
    hsv[..., 0] = angle * (180 / np.pi / 2)
    # Sets image saturation to maximum
    hsv[..., 1] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    # Sets image value according to the optical flow magnitude (normalized)
    hsv[..., 2] = 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


# TODO: Draw arrows depicting the provided `flow` on a 10x10 pixel grid.
#       You may use `cv2.arrowedLine(..)`.
# def drawArrows(img, flow, arrow_color=(0, 255, 0)):
#     outimg = img.copy()
#
#     # Turn grayscale to rgb if needed
#     if len(outimg.shape) == 2:
#         outimg = np.stack((outimg,) * 3, axis=2)
#
#     # Get start and end coordinates of the optical flow
#     flow_start = np.stack(np.meshgrid(range(flow.shape[1]), range(flow.shape[0])), 2)
#     flow_end = (flow[flow_start[:, :, 1], flow_start[:, :, 0], :1] * 3 + flow_start).astype(np.int32)
#
#     # Threshold values
#     norm = np.linalg.norm(flow_end - flow_start, axis=2)
#     norm[norm < 2] = 0
#
#     # Draw all the nonzero values
#     nz = np.nonzero(norm)
#     for i in range(0, len(nz[0]), 100):
#         y, x = nz[0][i], nz[1][i]
#         cv2.arrowedLine(outimg,
#                         pt1=tuple(flow_start[y, x]),
#                         pt2=tuple(flow_end[y, x]),
#                         color=arrow_color,
#                         thickness=1,
#                         tipLength=.2)
#     return outimg

# works with 10 x 10 pixels map
# # TODO: Draw arrows depicting the provided `flow` on a 10x10 pixel grid.
# #       You may use `cv2.arrowedLine(..)`.
def drawArrows(img, flow, arrow_color=(0, 255, 0)):
    out_img = img.copy()
    magnitude, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1], angleInDegrees=False)
    # magnitude = cv2.normalize(magnitude, None, 0, 10, cv2.NORM_MINMAX)
    for i in range(0, flow.shape[0], 10):
        for j in range(0, flow.shape[1], 10):
            increment_x, increment_y = (10, 10)
            if i + 10 > flow.shape[0]:
                increment_y = flow.shape[0] - i
            if j + 10 > flow.shape[1]:
                increment_x = flow.shape[1] - j
            avg_magnitude = np.mean(magnitude[i: i + increment_y, j: j + increment_x])
            avg_angle = np.mean(ang[i: i + increment_y, j: j + increment_x])
            flow_start = (j, i)
            flow_end = (int(j + avg_magnitude * np.cos(avg_angle))
                        if int(j + avg_magnitude * np.cos(avg_angle)) > 0 else 0,
                        int(i + avg_magnitude * np.sin(avg_angle))
                        if int(i + avg_magnitude * np.sin(avg_angle)) > 0 else 0)
            out_img = cv2.arrowedLine(out_img, flow_start, flow_end, color=arrow_color, tipLength=0.2)
    return out_img


# Calculate the angular error of an estimated optical flow compared to ground truth
def calculateAngularError(estimated_flow, groundtruth_flow):
    nom = groundtruth_flow[:, :, 0] * estimated_flow[:, :, 0] + groundtruth_flow[:, :, 1] * estimated_flow[:, :, 1] + \
          1.0
    denom = np.sqrt((groundtruth_flow[:, :, 0] ** 2 + groundtruth_flow[:, :, 1] ** 2 + 1.0) * (
            estimated_flow[:, :, 0] ** 2 + estimated_flow[:, :, 1] ** 2 + 1.0))
    return (1.0 / (estimated_flow.shape[0] * estimated_flow.shape[1])) * np.sum(np.arccos(np.clip(nom / denom, 0, 1)))


# Load a flow map from a file
def load_FLO_file(filename):
    if os.path.isfile(filename) is False:
        print("file does not exist %r" % str(filename))
    flo_file = open(filename, 'rb')
    magic = np.fromfile(flo_file, np.float32, count=1)
    if magic != 202021.25:
        print('Magic number incorrect. .flo file is invalid')
    w = np.fromfile(flo_file, np.int32, count=1)
    h = np.fromfile(flo_file, np.int32, count=1)
    # The float values for u and v are interleaved in row order, i.e., u[row0,col0], v[row0,col0], u[row0,col1],
    # v[row0,col1], ..., In total, there are 2*w*h flow values
    data = np.fromfile(flo_file, np.float32, count=2 * w[0] * h[0])
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))
    flo_file.close()
    # Some cleanup (remove cv-destroying large numbers)
    flow[np.sqrt(np.sum(flow ** 2, axis=2)) > 100] = 0
    return flow
