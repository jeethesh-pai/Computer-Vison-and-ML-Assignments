import numpy as np
import cv2
from flow_utils import *
from utils import *
import tqdm


# Task 2
#
# Implement Lucas-Kanade or Horn-Schunck Optical Flow.


# TODO: Implement Lucas-Kanade Optical Flow.
def LucasKanadeFlow(frames, Ix, Iy, It, kernel_size, eigen_threshold=0.01):
    """
    :param frames: the two consecutive frames
    :param Ix: Image gradient in the x direction
    :param Iy: Image gradient in the y direction
    :param It: Image gradient with respect to time
    :param kernel_size: kernel size
    :param eigen_threshold: threshold for determining if the optical flow is valid when performing Lucas-Kanade
    :return: returns the Optical flow based on the Lucas-Kanade algorithm
    """
    u = np.zeros_like(Ix)
    v = np.zeros_like(Ix)
    print('calculating Lukas Kanade Optical flow ... ')
    for i in tqdm.tqdm(range(int(kernel_size[0] / 2), Ix.shape[0])):
        for j in range(int(kernel_size[1] / 2), Ix.shape[1]):
            ix_kernel = Ix[i - int(kernel_size[0] / 2): i + int(kernel_size[0] / 2 + 1),
                           j - int(kernel_size[1] / 2): j + int(kernel_size[1] / 2) + 1]
            iy_kernel = Iy[i - int(kernel_size[0] / 2): i + int(kernel_size[0] / 2 + 1),
                           j - int(kernel_size[1] / 2): j + int(kernel_size[1] / 2) + 1]
            it_kernel = It[i - int(kernel_size[0] / 2): i + int(kernel_size[0] / 2) + 1,
                           j - int(kernel_size[1] / 2): j + int(kernel_size[1] / 2) + 1]
            ix_ix = np.sum(cv2.multiply(ix_kernel, ix_kernel))
            ix_iy = np.sum(cv2.multiply(ix_kernel, iy_kernel))
            iy_iy = np.sum(cv2.multiply(iy_kernel, iy_kernel))
            A = np.asarray([[ix_ix, ix_iy],
                            [ix_iy, iy_iy]])
            it_ix = - np.sum(cv2.multiply(ix_kernel, it_kernel))
            it_iy = - np.sum(cv2.multiply(iy_kernel, it_kernel))
            At = np.asarray([[it_ix],
                             [it_iy]])
            res = np.dot(np.linalg.pinv(A), At)
            u[i, j] = res[0]
            v[i, j] = res[1]
    return np.dstack([u, v])


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
def HornSchunckFlow(frames, Ix, Iy, It, max_iterations=1000, epsilon=0.002):
    """
    :param frames: the two consecutive frames
    :param Ix: Image gradient in the x direction
    :param Iy: Image gradient in the y direction
    :param It: Image gradient with respect to time
    :param max_iterations: maximum number of iterations allowed until convergence of the Horn-Schuck algorithm
    :param epsilon: the stopping criterion for the difference when performing the Horn-Schuck algorithm
    :return: returns the Optical flow based on the Horn-Schunck algorithm
    """
    u = np.zeros_like(Ix)
    v = np.zeros_like(Ix)
    lambda_factor = 0.15  # Euler lagrange factor in Horn Schunk optimization
    lambda_matrix = lambda_factor * np.ones_like(Ix)
    print('calculating Horn Schunk Optical flow....')
    for iteration in tqdm.tqdm(range(1, max_iterations)):
        avg_mask = [[1/12, 1/6, 1/12],
                    [1/6, 0, 1/6],
                    [1/12, 1/6, 1/12]]
        avg_mask = np.asarray(avg_mask, dtype=np.float64)
        u_avg = cv2.filter2D(u, -1, avg_mask)
        v_avg = cv2.filter2D(v, -1, avg_mask)
        # By solving the constraint equation from the reference
        # https://www.caam.rice.edu/~zhang/caam699/opt-flow/horn81.pdf pg 8
        denominator = lambda_matrix ** 2 + Ix ** 2 + Iy ** 2
        numerator = np.multiply(Ix, u_avg) + np.multiply(Iy, v_avg) + It
        u_new = u_avg - np.divide(np.multiply(Ix, numerator), denominator)
        v_new = v_avg - np.divide(np.multiply(Iy, numerator), denominator)
        if np.all(u_new - u < epsilon) and np.all(v_new - v) < epsilon:
            break
        u = u_new
        v = v_new
    return np.dstack([u, v])


# Load image frames
frames = [cv2.imread("resources/frame1.png"), cv2.imread("resources/frame2.png")]

# Load ground truth flow data for evaluation
flow_gt = load_FLO_file("resources/groundTruthOF.flo")
# Grayscale
gray = [(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) / 255.0).astype(np.float64) for frame in frames]

# Get derivatives in X and Y
xdk = np.array([[-1.0, 1.0], [-1.0, 1.0]])
ydk = xdk.T
fx = cv2.filter2D(gray[0], cv2.CV_64F, xdk) + cv2.filter2D(gray[1], cv2.CV_64F, xdk)
fy = cv2.filter2D(gray[0], cv2.CV_64F, ydk) + cv2.filter2D(gray[1], cv2.CV_64F, ydk)

# Get time derivative in time (frame1 -> frame2)
tdk1 = np.ones((2, 2))
tdk2 = tdk1 * -1
ft = cv2.filter2D(gray[0], cv2.CV_64F, tdk2) + cv2.filter2D(gray[1], cv2.CV_64F, tdk1)

# Ground truth flow
plt.figure(figsize=(5, 8))
showImages([("Ground truth flow", flowMapToBGR(flow_gt)),
            ("Ground truth field", drawArrows(frames[0], flow_gt))], 1, False)

# Lucas-Kanade flow

flow_lk = LucasKanadeFlow(gray, fx, fy, ft, [6, 6])
error_lk = calculateAngularError(flow_lk, flow_gt)
print(error_lk)

plt.figure(figsize=(5, 8))
showImages([("LK flow - angular error: %.3f" % error_lk, flowMapToBGR(flow_lk)),
            ("LK field", drawArrows(frames[0], flow_lk))], 1, False)

# Horn-Schunk flow
flow_hs = HornSchunckFlow(gray, fx, fy, ft)
error_hs = calculateAngularError(flow_hs, flow_gt)
plt.figure(figsize=(5, 8))
showImages([("HS flow - angular error %.3f" % error_hs, flowMapToBGR(flow_hs)),
            ("HS field", drawArrows(frames[0], flow_hs))], 1, False)

plt.show()
