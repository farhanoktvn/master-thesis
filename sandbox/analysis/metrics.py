import cv2
import numpy as np
from skimage.metrics import (
    structural_similarity,
    peak_signal_noise_ratio,
)
from matplotlib import pyplot as plt

# Gradient Magnitude Similarity Deviation (GMSD)
# https://www4.comp.polyu.edu.hk/~cslzhang/IQA/GMSD/GMSD.htm

GMSD_FILTER = (
    np.array(
        [
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1],
        ]
    )
    / 3
)
GMSD_T = 170
GMSD_DOWN_STEP = 2
AVG_FILTER = np.ones((2, 2)) / 4


# Gradient Magnitude Similarity Deviation (GMSD)
def gmsd(img_a, img_b, with_map=False):
    img_a = cv2.filter2D(img_a, -1, AVG_FILTER)
    img_b = cv2.filter2D(img_b, -1, AVG_FILTER)

    img_a = img_a[::GMSD_DOWN_STEP, ::GMSD_DOWN_STEP]
    img_b = img_b[::GMSD_DOWN_STEP, ::GMSD_DOWN_STEP]

    img_a_x = cv2.filter2D(img_a, -1, GMSD_FILTER)
    img_a_y = cv2.filter2D(img_a, -1, GMSD_FILTER.T)
    img_b_x = cv2.filter2D(img_b, -1, GMSD_FILTER)
    img_b_y = cv2.filter2D(img_b, -1, GMSD_FILTER.T)

    gradient_map_a = np.sqrt(img_a_x**2 + img_a_y**2)
    gradient_map_b = np.sqrt(img_b_x**2 + img_b_y**2)

    quality_map = (2 * gradient_map_a * gradient_map_b + GMSD_T) / (
        gradient_map_a**2 + gradient_map_b**2 + GMSD_T
    )

    mu = np.mean(quality_map)
    length = quality_map.shape[0] * quality_map.shape[1]
    stdev = np.sqrt(np.sum((quality_map - mu) ** 2) / (length - 1))

    if with_map:
        return stdev, quality_map
    return stdev


# Peak Signal-to-Noise Ratio (PSNR)
def psnr(img_a, img_b):
    return peak_signal_noise_ratio(img_a, img_b)


# Structural Similarity (SSIM)
def ssim(img_a, img_b):
    return structural_similarity(img_a, img_b)
