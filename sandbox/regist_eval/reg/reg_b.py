import cv2
import numpy as np
from PIL import Image, ImageEnhance

from zep.feature.affine_ransac import Ransac
from zep.feature.affine_transform import Affine


LOWES_RATIO = 0.8
SHARP_FILTER = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
ENHANCER = ImageEnhance.Sharpness
ENHANCE_FACTOR = 1


class RegB:
    def __init__(self, mode="sift", K=3, threshold=1):
        self.mode = mode
        self.K = K
        self.threshold = threshold

    def extract_SIFT(self, img):
        if self.mode == "orb":
            sift = cv2.ORB_create(500)
        if self.mode == "sift":
            sift = cv2.SIFT_create()
        if self.mode == "akaze":
            sift = cv2.AKAZE_create()
        kp, desc = sift.detectAndCompute(img, None)

        kp = np.array([p.pt for p in kp]).T

        return kp, desc

    def match_SIFT(self, desc_s, desc_t):
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc_s, desc_t, k=2)
        fit_pos = np.array([], dtype=np.int32).reshape((0, 2))

        matches_num = len(matches)
        for i in range(matches_num):
            if matches[i][0].distance <= LOWES_RATIO * matches[i][1].distance:
                temp = np.array([matches[i][0].queryIdx, matches[i][0].trainIdx])
                fit_pos = np.vstack((fit_pos, temp))

        return fit_pos

    def affine_matrix(self, kp_s, kp_t, fit_pos):
        kp_s = kp_s[:, fit_pos[:, 0]]
        kp_t = kp_t[:, fit_pos[:, 1]]

        _, _, inliers = Ransac(self.K, self.threshold).ransac_fit(kp_s, kp_t)

        if inliers is None:
            return None
        else:
            kp_s = kp_s[:, inliers[0]]
            kp_t = kp_t[:, inliers[0]]

            A, t = Affine().estimate_affine(kp_s, kp_t)
            M = np.hstack((A, t))

            return M

    def warp_image(self, source, target, M):
        # rows, cols, _ = target.shape
        rows, cols = target.shape
        warp = cv2.warpAffine(source, M, (cols, rows))
        merge = target * 0.5 + warp * 0.5
        return warp, merge

    def enhance(self, img):
        # img = cv2.filter2D(img, ddepth=-1, kernel=SHARP_FILTER)
        # img = Image.fromarray(img)
        # img = ENHANCER(img).enhance(ENHANCE_FACTOR)
        return np.array(img, dtype=np.uint8)

    def register(self, img_a, img_b):
        img_a_cp = img_a.copy()
        img_a_cp = cv2.equalizeHist(img_a_cp)
        img_b_cp = img_b.copy()
        img_b_cp = cv2.equalizeHist(img_b_cp)

        # sharpen image b (target/spectral image)
        img_b_cp = self.enhance(img_b_cp)

        kp_s, desc_s = self.extract_SIFT(img_a_cp)
        kp_t, desc_t = self.extract_SIFT(img_b_cp)

        if desc_s is not None and desc_t is not None:
            fit_pos = self.match_SIFT(desc_s, desc_t)
        else:
            fit_pos = None

        if len(kp_s) != 0 and len(kp_t) != 0 and fit_pos is not None:
            M = self.affine_matrix(kp_s, kp_t, fit_pos)
        else:
            M = None

        if M is None:
            return np.array(None)
        else:
            # warp_img, merge_img = self.warp_image(img_a, img_b, M)
            # return warp_img
            # return warp_img, merge_img, M
            return M, img_b.shape[1], img_b.shape[0]
            # return M
