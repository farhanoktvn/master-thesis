import cv2
import numpy as np
from PIL import Image, ImageEnhance


LOWES_RATIO = 0.7
SHARP_FILTER = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
ENHANCER = ImageEnhance.Sharpness
ENHANCE_FACTOR = 1
ITER_NUM = 2000


class Affine:
    def create_test_case(self, outlier_rate=0):
        A = 4 * np.random.rand(2, 2) - 2

        t = 20 * np.random.rand(2, 1) - 10

        num = 1000

        outliers = int(np.round(num * outlier_rate))
        inliers = int(num - outliers)

        pts_s = 100 * np.random.rand(2, num)

        pts_t = np.zeros((2, num))

        pts_t[:, :inliers] = np.dot(A, pts_s[:, :inliers]) + t

        pts_t[:, inliers:] = 100 * np.random.rand(2, outliers)

        rnd_idx = np.random.permutation(num)
        pts_s = pts_s[:, rnd_idx]
        pts_t = pts_t[:, rnd_idx]

        return A, t, pts_s, pts_t

    def estimate_affine(self, pts_s, pts_t):
        pts_num = pts_s.shape[1]

        M = np.zeros((2 * pts_num, 6))

        for i in range(pts_num):
            temp = [[pts_s[0, i], pts_s[1, i], 0, 0, 1, 0], [0, 0, pts_s[0, i], pts_s[1, i], 0, 1]]
            M[2 * i : 2 * i + 2, :] = np.array(temp, dtype=object)

        b = pts_t.T.reshape((2 * pts_num, 1))

        try:
            theta = np.linalg.lstsq(M, b, rcond=None)[0]

            A = theta[:4].reshape((2, 2))
            t = theta[4:]
        except np.linalg.linalg.LinAlgError:
            A = None
            t = None

        return A, t


class Ransac:
    def __init__(self, K=3, threshold=1):
        self.K = K
        self.threshold = threshold

    def residual_lengths(self, A, t, pts_s, pts_t):
        if not (A is None) and not (t is None):
            # Calculate estimated points:
            # pts_esti = A * pts_s + t
            pts_e = np.dot(A, pts_s) + t

            # Calculate the residual length between estimated points
            # and target points
            diff_square = np.power(pts_e - pts_t, 2)
            residual = np.sqrt(np.sum(diff_square, axis=0))
        else:
            residual = None

        return residual

    def ransac_fit(self, pts_s, pts_t):
        af = Affine()
        inliers_num = 0

        A = None
        t = None
        inliers = None

        for i in range(ITER_NUM):
            if pts_s.shape[1] <= 0:
                pass
            else:
                idx = np.random.randint(0, pts_s.shape[1], (self.K, 1))

                A_tmp, t_tmp = af.estimate_affine(pts_s[:, idx], pts_t[:, idx])

                residual = self.residual_lengths(A_tmp, t_tmp, pts_s, pts_t)

                if not (residual is None):
                    inliers_tmp = np.where(residual < self.threshold)

                    inliers_num_tmp = len(inliers_tmp[0])

                    if inliers_num_tmp > inliers_num:
                        inliers_num = inliers_num_tmp

                        inliers = inliers_tmp
                        A = A_tmp
                        t = t_tmp
                else:
                    pass

        return A, t, inliers


class AffineRegistration:
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
        img = cv2.filter2D(img, ddepth=-1, kernel=SHARP_FILTER)
        img = Image.fromarray(img)
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
            warp_img, merge_img = self.warp_image(img_a, img_b, M)
            return warp_img, M
            # return warp_img, merge_img, M
            # return M, img_b.shape[1], img_b.shape[0]
            # return M
