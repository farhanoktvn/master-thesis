import numpy as np


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
